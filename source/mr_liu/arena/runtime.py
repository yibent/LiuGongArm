"""Single simulation-thread skill orchestration over Arena's official IK action."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from queue import Queue, Empty
from uuid import uuid4

import numpy as np
from mr_liu.arena.arrays import numpy_data, pose_matrix
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import torch

from mr_liu.arena.perception import PerceptionBridge
from mr_liu.arena.holding import holding_measurement, HoldMonitor
from mr_liu.arena.free_space import choose_free_support, placement_is_free, support_grid
from mr_liu.arena.visual_refs import load_reference
from mr_liu.arena.failure import failure_feedback, LocalizationFailure
from mr_liu.arena.failure import PlacementSpaceUnavailable
from mr_liu.arena.placement_geometry import cell_fit
from mr_liu.arena.orientation import endpoint_vector, placement_rotations, transformed_payload, placement_pose
from mr_liu.arena.instances import InstanceConflict
from mr_liu.vision.worker import VisionWorker
from mr_liu.arena.cascade import run_cascade
from mr_liu.arena.fast import fast_pick_place, fast_place_held
from mr_liu.arena.contracts import ManipulationRequest, model_grasp_to_tcp, placement_to_tcp
from mr_liu.grasp.backends.graspgenx import ZmqGraspGenXTransport
from mr_liu.grasp.transforms import invert_transform, transform_points
from mr_liu.place.anyplace import AnyPlaceClient



class ArenaRuntime:
    def __init__(self, env, task, config, output):
        self.wrapped = env
        self.env = env.unwrapped
        self.runtime_id = uuid4().hex
        self.task, self.config = task, config
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.stop_requested = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose-inference")
        self.frames = {}; self.snapshot = {}; self.sequence = 0
        self.snapshot_requests = Queue()
        self.phase = "initializing"; self.current = None; self.held = None
        self.held_context = None
        self.held_geometry = None
        self.orientation_binding = None
        self.hold_monitor = HoldMonitor()
        self.observed_entities = {}
        self.prepared_clouds = {}
        self.body_entities = {name: {'name': name, 'prim_path': body.cfg.prim_path.replace(
            self.env.scene.env_regex_ns, self.env.scene.env_prim_paths[0])}
            for name, body in self.env.scene.rigid_objects.items()}
        self.bus_context = {}
        self.perception = PerceptionBridge(Path(__file__).resolve().parents[3]/'output/perception', config['vision']['service_url'])
        self.vision_worker = VisionWorker(self.perception.request)
        self.visual_result = None
        self.observing = False
        self.last_track = 0
        self.tracking = None
        self.tracking_stop = threading.Event()
        self.gripper = 1.
        self.goal = self.tcp_pose()
        self.events = []
        self.last_result = None
        self.selected_target = None
        self.target_name = None; self.initial_z = 0.; self.max_lift = 0.
        for _ in range(50): self.tick(check_stop=False)
        self.phase = "idle"
        self.refresh_snapshot()

    def tcp_pose(self):
        data = self.env.scene["ee_frame"].data
        return pose_matrix(numpy_data(data.target_pos_w)[0, 0],
                           numpy_data(data.target_quat_w)[0, 0])

    def object_pose(self, name):
        data = self.env.scene[name].data
        return pose_matrix(numpy_data(data.root_pos_w)[0], numpy_data(data.root_quat_w)[0])

    def locate(self, label, **vision_options):
        """Visual geometry first. Physical IDs are evaluation witnesses only."""
        points = self.cloud(label, associate=True, **vision_options)
        observation = self.visual_result
        name = observation['physical_witness']['instance_id']
        votes = observation['physical_witness']['votes']
        row = {**self.body_entities[name], 'label': observation['label'],
               'visual_ref': next((r['ref'] for r in observation.get('references', []) if r['kind'] == 'object'), None)}
        # Freeze the evaluator's body-frame shape at observation time. A grasp
        # can move/rotate the part before closure; a predicted held cloud must
        # not be relabelled as measured geometry in that later physical frame.
        row['evaluation_points_object'] = transform_points(invert_transform(self.object_pose(name)),points).tolist()
        self.observed_entities[name] = row
        self.prepared_clouds[name] = (points, observation)
        self.event('instance_observed', label=row['label'], instance_id=name,
                   observation_ref=observation['request_id'], witness_votes=votes,
                   association_source='model_mask_to_rendered_physical_instance')
        return row

    def locate_destination(self, request):
        if not request.destination: return None
        options = {'vision_mode': 'slow', 'slow_provider': 'sam3'} if request.placement_selection in {'auto', 'free_space'} else {}
        if request.destination_ref: options['visual_ref'] = request.destination_ref
        if request.cell_ref:
            cell = self.perception.resolve_reference(request.cell_ref)
            if cell.get('kind') != 'cell':
                raise ValueError('cell_ref must come from an observed grid cell')
            options.update(visual_ref=cell['container_ref'], inspect='grid')
        return self.locate(request.destination, **options)

    def placement_options(self, request):
        with np.load(self.perception.root/self.visual_result['request_id']/'frames.npz', allow_pickle=False) as frames:
            camera = next((key[:-2] for key in frames.files if key.endswith('_K')), None)
            view = {'K': frames[camera+'_K'].copy(), 'T': frames[camera+'_T'].copy()} if camera else None
        region = None
        if request.region_ref:
            row, directory = load_reference(self.perception.root, request.region_ref, self.perception.scene_id)
            with np.load(directory/'frames.npz', allow_pickle=False) as previous:
                region = {'box': row['box'], 'view': {'K': previous[row['camera']+'_K'].copy(),
                          'T': previous[row['camera']+'_T'].copy()}}
        return {'preference': request.placement_preference, 'view': view, 'region': region}

    def placement_support(self, request, destination, child, preferred):
        parent = self.cloud(destination['name'])
        centre = np.r_[np.quantile(parent[:, :2], [.02, .98], axis=0).mean(axis=0), np.quantile(parent[:, 2], .95)]
        destination['yaw_delta_rad'] = 0.
        destination.pop('selected_position_world_m', None)
        if request.cell_ref:
            selected = self.perception.resolve_reference(request.cell_ref)
            grid = self.visual_result.get('geometry', {})
            if grid.get('kind') != 'grid':
                raise PlacementSpaceUnavailable('需要重新观察料箱格网。')
            cell = next((row for row in grid.get('cells', []) if row['row']==selected['row'] and row['column']==selected['column']), None)
            if not cell or cell['occupancy'] != 'empty':
                raise PlacementSpaceUnavailable('指定格位被占用或底面尚不可见，需要重新观察或整理。')
            axes = np.asarray(grid['basis_xy'])
            projected = np.asarray(child)[:,:2]@axes
            extent = np.ptp(np.quantile(projected,[.02,.98],axis=0),axis=0)
            if np.any(extent+.004 > np.asarray(cell['interior_size_m'])):
                raise PlacementSpaceUnavailable('当前物体朝向的尺寸不适合指定格位，需要调整朝向或重新选择位置。')
            centre = np.asarray(cell['position_m'])
            destination.update(selected_position_world_m=centre.tolist(), grid_cell=cell,
                grid_basis_xy=grid['basis_xy'], yaw_delta_rad=0.)
            # This retained measured shape is used only by the simulator's
            # independent physical evaluator after release, never for planning.
            destination['evaluation_points_object'] = self.observed_entities[self.target_name]['evaluation_points_object']
            self.event('grid_cell_selected', observation_ref=self.visual_result['request_id'],
                position_world_m=centre.tolist(), row=cell['row'], column=cell['column'],
                footprint_radius_m=float(np.linalg.norm(cell['interior_size_m'])/2), source=grid['source'])
            return centre
        if request.placement_selection == 'center': return centre
        scene = self.perception.scene_cloud(self.visual_result)
        try:
            tool_offset = self.tcp_pose()[:2, 3]-np.quantile(child[:, :2], [.02, .98], axis=0).mean(0) if self.holding_status()['verified'] else None
            centre, details = choose_free_support(parent, scene, child, preferred, tool_offset=tool_offset, **self.placement_options(request))
        except RuntimeError:
            # A narrow pedestal can support an overhanging part; full footprint
            # containment is required for free-space packing, not all placement.
            size = np.ptp(np.quantile(parent[:, :2], [.02, .98], axis=0), axis=0)
            obstacles = scene[(np.linalg.norm(scene[:, :2]-centre[:2], axis=1) < .07) &
                              (scene[:, 2] > centre[2]+.009) & (scene[:, 2] < centre[2]+.12)]
            if request.placement_selection == 'auto' and not request.region_ref and min(size) < .08 and not len(obstacles):
                self.event('support_center_selected', reason='narrow_unoccupied_support_allows_overhang')
                return centre
            raise
        destination.update(selected_position_world_m=centre.tolist(), yaw_delta_rad=details['yaw_delta_rad'])
        self.event('free_space_selected', observation_ref=self.visual_result['request_id'], **details)
        return centre

    def remember_hold(self, row, geometry=None):
        self.held = row['name']
        self.held_geometry = geometry
        if geometry is not None:
            geometry['tcp_at_grasp'] = self.tcp_pose().copy()
            if self.orientation_binding:
                geometry['orientation'] = {**self.orientation_binding,
                    'axis_tcp': (self.tcp_pose()[:3,:3].T @ self.orientation_binding['axis_world']).tolist()}
        robot = self.env.scene['robot']
        fingers, _ = robot.find_joints('panda_finger_joint.*')
        self.held_context = {'instance_id': self.held, 'label': row['label'],
            'opening_at_grasp_m': float(numpy_data(robot.data.joint_pos)[0, fingers].sum()),
            'grasp_command_id': self.current, 'initial_z': self.initial_z,
            'max_lift_m': self.max_lift,
            'tcp_to_object_at_grasp': (invert_transform(self.tcp_pose()) @ self.object_pose(self.held)).tolist()}

    def holding_status(self):
        if not self.held_context or self.held != self.held_context['instance_id']:
            return {'verified': False, 'instance_id': self.held}
        robot = self.env.scene['robot']
        fingers, _ = robot.find_joints('panda_finger_joint.*')
        measured = holding_measurement(self.held_context['tcp_to_object_at_grasp'],
            invert_transform(self.tcp_pose()) @ self.object_pose(self.held),
            float(numpy_data(robot.data.joint_pos)[0, fingers].sum()),
            self.held_context.get('opening_at_grasp_m'))
        return {**self.held_context, **measured}

    def clear_hold(self):
        self.held = None
        self.held_context = None
        self.held_geometry = None

    def held_cloud(self):
        """Retain measured geometry in the tool frame while the grasp persists."""
        if not self.holding_status()['verified']:
            raise RuntimeError('夹持状态已改变，不能沿用持物几何。')
        if self.held_geometry is not None:
            self.event('held_geometry_reused', observation_ref=self.held_geometry['observation_ref'],
                       source='prior_rgbd_in_gripper_frame', points=len(self.held_geometry['points_tcp']))
            return transform_points(self.tcp_pose(), self.held_geometry['points_tcp'])
        points = self.cloud(self.held)
        self.held_geometry = {'points_tcp': transform_points(invert_transform(self.tcp_pose()), points),
                              'observation_ref': self.visual_result['request_id']}
        self.event('held_geometry_recorded', observation_ref=self.held_geometry['observation_ref'], points=len(points))
        return points

    def capabilities(self):
        return {"robot": "franka_panda", "execution": "isaaclab_arena.franka_ik",
                "skills": ["status", "capabilities", "select_target", "perceive", "grasp", "pick_place", "place_held", "stop", "hold", "home"],
                "grasp": {"backend": "official_pick_place_or_graspgenx", "ready": True},
                "placement": {"backend": "official_pick_place_or_anyplace", "relations": ["on"], "place_held": True,
                    "selection": ["auto", "center", "free_space"], "preferences": ["nearest", "left", "right", "near", "far", "center", "compact"],
                    "visual_references": True, "region_reference": True,
                    "cell_reference": True, "endpoint_reorientation": True,
                    "orientation_input": {"axis_ref":"inspect_object axis_ref", "endpoint":[0,1], "direction":["up","down"]}},
                "routing": "fast_first_then_models_once; complex_tasks_use_models_directly",
                "vla_loaded": False, "perception_source": "task_routed_rgbd",
                "vision": {"architecture": "task_routed_fast_slow", "visual_tracking": True,
                    "collection_observation": True, "spatial_groups": True,
                    "image_box_grounding": True,
                    "geometry_inspection": ["axis", "grid"],
                    "grid_cell_placement": True,
                    "persistent_memory": True, "fast": ["yoloe_text", "yoloe_visual", "sam2_tiny", "lk"],
                    "slow_localizer": self.config['vision'].get('slow_localizer'),
                    "scene_description": "optional_florence2", "cloud_reasoning": "busagent_on_demand"},
                "frame": "world", "quaternion": "xyzw", "units": "metres",
                "configured_label_required": False,
                "target_source": "image_model_mask_and_rgbd",
                "objects": [{k:v for k,v in row.items() if k not in {'evaluation_points_object','orientation_evaluation'}} for row in self.observed_entities.values()],
                "destinations": [{k:v for k,v in row.items() if k not in {'evaluation_points_object','orientation_evaluation'}} for row in self.observed_entities.values()]}

    def refresh_snapshot(self):
        robot = self.env.scene["robot"]
        tcp = self.tcp_pose()
        self.snapshot = {"ready": True, "robot": "franka_panda", "phase": self.phase,
                         "runtime_id": self.runtime_id,
                         "sequence": self.sequence, "timestamp": time.time(), "capabilities": self.capabilities(),
                         "held_object": self.held, "command_id": self.current,
                         "holding": self.holding_status(),
                         "tcp_pose_world": tcp.tolist(), "prompt": self.selected_target,
                         "last_result": self.last_result,
                         "vision": self.visual_result,
                         "visual_candidates": list(self.perception.references.values())[-64:],
                         "world": self.perception.world.snapshot(),
                         "visual_tracking": {"enabled": self.tracking is not None,
                             "target": self.tracking['label'] if self.tracking else None},
                         "motion": {"mode": "hold" if self.phase == "idle" else "moving",
                                    "active_command_id": self.current,
                                    "joint_positions_deg": dict(zip(robot.joint_names[:7], np.rad2deg(numpy_data(robot.data.joint_pos)[0, :7]).tolist())),
                                    "tool_position_world_m": tcp[:3, 3].tolist(),
                                    "last_command": self.last_result},
                         "last_event": self.events[-1] if self.events else None}

    def tick(self, *, check_stop=True):
        if check_stop and self.stop_requested.is_set():
            self.goal = self.tcp_pose()
            raise InterruptedError("Stopped; gripper state preserved")
        robot = self.env.scene["robot"].data
        base = pose_matrix(numpy_data(robot.root_pos_w)[0], numpy_data(robot.root_quat_w)[0])
        target = invert_transform(base) @ self.goal
        action = np.r_[target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_quat(), self.gripper]
        self.wrapped.step(torch.as_tensor(action[None], dtype=torch.float32, device=self.env.device))
        self.sequence += 1
        if self.target_name:
            self.max_lift = max(self.max_lift, float(self.object_pose(self.target_name)[2, 3] - self.initial_z))
        if self.sequence % 6 == 0:
            for key, camera_name in [("scene", "scene_camera"), ("side", "side_camera"), ("wrist", "wrist_camera")]:
                rgb = self.env.scene[camera_name].data.output["rgb"][0, :, :, :3].detach().cpu().numpy()
                picture = Image.fromarray(rgb.astype(np.uint8))
                for view in (self.visual_result or {}).get('views', []):
                    if view['camera'] == camera_name and view.get('box') and self.sequence-view['sequence'] < 90:
                        draw = ImageDraw.Draw(picture)
                        draw.rectangle(view['box'], outline='#efc651', width=2)
                        draw.text((view['box'][0], max(0,view['box'][1]-14)), view['label']+' / '+str(view['sequence']), fill='#efc651')
                encoded = BytesIO(); picture.save(encoded, "JPEG", quality=80)
                self.frames[key] = encoded.getvalue()
                if not hasattr(self, 'frame_packets'): self.frame_packets = {}
                self.frame_packets[key] = (self.frames[key], self.sequence, time.time())
            self.refresh_snapshot()
            if self.hold_monitor.update(bool(self.snapshot['holding'].get('verified')), self.phase, self.gripper < 0):
                self.goal = self.tcp_pose()
                self.event('holding_lost', during=self.phase, measurement=self.snapshot['holding'])
                raise RuntimeError('夹持状态已改变，检测到持物滑落；停止运输并重新观察。')
        # HTTP requests only enqueue work. Sensor arrays are copied on the sim
        # thread even while a long physical action or model inference is running.
        try:
            request = self.snapshot_requests.get_nowait()
        except Empty:
            request = None
        if request is not None and request['future'].set_running_or_notify_cancel():
            try:
                packet = self.perception.capture(self, None, cameras=[request['camera']], scene_mode='frame')
                request['future'].set_result(packet)
            except Exception as error:
                request['future'].set_exception(error)
        if hasattr(self, 'vision_worker'):
            if self.tracking_stop.is_set():
                self.tracking = None
                self.tracking_stop.clear()
            try:
                result = self.vision_worker.poll()
                expected = self.current or (self.tracking['command_id'] if self.tracking else None)
                if result and result.get('command_id') == expected:
                    self.visual_result = result
                    if self.tracking and not self.current:
                        # One slow recovery per loss episode; keep looking with the fast detector afterwards.
                        self.tracking['lost'] = not result.get('ok', False)
            except Exception as error:
                self.visual_result = {'ok': False, 'error': str(error), 'views': []}
            track_label = self.target_name if self.current else (self.tracking['label'] if self.tracking else None)
            if (track_label and not self.observing and self.vision_worker.available
                    and self.sequence-self.last_track >= self.config['vision']['track_every_steps']):
                self.last_track = self.sequence
                packet = self.perception.capture(self, track_label, cameras=['scene_camera'], transient=True,
                    vision_mode='fast' if self.tracking and self.tracking['lost'] else 'auto')
                if self.tracking and not self.current:
                    packet.update(command_id=self.tracking['command_id'], **self.tracking['context'])
                    (self.perception.root/packet['request_id']/'request.json').write_text(json.dumps(packet))
                self.vision_worker.submit(packet)

    def event(self, phase, **data):
        self.phase = phase
        self.events.append({"phase": phase, "sequence": self.sequence, "time": time.time(), **data})
        print(json.dumps(self.events[-1], ensure_ascii=False), flush=True)
        self.refresh_snapshot()

    def cloud(self, name, *, associate=False, **vision_options):
        if not vision_options and name in self.prepared_clouds:
            points, self.visual_result = self.prepared_clouds.pop(name)
            return points
        # Instance IDs identify physical bodies; only observed language labels
        # are valid prompts when a later command needs fresh RGB-D geometry.
        row = self.observed_entities.get(name, {})
        label = row.get('label', name)
        if row.get('visual_ref') and 'visual_ref' not in vision_options:
            vision_options['visual_ref'] = row['visual_ref']
        self.observing = True
        try:
            # Retire a tracking request before requesting a geometric observation.
            while not self.vision_worker.available: self.tick()
            for recovery in range(2):
                for _ in range(self.config['camera']['render_interval']): self.tick()
                packet = self.perception.capture(self, label, refine=True, **vision_options)
                self.event('visual_observation', target=label, request_id=packet['request_id'])
                result = self.infer(self.perception.request, packet)
                if not result.get('ok') and self.held == name:
                    packet = self.perception.capture(self, label, refine=True,
                        **{**vision_options, 'cameras': ['scene_camera', 'side_camera']})
                    result = self.infer(self.perception.request, packet)
                self.visual_result = result
                if not result.get('ok'):
                    states = {v.get('status') for v in result.get('views', [])}
                    message = ('目标不唯一，请指定其中一个' if 'ambiguous' in states else
                               '所选视觉模型不可用' if 'provider_unavailable' in states else
                               '视觉服务调用失败' if result.get('error') else '本次画面未找到目标')
                    raise LocalizationFailure(f'{message}：{label}')
                if associate or name in self.observed_entities:
                    try:
                        witness, votes = self.perception.witness(result, self.body_entities)
                        if name in self.observed_entities and witness != name:
                            raise InstanceConflict('观测切换到了另一实例，需要重新识别。')
                        result['physical_witness'] = {'instance_id': witness, 'votes': votes}
                    except InstanceConflict as error:
                        if recovery or vision_options.get('vision_mode') == 'slow':
                            raise
                        self.event('visual_relocalization', reason=str(error),
                                   previous_observation_ref=result['request_id'], provider='sam3')
                        vision_options.update(vision_mode='slow', slow_provider='sam3')
                        continue
                points, views = self.perception.cloud(result)
                self.event('observation', target=name, request_id=packet['request_id'],
                           perception_source=result['perception_source'], views=views, models=result['views'])
                return points
        finally:
            self.observing = False
            self.last_track = self.sequence

    def infer(self, function, *args, **kwargs):
        future = self.pool.submit(function, *args, **kwargs)
        while not future.done():
            self.tick()
            time.sleep(.002)
        return future.result()

    def move(self, target, *, label, steps=None, until_contact=None):
        self.event(label, target_world=target.tolist())
        start = self.tcp_pose()
        travel = np.linalg.norm(target[:3, 3] - start[:3, 3])
        rotation = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).magnitude()
        waypoints = max(1, int(np.ceil(max(travel / .003, rotation / .035))))
        interpolation = Slerp([0, 1], Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]])))
        progress = []
        for index in range(steps or self.config["controller"]["max_steps"]):
            actual = self.tcp_pose()
            if until_contact is not None and until_contact():
                self.goal = actual
                self.event("support_contact", measured_world=actual.tolist())
                return
            delta = target[:3, 3] - actual[:3, 3]
            angle = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
            progress.append(float(np.linalg.norm(delta)+.1*angle))
            if (np.linalg.norm(delta) < self.config["controller"]["position_tolerance_m"]
                    and angle < np.deg2rad(self.config["controller"]["rotation_tolerance_deg"])):
                self.goal = target.copy()
                for _ in range(10): self.tick()
                if self.held_context:
                    self.event('motion_arrived', during=label, holding_measurement=self.holding_status())
                return
            if index > waypoints+35 and len(progress) >= 30:
                robot = self.env.scene['robot']
                joints = numpy_data(robot.data.joint_pos)[0,:7]
                limits = numpy_data(robot.data.joint_pos_limits)[0,:7]
                saturated = np.any(np.minimum(joints-limits[:,0],limits[:,1]-joints)<.005)
                if saturated and min(progress[-30:]) > min(progress[:-30])-.001:
                    break  # Switch candidate when a real joint limit stalls IK.
            # Time-parameterized Cartesian waypoints tracked by Arena IK. Advancing
            # from measured pose every tick would repeatedly reset the ramp and
            # stall behind the actuator's small tracking lag.
            fraction = min(1., (index + 1) / waypoints)
            self.goal = start.copy()
            self.goal[:3, 3] += fraction * (target[:3, 3] - start[:3, 3])
            self.goal[:3, :3] = interpolation(fraction).as_matrix()
            self.tick()
        robot = self.env.scene["robot"]
        self.event("motion_failed", target_world=target.tolist(), measured_world=self.tcp_pose().tolist(),
                   joints_rad=numpy_data(robot.data.joint_pos)[0].tolist(),
                   joint_limits_rad=numpy_data(robot.data.joint_pos_limits)[0].tolist(),
                   joint_velocities=numpy_data(robot.data.joint_vel)[0].tolist())
        raise RuntimeError(f"Arena IK did not reach {label}: position error {np.linalg.norm(delta):.4f} m, rotation error {np.rad2deg(angle):.2f} deg")

    def _grasp_candidates(self, points):
        cfg = self.config["graspgenx"]
        # Official franka_panda gripper description, not the old SO-101 sweep.
        sweep = {"extents_open": np.array([.08, .018, .018], np.float32),
                 "offset_open": np.array([0., 0., .1034], np.float32),
                 "extents_mid": np.array([.04, .018, .018], np.float32),
                 "offset_mid": np.array([0., 0., .1034], np.float32),
                 "gripper_type": 0, "fingertip_depth": .1034}
        def generate():
            # The worker owns its ZMQ socket, including cleanup after a stop.
            transport = ZmqGraspGenXTransport(cfg["host"], cfg["port"], cfg["timeout_ms"])
            try:
                return transport.infer_object(points, sweep, planner="diffusion",
                    num_grasps=cfg["num_grasps"], topk_num_grasps=cfg["num_grasps"], grasp_threshold=.1)
            finally:
                transport.close()
        poses, scores = self.infer(generate)
        np.savez(self.active_directory / "graspgenx_candidates.npz", object_points=points, poses=poses, scores=scores)
        self.event("grasp_candidates", backend="graspgenx", count=len(poses), scores=scores.tolist())
        current = self.tcp_pose()[:3, :3]
        symmetry = np.diag([-1., -1., 1., 1.])
        candidates = []
        for pose, score in zip(poses, scores):
            tcp = model_grasp_to_tcp(pose)
            # Panda's parallel jaws are invariant to a half-turn about approach
            # Z. Choose the equivalent wrist orientation closest to the current
            # one; preserve every other part of the generated full pose.
            tcp = min((tcp, tcp @ symmetry), key=lambda t: Rotation.from_matrix(t[:3, :3] @ current.T).magnitude())
            candidates.append((tcp, float(score)))
        return candidates

    def prepare_task(self, request):
        if self.held_context and not self.holding_status()['verified']:
            self.clear_hold()
        if self.held is not None:
            raise ValueError('夹爪仍持有物体，可使用 place_held 指定目的地继续放置。')
        self.gripper = 1.
        row = self.locate(request.target, **({'visual_ref': request.target_ref} if request.target_ref else {}))
        destination = self.locate_destination(request)
        if destination and destination['name'] == row['name']:
            raise ValueError('抓取对象和支撑对象相同，请指定另一个放置对象。')
        self.target_name = row["name"]
        self.selected_target = row["label"]
        self.initial_z = float(self.object_pose(row["name"])[2, 3]); self.max_lift = 0.
        self.orientation_binding = None
        return row, destination

    def bind_orientation(self, request, row, points):
        if not request.orientation: return None
        ref, directory = load_reference(self.perception.root, request.orientation['axis_ref'], self.perception.scene_id)
        observed = json.loads((directory/'result.json').read_text())
        geometry = observed.get('geometry', {})
        if geometry.get('kind') != 'axis' or geometry.get('axis_confidence', 0) < 1.5:
            raise LocalizationFailure('该引用没有可靠的长轴观测，需要 inspect_object(kind=axis)。')
        # Bind a visual reference to the same physical instance, never use its
        # configured shape/orientation to decide which semantic end is up.
        witness, _ = self.perception.witness(observed, self.body_entities)
        if witness != row['name']:
            raise LocalizationFailure('朝向引用属于另一个物体，需要重新观察当前目标。')
        old_points, _ = self.perception.cloud(observed)
        old_centre = np.quantile(old_points,[.02,.98],axis=0).mean(0)
        centre = np.quantile(points,[.02,.98],axis=0).mean(0)
        if np.linalg.norm(old_centre-centre) > .03:
            raise LocalizationFailure('端点观察后物体已移动，需要重新检查端点。')
        axis = np.asarray(geometry['axis_world'])
        self.orientation_binding = {**request.orientation, 'direction':request.orientation.get('direction','up'),
            'axis_world':axis.tolist(), 'axis_object':(self.object_pose(row['name'])[:3,:3].T @ axis).tolist()}
        return self.orientation_binding

    def oriented_place_held(self, request, row, destination):
        """Execute the observed endpoint constraint using Arena's full-pose IK."""
        child = self.held_cloud(); tcp = self.tcp_pose()
        self.prepared_clouds.pop(destination['name'], None)
        parent = self.cloud(destination['name'], **({'inspect':'grid'} if request.cell_ref else {}))
        self.prepared_clouds[destination['name']] = (parent,self.visual_result)
        binding = self.held_geometry.get('orientation')
        if not binding or binding['axis_ref'] != request.orientation['axis_ref']:
            # A user may inspect, grasp, then request orientation in a later
            # command. Relate that pregrasp observation to the retained shape.
            at_grasp = self.held_geometry['tcp_at_grasp']
            prior = transform_points(at_grasp, self.held_geometry['points_tcp'])
            _, origin = load_reference(self.perception.root,request.orientation['axis_ref'],self.perception.scene_id)
            axis_result = json.loads((origin/'result.json').read_text())
            observed_points,_ = self.perception.cloud(axis_result)
            centre = np.quantile(observed_points,[.02,.98],axis=0).mean(0)
            current_distance = np.linalg.norm(centre-np.quantile(child,[.02,.98],axis=0).mean(0))
            prior_distance = np.linalg.norm(centre-np.quantile(prior,[.02,.98],axis=0).mean(0))
            basis, shape = (tcp,child) if current_distance <= prior_distance else (at_grasp,prior)
            binding = self.bind_orientation(request, row, shape)
            binding = {**binding, 'axis_tcp':(basis[:3,:3].T @ binding['axis_world']).tolist()}
            binding['axis_object'] = (self.object_pose(row['name'])[:3,:3].T @ tcp[:3,:3] @ binding['axis_tcp']).tolist()
            self.held_geometry['orientation'] = binding
        directed = endpoint_vector(tcp[:3,:3] @ np.asarray(binding['axis_tcp']),
                                   request.orientation['endpoint'], request.orientation.get('direction','up'))
        candidates = []
        for delta in placement_rotations(directed, tcp[:3,:3]):
            rotated = transformed_payload(child, tcp, delta)
            # Free-space/cell search evaluates the intended shape, not the
            # horizontal pregrasp footprint of a part that will stand upright.
            if not candidates:
                support = self.placement_support(request, destination, rotated, tcp[:3,3])
            pose, placed = placement_pose(child, tcp, delta, support)
            if request.cell_ref and not cell_fit(placed,destination['grid_cell'],destination['grid_basis_xy'])['fits']:
                continue
            # The real Panda hand extends 103.4 mm behind the controlled TCP.
            # Reject poses putting its palm under the observed support plane.
            if (pose[:3,3]-.1034*pose[:3,2])[2] < support[2]+.012: continue
            candidates.append(pose)
        destination['orientation_evaluation'] = {**binding, **request.orientation}
        if request.cell_ref:
            destination['evaluation_points_object'] = row['evaluation_points_object']
        if not candidates:
            raise RuntimeError('当前抓法无法以所需端点朝向释放，需要从侧面重新抓取。')
        candidates.sort(key=lambda pose: Rotation.from_matrix(pose[:3,:3]@tcp[:3,:3].T).magnitude())
        def approach(pose):
            value = pose.copy();value[2,3] = max(tcp[2,3],pose[2,3]+.16)
            return value
        goal = self.try_precontact_candidates(candidates, approach, phase='reorient_transport', holding=True)
        self.event('orientation_selected', axis_ref=binding['axis_ref'], endpoint=request.orientation['endpoint'],
                   direction=request.orientation.get('direction','up'), pose_world=goal.tolist())
        # Side-wall contact is not proof of seating in a cell. Wait for the
        # measured full TCP pose before opening, as on the ordinary fast path.
        self.move(goal,label='place_approach')
        self.event('release',holding_measurement=self.holding_status());self.gripper=1.
        for _ in range(50): self.tick()
        self.clear_hold()
        retreat=self.tcp_pose();retreat[2,3]+=.14
        self.move(retreat,label='retreat')
        before=self.object_pose(row['name'])[:3,3]
        for _ in range(60):self.tick()
        return self.task.evaluate(self.env,row['name'],self.initial_z,destination,released=True,
            max_lift=self.max_lift,stability=float(np.linalg.norm(self.object_pose(row['name'])[:3,3]-before)))

    def try_precontact_candidates(self, candidates, make_approach, *, phase, holding=False):
        last_error = None
        for index, candidate in enumerate(candidates[:3]):
            if holding and not self.holding_status()['verified']:
                raise RuntimeError('夹持状态已改变，需要重新观察。')
            self.event('candidate_attempt', index=index, total=len(candidates), inference_reused=index > 0)
            try:
                self.move(make_approach(candidate), label=phase)
                return candidate
            except InterruptedError:
                raise
            except RuntimeError as error:
                if 'Arena IK did not reach' not in str(error): raise
                last_error = error
                self.event('candidate_rejected', index=index, reason=str(error), retry_scope='precontact_only')
        raise last_error or RuntimeError('No reachable pose candidates')

    def _pick_place(self, request):
        row, destination = self.prepare_task(request)
        points = self.cloud(row["name"])
        observation_ref = self.visual_result['request_id']
        binding = self.bind_orientation(request, row, points)
        self.event("planning", route=request.route(), object_points=len(points))
        candidates = self._grasp_candidates(points)
        if not candidates:
            raise RuntimeError("GraspGenX returned no grasp candidates")
        current = self.tcp_pose()
        # Prefer a short motion among model proposals, without projecting poses
        # onto a manually permitted direction or hard-coded workspace box.
        def motion_cost(pose):
            orientation_cost = 0.
            if binding:
                axis = endpoint_vector(binding['axis_world'], binding['endpoint'], binding['direction'])
                delta = placement_rotations(axis, pose[:3,:3])[0]
                # Prefer a grasp whose palm can remain above the support after
                # reorientation; don't start a top-down grasp for a full flip.
                orientation_cost = 8.*max(0., (delta @ pose[:3,2])[2]-.15)
            return (orientation_cost + np.linalg.norm(pose[:3, 3] - current[:3, 3])
                    + Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).magnitude())
        ranked = sorted(candidates, key=lambda item: motion_cost(item[0]) - .15*(item[1] or 0.))
        def pregrasp(item):
            pose = item[0].copy(); pose[:3, 3] -= pose[:3, 2]*.12
            return pose
        grasp, score = self.try_precontact_candidates(ranked, pregrasp, phase='pregrasp')
        self.event("selected_grasp", pose_world=grasp.tolist(), model_score=score, raw_candidates=len(candidates))
        self.move(grasp, label="approach")
        self.event("close_gripper"); self.gripper = -1.
        for _ in range(50): self.tick()
        self.held = row["name"]  # possible holding is preserved even on failed lift
        geometry = {'points_tcp': transform_points(invert_transform(self.tcp_pose()), points),
                    'observation_ref': observation_ref}
        self.remember_hold(row, geometry)
        lift = self.tcp_pose(); lift[2, 3] += .16
        self.move(lift, label="lift")
        if self.max_lift < .04:
            raise RuntimeError("Physical lift verification failed")
        if not self.holding_status()['verified']:
            raise RuntimeError('夹持状态已改变，抬升后物体未跟随夹爪。')
        self.held_context['max_lift_m'] = self.max_lift
        self.event("lift_verified", lift_m=self.max_lift)
        if destination is None:
            return self.task.evaluate(self.env, row["name"], self.initial_z, None,
                                      released=False, max_lift=self.max_lift, stability=0.)
        return self._place_held(request, row, destination)

    def _place_held(self, request, row, destination):
        if request.orientation:
            return self.oriented_place_held(request, row, destination)
        # Geometry below comes from RGB-D. Simulator object pose is used only by
        # evaluation; the held reference is estimated from measured cloud bounds.
        child = self.held_cloud()
        object_input = np.eye(4)
        object_input[:3, 3] = np.quantile(child, [.02, .98], axis=0).mean(axis=0)
        tcp_to_object = invert_transform(self.tcp_pose()) @ object_input
        # Placement needs the named support surface, excluding other objects
        # inside its box. Use semantic segmentation for this geometry task.
        parent = self.cloud(destination["name"], vision_mode='slow', slow_provider='sam3',
                            **({'inspect':'grid'} if request.cell_ref else {}))
        free_patch = None
        scene = self.perception.scene_cloud(self.visual_result)
        if request.cell_ref or request.placement_selection in {'auto', 'free_space'}:
            self.prepared_clouds[destination['name']] = (parent, self.visual_result)
            centre = self.placement_support(request, destination, child, self.tcp_pose()[:3, 3])
            if destination.get('selected_position_world_m') is not None:
                free_patch = self.events[-1]
                full_parent = parent.copy()
                parent = parent[(np.linalg.norm(parent[:, :2]-centre[:2], axis=1) < free_patch['footprint_radius_m']+.02)
                                & (np.abs(parent[:,2]-centre[2]) < .008)]
        cfg = self.config["anyplace"]
        self.event("placement_inference", backend="anyplace", parent_points=len(parent), child_points=len(child))
        client = AnyPlaceClient(cfg["url"], cfg["timeout_s"])
        sequence = self.sequence
        answer = self.infer(client.infer, sequence=sequence, parent=parent.tolist(), child=child.tolist(),
                            candidates=cfg["candidates"], iterations=cfg["iterations"],
                            input_geometry="partial_multiview_rgbd", init_current_orientation=True)
        np.savez(self.active_directory / "anyplace_candidates.npz", parent=parent, child=child,
                 relative=np.asarray(answer["transforms"]), object_input=object_input, tcp_to_object=tcp_to_object)
        self.event("placement_candidates", backend="anyplace", count=len(answer["transforms"]),
                   inference_s=answer.get("inference_s"), checkpoint_sha256=answer.get("checkpoint_sha256"),
                   input_sampling=answer.get('input_sampling'))
        candidates = [placement_to_tcp(np.asarray(relative), object_input, tcp_to_object)
                      for relative in answer["transforms"]]
        if free_patch is not None:
            candidates = []
            for relative in answer['transforms']:
                placed = transform_points(np.asarray(relative), child)
                low, high = np.quantile(placed, [.02, .98], axis=0)
                centre = (low + high) / 2
                radius = max(.05, np.linalg.norm(high[:2] - low[:2]) / 2 + .015)
                pose = placement_to_tcp(np.asarray(relative), object_input, tcp_to_object)
                fits = cell_fit(placed,destination['grid_cell'],destination['grid_basis_xy'])['fits'] if request.cell_ref else placement_is_free(full_parent, scene, placed, pose[:3, 3])
                if (np.linalg.norm(centre[:2] - np.asarray(free_patch['position_world_m'])[:2]) < .04 and fits):
                    candidates.append(pose)
        if not candidates:
            raise RuntimeError("AnyPlace returned no placement candidates fitting the requested support")
        current = self.tcp_pose()
        ranked = sorted(candidates, key=lambda pose: np.linalg.norm(pose[:3, 3] - current[:3, 3])
                   + Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).magnitude())
        def preplace(pose):
            approach = pose.copy(); approach[2,3] = max(pose[2,3]+.13, self.tcp_pose()[2,3])
            return approach
        goal = self.try_precontact_candidates(ranked, preplace, phase='transport', holding=True)
        self.event("selected_placement", pose_world=goal.tolist(), backend=request.route()["placement"])
        self.move(goal, label="place_approach", until_contact=lambda:
                  self.task.support_contact(self.env, row["name"], destination))
        self.event("release"); self.gripper = 1.
        for _ in range(50): self.tick()
        self.clear_hold()
        retreat = self.tcp_pose(); retreat[2, 3] += .13
        self.move(retreat, label="retreat")
        before = self.object_pose(row["name"])[:3, 3]
        for _ in range(60): self.tick()
        stability = float(np.linalg.norm(self.object_pose(row["name"])[:3, 3] - before))
        return self.task.evaluate(self.env, row["name"], self.initial_z, destination,
                                  released=True, max_lift=self.max_lift, stability=stability)

    def recover_fast(self, request):
        if self.target_name is None:
            return None  # No grasp was started; retain the original localization failure.
        row = self.observed_entities[self.target_name]
        self.prepared_clouds.clear()
        if self.holding_status()['verified']:
            destination = self.locate_destination(request)
            self.event("fallback_keep_grasp", target=row["name"])
            if destination is None:
                return lambda task: self.task.evaluate(self.env, row["name"], self.initial_z, None,
                    released=False, max_lift=self.max_lift, stability=0.)
            return lambda task: self._place_held(task, row, destination)
        # No successful lift: open the jaws and move above the observed scene,
        # then GraspGenX observes the current object rather than replaying a pose.
        self.event("fallback_reobserve")
        self.gripper = 1.
        for _ in range(25): self.tick()
        self.clear_hold()
        retreat = self.tcp_pose(); retreat[2, 3] += .12
        self.move(retreat, label="fallback_retreat")
        return None

    def place_held(self, request, attempts=None):
        measured = self.holding_status()
        if not measured['verified']:
            if self.held_context:
                self.clear_hold()
            raise RuntimeError('当前没有确认仍在夹爪中的物体，未执行放置。')
        row = self.observed_entities[self.held]
        destination = self.locate_destination(request)
        if row['name'] == destination['name']:
            raise ValueError('持有物体不能作为它自身的放置目标。')
        self.target_name = row['name']; self.selected_target = row['label']
        self.initial_z = self.held_context['initial_z']
        self.max_lift = self.held_context['max_lift_m']
        self.event('holding_resumed', **measured)

        def recover(_):
            if not self.holding_status()['verified']:
                raise RuntimeError('放置未通过评测，物体已释放；未自动重新抓取。')
            self.prepared_clouds.clear()
            self.event('fallback_keep_grasp', target=row['name'])
            return lambda task: self._place_held(task, row, destination)

        evaluation, route, attempts = run_cascade(request,
            lambda task: fast_place_held(self, task, row, destination),
            lambda task: self._place_held(task, row, destination), recover, self.event, attempts=attempts)
        route['grasp'] = 'retained_from_previous_command'
        return {'ok': evaluation['physical_success'], 'evaluation': evaluation,
                'route': route, 'attempts': attempts, 'fallback_used': len(attempts) > 1,
                'message': 'Physical placement verified' if evaluation['physical_success'] else 'Physical placement failed verification'}

    def target_value(self, value):
        if isinstance(value, dict):
            if value.get('cell_ref'):
                cell = self.perception.resolve_reference(value['cell_ref'])
                return cell['label'], cell['container_ref']
            if value.get('ref'):
                return self.perception.resolve_reference(value['ref'])['label'], value['ref']
            label = value.get('label', value.get('category', ''))
            color = value.get('attributes', {}).get('color', '')
            return (color + ' ' + label if color and color not in label else label), None
        return value or '', None

    def post_action_observation(self, result, request):
        """Capture new evidence; uncertain visual checks request semantic review.

        Ordinary actions do not invoke an LLM or another image model. A cell
        placement additionally measures its occupancy with SAM2/RGB-D.
        """
        self.observing = True
        try:
            inspect_cell = bool(request and request.cell_ref and result.get('ok'))
            options = {'cameras':['scene_camera','side_camera'],'scene_mode':'frame'}
            if inspect_cell:
                cell = self.perception.resolve_reference(request.cell_ref)
                options = {'visual_ref':cell['container_ref'],'inspect':'grid'}
                while not self.vision_worker.available: self.tick()
            packet = self.perception.capture(self, request.destination if inspect_cell else None, **options)
            result['post_action_snapshot'] = {'observation_ref':packet['request_id'],
                'observed_at':packet['observed_at'],'frame_sequence':self.sequence,
                'cameras':[v['camera'] for v in packet['views']]}
            if inspect_cell:
                observed = self.infer(self.perception.request,packet)
                self.visual_result = observed
                grid = observed.get('geometry',{})
                selected = next((c for c in grid.get('cells',[]) if c['row']==cell['row'] and c['column']==cell['column']),None)
                verified = bool(observed.get('ok') and selected and selected['occupancy']=='occupied')
                result['postconditions'] = {'cell_visually_occupied':{'satisfied':verified,
                    'status': selected['occupancy'] if selected else 'unknown',
                    'source':'fresh_rgbd','observation_ref':packet['request_id']}}
                if not verified:
                    result.update(review_required=True, review_reason='物理格位评测通过，但新图像未能确认占用，需要核对，不能重放已完成抓放。')
            self.event('post_action_observed',**result['post_action_snapshot'],
                review_required=result.get('review_required',False))
        finally:
            self.observing = False

    def execute(self, command):
        self.current = command["command_id"]; self.events = []; started = time.time()
        self.tracking = None
        self.visual_result = None
        self.prepared_clouds.clear()
        self.bus_context = {key:command[key] for key in ('task_id','task_version','correlation_id','causation_id') if key in command}
        attempts = []
        request = None
        directory = self.output / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
        directory.mkdir()
        self.active_directory = directory
        try:
            params = command.get("params", {}); skill = command["skill"]
            if skill == 'place_held':
                dest = params.get('destination', {})
                label, destination_ref = self.target_value(dest)
                if not label:
                    raise ValueError('请指定放置目的地。')
                request = ManipulationRequest((self.held_context or {}).get('label', 'held object'), label,
                    params.get('mode', 'auto'), params.get('unfamiliar', False), params.get('cluttered', False),
                    params.get('precise', False), params.get('relation', 'on'),
                    dest.get('selection', 'auto') if isinstance(dest, dict) else 'auto',
                    destination_ref=destination_ref, region_ref=dest.get('region_ref') if isinstance(dest, dict) else None,
                    placement_preference=dest.get('preference', 'nearest') if isinstance(dest, dict) else 'nearest',
                    cell_ref=dest.get('cell_ref') if isinstance(dest, dict) else None, orientation=params.get('orientation'))
                result = self.place_held(request, attempts)
            elif skill in {"grasp", "pick_place"}:
                target = params.get("target", {})
                label, target_ref = self.target_value(target)
                dest = params.get("destination", {})
                dest_label, destination_ref = self.target_value(dest)
                if skill == "pick_place" and not dest_label:
                    raise ValueError("pick_place requires a destination label")
                request = ManipulationRequest(label, dest_label if skill == "pick_place" else None,
                    params.get("mode", "auto"), params.get("unfamiliar", False), params.get("cluttered", False),
                    params.get("precise", False), params.get("relation", "on"),
                    dest.get('selection', 'auto') if isinstance(dest, dict) else 'auto',
                    target_ref=target_ref, destination_ref=destination_ref,
                    region_ref=dest.get('region_ref') if isinstance(dest, dict) else None,
                    placement_preference=dest.get('preference', 'nearest') if isinstance(dest, dict) else 'nearest',
                    cell_ref=dest.get('cell_ref') if isinstance(dest, dict) else None, orientation=params.get('orientation'))
                evaluation, route, attempts = run_cascade(request,
                    lambda task: fast_pick_place(self, task), self._pick_place, self.recover_fast, self.event,
                    attempts=attempts)
                result = {"ok": evaluation["physical_success"], "evaluation": evaluation, "route": route,
                          "attempts": attempts, "fallback_used": len(attempts) > 1,
                          "message": "Physical task verified" if evaluation["physical_success"] else "Physical task failed verification"}
            elif skill in {"select_target", "perceive"}:
                if params.get('scope') == 'scene':
                    self.selected_target = None
                    self.observing = True
                    try:
                        while not self.vision_worker.available: self.tick()
                        packet = self.perception.capture(self, None, scene_mode=params.get('scene_mode', 'inventory'), cameras=params.get('cameras'), vision_mode=params.get('vision_mode', 'auto'), slow_provider=params.get('slow_provider'))
                        self.event('visual_observation', scope='scene', request_id=packet['request_id'])
                        observed = self.infer(self.perception.request, packet)
                        self.visual_result = observed
                        if not observed.get('ok'): raise RuntimeError(str(observed.get('error', 'Scene observation failed')))
                        descriptions = {'detected_objects': sorted({o['label'] for v in observed['views'] for o in v.get('objects', [])}),
                                        'region_descriptions': [r['description'] for v in observed['views'] for r in v.get('regions', [])]}
                        result = {'ok': True, 'message': 'Fresh scene observation; detected_objects are model detections from this image. Regions may overlap and inventory may be incomplete. Empty results do not prove the scene is empty: '+json.dumps(descriptions),
                                  'scope': 'scene'}
                    finally:
                        self.observing = False
                else:
                    label, visual_ref = self.target_value(params if params.get("ref") else params.get("category") or self.selected_target)
                    if not label: raise ValueError('Specify a target or scope=scene')
                    color = params.get("attributes", {}).get("color", "")
                    if color and color not in label: label = color + " " + label
                    self.selected_target = label
                    collection = not visual_ref and not params.get('grounding') and not params.get('tracking') and params.get('selection', 'all') != 'one'
                    if collection:
                        self.observing = True
                        try:
                            while not self.vision_worker.available: self.tick()
                            packet = self.perception.capture(self, label, collection=True,
                                vision_mode=params.get('vision_mode', 'auto'), slow_provider=params.get('slow_provider'), cameras=params.get('cameras'))
                            self.event('visual_observation', scope='collection', request_id=packet['request_id'])
                            self.visual_result = self.infer(self.perception.request, packet)
                            if not self.visual_result.get('ok'):
                                raise LocalizationFailure(str(self.visual_result.get('error', 'Collection provider unavailable')))
                            result = {'ok': True, 'scope': 'collection', 'collection': self.visual_result['collection'],
                                'message': 'Observed object collection. Choose actual member refs for manipulation; counts are visible estimates, not proof of exhaustive inventory.'}
                        finally:
                            self.observing = False
                    else:
                        cloud = self.cloud(label, vision_mode=params.get('vision_mode', 'auto'),
                                       slow_provider=params.get('slow_provider'), cameras=params.get('cameras'), visual_ref=visual_ref,
                                       grounding=params.get('grounding'), inspect=params.get('inspect'))
                        semantic = self.visual_result.get('semantic_status', 'unknown')
                        result = {"ok": True, "message": "Target matched by detector" if semantic == 'detected' else
                            "Only a provisional visual region was located. The queried category is unconfirmed; do not report it as a confirmed object.",
                            "points": len(cloud), "target": label, "semantic_status": semantic}
                        if self.visual_result.get('geometry'):
                            result['geometry'] = self.visual_result['geometry']
                        if params.get('tracking'):
                            self.tracking = {'label': label, 'command_id': self.current,
                                         'context': self.bus_context.copy(), 'lost': False}
                            result.update(visual_tracking=True,
                            message='Visual tracking enabled; robot pose is unchanged. Target semantics: '+semantic)
            elif skill == "home":
                goal = np.eye(4); goal[:3, :3] = np.diag([1., -1., -1.]); goal[:3, 3] = [.4, 0., .3]
                self.move(goal, label="home")
                result = {"ok": True, "message": "Panda reached home pose"}
            else:
                raise ValueError(f"Unsupported skill: {skill}")
            result["state"] = "completed" if result["ok"] else "failed"
        except Exception as exc:
            self.goal = self.tcp_pose()
            result = {"ok": False, "state": "cancelled" if isinstance(exc, InterruptedError) else "failed",
                      "message": str(exc), "error_type": type(exc).__name__, "held_object": self.held,
                      "failure_phase":next((e['phase'] for e in reversed(self.events)
                          if e['phase'] not in {'attempt_started','attempt_finished'}),self.phase)}
            if attempts:
                result.update(attempts=attempts, fallback_used=len(attempts) > 1)
            if getattr(exc,'evaluation',None) is not None:
                result['evaluation'] = exc.evaluation
        if command['skill'] in {'grasp','pick_place','place_held'} and result.get('state') != 'cancelled':
            try:
                self.post_action_observation(result,request)
            except Exception as error:
                # Do not erase a completed physical action if its new camera
                # evidence is unavailable; the queue must inspect before advancing.
                result.update(review_required=True,review_reason='动作后观察不可用：'+str(error))
        if self.held_context and not self.holding_status()['verified']:
            self.clear_hold()
            if 'held_object' in result: result['held_object'] = None
        result['holding'] = self.holding_status()
        if not result.get('ok') and result.get('state') != 'cancelled':
            result['failure'] = failure_feedback(result.get('message', ''), result.get('failure_phase',self.phase), result['holding'],result.get('evaluation'))
        result.update(command_id=self.current, skill=command["skill"], elapsed_s=time.time() - started,
                      evidence_dir=str(directory), events=self.events, vision=self.visual_result, tcp_pose_world=self.tcp_pose().tolist())
        self.last_result = {key: value for key, value in result.items() if key != "events"}
        (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        for name, data in self.frames.items(): (directory / f"{name}.jpg").write_bytes(data)
        self.phase = "idle"; self.current = None; self.target_name = None
        self.refresh_snapshot()
        return result
