"""Single simulation-thread skill orchestration over Arena's official IK action."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np
from mr_liu.arena.arrays import numpy_data, pose_matrix
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp
import torch

from mr_liu.arena.perception import PerceptionBridge
from mr_liu.arena.entities import scene_entities, resolve_entity
from mr_liu.vision.worker import VisionWorker
from mr_liu.arena.cascade import run_cascade
from mr_liu.arena.fast import fast_pick_place
from mr_liu.arena.contracts import ManipulationRequest, model_grasp_to_tcp, placement_to_tcp
from mr_liu.grasp.backends.graspgenx import ZmqGraspGenXTransport
from mr_liu.grasp.transforms import invert_transform
from mr_liu.place.anyplace import AnyPlaceClient



class ArenaRuntime:
    def __init__(self, env, task, config, output):
        self.wrapped = env
        self.env = env.unwrapped
        self.task, self.config = task, config
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.stop_requested = threading.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pose-inference")
        self.frames = {}; self.snapshot = {}; self.sequence = 0
        self.phase = "initializing"; self.current = None; self.held = None
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

    def resolve(self, label, kind="objects"):
        return resolve_entity(self.config, label, graspable=kind == 'objects')

    def capabilities(self):
        return {"robot": "franka_panda", "execution": "isaaclab_arena.franka_ik",
                "skills": ["status", "capabilities", "select_target", "perceive", "grasp", "pick_place", "stop", "hold", "home"],
                "grasp": {"backend": "official_pick_place_or_graspgenx", "ready": True},
                "placement": {"backend": "official_pick_place_or_anyplace", "relations": ["on"]},
                "routing": "fast_first_then_models_once; complex_tasks_use_models_directly",
                "vla_loaded": False, "perception_source": "task_routed_rgbd",
                "vision": {"architecture": "task_routed_fast_slow", "visual_tracking": True,
                    "persistent_memory": True, "fast": ["yoloe_text", "yoloe_visual", "sam2_tiny", "lk"],
                    "slow_localizer": self.config['vision'].get('slow_localizer'),
                    "scene_description": "florence2", "planned_disabled": ["qwen_multimodal"]},
                "frame": "world", "quaternion": "xyzw", "units": "metres",
                "objects": self.config["objects"], "destinations": scene_entities(self.config)}

    def refresh_snapshot(self):
        robot = self.env.scene["robot"]
        tcp = self.tcp_pose()
        self.snapshot = {"ready": True, "robot": "franka_panda", "phase": self.phase,
                         "sequence": self.sequence, "timestamp": time.time(), "capabilities": self.capabilities(),
                         "held_object": self.held, "command_id": self.current,
                         "tcp_pose_world": tcp.tolist(), "prompt": self.selected_target,
                         "last_result": self.last_result,
                         "vision": self.visual_result,
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
            self.refresh_snapshot()
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

    def cloud(self, name, **vision_options):
        self.observing = True
        try:
            # Retire a tracking request before requesting a geometric observation.
            while not self.vision_worker.available: self.tick()
            for _ in range(self.config['camera']['render_interval']): self.tick()
            packet = self.perception.capture(self, name, refine=True, **vision_options)
            self.event('visual_observation', target=name, request_id=packet['request_id'])
            result = self.infer(self.perception.request, packet)
            if not result.get('ok') and self.held == name:
                packet = self.perception.capture(self, name, refine=True, cameras=['scene_camera', 'side_camera'], **vision_options)
                result = self.infer(self.perception.request, packet)
            self.visual_result = result
            if not result.get('ok'):
                states = {v.get('status') for v in result.get('views', [])}
                message = ('目标不唯一，请指定其中一个' if 'ambiguous' in states else
                           '所选视觉模型不可用' if 'provider_unavailable' in states else
                           '视觉服务调用失败' if result.get('error') else '本次画面未找到目标')
                # Full diagnostics remain in result.vision, not in the console's action label.
                raise RuntimeError(f'{message}：{name}')
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
        for index in range(steps or self.config["controller"]["max_steps"]):
            actual = self.tcp_pose()
            if until_contact is not None and until_contact():
                self.goal = actual
                self.event("support_contact", measured_world=actual.tolist())
                return
            delta = target[:3, 3] - actual[:3, 3]
            angle = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
            if (np.linalg.norm(delta) < self.config["controller"]["position_tolerance_m"]
                    and angle < np.deg2rad(self.config["controller"]["rotation_tolerance_deg"])):
                self.goal = target.copy()
                for _ in range(10): self.tick()
                return
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
        if self.held is not None:
            raise ValueError("A previous object may still be held; do not start another grasp")
        row = self.resolve(request.target)
        destination = self.resolve(request.destination, "destinations") if request.destination else None
        if destination and destination['name'] == row['name']:
            raise ValueError('抓取对象和支撑对象相同，请指定另一个放置对象。')
        self.target_name = row["name"]
        self.selected_target = row["label"]
        self.initial_z = float(self.object_pose(row["name"])[2, 3]); self.max_lift = 0.
        return row, destination

    def _pick_place(self, request):
        row, destination = self.prepare_task(request)
        points = self.cloud(row["name"])
        self.event("planning", route=request.route(), object_points=len(points))
        candidates = self._grasp_candidates(points)
        if not candidates:
            raise RuntimeError("GraspGenX returned no grasp candidates")
        current = self.tcp_pose()
        # Prefer a short motion among model proposals, without projecting poses
        # onto a manually permitted direction or hard-coded workspace box.
        def motion_cost(pose):
            return (np.linalg.norm(pose[:3, 3] - current[:3, 3])
                    + Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).magnitude())
        grasp, score = min(candidates, key=lambda item: motion_cost(item[0]) - .15 * (item[1] or 0.))
        self.event("selected_grasp", pose_world=grasp.tolist(), model_score=score,
                   raw_candidates=len(candidates))
        pre = grasp.copy(); pre[:3, 3] -= grasp[:3, 2] * .12
        self.move(pre, label="pregrasp")
        self.move(grasp, label="approach")
        self.event("close_gripper"); self.gripper = -1.
        for _ in range(50): self.tick()
        self.held = row["name"]  # possible holding is preserved even on failed lift
        lift = self.tcp_pose(); lift[2, 3] += .16
        self.move(lift, label="lift")
        if self.max_lift < .04:
            raise RuntimeError("Physical lift verification failed")
        self.event("lift_verified", lift_m=self.max_lift)
        if destination is None:
            return self.task.evaluate(self.env, row["name"], self.initial_z, None,
                                      released=False, max_lift=self.max_lift, stability=0.)
        return self._place_held(request, row, destination)

    def _place_held(self, request, row, destination):
        # Geometry below comes from RGB-D. Simulator object pose is used only by
        # evaluation; the held reference is estimated from measured cloud bounds.
        child = self.cloud(row["name"])
        object_input = np.eye(4)
        object_input[:3, 3] = np.quantile(child, [.02, .98], axis=0).mean(axis=0)
        tcp_to_object = invert_transform(self.tcp_pose()) @ object_input
        # Placement needs the named support surface, excluding other objects
        # inside its box. Use semantic segmentation for this geometry task.
        parent = self.cloud(destination["name"], vision_mode='slow', slow_provider='sam3')
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
                   inference_s=answer.get("inference_s"), checkpoint_sha256=answer.get("checkpoint_sha256"))
        candidates = [placement_to_tcp(np.asarray(relative), object_input, tcp_to_object)
                      for relative in answer["transforms"]]
        if not candidates:
            raise RuntimeError("AnyPlace returned no placement candidates")
        current = self.tcp_pose()
        goal = min(candidates, key=lambda pose: np.linalg.norm(pose[:3, 3] - current[:3, 3])
                   + Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).magnitude())
        self.event("selected_placement", pose_world=goal.tolist(), backend=request.route()["placement"])
        preplace = goal.copy(); preplace[2, 3] = max(goal[2, 3] + .13, self.tcp_pose()[2, 3])
        self.move(preplace, label="transport")
        self.move(goal, label="place_approach", until_contact=lambda:
                  self.task.support_contact(self.env, row["name"], destination))
        self.event("release"); self.gripper = 1.
        for _ in range(50): self.tick()
        self.held = None
        retreat = self.tcp_pose(); retreat[2, 3] += .13
        self.move(retreat, label="retreat")
        before = self.object_pose(row["name"])[:3, 3]
        for _ in range(60): self.tick()
        stability = float(np.linalg.norm(self.object_pose(row["name"])[:3, 3] - before))
        return self.task.evaluate(self.env, row["name"], self.initial_z, destination,
                                  released=True, max_lift=self.max_lift, stability=stability)

    def recover_fast(self, request):
        row = self.resolve(request.target)
        if self.held and self.task.evaluate(self.env, row["name"], self.initial_z, None,
                released=False, max_lift=self.max_lift, stability=0.)["physical_success"]:
            destination = self.resolve(request.destination, "destinations") if request.destination else None
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
        self.held = None
        retreat = self.tcp_pose(); retreat[2, 3] += .12
        self.move(retreat, label="fallback_retreat")
        return None

    def execute(self, command):
        self.current = command["command_id"]; self.events = []; started = time.time()
        self.tracking = None
        self.visual_result = None
        self.bus_context = {key:command[key] for key in ('task_id','task_version','correlation_id','causation_id') if key in command}
        attempts = []
        directory = self.output / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
        directory.mkdir()
        self.active_directory = directory
        try:
            params = command.get("params", {}); skill = command["skill"]
            if skill in {"grasp", "pick_place"}:
                target = params.get("target", {})
                label = target.get("category", "") if isinstance(target, dict) else target
                color = target.get("attributes", {}).get("color", "") if isinstance(target, dict) else ""
                if color and color not in label: label = color + " " + label
                dest = params.get("destination", {})
                dest_label = dest.get("label") if isinstance(dest, dict) else dest
                if skill == "pick_place" and not dest_label:
                    raise ValueError("pick_place requires a destination label")
                request = ManipulationRequest(label, dest_label if skill == "pick_place" else None,
                    params.get("mode", "auto"), params.get("unfamiliar", False), params.get("cluttered", False),
                    params.get("precise", False), params.get("relation", "on"))
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
                        packet = self.perception.capture(self, None, scene_mode=params.get('scene_mode', 'describe'))
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
                    label = params.get("category") or self.selected_target
                    if not label: raise ValueError('Specify a target or scope=scene')
                    color = params.get("attributes", {}).get("color", "")
                    if color and color not in label: label = color + " " + label
                    self.selected_target = label
                    cloud = self.cloud(label, vision_mode=params.get('vision_mode', 'auto'),
                                       slow_provider=params.get('slow_provider'))
                    semantic = self.visual_result.get('semantic_status', 'unknown')
                    result = {"ok": True, "message": "Target matched by detector" if semantic == 'detected' else
                        "Only a provisional visual region was located. The queried category is unconfirmed; do not report it as a confirmed object.",
                        "points": len(cloud), "target": label, "semantic_status": semantic}
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
                      "message": str(exc), "error_type": type(exc).__name__, "held_object": self.held}
            if attempts:
                result.update(attempts=attempts, fallback_used=len(attempts) > 1)
        result.update(command_id=self.current, skill=command["skill"], elapsed_s=time.time() - started,
                      evidence_dir=str(directory), events=self.events, vision=self.visual_result, tcp_pose_world=self.tcp_pose().tolist())
        self.last_result = {key: value for key, value in result.items() if key != "events"}
        (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        for name, data in self.frames.items(): (directory / f"{name}.jpg").write_bytes(data)
        self.phase = "idle"; self.current = None; self.target_name = None
        self.refresh_snapshot()
        return result
