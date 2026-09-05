"""Deterministic, simulation-only industrial staging and pick/stack playback.

No perception or grasp model is invoked. Props are explicitly kinematic;
attachment and placement are animation, not evidence of physical grasp success.
"""
from __future__ import annotations

import copy
import math
import time
import uuid
from collections import deque

import numpy as np


def layout(count=48):
    if not 1 <= count <= 96:
        raise ValueError('demo count must be between 1 and 96')
    return [dict(index=i, source=[.33 + (i % 4)*.052, -.19 + ((i//4)%6)*.022,
                                 1.059 + (i//24)*.018],
                 destination=[.335 + (i % 4)*.046, .055 + ((i//4)%6)*.020,
                              1.065 + (i//24)*.018],
                 yaw=(-12, 8, -5, 15, 0, -8)[i % 6]) for i in range(count)]


def spawn(stage, count=48, clutter=96):
    from pxr import Gf, Sdf, Semantics, UsdGeom, UsdPhysics, UsdShade
    if not 0 <= clutter <= 5000:
        raise ValueError('demo clutter must be between 0 and 5000')
    root = '/World/IndustrialDemo'
    material = UsdShade.Material.Define(stage, root+'/Steel')
    shader = UsdShade.Shader.Define(stage, root+'/Steel/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.46, .50, .55))
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.9)
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.25)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')

    def box(name, position, size, color):
        geom = UsdGeom.Cube.Define(stage, root+'/'+name)
        geom.CreateSizeAttr(1)
        geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xf = UsdGeom.Xformable(geom)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddScaleOp().Set(Gf.Vec3d(*size))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

    # Open industrial tote: solid floor, four walls and visible ribs.
    box('Basket/Floor', [.404, .105, 1.053], [.20, .14, .006], (.07,.22,.34))
    for x in [.301, .507]:
        box(f'Basket/Side{int(x*1000)}', [x,.105,1.095], [.006,.146,.09], (.08,.30,.43))
    for y in [.032, .178]:
        box(f'Basket/End{int(y*1000)}', [.404,y,1.095], [.212,.006,.09], (.08,.30,.43))
    for i in range(10):
        box(f'Basket/Rib{i}', [.312+i*.020,.028,1.095], [.005,.006,.085], (.12,.39,.51))
    items = layout(count)
    ops = []
    for item in items:
        geom = UsdGeom.Cylinder.Define(stage, root+f'/Column{item["index"]:03}')
        geom.CreateAxisAttr('X')
        geom.CreateRadiusAttr(.008)
        geom.CreateHeightAttr(.040)
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(geom.GetPrim()).CreateKinematicEnabledAttr(True)
        xf = UsdGeom.Xformable(geom)
        translate = xf.AddTranslateOp()
        translate.Set(Gf.Vec3d(*item['source']))
        rotate = xf.AddRotateZOp()
        rotate.Set(item['yaw'])
        sem = Semantics.SemanticsAPI.Apply(geom.GetPrim(), 'Semantics')
        sem.CreateSemanticTypeAttr().Set('class')
        sem.CreateSemanticDataAttr().Set('metal cylinder')
        ops.append((translate, rotate))
    # Separate rigid-body pile increases render/physics load without changing
    # the deterministic demonstration inventory or its fixed pickup positions.
    for i in range(clutter):
        geom = UsdGeom.Cylinder.Define(stage, root+f'/Stock/Part{i:05}')
        geom.CreateRadiusAttr(.006)
        geom.CreateHeightAttr(.024)
        xf = UsdGeom.Xformable(geom)
        xf.AddTranslateOp().Set(Gf.Vec3d(.60+(i%12)*.017, -.19+((i//12)%20)*.018,
                                        1.07+(i//240)*.026))
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(material)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(geom.GetPrim())
        UsdPhysics.MassAPI.Apply(geom.GetPrim()).CreateMassAttr(.015)
    return items, ops


class IndustrialDemo:
    def __init__(self, control, items, drive, clock, *, phase_seconds=1.0):
        if not math.isfinite(phase_seconds) or phase_seconds <= 0:
            raise ValueError('phase duration must be positive and finite')
        self.control, self.motion = control, control.motion
        self.items, self.drive, self.clock = items, drive, clock
        self.duration = phase_seconds
        self.remaining = deque(range(len(items)))
        self.placed = 0
        self.job = None
        self.samples = deque(maxlen=10000)
        self.last_wall = time.monotonic()
        self.warmup = 30

    def capabilities(self):
        return dict(demo=True, model_free=True, physical_grasp_verified=False,
                    scenario='industrial_pick_stack', columns=len(self.items),
                    remaining=len(self.remaining), placed=self.placed)

    def status(self):
        with self.motion.lock:
            record = next((r for r in reversed(self.motion.records.values()) if r.get('demo')), {})
            samples = np.asarray(self.samples)
            return copy.deepcopy({**record, **self.capabilities(), 'performance': {
                'sample_count': len(samples), 'window': 'last 10000 app updates after 30 warmup updates',
                'app_updates_per_second': float(1/samples.mean()) if len(samples) else None,
                'p95_update_ms': float(np.percentile(samples, 95)*1000) if len(samples) else None}})

    def submit(self, params, command_id=None):
        with self.motion.lock:
            cid = command_id or str(uuid.uuid4())
            if cid in self.motion.records:
                return self.motion.result(cid)
            if self.motion.external_owner or self.motion.active or self.motion.pending or self.motion.interruption:
                return dict(ok=False, message='BUSY：机械臂正在执行其他动作。')
            category = str((params.get('target') or {}).get('category', '')).lower()
            if not any(word in category for word in ('金属柱', '圆柱', '钢柱', 'metal', 'cylinder', '柱体')):
                return dict(ok=False, message='工业演示仅支持金属柱，请说“把金属柱放入篮子”。')
            count = len(self.remaining) if params.get('demo_stack') else 1
            if not self.remaining:
                return dict(ok=False, message='金属柱已全部码放，请重启场景复位。')
            self.control.set_follow_enabled(False)
            self.motion.saved = None
            self.motion.gripper_hold_target = None
            self.motion.external_owner = cid
            self.job = dict(cid=cid, left=count, phase=-1, start=self.clock(), cancelled=False)
            self.motion.records[cid] = dict(ok=True, state='accepted', command_id=cid,
                skill='grasp', demo=True, progress_seq=0, message='已启动金属柱固定取放演示。')
            for old in list(self.motion.records):
                if len(self.motion.records) > 256 and old != cid:
                    del self.motion.records[old]
            return self.motion.result(cid)

    def cancel(self):
        with self.motion.lock:
            if self.job:
                self.job['cancelled'] = True

    def _finish(self, state, message):
        self.motion._finish(self.job['cid'], state, message)
        self.motion.external_owner = None
        self.motion.holding = True
        self.motion.hold_target = None  # next tick captures current measured pose
        self.job = None

    def poll(self):
        now = time.monotonic()
        with self.motion.lock:
            if self.warmup:
                self.warmup -= 1
            else:
                self.samples.append(now-self.last_wall)
            self.last_wall = now
            if not self.job:
                return
            job = self.job
            if job['cancelled'] or self.motion.interruption:
                self._finish('cancelled', '工业演示已停止；已放置的物体保留，当前物体暂停。')
                return
            if not self.motion.snapshot.get('simulation_playing', False):
                return
            try:
                elapsed = self.clock()-job['start']
                if job['phase'] < 0 or elapsed >= self.duration:
                    if job['phase'] >= 0:
                        self.drive.step(1.)
                    job['phase'] += 1
                    if job['phase'] == 8:
                        self.remaining.popleft()
                        self.placed += 1
                        job['left'] -= 1
                        if job['left'] == 0:
                            self._finish('completed', f'金属柱取放演示完成，篮内共 {self.placed} 根。')
                            return
                        job['phase'] = 0
                    job['start'] = self.clock()
                    elapsed = 0
                    record = self.motion.records[job['cid']]
                    record.update(state='started', phase=['approach','descend','grip','lift','transfer','lower','release','retreat'][job['phase']],
                                  progress_seq=record['progress_seq']+1,
                                  message=f'正在演示第 {self.placed+1} 根金属柱取放。')
                    self.drive.begin(self.items[self.remaining[0]], job['phase'])
                self.drive.step(min(1., elapsed/self.duration))
            except Exception as exc:
                self._finish('failed', f'演示动作失败：{exc}')

    def close(self):
        self.cancel()
        self.poll()


def solve_waypoint(kinematics, point, start, base, defaults):
    # A fixed second seed avoids branch-local minima at near-base positions.
    try:
        return kinematics.solve(point, None, start, base)
    except ValueError:
        return kinematics.solve(point, None, {**start, **defaults}, base)


class IsaacDemoDrive:
    """IK computes fixed waypoint joint poses; playback uses direct articulation
    poses and kinematic props. This deliberately bypasses contact grasping.
    """
    def __init__(self, arm, motion, base, ops, items):
        self.arm, self.motion, self.base, self.ops = arm, motion, base, ops
        self.cache = {}
        # Solve preset poses before rendering the demonstration to avoid
        # per-phase solver stalls. These are classical IK, never model calls.
        for item in items:
            for key in ('source', 'destination'):
                for dz in (0., .12):
                    point = np.array(item[key]) + [0, 0, dz]
                    self.cache[tuple(point)] = solve_waypoint(motion.kinematics, point,
                        motion.defaults, base, motion.defaults)

    def begin(self, item, phase):
        if phase == 0:
            item["source"] = list(self.ops[item["index"]][0].Get())
        self.item, self.phase = item, phase
        self.start = dict(zip(self.arm.dof_names, self.arm.articulation.get_dof_positions().numpy().reshape(-1)))
        src, dst = np.array(item['source']), np.array(item['destination'])
        high_src, high_dst = src+[0,0,.12], dst+[0,0,.12]
        point = [high_src, src, src, high_src, high_dst, dst, dst, high_dst][phase]
        cached = self.cache.get(tuple(point))
        self.goal = dict(cached) if cached is not None else solve_waypoint(
            self.motion.kinematics, point, self.start, self.base, self.motion.defaults)
        self.goal['gripper'] = .15 if 2 <= phase <= 5 else 1.0

    def step(self, fraction):
        from pxr import Gf
        t = fraction**3 * (10-15*fraction+6*fraction*fraction)
        q = {name: self.start[name]+(self.goal[name]-self.start[name])*t for name in self.start}
        values = [[q[name] for name in self.arm.dof_names]]
        self.arm.articulation.set_dof_positions(values)
        self.arm.articulation.set_dof_position_targets(values)
        self.arm.articulation.set_dof_velocities([[0.]*len(self.arm.dof_names)])
        translate, rotate = self.ops[self.item['index']]
        if 2 <= self.phase <= 5:
            point = self.motion.kinematics.forward(q, self.base)[:3,3]
            translate.Set(Gf.Vec3d(*point))
            rotate.Set(0.)
        elif self.phase >= 6:
            translate.Set(Gf.Vec3d(*self.item['destination']))
            rotate.Set(0.)
