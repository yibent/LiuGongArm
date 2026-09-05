# SO-101 运动稳定性调整（2026-09-05）

## 当前实现

- `MotionCommands.tick()` 在 Isaac `PHYSICS_POST_STEP` 回调执行；轨迹与到位持续时间使用 `SimulationManager.get_simulation_time()`，墙钟仅用于审计时间戳和 120 秒最终超时。
- 物理步长 1/120 秒。它是仿真时间步长，不代表服务器能以现实时间 120 Hz 运行。
- 双相机推理和 JPEG 编码在单工作线程执行；只允许一个任务在途，不排队积累旧帧。摄像头读取、USD 操作、关节读写仍在 Isaac 线程。
- RGB、深度、语义标签、反投影位姿和观察时间一起冻结；异步返回后不使用相机的新位姿解释旧深度，旧 prompt 结果不作为新目标定位结果。
- 采用五次平滑插值（起止速度、加速度均为零）。运动时长同时满足峰值速度 28.65°/s 和峰值加速度 60°/s²；`set_speed` 仍可修改速度。
- 到位要求：所有关节误差小于 1°，帧间实测速度小于 1°/s，连续稳定 0.35 仿真秒。轨迹结束后最多等待 10 仿真秒。
- 保留原 USD 刚度及 10 Nm 最大驱动力，显式配置关节阻尼；API 单位为 Nm/rad 和 Nm·s/rad，不能把 USD 的每度数值直接填入。运行时实际值位于 `/api/status.motion.drives`。
- 复用资产自带 `/World/SO101/root_joint`，按底座世界位姿设置锚点，不再新增第二个固定约束。求解器位置迭代 32、速度迭代 1；关节物理速度上限 60°/s。

## TGS 速度的处理

实测时，关节位置已稳定，TGS 返回的末子步速度仍可能达到约 4.8°/s。它与相邻物理帧位置变化率不是同一个量。因此保留 `joint_velocities_deg_s` 原始遥测，同时增加 `joint_frame_velocities_deg_s`；生产配置 `settle_velocity_source: position_delta` 使用后者判断可见帧尺度的停稳，不使用指令速度假装实测速度。

这一差异与 [NVIDIA 对 TGS 静止速度的解释](https://forums.developer.nvidia.com/t/dof-velocity-offset-at-rest/205799/13)一致。求解器采用少量速度迭代，参考 [NVIDIA Simulation Control](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/simulation_control/simulation_control.html)。这不代表内部子步完全没有运动，也不是硬件机械臂的速度测量方案。

## 验证及边界

41 项 Python 回归测试通过，包括原动作生命周期、暂停/继续、未到位超时、仿真时钟、轨迹限速限加速度、停稳判定、真实位置抖动拒绝完成、异步推理不阻塞、相机位姿冻结。

线上 `scripts/verify_motion_settling.py` 底座、肩部、腕部各做 5° 往返，6 次均完成并停稳。底座过冲约 0.0045°；肩部最终静态误差约 0.24°。旧版同类底座测试过冲约 3.74° 并超时。**重启前后的起始姿态不同，这些数值不是严格同姿态 A/B 对照，不能推算所有姿态的改善比例。**

补测归位、末端向上 5 mm、再次归位均完成并停稳，前两项到位时最大关节误差约 0.054°。绿色方块 RGB-D 定位也通过，确认只返回坐标而没有触发动作；测试结束后恢复此前感知目标。

状态采样在最终小幅测试中的观测间隔中位数约 37–41 ms；HTTP 采样会跳过物理步，不能将它等同于物理回调频率。渲染与推理仍共享 GPU，现实时间速度受负载影响。

本次没有加入抓取、碰撞规避或重力补偿；静态重力偏差仍存在，复杂姿态、接触和关节极限附近仍需继续测试。现有自碰撞检测没有关闭。

## 运维

改 `configs/motion.yaml`、`configs/robot_so101.yaml` 或上述控制代码后，需要重启 Isaac 控制进程，而非只重启 BusAgent。

小幅实测会真实驱动仿真机械臂，应在空闲、停止自动跟随时运行：

```bash
python3 scripts/verify_motion_settling.py
python3 scripts/verify_motion_settling.py --joint shoulder_lift
python3 scripts/verify_motion_settling.py --joint wrist_roll
```

服务器测试须使用 Isaac Python 的正确动态库路径；裸 `.venv/bin/python` 会产生 ctypes 符号冲突：

```bash
LD_LIBRARY_PATH=/root/isaacsim/kit/python/lib:$LD_LIBRARY_PATH /root/isaacsim/python.sh -c 'import sys,unittest; sys.path.append("/root/gpufree-data/LiuGongArm/.venv/lib/python3.12/site-packages"); result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover("tests")); sys.exit(not result.wasSuccessful())'
```

部署前备份：`/root/gpufree-data/deploy-backups/before-motion-stability-20260905.tgz`。最终 Isaac 日志：`logs/isaac-motion-stability-final.log`。
