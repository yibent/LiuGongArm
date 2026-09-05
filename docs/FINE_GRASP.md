# FineGrasp 工程指南

本文说明 BusAgent 的通用精细抓取节点、eye-in-hand 数据契约、闭环执行逻辑、Isaac Sim 运行与真机接入前置条件。模型选择与许可证证据见 [MODEL_EVALUATION.md](MODEL_EVALUATION.md)。

## 能力边界与当前证据

`GeneralGraspNode` 接管机械臂已经到达目标附近后的阶段。它接收上游目标提示，用腕部 RGB-D 的新观测完成局部分割、抓取候选生成、可行性筛选、pre-grasp、逐步视觉伺服、闭合、抬升和验证。候选生成器、相机、分割器、运动执行器、夹爪和验证器都是可替换依赖；BusAgent 的请求与结果类型不随底层模型改变。

截至 2026-09-05，证据应按下面的范围解读：

- 本机 Isaac Sim 已保存一次由 **GraspGenX primary 候选直接驱动、未使用 fallback** 的默认未知立方体闭环成功 run：模型 687 ms、19 个视觉伺服步、最终平移/姿态误差 3.39 mm/0.15°、视觉验证抬升 82.05 mm、Isaac 刚体实际抬升 82.06 mm、FineGrasp 节点循环 17.875 s（不含 server/Isaac 启动）。`selected_grasp.backend=graspgenx` 且没有 `fallback_reason`。证据目录为 `output/fine_grasp_runs/20260905_072102_graspgenx_cube/`；`output/` 被 Git 忽略，这是保留在本机的运行证据，不属于分支内容。这是单次工程回归，不是 unseen 泛化成功率。
- unseen smoke 的球体单次试验以 `gripper_width_infeasible` 结束，物体没有抬升。该失败记录在 `output/unseen_benchmark_smoke/summary.json`，说明完整 unseen suite 仍需继续收敛。
- 反光金属圆柱只执行过 geometric dry-run：产生可行候选（宽度 31.37 mm、score 0.577），没有闭合或抬升，因此不能计为抓取成功。证据在 `output/metallic_benchmark_smoke/`。
- 第二台机器在较早提交 `b0b1b02` 上通过了 24 个无 GUI 单元测试和 eye-in-hand 相机运动检查；没有完成远端抓取成功验证。证据在 `output/remote/eye_in_hand/`。
- LeRobot 接口目前有可注入的相机、夹爪和规划运动桥及 fake-robot 单元测试，但尚未在真机上通电执行抓取。真机章节是接入清单，不是已完成验证声明。
- GraspGenX 已完成本地权重推理、ZMQ、时延、显存、输出稳定性实测和上述本项目 Isaac 物理仿真闭环成功。尚未做真机抓取。VGN 有官方仿真环境的小规模实测，但尚未接入运行时节点。

## 一键运行 Isaac 闭环 demo

从仓库根目录 `D:\liuGong` 运行：

```powershell
scripts\run_fine_grasp_demo.bat
```

该 wrapper 默认选择 `graspgenx`：检查本机 Isaac/model Python、vendor code、checkpoint 和 gripper description；如果指定端口没有服务，则隐藏启动官方 ZMQ server，等待端口可连接，再执行 warmup。wrapper 最多等待 warmup 45 s，并以 warmup JSON 的 `status=ready` 作为硬性就绪契约，之后才启动 Isaac；如果已收到 ready 但 Isaac Python 返回非零 Windows shutdown code，只记录 warning。结束时只停止自己创建的 server。服务和 warmup 日志位于 `output/graspgenx_server/`。路径目前按本机 `D:\isaac\env_isaacsim60` 和仓库内 `_envs/_vendor/_models` 布局配置。若端口启动前已可连接，wrapper 复用已有服务并跳过自身的依赖检查/warmup，因此操作者必须确认该端口确为已就绪的 GraspGenX。

每次 wrapper 运行默认创建按秒时间戳目录 `output/fine_grasp_runs/<yyyyMMdd_HHmmss>_<backend>/` 并打印绝对路径，可用 `-Output <path>` 指定其他目录。常规串行运行由此保留每次 `report.json` 和图像；同一 backend 若在同一秒并发启动会命中同一路径，并发测试必须显式传不同的 `-Output`。判断实际候选来源仍应读取 `selected_grasp.backend` 和可选的 `selected_grasp.metadata.fallback_reason`。

wrapper 默认是 `graspgenx`，直接运行 `run_fine_grasp_demo.py` 的默认则是 `geometric`，两者不要混淆。单次 demo 的独立物理仿真判据是：

```text
report.result.success == true
AND report.actual_target_lift_m >= 0.025
```

第二项必须用 Isaac ground-truth rigid-body 位移；`result.metrics.target_lift_m` 是视觉验证量，不能替代它。当前 demo 的进程返回逻辑只看节点 success，因此严格回归使用 batch runner。

只验证不依赖权重服务的当前闭环基线：

```powershell
scripts\run_fine_grasp_demo.bat -Backend geometric
```

直接调试 Python runner 时可用：

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'; & 'D:\isaac\env_isaacsim60\python.exe' `
  scripts\run_fine_grasp_demo.py --headless --backend graspgenx --graspgenx-port 5556
```

Kit/Isaac 的宿主进程退出码在部分失败路径上可能仍为 0。自动化必须读取 wrapper 打印的 run 目录内 `report.json` 的 `result.success` 和 `actual_target_lift_m`，不能只看进程退出码。若直接运行 Python 且未传 `--output`，才会写入可覆盖的旧默认目录 `output/fine_grasp_demo/`。若 Python/Kit 抛出异常，查看相同 run 目录中的 `error.txt`。

运行参数：

```text
--headless / --no-headless
--backend geometric / --backend graspgenx
--graspgenx-port <port>
--dry-run
--case-json <case.json>
--output <per-run-directory>
```

`--dry-run` 会完成观测、候选筛选、IK/碰撞可行性检查并返回候选，不执行 pre-grasp、夹爪或抬升。它不是物理成功。

## Unseen-object 回归

批量脚本为每个 case 启动独立 Isaac 进程，避免仿真状态跨 case 泄漏：

```powershell
& 'D:\isaac\env_isaacsim60\python.exe' scripts\run_fine_grasp_benchmark.py `
  --isaac-python 'D:\isaac\env_isaacsim60\python.exe' `
  --cases cube,sphere,cylinder,thin,metallic_part `
  --seeds 0 `
  --backend geometric `
  --output output\unseen_benchmark
```

默认集合覆盖立方体、球体、圆柱、7 mm 薄片和反光金属圆柱。case JSON 可用 `--case-json` 追加。一次 trial 只有同时满足以下条件才算成功：

1. `GeneralGraspNode` 返回 `success=true`；
2. Isaac 中目标刚体相对初始位置真实抬升至少 25 mm。

聚合结果写入 `summary.json` 和 `SUMMARY.md`。不要用 dry-run 结果计算物理成功率。`summary.json` 的每个 run 同时记录 `backend`（实际 selected backend）、`configured_backend`（配置的 backend/wrapper 名）和 `fallback_reason`，因此可以把 GraspGenX primary 成功与 geometric fallback 成功分开统计；当前 `SUMMARY.md` 表格只显示实际 `backend`，完整归因以 JSON 和原始 `report.json` 为准。

## BusAgent 稳定接口

稳定数据契约在 `source/mr_liu/grasp/contracts.py`：

- 输入：`FineGraspRequest(target=TargetSpec(...), request_id, timeout_s, dry_run)`。
- 目标：`TargetSpec.object_id` 必填；可附带 `semantic_label`、基座系粗位置和 `ObjectProperties`。
- 输出：`FineGraspResult(success, phase, failure, message, selected_grasp, metrics, debug_artifacts)`。
- 取消：`FineGraspSkill.cancel()` 转发给节点，节点设置取消标志并立即调用 `motion.stop()`。
- BusAgent 包装：`source/mr_liu/control/skills/fine_grasp.py` 的 `FineGraspSkill`。

上游粗位置只用于在最新腕部图像中引导局部分割，不是一次性抓取位姿。`SelectedGrasp` 会记录候选来源、观测序号、基座系 grasp/pre-grasp 以及 `T_object_grasp`，便于闭环跟踪和复盘。

节点通过以下 Protocol 注入具体实现：

| 边界 | 核心方法 | 当前实现 |
|---|---|---|
| `WristCamera` | `capture(target)` | `IsaacWristCamera`、`LeRobotWristCamera` |
| `TargetSegmenter` | `segment(observation, target)` | observation mask、seeded depth、appearance+depth tracker |
| `GraspBackend` | `generate(observation, points, target)` | GraspGenX、geometric antipodal；VGN 尚未接入 |
| `MotionExecutor` | reachability、collision、move、servo、stop | Isaac cuMotion、planned LeRobot bridge |
| `GripperController` | open、close、state | Isaac SO-101、LeRobot SO-101 |
| `GraspVerifier` | verify closed/lifted | wrist vision + gripper state |

构造时直接注入这些依赖，模型工厂只负责候选后端：

```python
from mr_liu.grasp.backends.factory import create_grasp_backend
from mr_liu.grasp.node import GeneralGraspNode

backend = create_grasp_backend(fine_grasp_config())
node = GeneralGraspNode(
    camera=camera,
    segmenter=segmenter,
    backend=backend,
    motion=motion,
    gripper=gripper,
    verifier=verifier,
    settings=settings,
    trace=recorder.trace,
)
result = node.execute(request)
```

`configs/fine_grasp.yaml` 的 `backend` 目前接受 `geometric` 或 `graspgenx`。GraspGenX 可配置 `fallback_backend: geometric`；服务异常或空输出时，候选 metadata 会带 `fallback_reason`。如要禁止降级，将 fallback 设置为 `none`，让模型故障显式返回。

## Eye-in-hand 坐标链

所有齐次变换使用 `T_destination_source` 命名，即把 source 坐标表达转换到 destination。对第 `k` 个同步观测：

```text
T_base_camera(k) = T_base_ee(k) @ T_ee_camera
T_base_grasp(k)  = T_base_camera(k) @ T_camera_grasp(k)
T_object_grasp   = inverse(T_base_object(k)) @ T_base_grasp(k)
```

相机光学坐标采用 OpenCV 约定：`+X` 向图像右、`+Y` 向图像下、`+Z` 向前。候选抓取坐标采用 `+X` 夹爪闭合轴、`+Z` 从夹爪指向物体的接近轴、`+Y` 构成右手系。

`RGBDObservation` 同时携带 `T_base_camera`，并可携带 `T_base_ee` 与 `T_ee_camera`。若后两者存在，构造器会验证：

```text
T_base_camera ≈ T_base_ee @ T_ee_camera    (atol = 2e-4)
```

候选必须与当前 observation 的 `sequence` 完全相同。选择器拒绝旧序号候选，节点也拒绝非递增帧；机械臂运动后不能把旧 `T_camera_grasp` 与新相机位姿相乘。

GraspGenX 的 object-points 路径默认设置 `inference_frame: base`。最新腕部点云先按同一观测的位姿转换：

```text
P_base = T_base_camera(k) @ P_camera
T_camera_grasp = inverse(T_base_camera(k)) @ T_base_grasp_model
```

这是因为 GraspGenX 的 MoE 分支把输入坐标的重力/竖直轴用于抓取族构造。腕部相机随手臂旋转，直接把 camera `+Z` 当作固定重力轴会生成方向错误的候选；基座系 `+Z` 才是本场景稳定的重力轴。模型输出转回当前 camera frame 后仍满足 BusAgent 的 `GraspCandidate` 契约。`scene_depth` 模式保留相机系像素/内参语义，配置校验要求它使用 `inference_frame: camera`。

Isaac 适配器每次 capture 先推进两个 render frame，再同时读取 RGB、深度、相机 world pose 和 EE pose，以补偿 Isaac 6.0 RTX annotator 的一帧延迟。它还检查推导出的 `T_ee_camera` 在运动中是否保持刚性。仿真 CPU PhysX 用于保持 articulation USD transform 与 RTX sensor 同步；cuMotion 仍可使用 CUDA。

真机必须把 RGB、深度和用于 FK 的关节状态同步到同一采样时刻。`LeRobotWristCamera` 默认 `require_synchronized_snapshot=True`，接受两种严格输入：注入返回 `SynchronizedLeRobotRGBD(observation, rgb, depth, timestamp_s)` 的 driver-owned provider；或在同一个 LeRobot observation 中提供 `<camera_key>`、`<camera_key>_depth` 和 `<camera_key>_timestamp`。传给 `ee_pose_from_observation` 的正是该 snapshot 的 observation。缺少同步 provider/同 observation timestamp 时默认直接报错，不再把锁外 camera buffer 与旧关节拼接。

## 闭环状态机

```text
OBSERVE -> GENERATE -> PREGRASP -> SERVO -> CLOSE -> VERIFY_CLOSE
   ^                         |                          |
   |                         +-- drift/IK change ------+
   |                                                    v
   +--------------- replan                         LIFT -> VERIFY_LIFT
                                                       |          |
                                                       v          v
                                                    FAILED     SUCCEEDED
```

具体行为：

1. `OBSERVE`：读取新腕部 RGB-D，检查时间、序号、目标 mask、有效深度比例，去除 MAD 深度离群点并反投影点云。
2. `GENERATE`：低频后端从当前腕部点云生成多候选。GraspGenX object-points 默认先转到基座/重力对齐坐标供 MoE 推理，再把抓取转回当前相机系，并做 SE(3) 清洗、聚类、共识加分和跨观测滞回；geometric 后端直接用当前相机系目标点云 PCA 产生侧向 antipodal 候选。
3. 选择器依次过滤旧候选、低分、夹爪宽度、桌面间隙、pre-grasp/grasp IK、碰撞和 pre-grasp 到 grasp 的局部路径。目标接触只在 grasp/servo 阶段允许。
4. `PREGRASP`：夹爪打开后移动到沿抓取接近轴后退 60 mm 的位姿。
5. `SERVO`：每一步都重新读取腕部图像。轻量 tracker 用新观测更新目标平移，保留低频模型的姿态和接触偏置；单步平移不超过 6 mm、旋转不超过 3°。默认要求误差小于 4 mm/5° 连续 3 帧。
6. 当 active target 被轻量 tracker 修正到相对原方案平移超过 15 mm，且距离上次模型调用至少 0.5 s 时，退出快环重新生成。配置还保留 12° 姿态阈值，但当前 tracker 只更新平移，因此该阈值要到接入可靠的局部姿态 tracker 后才会实际触发。单帧 centroid jump 超过 35 mm 时该样本不更新 active target，只记录 outlier；正常样本的创新再限幅到 4 mm。默认最多 6 次 replan、40 个伺服步、总计 45 s。
7. `CLOSE`：按物品策略闭合。`VERIFY_CLOSE` 组合夹爪非空宽度、stall/contact 和腕部视觉；允许小物体被手指完全遮挡。
8. `LIFT`：基座 `+Z` 抬升 80 mm。`VERIFY_LIFT` 比较最新腕部图像中的目标基座系高度，并结合夹爪状态。

重模型只负责低频抓取族提议。最后几厘米的位置更新和运动约束由高频几何 tracker、视觉伺服、IK 与碰撞检查负责。

## 目标属性策略

当前策略来自结构化 `ObjectProperties`，并兼容少量中英文标签提示：

| 属性 | 执行变化 |
|---|---|
| 水果/`fragile` | 夹持力最多 6 N、闭合速度最多 10 mm/s、运动速度比例 0.35 |
| 薄物体/`thin` | 桌面间隙至少 12 mm、运动速度比例 0.35；不满足夹爪或桌面几何时返回不可行 |
| 金属/`reflective` | 仅填补 mask 内面积不超过 160 px、边界深度跨度不超过 12 mm 的小孔；大面积缺深仍失败 |
| 未知 | 默认 12 N、25 mm/s、运动速度比例 0.55；应在真机标定后再确认这些值 |

`max_grip_force_n` 可进一步下压力上限。当前 `fragile/deformable` 不等价于触觉顺应控制；没有经标定的力限制时不得宣称安全夹持易损品。

## 调试产物

wrapper 单次 demo 默认写入唯一的 `output/fine_grasp_runs/<timestamp>_<backend>/`；下表中的路径均相对该 run 目录。直接运行 Python 且不传 `--output` 时才使用 `output/fine_grasp_demo/`：

| 文件 | 用途 |
|---|---|
| `report.json` | 节点结果、最终夹爪、刚体实际抬升、关节/EE、运动学诊断、完整 trace |
| `debug/trace.jsonl` | 每次 observe/generate/pregrasp/servo/close/lift/verify 的增量事件 |
| `debug/obs_NNNN_rgb.png` | 原始腕部 RGB |
| `debug/obs_NNNN_depth.png` | 每帧自适应色标的深度可视化，不可用于读取绝对深度 |
| `debug/obs_NNNN_overlay.png` | 目标 mask 叠加图 |
| `debug/obs_NNNN.json` | 时间戳、mask 像素、有效深度和 `T_base_camera` |
| `initial_wrist.png`、`final_wrist.png` | 入口/结束快速对照 |
| `error.txt` | 未被节点分类的 Kit/Python 异常 |

优先检查顺序：`result.failure` → trace 的最后一次 phase/event → 对应 observation overlay/depth → `rejected_*` metrics → `kinematic_diagnostics` → 目标实际抬升。节点 success 和物理抬升不一致时，batch runner 会分类成 `false_positive_no_physical_lift` 或 `verification_false_negative`。

## 失败码

| 类别 | 失败码 | 含义/首要检查 |
|---|---|---|
| 请求与控制 | `invalid_request`、`aborted`、`internal_error` | 请求字段、取消来源、异常栈 |
| 观测 | `no_observation`、`stale_observation`、`target_not_visible`、`inconsistent_observation`、`insufficient_depth`、`calibration_invalid` | 相机流、时间域、sequence、mask/depth、手眼外参 |
| 模型 | `no_grasp_candidates`、`model_unavailable`、`model_timeout`、`model_protocol_error`、`model_inference_failed` | 服务进程、端口、权重、响应数组、阈值；确认是否发生 fallback |
| 几何可行性 | `gripper_width_infeasible`、`table_clearance`、`collision`、`ik_unreachable` | 候选宽度、指尖/桌面、world obstacle、IK/路径与 EE-pinch 标定 |
| 闭环 | `servo_timeout`、`target_moved` | 新帧率、tracker 跳变、稳定帧、replan 上限 |
| 夹取验证 | `gripper_failure`、`grasp_not_detected`、`lift_verification_failed` | 力限制、stall 误判、遮挡、物体是否随夹爪抬升 |

并非所有枚举当前都有独立触发路径；保留它们是为了不同相机、校准器和后端统一上报。不要把不可达或夹爪宽度不符改写成网络“猜一个动作”。

## 真机 LeRobot 接入前置

真机桥位于 `source/mr_liu/grasp/adapters/lerobot.py`。它只依赖 LeRobot 公共形状 `get_observation()` / `send_action()`；导入模块不会连接串口或使能电机。接入前必须完成以下工作。

### 1. RGB-D 与时钟

- 腕部相机驱动必须启用 depth，并确保 RGB 与 depth 已配准到同一内参与分辨率。
- 明确深度单位并设置 `depth_scale_m`；0、负值和非有限值会转为 NaN。
- 将相机时间戳和控制器 clock 放在同一时间域。生产默认必须注入 `SynchronizedLeRobotRGBD` provider，或让同一个 observation 同时携带 RGB、depth、camera timestamp 和对应关节；缺失时间戳会被严格模式拒绝。
- `ee_pose_from_observation` 必须使用同步 snapshot 中传入的 observation 做 FK，不能再读取一遍“当前关节”。只有显式设置 `require_synchronized_snapshot=False` 才能进入无源时间戳的 legacy/test fallback；该 opt-out 不得用于真机最后接近。

### 2. 内参与手眼标定

- 标定并版本化相机内参、畸变模型和 RGB-depth 外参；输入节点前完成去畸变/对齐。
- 标定固定的 `T_ee_camera`，注意方向必须是 camera 到 EE 的表达转换；用 `T_base_ee @ T_ee_camera` 重建相机 pose。
- 在多个腕部位置和姿态观测固定标定物，检查重建的基座系点是否稳定。真实系统尚未建立合格阈值；阈值应显著小于 4 mm 伺服容差，并作为设备校准记录，而不是沿用仿真数字。
- 重新安装相机、改变指尖、碰撞或维护后必须重标定。外参不确定时禁止进入最后接近。

### 3. IK、碰撞与工具几何

- 向 `PlannedLeRobotMotionExecutor` 注入实现 `KinematicsCollisionPlanner` 的 cuRobo、MoveIt 或等价规划器；网络不得直接输出关节命令。
- 确认关节名、弧度/角度转换、限位、base/EE frame 和 tool frame 与真机一致。
- 标定“模型的对称 pinch center”到 SO-101 EE 的宽度相关偏置。默认 `pinch_center_x_per_width=-0.5` 只是当前单动指 SO-101 几何近似，换指尖或换夹爪必须重测。
- world model 至少包含桌面、基座、相机和静态障碍；最后接触阶段只能豁免目标物体，不能豁免桌面或机器人自碰撞。

### 4. 夹爪与力

- 标定 `LeRobotGripperCalibration` 的开/闭位置和实际开口宽度。
- LeRobot position action 本身不表达牛顿力。默认 `LeRobotGripperController` 要求注入 `set_force_limit_n`；缺失时 close 返回 false。
- `FeetechTorqueLimitBridge` 的 `full_scale_force_n` 必须通过测力计实测。默认寄存器上限 500 只是保守限幅，不是力标定。
- 当前 contact 主要来自位置受载后的 stall 推断，不是六轴力传感器。应分别验证空闭合、物体接触、桌面碰撞的区分能力。

### 5. 上机安全门

- 首次运行先用 `dry_run`，再在远离桌面的位置限速验证 FK、相机链和夹爪方向。
- 配置硬件急停、软限位、最大关节步长、工作区、watchdog 和通信丢失停止；操作者必须能立即断力。
- 从低力、低速、软物体和无旁人工作区开始。薄片、反光/透明物、尖锐工具以及未知负载在单独验证前不进入无人值守运行。
- 任何 stale frame、校准不一致、IK/碰撞失败、夹爪不可行或验证失败都应停止/退回，不得开环继续旧轨迹。

## 维护规则

- 修改 frame、指尖或夹爪时，同时更新 URDF/XRDF、`T_ee_camera`、sweep volume、pinch-center 偏置和回归 case。
- 新后端的核心边界是实现 `GraspBackend.generate()` 并输出当前 camera frame 的候选，不在后端内控制机器人。若要支持 YAML/CLI 选择，还需同步注册 factory、runner choices、配置校验和 contract/backend 测试。
- 新相机适配器必须提供单调 sequence、同帧 RGB-D、同时间 EE pose 和米制深度。
- 任何性能或成功率声明都必须指向可复现 artifact，并区分模型推理、Isaac 物理抓取、远端回归和真机测试。
- 合入前至少运行无 GUI 单测、默认 cube 闭环和 unseen suite；以 batch summary 的节点+物理双条件作为回归门。
