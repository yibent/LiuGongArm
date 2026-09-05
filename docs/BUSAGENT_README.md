# BusAgent 接入 README：GeneralGrasp / FineGrasp

更新：2026-09-05。代码基线：`2c7b53f`，开发分支：`codex/fine-grasp-loop`。

这是可接入 BusAgent 的**精细抓取开发模块**，不是完整 VLA，也不是已经通过
工业场景或真机泛化验收的产品。上游负责目标选择、粗定位、避障和快速接近；
本模块接管目标附近的腕部感知、抓取候选、最后接近、夹爪闭合、抬升与验证。

本文是当前交接入口；历史单次成功不能代替最新基线结果。

合并后的实时接入见 [BusAgent 抓取运行说明](BUSAGENT_GRASP_RUNTIME.md)：现有场景已经增加
HTTP `grasp`、机械臂队列占用、中断及阶段反馈。下文模型评估数字仍是抓取分支的历史基线，
不代表新运行入口已通过 GPU 实测。

## 1. 我们开发了什么

| 组件 | 已完成的工作 | 当前边界 |
|---|---|---|
| BusAgent 接口 | `FineGraspSkill` 包装 `GeneralGraspNode`；实时入口增加 HTTP `grasp` 调度 | 新入口待 GPU 验证；无 ROS 服务 |
| 抓取模型 | GraspGenX 本地权重推理、独立 ZMQ 服务、候选转换与筛选；在线禁止几何回退 | 工程选定 primary，不是已证明优于所有模型 |
| 腕部闭环 | 同帧 RGB-D/渲染位姿、手眼坐标链、逐步重新观察、目标平移跟踪、pose servo | 姿态突变检测、长遮挡跟踪和时延控制仍需增强 |
| 执行约束 | IK、world collision、桌面间隙、夹爪开口和接近检查 | SO-101 指尖/扫掠体含近似，不能视为完备碰撞保证 |
| 属性策略 | 易损品限力限速、薄件间隙检查、反光物体小深度孔洞恢复 | 规则策略，不是逐物体训练；真机力值必须标定 |
| 双 RGB-D | 仿真头顶和腕部均输出 RGB-D；保留旧头顶 prim 路径 | 头顶数据尚不参与 FineGrasp 决策，两路同时曝光/融合未完成 |
| 实验多视角 | 显式启用的 2–3 个腕部安全视点、局部表面融合、SAM 2/几何身份约束 | 原型，不是完整 TSDF；不是失败后自动重观察/重抓 |
| 真机接口 | LeRobot 腕部、夹爪、规划执行桥及 fake-device 测试 | 未完成真实设备标定和通电抓取验收 |
| 调试/测试 | 每帧 RGB-D、mask、变换、trace、失败分类、独立物理抬升评估 | 基线 82 项单测通过；本次增加 README 示例/链接测试后共 84 项通过，不等于真实抓取成功率 |

基础执行关系：

```text
BusAgent 目标 + 已接近的机械臂
  → 最新腕部 RGB-D → 局部分割/点云 → 低频 GraspGenX 候选
  → IK/碰撞/夹爪筛选 → pre-grasp
  → [新腕部观测 → 小步修正 → 再观察]，必要时重新生成候选
  → 闭爪 → 检查夹爪/视觉 → 抬升 → 再验证 → FineGraspResult
```

末端位置不是网络直接输出的关节动作。仿真使用现有 Isaac cuMotion 执行器；
真机规划桥允许注入 cuRobo/等价 IK 与碰撞规划器，不代表真机 cuRobo 已部署。

## 2. 模型选择与泛化证据

- 已实际运行 **GraspGenX 和 VGN**。GraspGenX 集成进运行节点；VGN 只做过
  独立官方仿真实验，没有 `--backend vgn`。两者测试环境不同，不能把成功率直接排名。
- 当前 GraspGenX 使用 GraspMoE：diffusion 与几何 OBB 候选合并，由学习到的
  discriminator 评分。本项目不少选中候选来自 OBB 分支，不能描述成纯 diffusion 成功。
- 没有在本项目为苹果、杯子、工具等分别采集 demonstrations 或微调模型。
  目标是复用预训练几何抓取能力，但**未见物体泛化仍须独立测试证明**。
- 历史模型独立热推理：GraspGenX 100 候选 p50 约 689 ms、p95 707 ms，
  GPU allocated/reserved 约 684/812 MiB；不含 Isaac、图像处理和整个进程显存。
  VGN 官方仿真为 9/20 次抬升，p50 8.38 ms；不是与 GraspGenX 的同条件比较。
- 历史多 seed 代理物体试验合计 20/24 次可行抓取成功，但跨开发版本、姿态扰动小；
  不能当作当前版本成绩，更不能当作真实工业工具/水果/咖啡杯泛化指标。
- **最新 `2c7b53f` 基础 cube 回归没有通过完整成功判据**：刚体实际抬升
  80.24 mm，节点视觉随动误差 27.06 mm，超过 25 mm 阈值，返回
  `lift_verification_failed`。模型 641 ms，20 次伺服，节点耗时 13.281 s。
  本次没有放宽阈值或用仿真真值替代视觉判断。

来源与版本见 [模型评估](MODEL_EVALUATION.md)、[泛化审计](FINE_GRASP_GENERALIZATION_AUDIT.md)
和[基线实测记录](FINE_GRASP_BASELINE.md)。尚无真机 unseen-object 成功率。

## 3. 怎么运行

### 已配置的本机

在仓库根目录 `D:\liuGong`，PowerShell 执行：

```powershell
# 基础版本：GraspGenX + 腕部闭环，不主动多视角，不使用头顶深度决策
scripts\run_fine_grasp_demo.bat -Backend graspgenx -Perception single -Segmenter depth

# 检查两路 RGB-D（不执行抓取）
& D:\isaac\env_isaacsim60\python.exe scripts\verify_cameras.py

# 无需模型权重的几何基线；不用于证明神经模型性能
scripts\run_fine_grasp_demo.bat -Backend geometric -Perception single

# 单元测试，不连接真实硬件
& D:\isaac\env_isaacsim60\python.exe -m unittest discover -s tests -v
```

demo 会创建仿真场景并移动**仿真**机械臂，不会自动连接真机。默认 headless，
需要窗口加 `-NoHeadless`。wrapper 默认使用 GraspGenX，直接运行 Python runner
时应显式指定后端；`configs/fine_grasp.yaml` 与 BusAgent 在线服务默认 GraspGenX，禁止自动几何回退。

wrapper 会在 `127.0.0.1:5556` 没有服务时启动本仓库 `serve_graspgenx.py`，
warmup 后运行 Isaac，结束时只关闭自己创建的服务。复用现有端口时需自行确认
服务版本、权重和就绪状态；端口监听本身不是健康保证。直接构造 Python skill
不会自动启动模型服务，须由应用生命周期管理该服务。不要把无鉴权 ZMQ 暴露到公网。

每次默认输出 `output/fine_grasp_runs/<timestamp>_<backend>/`；使用 `-Output`
时应指定新目录，避免覆盖旧证据。进程码：0 表示节点成功、2 表示节点失败、
1 表示捕获到的 Python 异常；自动化仍必须检查报告，dry-run 的成功也不是物理成功。

### 新电脑需要什么

Git clone 不包含运行环境、模型权重或原始测试记录。当前 wrapper 是本机路径布局，
不是跨平台一键安装器。需要先配置：

```powershell
git clone --branch codex/fine-grasp-loop https://github.com/yibent/LiuGongArm.git
cd LiuGongArm
```

本次已测硬件为 RTX 4070 Laptop 8 GB（驱动 596.21）、16 GB 系统内存；这是测试
配置，不是测定的最低要求。Isaac 环境基础安装步骤见[仓库 README](../README.md)，
模型环境须另行按固定版本准备。

| 位置 | 内容 |
|---|---|
| `D:\isaac\env_isaacsim60\python.exe` | 已测 Isaac Sim 6.0.1 / Python 3.12；其他路径需修改 PS1 的 `$IsaacPython` |
| `_envs/graspgenx/python.exe` | 独立模型环境；本机 Python 3.11 / PyTorch 2.6.0+cu124，已安装 GraspGenX 与 ZMQ 依赖 |
| `_vendor/GraspGenX/` | [官方代码](https://github.com/NVlabs/GraspGenX)，已测提交 `b9429097728cb1c430dd78b92edf17ba318aad03` |
| `_models/graspgenx/checkpoints/release/` | [官方权重](https://huggingface.co/adithyamurali/GraspGenXModel)，含 `gen/`、`dis/`；revision `7c834043c11a11417e31d6d5ea9355801e40a2c1` |
| `_vendor/GraspGenX/ext/gripper_descriptions/` | 官方夹爪描述/资产，来自 [companion repository](https://huggingface.co/datasets/adithyamurali/gripper_descriptions) |

Isaac 客户端额外依赖见 `requirements-graspgenx-client.txt`，核心代码还使用 NumPy、
SciPy、PyYAML、OpenCV（当前 Isaac 环境已有）。不要把模型环境的旧 Torch/transformers
依赖装进 Isaac 环境。Windows 官方完整依赖安装存在 SharedArray 编译问题；已测推理
安装 workaround 和权重哈希见模型评估，尚未交付全新 Windows 机器的安装自动化。

GraspGenX 代码为 Apache-2.0，权重为 NVIDIA Open Model License；两者不是同一许可证。
本仓库不再分发上游权重、第三方仓库或仿真资产；项目自身尚缺明确根 LICENSE。
模型可在资产准备后本地推理，但 Isaac 首次加载场景资产仍可能联网下载。
场景通过 `source/mr_liu/sim/assets.py` 解析 NVIDIA Isaac 资产服务中的桌子、SO-101
和 HDRI；离线部署还需预先缓存/配置这些 USD 及其依赖资源，不能只复制一个场景文件。

## 4. BusAgent 如何调用

将仓库 `source/` 加入 BusAgent Python 路径。以下代码只装配接口，不连接硬件；
`camera`、`motion`、`gripper` 必须由宿主提供。每个机械臂同一时间只允许一个抓取执行者，
进入 FineGrasp 前先停止上游快环对该机械臂的命令输出。

上游可在距离目标约 5–8 cm 时安排移交，但节点没有自动判定这一距离的接管器。
必须确认目标在腕部视野内，粗位置能初始化目标区域、相机/工具没有近距离遮挡碰撞。
当前配置的有效相机深度范围为 0.025–0.60 m，至少 80 个 mask 像素、64 个有效
深度像素且有效比例至少 0.35；这些是拒绝坏观测的门槛，不是成功保证。
EE 到目标的距离不等于腕部相机的 optical-Z 深度。条件不满足时应先重新定位/观察。

```python
from pathlib import Path

from mr_liu.config import fine_grasp_config
from mr_liu.control.skills.fine_grasp import FineGraspSkill
from mr_liu.grasp.backends.factory import create_grasp_backend
from mr_liu.grasp.contracts import FineGraspRequest, ObjectProperties, TargetSpec
from mr_liu.grasp.debug import DebugSegmenter, FineGraspDebugRecorder
from mr_liu.grasp.node import GeneralGraspNode
from mr_liu.grasp.segmentation import AppearanceDepthTrackerSegmenter, SeededDepthSegmenter
from mr_liu.grasp.settings import FineGraspSettings
from mr_liu.grasp.verification import VisionGripperVerifier


def build_fine_grasp_skill(*, camera, motion, gripper, run_dir, backend="graspgenx"):
    # 每个请求使用新 run_dir 和新 tracker，避免继承上一次失败的局部状态。
    config = fine_grasp_config()
    config["backend"] = backend
    config["fusion"] = {"enabled": False}
    config["active_views"] = {"enabled": False}
    recorder = FineGraspDebugRecorder(Path(run_dir) / "debug")
    segmenter = DebugSegmenter(
        AppearanceDepthTrackerSegmenter(SeededDepthSegmenter()), recorder
    )
    node = GeneralGraspNode(
        camera=camera,
        segmenter=segmenter,
        backend=create_grasp_backend(config),
        motion=motion,
        gripper=gripper,
        verifier=VisionGripperVerifier(),
        settings=FineGraspSettings.from_mapping(config),
        trace=recorder.trace,
    )
    return FineGraspSkill(node)


def execute_nearby_target(skill, *, object_id, label, coarse_position_base_m,
                          request_id, properties=None, dry_run=True):
    target = TargetSpec(
        object_id=object_id,
        semantic_label=label,
        coarse_position_base_m=tuple(coarse_position_base_m),
        properties=properties if properties is not None else ObjectProperties(),
    )
    return skill.execute(FineGraspRequest(
        target=target, request_id=request_id, timeout_s=45.0, dry_run=dry_run
    ))
```

首次联调用 `dry_run=True`：只观察、生成/筛选，不闭爪、不运动；此时 `success=True`
只代表找到可行候选。确认设备、校准和安全门后，显式设置 `dry_run=False` 才执行抓取。
每次请求都重新调用 `build_fine_grasp_skill()`，再调用 `execute_nearby_target()`；
不要将示例构造出的 skill 作为跨失败尝试永久复用的 tracker。返回的 result 没有
`dry_run` 字段，宿主应以 `request_id` 保存请求模式及结果，只有
`not request.dry_run and result.success` 才可解释为节点宣称的抓取成功。
上述代码使用仓库 SO-101 仿真参数；真机接入必须修改桌面高度、工作区、夹爪、工具
偏置和力/速度限制，不能照抄仿真配置直接通电。

`execute()` 是阻塞调用；`timeout_s` 限制在配置总时长以内，但底层阻塞 I/O/运动仍需
驱动自己的 timeout/watchdog，不能当作硬实时截止保证。`skill.cancel()` 请求取消并
调用 `motion.stop()`，不是硬件急停，也不会保证中断正在执行的模型 RPC。
skill 名为 `fine_grasp`，aliases 是 `general_grasp`、`grasp`；BusAgent 需自行注册，
不要假定这些字符串已经在上游 Agent 的调度表中生效。

### 输入与输出要点

- `object_id` 是目标实例标识，不是类别名；`semantic_label` 是可选语义。
- `coarse_position_base_m` 是同一个 base 坐标系中的米制位置。类型允许省略，
  但上述默认 depth segmenter 在没有粗位置且没有外部 mask 时无法初始化。
  只传“抓咖啡杯”并不能让当前基础版自动从整桌定位目标。
- `ObjectProperties` 支持 `fragile`、`thin`、`reflective`、`deformable`、
  `max_grip_force_n`、`preferred_grasp`、`metadata`。当前直接策略使用前三个标志
  和最大力值。`deformable` 尚未实现专门顺应控制，`preferred_grasp` 不保证按指定
  手柄/区域抓取，`metadata` 为扩展透传信息，不保证所有后端都使用。
- 返回 `FineGraspResult`：`success`、`phase`、`failure`、`message`、
  `selected_grasp`、`metrics`、`debug_artifacts`。这些是 dataclass/Enum/NumPy 类型，
  不是自动可序列化的 HTTP JSON；可参考 demo 的 `_jsonable()`。
- `selected_grasp` 包含 base 系位姿、夹爪宽度、后端及观测序号，用于调试；
  **不能在调用返回后把它当作永久有效的位姿再次开环执行**。
- 构造函数的 `backend` 或 `metrics.backend` 不等于实际选中候选来源。请看
  `selected_grasp.backend`、`metadata.branch`、`metadata.fallback_reason`；
  `backends.graspgenx.fallback_backend` 默认是 `none`，BusAgent 在线会话也强制禁止几何降级。

## 5. 相机、机器人和模型的替换边界

稳定 Protocol 均定义在 [`contracts.py`](../source/mr_liu/grasp/contracts.py)。

| 注入项 | BusAgent/驱动必须提供 |
|---|---|
| `WristCamera` | `capture(target)` 返回同时间 RGB-D、内参、位姿、单调序号的 `RGBDObservation` |
| `TargetSegmenter` | 当前观测下的目标实例 mask，不是整类物体合并 mask |
| `GraspBackend` | `generate(observation, points_camera, target)` 返回本帧 `GraspCandidate`，不得直接驱动机械臂 |
| `MotionExecutor` | 当前 robot state、IK、collision、move、servo、stop；内部负责路径与反馈 |
| `GripperController` | `open`、带 force/speed 的 `close`、`state`；无真实力限制不能假报支持 |
| `GraspVerifier` | 闭合与抬升后的目标证据和夹爪状态检查 |

所有变换为 `T_destination_source`，即把 source 表达转换至 destination：

```text
T_base_camera(t) = T_base_ee(t) @ T_ee_camera
T_base_grasp(t)  = T_base_camera(t) @ T_camera_grasp(t)
```

RGB-D 深度为米制 optical-Z；相机轴为右/下/前。候选轴为 `+X` 闭合、`+Z` 接近。
不要组合“上一帧图像”和“当前 EE 位姿”。真机驱动必须对齐 RGB/depth/FK 的采样
时间，并与节点 `time.monotonic()` 使用同一时间域；不能拿墙钟时间直接比较。
生产接入应同时提供 `T_base_ee` 和标定的 `T_ee_camera`，而不只提供拼好的相机位姿。

仿真头顶保留 `/World/Cameras/TableTopRGB` 路径，但已输出深度。
`SceneCamera.rgbd_frame()` 返回 `CameraRGBDFrame`，含 `T_world_camera` 和渲染时间。
它不推进仿真，也不承诺每次读取是新帧；消费方须检查 `acquisition_id`/时间递增。
`world` 不必等于真机 `base`，两路相机各自的同帧采集不等于跨相机同步。

真机桥见 [`adapters/lerobot.py`](../source/mr_liu/grasp/adapters/lerobot.py)：
需要同步 RGB-D/关节 snapshot、手眼标定、FK/IK/碰撞规划器，以及经实测的夹爪
宽度和力限制 callback。原生 RGB 摄像头不能通过改 YAML 变成 RGB-D 硬件。
硬件急停、通信失效停止、watchdog、限位和有人值守测试仍是部署前置条件。

## 6. 失败该如何交给 BusAgent

失败时节点会请求 `motion.stop()`，**没有自动安全退让、松爪或再次抓取流程**。
若返回验证失败，物体可能已经握在夹爪里；上游不能简单 `open()` 后重试。

| 结果/失败码 | 上游应如何理解 |
|---|---|
| `success=True` 且非 dry-run | 节点完成了自己的验证；当前仍需要独立验收防范验证误差 |
| `no_observation` / `stale_observation` / `calibration_invalid` | 检查相机、时间与标定；不要继续旧轨迹 |
| `target_not_visible` / `identity_uncertain` / `geometry_uncertain` / `insufficient_depth` | 观测不足或身份不确定，不等于物体物理不可抓 |
| `gripper_width_infeasible` / `table_clearance` / `collision` / `ik_unreachable` | 当前候选/夹爪/姿态受约束，不证明换角度或换夹爪也永远不可行 |
| `model_*` / `no_grasp_candidates` | 检查模型服务与候选，记录是否使用 fallback |
| `target_moved` / `servo_timeout` | 停止当前尝试，重新观察后再决定；不能盲目复用候选 |
| `grasp_not_detected` / `lift_verification_failed` | 尚未确认成功；先分辨空抓、滑落与视觉遮挡，保留夹爪状态 |

失败枚举保留了不同驱动的扩展空间，不是每个枚举当前都有独立触发路径。

## 7. 调试与回归

优先看 `report.json` 的 `result.failure`，再看 `debug/trace.jsonl` 和对应帧：
`obs_NNNN_rgbd.npz`（原始 RGB、米制 depth、mask、K、变换）、RGB/depth/overlay PNG、
观测 JSON。PNG 的深度色标每帧可能不同，不能用亮度比较绝对距离。
`debug_artifacts` 目前可能为空，使用调用方保存的 `run_dir` 寻找文件；Python skill
不会自动写 demo 顶层 `report.json`，宿主需自行序列化并保存结果。

开发回归示例（多次完整仿真，会花费较长时间）：

```powershell
scripts\run_fine_grasp_demo.bat -Benchmark -Backend graspgenx -Perception single -Segmenter depth -Seeds 0,1,2 -Cases cube,apple,hammer,wrench,coffee_mug
```

物理成功统计同时要求节点成功和目标刚体抬升至少 25 mm；正确拒绝预期不可行 case
是 task outcome，不是抓取成功。当前案例中的水果、工具、杯子多为几何代理，不能
等同真实物品。VGN 官方 mesh 资产也不能保证未出现在 GraspGenX 预训练集中。
冻结 manifest 位于 `configs/eval/fine_grasp_v2/`，但跨 Git checkout 的换行/哈希
可移植性尚待修正；验收集未完成运行，不要重生成 manifest 来掩盖校验失败。

## 8. 接下来增强什么

保持请求/结果契约不变，单独开发和提交：

1. 可靠地区分空抓、滑落与“已抓住但视觉遮挡”，修复当前验证问题。
2. 失败或低置信度时安全退让；头顶 RGB-D 重新找目标，腕部选择信息更充分的视点。
3. 跨相机校准/时间门控、目标身份一致性、局部融合和目标移动后失效处理。
4. 根据缺失的接触几何选择新抓取，排除已失败方案，最后仍使用新腕部观测闭环。
5. 有上限的重试与失败记忆；独立统计首次/重试成功率、误报、不可行拒绝和总耗时。
6. 冻结多实例、多姿态、多光照、多种子测试，覆盖工业工具/零件、咖啡杯、水果；
   先仿真，再完成真机标定后做有人值守测试。

现有 `-Perception multiview -Segmenter sam2` 只是实验选项，需要额外 SAM 2 环境与
`_models/sam2/sam2.1_hiera_tiny.pt`；不能把这个开关当成以上优化已经完成。
GroundingDINO 目前仅有可选 lazy-load 路径，未完成实际泛化验收；未缓存权重时会联网。

本次按用户明确要求把 README 和开发基线推送到 GitHub，供 BusAgent 接入与继续开发；
**推送不代表泛化验收通过，不代表可无人值守上真机。**
