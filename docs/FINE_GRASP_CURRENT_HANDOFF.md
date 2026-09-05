# BusAgent 精细抓取：当前成果与工程交接

更新：2026-09-05。代码快照：`e3e0592`；开发分支：`codex/active-grasp-recovery`。
本文是当前开发线入口；[旧版接入 README](BUSAGENT_README.md) 保留基线接口与安装说明，
[主动恢复说明](ACTIVE_GRASP_RECOVERY.md) 保留分阶段实验记录。

合并说明：本文的相机安装参数、30 Hz 频率和实验结果记录的是上述开发分支快照。
合入主线时保留了 Web 运行配置：两路相机 15 Hz，腕部局部位移 `[0.150, 0, -0.100]`，
顶部相机继续输出语义分割；另加入可选 `recovery_oblique` 安装参数。
因此下文历史实验成绩不能直接视为合并后配置的验收结果。
主线 Web 已装配默认双相机辅助；与下文 demo 的自动恢复不同，Web 失败后先保持，
通过对话收到明确重试要求后才执行恢复观察与第二次尝试。当前未提供放置执行器。

## 1. 当前能做什么，不能承诺什么

已实现一个可由 BusAgent 装配的精细抓取模块：上游把空手机械臂带到目标附近后，
模块使用腕部 RGB-D 持续观察、生成候选、约束筛选、小步伺服、闭爪、抬升和验证。
增强模式会在符合恢复条件的失败后，利用固定 RGB-D 重定位，安全退让，
再从腕部安全侧向视点采集局部几何，重新尝试。

当前是 **Isaac Sim 已跑通部分场景的工程开发版**，不是完整 VLA，不是生产级通用抓取。
真实工具、真实水果/杯子、强反光/透明物体、密集杂乱场景及真机泛化均未验收。
主动观察次数由可行性筛选决定，不保证每次都会“左右各看一次”。

| 已实现 | 实际边界 |
|---|---|
| `GeneralGraspNode` + `FineGraspSkill` 稳定调用接口 | Python 进程内接口；BusAgent 必须自行装配和注册 |
| GraspGenX 本地服务与可替换 backend | 混合候选，不是所有成功都来自神经生成分支 |
| 腕部同帧 RGB-D / 位姿、手眼变换、小步重新观察 | 不可把旧抓取位姿用于后续开环执行 |
| IK、碰撞、开口、桌面间隙、局部手指扫掠检查 | 几何近似，不是完备接触稳定性保证 |
| 双 RGB-D 辅助验证和有界失败恢复 | 持物不确定时禁止自动松爪、横向扫视或盲目重试 |
| 腕部多视点局部表面融合 | 不是完整 TSDF，也不是两路相机联合稠密重建 |
| 易损物限力限速、薄件和深度缺失策略 | 无质量重心/偏载力矩自适应，真机力值尚需标定 |
| 无头 MP4、连续画廊、事件/候选/深度调试记录 | 录制增加开销，不用于无录制性能排名 |

## 2. 模型方案与已完成的比较

当前运行时默认由启动 wrapper 选择 **GraspGenX / GraspMoE**，独立模型进程通过
本机 ZMQ 提供候选。GraspMoE 包含 learned diffusion 和几何 OBB 候选，再由 discriminator
评分；外层另有独立的 `geometric_antipodal` fallback。三者不能混为一谈。
本次画廊四段及 5 cm 扰动试验实际选择的都是 `metadata.branch=obb`。

历史上已实际运行 GraspGenX 和 VGN；VGN 在官方 held-out 仿真小样本中为 9/20 抬升成功，
热推理约 8.38 ms；GraspGenX 的独立 100 候选热推理 p50 约 689 ms。
两者不是同一 Isaac 场景、同一控制器下的公平成功率对比，不得据此宣称 GraspGenX 已证明最优。
VGN 尚未接入运行时；工程选择、版本、权重校验与许可证见 [模型评估](MODEL_EVALUATION.md)。

GraspGenX 代码为 Apache-2.0，权重为 NVIDIA Open Model License；VGN 权重再分发授权仍待核对。
仓库不包含权重、第三方代码和 Isaac 资产；项目自身根许可证也尚待明确。

## 3. 相机放在哪里、各自负责什么

所有位置以米为单位，配置见 [cameras.yaml](../configs/cameras.yaml)。

| 相机 | 安装方式 | 当前职责 |
|---|---|---|
| 腕部 RGB-D | `/World/SO101/gripper/WristRGBD`，相对夹爪局部位移 `[0.060, 0, -0.030]`，朝向夹持区域 | 目标局部点云、抓取姿态、最后几厘米闭环与腕部验证 |
| 固定 RGB-D | `/World/Cameras/TableTopRGB`；`oblique` 时位置 `[0.65, -0.65, 1.75]`，看向 `[0.35, -0.15, 1.08]` | 失败后重新定位、辅助判断目标是否随抬升运动 |
| 录制专用视角 | 仅 `-RecordVideo` 创建 `/World/DemoOnly/Overview` | 只生成第三人称视频，绝不输入抓取决策 |

实际抓取传感器仍是 **两路 RGB-D**，各为 640×480、配置频率 30 Hz。
默认固定相机为 `overhead`：位置 `[0.50, 0, 2.30]`；本次视频显式选 `oblique`。
USD Camera 没有相机外壳模型，所以主画面看不到摄像头盒子；这不是已经完成真机改装。

变换记号为 `T_目标坐标系_源坐标系`：

```text
T_base_camera(t) = T_base_ee(t) @ T_ee_camera
T_base_grasp(t)  = T_base_camera(t) @ T_camera_grasp(t)
T_base_scene(t)  = T_base_world @ T_world_scene_camera(t)
```

腕部变换必须与该帧 RGB-D 的采样时间一致。深度为米制 optical-Z，相机光学轴为右/下/前。
不能用旧图像配当前关节位姿，也不能把随腕部转动的 camera +Z 当固定重力轴。
当前 object-points 模式先按本帧变换转至 base/gravity 系推理，再把候选转回相机系。
固定相机与腕部各自有新鲜度检查，但没有严格跨相机同步保证。

## 4. 执行逻辑与恢复门槛

```text
BusAgent：选目标、粗定位、避障、接近、确认空手、停止快环指令
  ↓ FineGraspRequest
最新腕部 RGB-D → 目标 mask / 局部点云 → 低频抓取候选
  ↓ IK / 碰撞 / 夹爪 / 桌面间隙筛选
pre-grasp → 新观测 → 小步 pose servo → 新观测 → 闭合 → 抬升 → 验证
  ├─ 成功：返回结果给 BusAgent
  └─ 失败：停止；检查失败类型、夹爪与持物状态
       ├─ 持物不明 / 标定或不可恢复错误：保留状态，交给上游
       └─ 可恢复且空手证据足够：固定 RGB-D 重定位 + 稳定性确认
            → 验证退让路径 → 安全退让 → 新建尝试节点
            → 安全侧向视点观察 / 融合 → 重新生成 → 再闭环抓取
```

模型在入口或候选失效时低频运行，伺服逐次读取最新腕部观测。
模型/IK 耗时后也复核观测；目标变化或局部几何不一致时不能继续旧轨迹。
这里的“较高频伺服”是相对重模型而言，不是已保证硬实时 Hz 数。

`off` 为原始单次节点；`assisted` 增加自遮罩和固定相机验证但不重试；
`active` 在辅助机制上增加有界重试。默认 `Recovery=off`。
默认 `Perception=single`；`-Perception multiview` 会让第一次尝试也主动观察。
增强 demo 默认最多两次尝试、恢复总预算 110 s、退让 4 cm、恢复速度比例 0.25；
单次节点仍受 45 s 配置限制，调用者可以设更短的请求预算。
底层阻塞 I/O 仍需要独立 watchdog，以上时限不是强制抢占保证。

固定相机当前依靠颜色、连通域与几何一致性，不是已验证的开放词汇语义跟踪。
机器人遮罩使用已知机器人路径的仿真 instance-ID，不使用目标真值进行分割。
但 demo 仍有场景先验：上游粗位置、物品属性、桌面高度、按对象高度调整交接姿态，
以及规划场景中的目标路径排除；不能宣称端到端无先验盲抓。

## 5. 怎么运行

### 已配置本机

在 `D:\liuGong` 的 PowerShell 执行。默认自动建立时间戳目录；显式传 `-Output` 时使用新目录，避免覆盖旧记录。

```powershell
# 无头：双相机辅助 + 失败后主动恢复；正常成功不会额外扫视
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -SceneView oblique

# GUI：准备 15 秒再开始，结束后保留窗口、暂停物理
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -SceneView oblique -NoHeadless -KeepOpen -StartDelayS 15

# 无头录制：首次闭爪前瞬移 5 cm，不修改上游粗位置或候选
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -SceneView oblique -TestTargetShiftM 0.05 -RecordVideo

# 连续四段：方块 4 cm 故障恢复、水果、咖啡杯、锤子；保留失败
& D:/isaac/env_isaacsim60/python.exe scripts/run_fine_grasp_gallery.py --output output/video_gallery/my_new_run

# 单元测试；不连接硬件
& D:/isaac/env_isaacsim60/python.exe -m unittest discover -s tests
```

`-TestTargetShiftM` 仅用于仿真，默认 0，绝对值上限 0.05 m；
只有 `test_target_shift_applied=true` 才证明到达触发阶段并实际注入。
自定义对象用 `-CaseJson <case.json>`，字段见 [BenchmarkCase](../source/mr_liu/grasp/benchmark.py)。
切勿把上述测试扰动接到真机驱动。所有 demo 只控制仿真机械臂。

画廊输出 `BusAgent_grasp_gallery.mp4`、`gallery.json` 及逐段报告，支持章节信息；
已有非空输出目录需使用 `--resume` 复用完整片段，或换新目录。
视频为 H.264、1280×720、15 fps；右侧显示两路相机最新 RGB 缓冲，不是严格同步采集。
录制增加渲染/编码开销，报告有 `video.recording_overhead=true`；
不可与无录制性能 benchmark 混算。系统待机会计入节点超时，录制期间保持电脑唤醒。

### 环境与新机器

新机器获取本次增强开发线：

```powershell
git clone --branch codex/active-grasp-recovery https://github.com/yibent/LiuGongArm.git
cd LiuGongArm
```

| 路径/依赖 | 当前本机配置 |
|---|---|
| `D:/isaac/env_isaacsim60/python.exe` | Isaac Sim 6.0.1、Python 3.12；不使用本机旧 Isaac Lab 环境 |
| `_envs/graspgenx/python.exe` | 独立模型环境，Python 3.11、PyTorch 2.6.0+cu124 |
| `_vendor/GraspGenX/` | 上游代码、server、夹爪描述资产 |
| `_models/graspgenx/checkpoints/release/` | generator/discriminator 权重 |
| `requirements-graspgenx-client.txt` | Isaac 侧轻量 ZMQ 客户端依赖 |
| `requirements-demo-video.txt` | 可选录制依赖；在 Isaac 环境中安装 |

```powershell
& D:/isaac/env_isaacsim60/python.exe -m pip install -r requirements-demo-video.txt
```

wrapper 在 `127.0.0.1:5556` 没有服务时启动模型、等待 warmup，并仅回收自己创建的进程。
已有监听端口会被复用，操作者须确认服务版本与健康状态。不要把无鉴权 ZMQ 暴露到公网。
换机器须修改 PS1 中的 Isaac Python 路径，分别准备环境和权重；clone 不是一键安装。
完整模型来源与校验见 [旧版安装说明](BUSAGENT_README.md#新电脑需要什么)；
其中 `codex/fine-grasp-loop` 的 clone 命令只适用于历史基线，本次应使用上面的增强分支命令。
Isaac 桌子、SO-101、HDRI 首次加载可能联网，离线运行要事先缓存全部依赖资产。

## 6. BusAgent 应如何接入

将 `source/` 加入宿主 Python 路径。上层只依赖
[FineGraspSkill](../source/mr_liu/control/skills/fine_grasp.py) 和
[请求/结果契约](../source/mr_liu/grasp/contracts.py)，不依赖 GraspGenX 内部模型实现。
同一机械臂同一时刻只允许一个执行者；约 5–8 cm 的交接距离由上游判断，节点不自动接管。
上游需提供实例 `object_id`、粗位置以及可选语义/属性，并确认空手、相机视野及控制权交接。

以下是**调用示例，不是完整设备装配脚本**；`assembled_node` 必须已由宿主安全装配：

```python
from mr_liu.control.skills.fine_grasp import FineGraspSkill
from mr_liu.grasp.contracts import FineGraspRequest, ObjectProperties, TargetSpec

def grasp_nearby(assembled_node, *, object_id, label, coarse_xyz_m, request_id,
                 fragile=False, dry_run=True):
    skill = FineGraspSkill(assembled_node)
    request = FineGraspRequest(
        target=TargetSpec(
            object_id=object_id, semantic_label=label,
            coarse_position_base_m=tuple(coarse_xyz_m),
            properties=ObjectProperties(fragile=fragile),
        ),
        request_id=request_id, timeout_s=110.0, dry_run=dry_run,
    )
    return skill.execute(request)
```

单次节点装配可参考旧 README；主动恢复还须注入：

- `SceneTargetObserver`：已标定固定 RGB-D、新鲜目标对应、机器人遮罩。
- `attempt_factory(index, failed_grasps, target)`：每次新建 tracker/node/verifier；
  后续尝试启用腕部局部融合与安全视点，并保留失败候选排除。
- motion：IK/碰撞、真实状态、servo/stop、`clear_grasp_plan()`，以及能校验完整实际
  关节路径和局部手指扫掠体的 `is_observation_path_safe(...)`。
- `SceneAssistedVerifier`、可靠夹爪状态与持物保护；最终组装 `RecoveringGraspNode`。

可运行装配位于 [run_fine_grasp_demo.py](../scripts/run_fine_grasp_demo.py) 的 `attempt_factory`
及其外围；不能只把 `GeneralGraspNode` 改个类名就认为恢复功能已接好。
`execute()` 阻塞，`cancel()` 请求停止但不是硬件急停；上游须自行注册
`fine_grasp`（别名 `general_grasp`、`grasp`）。

首次联调保持 `dry_run=True`，只评估候选，不实际运动/闭爪。
保存请求模式：结果本身没有 `dry_run` 字段。只有非 dry-run 的成功结果才代表节点宣称抓取成功。
`selected_grasp` 只用于诊断，不可在返回后再次按旧位姿开环执行。
真机适配桥已有，但手眼、RGB-D/FK 同步、夹爪宽度/力、工具偏置、碰撞几何、watchdog
与有人值守的实抓测试尚未完成；不可照搬仿真配置通电。

## 7. 最新实测结果：必须分开“抬起”和“成功”

以下都是开发集几何代理、单种子、开启录制，不是泛化成功率统计。
可随 Git 查看 [精简报告与原始报告 SHA-256](benchmarks/fine_grasp_handoff_20260905.json)。
四段画廊产于录制功能开发工作区（报告 HEAD 为 `e66e7f6`），5 cm 测试报告 HEAD 为 `e3e0592`；
不能追溯性地把所有试验标成同一个干净提交。各报告的 provenance 均原样保留。

| 场景 | 节点结果 | 最终刚体原点抬升 | 观察/解释 |
|---|---|---:|---|
| 方块首次闭爪前瞬移 4 cm | 成功 | 82.23 mm | 两次闭爪、两次安全侧向观察；节点周期 48.56 s |
| 水果球形代理，无注入 | 成功 | 81.14 mm | 一次闭爪；节点周期 29.61 s |
| 咖啡杯代理，无注入 | 成功 | 50.98 mm | 一次闭爪；节点周期 30.09 s |
| 锤子代理，无注入 | 失败：`lift_verification_failed` | 25.45 mm | 未稳定随腕部运动；`payload_uncertain_do_not_open` 停止 |
| 方块首次闭爪前瞬移 5 cm | 失败：`ik_unreachable` | 77.94 mm | 两次闭爪、一个安全侧向视点；抬升 motion 返回 false，抓后验证也失败 |

5 cm 测试第二次 `attempt_end` 时为 82.10 mm，最终报告采样为 77.94 mm。
这说明采样时刻不同；此前演示中的“约 8.2 cm”指尝试结束时，不是持续稳定持物证明。
该失败码由“抬升 motion 返回 false 且抓后验证失败”的组合路径产生，
不能仅凭名字断言 IK 数学无解，也不能因视频看到抬起就改记成功。

当前节点以抓后验证结果作最终成功判定：`lift_motion_completed=0` 不会单独否决
已通过的抓后验证。上述方块 4 cm、水果、杯子三个成功样本均为该组合，由固定相机辅助
确认目标随抬升运动。因此“节点成功”也不应被解释为所有运动完成标志均为 true；
这组运动完成标志与验证的差异，是仍需定位和改进的工程问题，而非本次修改过成功阈值。

锤子实际移动了，但这不等于完整离桌、稳定夹持：腕部跟随误差约 41.58 mm，
固定相机验证不确定。夹爪最终宽度为 0、堵转标志为 true、effort 读数为 0，
不是足以证明稳定接触的可靠力证据。

### 锤子暴露的适配缺口

当前排序轻度偏向几何中心，不是真实质量重心；没有显式质量分布、偏载力矩、
摩擦或抗旋转稳定性评分。当前 `stable_region=None`，虽然保留找细长手柄区域的函数，
这次实际 `region_strategy=model_pose`，未启用手柄重定位。
`metadata.preferred_region=handle` 不代表已经实现工具重心适配。

“锤头偏重造成旋转/滑脱”是合理假设，但现有记录不能排除抓点、摩擦和夹持力的影响，
没有完成因果消融。后续应增加通用载荷不确定性/接触稳定性评估与试抬升反馈，
并在安全放回后重新选抓点；不通过逐物体写死位姿来宣称泛化。

### 本机视频位置（不随 Git 分发）

- `D:/liuGong/output/video_gallery/20260905_v2/BusAgent_grasp_gallery.mp4`：2 分 24 秒，
  四段依次从约 00:00、00:52、01:25、01:58 开始；锤子失败片段完整保留。
- `D:/liuGong/output/video_gallery/20260905_shift_5cm_144639/demo.mp4`：约 54 秒，5 cm 扰动。
- `20260905_v1/01_cube_recovery` 是受 Windows 待机影响的超时录制，保留在本地，
  不纳入上述正常运行结果，也不是模型推理用了 10 分钟的证据。

`output/` 被 Git 忽略。远端 clone 可以得到本文、精简证据和复现脚本，但不会得到 MP4、
原始点云/图像、完整报告或权重；这些产物需从本机另行归档。

## 8. 失败处理与调试入口

失败结果不代表空手，`effort_n=0` 也不代表无接触。BusAgent 不得统一执行“松爪再试”。
特别是 `payload_uncertain_do_not_open`，必须保留持物状态并进入上游安全处置；
没有这个标志也不保证空手，因为某些不可重试错误会更早结束恢复流程。

| 类别 | 处理原则 |
|---|---|
| 无观测、过期、标定异常 | 停止；检查采样时间与手眼标定，不沿旧轨迹继续 |
| 身份/几何/深度不确定 | 与物理不可抓区分；只有满足恢复安全条件才能换视点 |
| 碰撞、桌面间隙、开口、IK 约束 | 当前方案不可执行；不能强行闭爪，也不证明所有姿态都不可行 |
| 闭爪或抬升验证失败 | 物体可能挂住或已在手里；先确认持物与安全处置 |
| 模型不可用/超时 | 检查本地服务与候选实际来源；fallback 成功不算神经模型成功 |

检查顺序：`report.json` → `attempt_results`/`attempt_physics` → `debug/trace.jsonl` →
`attempt_N/debug/` 的 RGB-D/mask/变换/候选 → `overhead/` 固定相机证据 → `console.log`。
`actual_target_lift_m` / `attempt_physics` 只用于独立仿真评估，不输入控制或验证器。
录制另有 `demo.video.json`、`demo.ffmpeg.log` 和结果预览图。

代码入口：

- [node.py](../source/mr_liu/grasp/node.py)：单次闭环与验证。
- [recovery.py](../source/mr_liu/grasp/recovery.py)：有界恢复与持物保护。
- [scene_observer.py](../source/mr_liu/grasp/scene_observer.py)：固定 RGB-D 定位/辅助验证。
- [selection.py](../source/mr_liu/grasp/selection.py)：模型无关约束与排序。
- [graspgenx.py](../source/mr_liu/grasp/backends/graspgenx.py)：候选协议与模型接入。
- [run_fine_grasp_gallery.py](../scripts/run_fine_grasp_gallery.py)：连续无头录制。

本次交接重新运行单元测试；当前 126 项通过。单测、视频和单次成功均不能代替真机验收。
下一阶段优先处理偏载接触稳定性、5 cm 案例的运动完成/验证失败分类，再做冻结对象、姿态、
光照、多随机种子的配对验收，以及有人值守的真机迁移。
