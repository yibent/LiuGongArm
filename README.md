# MR Liu + YOLOE

## BusAgent：精细抓取模块交接入口

**当前开发线请先读 [当前成果与工程交接](docs/FINE_GRASP_CURRENT_HANDOFF.md)**：
包含双 RGB-D 分工、主动恢复逻辑、一键 GUI/无头视频、BusAgent 装配与失败处理，
以及最新 4/5 cm 扰动、杯子/水果/锤子结果。下文保留旧基线背景，不作为最新状态。

现有 Web/语音控制链路的抓取接入见 [运行说明](docs/BUSAGENT_GRASP_RUNTIME.md)：
单物体定位、接近、腕部抓取、抬升验证与中断已接入代码。
Web 运行时已装配默认双相机辅助，失败后等待对话指令再恢复；当前控制器尚不支持放置。

新增 `GeneralGraspNode` / `FineGraspSkill`：机械臂靠近目标后，以腕部 RGB-D 持续观测，
通过 GraspGenX 候选、IK/碰撞筛选和小步伺服完成接近、闭爪、抬升与验证。
已发布旧基线的头顶相机已升级 RGB-D，但当时尚未参与抓取决策。

新开发线 `codex/active-grasp-recovery` 增加了实验性的双 RGB-D 主动恢复：
失败后检查持物状态、安全退让、腕部多视角重新观察，再生成抓取并闭环执行。
机制、运行方式、验证证据和安全边界见 [主动恢复开发说明](docs/ACTIVE_GRASP_RECOVERY.md)。
下文的 `2c7b53f` 状态指已发布的旧基线，不代表新开发线的最新结果。

**请先读 [BusAgent 接入 README](docs/BUSAGENT_README.md)**：包含已开发内容、
实际模型评估、环境要求、一键 demo、Python 接入代码、失败处理和后续增强计划。

已发布旧基线（`2c7b53f`）不是泛化验收通过的发布版。该分支合并前记录：82 项单测和双 RGB-D
运行检查通过；交接文档增加 2 项示例/链接测试，共 84 项通过（不是合并后全仓库测试数量）。
旧基线 `2c7b53f` 的 cube 实际抬升约 8 cm，但节点视觉验证失败。
真实工业工具、咖啡杯、水果的 unseen-object 抓取能力尚未验收。

以下保留原仿真/视觉工程说明。

机械臂仿真工程：Isaac Sim 桌面 SO-101 + cuMotion follow-target，接入共享 Florence-2 FIND 与双路独立 YOLOE / OpenCV TRACK，并提供桌面顶视与腕部两路 RGB-D 相机；顶视保留语义分割，支持现有物体定位链路。

合并自：

- 仿真：[SQLsleeping/MR_Liu](https://github.com/SQLsleeping/MR_Liu)
- 视觉：[SQLsleeping/yoloe](https://github.com/SQLsleeping/yoloe)

本机 `D:\isaac\env_isaaclab` 是 Isaac Sim **5.1 + Isaac Lab**，**不要**用来跑本仓库。NVIDIA 当前公开的 6.x pip 包是 **6.0.1.0**（没有 6.1 轮子），需要 **Python 3.12**。

## 目录

```
liuGong/
├── configs/                      # 场景、机器人、规划、相机
├── assets/robots/so101/          # URDF / XRDF / rmp_flow.yaml
├── source/mr_liu/                # sim / robot / motion / vision / perception / app
├── source/find_and_track/        # Florence-2 FIND + YOLOE/CV TRACK + WebUI
├── BusAgent/                     # 语音指令、任务规划与机器人控制总线
├── samples/                      # 视觉示例图/视频
├── scenes/world.usda
├── scripts/                      # 启动脚本（不含业务逻辑）
├── tests/                        # 关节名映射等无 GUI 测试
├── extensions/mr_liu.project/    # Kit 薄扩展
├── vision_main.py                # 独立视觉 CLI / WebUI
├── requirements-vision.txt
└── isaac_env.bat                 # 指向本机 Isaac Sim 6.0.1
```

## 环境

### 1. Isaac Sim 6.0.1（仿真必需）

```bat
conda create -y -p D:\isaac\env_isaacsim60 python=3.12
D:\isaac\env_isaacsim60\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
D:\isaac\env_isaacsim60\python.exe -m pip install isaacsim[all,extscache]==6.0.1.0 --extra-index-url https://pypi.nvidia.com
```

体积很大（`extscache-kit` 约 5.9 GB + kit-sdk 约 0.7 GB + 其它）。换机器时编辑 `isaac_env.bat` 里的 `ISAAC_ENV`。启动脚本会设置 `OMNI_KIT_ACCEPT_EULA=YES`。

### 2. 视觉依赖（WebUI / vision follow）

```bat
D:\isaac\env_isaacsim60\python.exe -m pip install -r requirements-vision.txt
```

Windows 本地也可以使用仓库内的独立视觉环境（不修改 Isaac Sim 的 Python）：

```bat
conda create -y -p E:\LiuGongArm\.venv python=3.12
E:\LiuGongArm\.venv\python.exe -m pip install torch torchvision
E:\LiuGongArm\.venv\python.exe -m pip install -r requirements-vision.txt ultralytics opencv-python
```

Florence 权重放在 `.cache\huggingface\Florence-2-large`、YOLOE 权重放在仓库根目录后，
可运行 `scripts\run_florence_local.bat` 启动离线 WebUI；该入口默认使用 Florence + YOLOE，
缺少 YOLOE 权重时自动回退到 CV 跟踪。

视觉权重不进 Git。把 `yoloe-26x-seg.pt` 放到仓库根目录。没有权重时，仿真仍可手动拖绿立方体。

Linux/服务器建议把视觉依赖放在仓库自己的 venv 中：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements-vision.txt
```

`scripts/run_vision_follow.py` 会先初始化 Isaac Sim，再把 `.venv` 的
`site-packages` 追加到搜索路径，从而同时使用 Isaac SDK 与 Florence/YOLOE。
不要把 venv 通过 `PYTHONPATH` 放到 Isaac 自带依赖之前，否则可能覆盖其
Torch/渲染组件。Florence 权重首次运行时从 Hugging Face 下载，建议将
`HF_HOME` 指向仓库内可持久化的缓存目录。

## 启动

FineGrasp 的架构、Isaac 闭环 demo、失败排查和真机前置条件见
[`docs/FINE_GRASP.md`](docs/FINE_GRASP.md)；模型实测与选型证据见
[`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md)。
当前开发基线与后续主动观察版本的边界见
[`docs/FINE_GRASP_BASELINE.md`](docs/FINE_GRASP_BASELINE.md)；尚未通过真实物体泛化验收。

| 做什么 | 命令 |
|---|---|
| 腕部 RGB-D 精细抓取闭环（默认 GraspGenX，不自动几何回退） | `scripts\run_fine_grasp_demo.bat` |
| 拖绿立方体，臂跟随（一期） | `scripts\run_follow_target.bat` |
| 场景相机 → FIND/TRACK → 立方体 → 臂 | `scripts\run_vision_follow.bat` |
| 只加载桌 + SO-101 | `run_hello_world.bat` |
| Kit GUI + 工程扩展 | `launch_isaac_sim.bat` |
| 独立视觉 WebUI（不开仿真，默认 :7860） | `scripts\run_yoloe_webui.bat` |
| 本地 Florence 离线 WebUI（不开仿真，默认 :7860） | `scripts\run_florence_local.bat` |
| 关节名自检（无 GUI） | `scripts\run_tests.bat` |

FineGrasp wrapper 在自己启动模型服务时会等待 GraspGenX warmup 返回 `status=ready`，并把
串行运行结果保存到按秒时间戳的 `output/fine_grasp_runs/<timestamp>_<backend>/`。本机已保存
早期版本的一次无 fallback 的 GraspGenX Isaac cube 闭环成功证据：模型 687 ms、最终
3.39 mm/0.15°、腕部视觉验证抬升 82.05 mm、Isaac 刚体实际抬升 82.06 mm。
这不代表当前基线已通过验收；最新双 RGB-D 基线回归实际抬升 80.24 mm，
但节点视觉随动验证失败，完整结果和限制见上述基线说明。

双 RGB-D 相机运行时验证（本机 PowerShell；不会执行抓取）：

```powershell
& D:\isaac\env_isaacsim60\python.exe scripts\verify_cameras.py
```

输出目录 `output/camera_verify/<timestamp>/` 保存两路 RGB、深度图、带渲染时间及
OpenCV 光学坐标系位姿的 RGB-D NPZ 和检测报告。脚本验证两路流各自帧号递增，
不表示跨相机时间同步或双相机融合已完成。

相机运行时验证（Linux）：

```bash
PYTHONEXE="$PWD/.venv/bin/python" /root/isaacsim/python.sh scripts/verify_cameras.py
```

桌面物品物理与俯视相机验证（Linux）：

```bash
/root/isaacsim/python.sh scripts/verify_tabletop_props.py
```

Play 之后拖动 `/World/TargetCube`，SO-101 用 cuMotion RMPflow 跟随。Stop/Play 会重置控制器。

### 仿真运行中切换视觉目标

双视角视觉跟随默认在 `127.0.0.1:7861` 提供控制页和 JSON API。顶部与腕部画面共享一个 Florence 慢速检测模型，但各自维护独立的 YOLOE 跟踪器；修改提示后两路都会重新 FIND，后续帧恢复快速 TRACK。

```text
顶部 RGB ─┐                         ┌─ 顶部 YOLOE/BYTETrack
          ├─ 共享 Florence-2 FIND ─┤
腕部 RGBD ┘                         └─ 腕部 YOLOE/BYTETrack
```

```bash
/root/isaacsim/python.sh scripts/run_vision_follow.py \
  --headless --prompt "red cube" --slow-interval 0

# 感知基准模式：仿真继续运行，保留 USD 初始臂姿态且不追随检测框
/root/isaacsim/python.sh scripts/run_vision_follow.py \
  --headless --no-follow --prompt "red cube" --slow-interval 0

# 运行中切换目标（支持用分号指定多个目标）
curl -X POST http://127.0.0.1:7861/api/prompt \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"power drill"}'

# 不改提示，要求两路立即重新 FIND
curl -X POST http://127.0.0.1:7861/api/find \
  -H 'Content-Type: application/json' -d '{}'

curl http://127.0.0.1:7861/api/status

# BusAgent 使用的统一语义控制接口
curl -X POST http://127.0.0.1:7861/api/command \
  -H 'Content-Type: application/json' \
  -d '{"command_id":"demo-1","skill":"select_target","params":{"category":"power drill"}}'
```

远程查看控制页可建立 SSH 隧道：`ssh -L 7861:127.0.0.1:7861 -p 30133 root@183.147.142.40`，然后打开 `http://127.0.0.1:7861`。接口还提供 `/api/frame/scene.jpg` 与 `/api/frame/wrist.jpg` 两路实时标注快照。

### 目标记忆与恢复

完整的模块关系、状态转换、记忆文件格式、线程边界和接入 API 见 [视觉记忆方案说明](docs/VISION_MEMORY_ARCHITECTURE.md)。

视觉链路按“YOLOE 先试、Florence 慢环兜底、CV 持续跟踪”的策略运行。YOLOE 先使用文本标签库和置信度尝试定位；置信度不足时才调用 Florence。Florence 的框会同时初始化 CV 和 YOLOE 视觉提示。CV 丢失达到 `lost_patience` 后，系统再次调用 Florence，并用新框重建 YOLOE 提示后恢复 CV 跟踪。

记忆保存在 `runs/memories/`，每条记忆包含标签、时间、多视角裁剪图和原始框坐标。机械臂运行时可通过统一控制接口采集和回放：

```bash
# 采集当前顶部与腕部快照；重复使用同一个 memory_id 可追加新的视角
curl -X POST http://127.0.0.1:7861/api/command \
  -H 'Content-Type: application/json' \
  -d '{"skill":"remember","params":{"label":"red mug"}}'

# 回放记忆：下一帧直接用记忆图片建立 YOLOE visual prompt
curl -X POST http://127.0.0.1:7861/api/command \
  -H 'Content-Type: application/json' \
  -d '{"skill":"recall","params":{"memory_id":"<memory_id>","view":"wrist"}}'

curl http://127.0.0.1:7861/api/status
curl -X POST http://127.0.0.1:7861/api/command \
  -H 'Content-Type: application/json' \
  -d '{"skill":"forget","params":{"memory_id":"<memory_id>"}}'
```

记忆采集与回放也分别提供 `/api/memory`、`/api/recall`、`/api/memories` 和 `/api/forget` 接口。`memory_id` 回放时不要求 Florence 首次重新圈定，YOLOE 无法恢复时仍会回退到 Florence。

`--slow-interval 0` 表示只在首次检测、目标丢失、提示词变化或手动请求时
运行 Florence；设置为正数则按相应帧数周期重新 FIND。`--no-follow` 只关闭
检测框到机械臂目标的映射，物理仿真、双相机和推理仍正常运行。

### 接入 BusAgent 语音控制

先保持上述 `run_vision_follow.py` 运行，再启动独立仓库 `BusAgent/backend` 和 Web 前端。BusAgent 会把语音指令转换为结构化技能，通过 `/api/command` 下发；当前支持目标识别、跟随、基础运动和已接入控制器的单物体抓取，放置未实现。视觉记忆已提供控制 API，但尚未接入 BusAgent 的记忆语音意图，不能仅凭接口存在就宣称支持语音记忆。实际能力以 `/api/capabilities` 为准，详见 [BusAgent 说明](docs/BUSAGENT_README.md)。

### 验证结果

在 Isaac Sim 6.0.1、CUDA 和 `yoloe-26x-seg.pt` 下实测：

- 初始 `red cube`：顶部与腕部均完成慢 FIND 并进入快 TRACK；
- 运行中切换到 `power drill`：两路在同一视觉周期重新 FIND，分别耗时
  278.1 ms 和 275.4 ms；
- 稳态 YOLOE TRACK：顶部约 25 ms/帧，腕部约 24–26 ms/帧；
- 连续运行超过 5000 个视觉帧后，两路仍能正确追踪电钻；
- `python -m unittest discover -s tests -v`：17 项测试全部通过。

独立视觉 CLI：

```bat
D:\isaac\env_isaacsim60\python.exe vision_main.py --source samples\bus.jpg --prompt "bus"
D:\isaac\env_isaacsim60\python.exe vision_main.py --webui --host 127.0.0.1 --port 7860
```

`--prompt` 多个目标用分号分隔；`--fast yoloe|cv`。

## 资产与配置

- 桌子：`Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd`
- 机械臂 USD：`Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd`
- 规划：`assets/robots/so101/robot.urdf` + `robot.xrdf`
- 基座通过 `/World/SO101Mount` Fixed Joint 焊在桌面
- 机械臂根节点默认绕世界 Z 轴逆时针旋转 90°，使前半周工作区朝向桌面操作区域
- 目标：`/World/TargetCube`（`configs/scene.yaml`）
- 桌面可操作物品：官方红色积木、YCB 电钻、Factory M20 螺栓/螺母与小齿轮，外加程序化复合刚体扳手
- 所有桌面物品位于 `/World/TabletopProps`，带质量、碰撞和 `class` 语义标签

| 文件 | 内容 |
|---|---|
| `configs/scene.yaml` | 桌、臂、灯光、TargetCube、桌面物品及其质量/位姿 |
| `configs/robot_so101.yaml` | 关节名 / 初始姿态 |
| `configs/motion.yaml` | physics dt、device、RMPflow |
| `configs/cameras.yaml` | 场景相机 / 腕部相机 |

相机 rig 默认启用：`/World/Cameras/TableTopRGB` 位于桌面正上方并输出 RGB 与米制深度（保留旧 prim 路径兼容已有消费者）；
`/World/SO101/gripper/WristRGBD` 挂载在夹爪节点下，随机械臂运动并输出 RGB 与米制深度。
