# MR Liu + YOLOE

## BusAgent：精细抓取模块交接入口

新增可选 [标签到完整抓取入口](docs/LABEL_TO_GRASP.md)：
`scripts\run_label_grasp.ps1 -Label 'red cube' -RecordVideo`。
Florence/YOLOE 定位 → 真正的 LK 光流 → 经安全检查的粗接近 → 原腕部 FineGrasp；
实测结果与尚未解决的泛化边界见该文档。

**当前开发线请先读 [当前成果与工程交接](docs/FINE_GRASP_CURRENT_HANDOFF.md)**：
包含双 RGB-D 分工、主动恢复逻辑、一键 GUI/无头视频、BusAgent 装配与失败处理，
以及最新 4/5 cm 扰动、杯子/水果/锤子结果。下文保留旧基线背景，不作为最新状态。

新增 `GeneralGraspNode` / `FineGraspSkill`：机械臂靠近目标后，以腕部 RGB-D 持续观测，
通过 GraspGenX 候选、IK/碰撞筛选和小步伺服完成接近、闭爪、抬升与验证。
已发布旧基线的头顶相机已升级 RGB-D，但当时尚未参与抓取决策。

新开发线 `codex/active-grasp-recovery` 增加了实验性的双 RGB-D 主动恢复：
失败后检查持物状态、安全退让、腕部多视角重新观察，再生成抓取并闭环执行。
机制、运行方式、验证证据和安全边界见 [主动恢复开发说明](docs/ACTIVE_GRASP_RECOVERY.md)。
下文的 `2c7b53f` 状态指已发布的旧基线，不代表新开发线的最新结果。

**请先读 [BusAgent 接入 README](docs/BUSAGENT_README.md)**：包含已开发内容、
实际模型评估、环境要求、一键 demo、Python 接入代码、失败处理和后续增强计划。

已发布旧基线（`2c7b53f`）不是泛化验收通过的发布版。基线 82 项单测和双 RGB-D
运行检查通过；本次交接文档增加 2 项示例/链接测试，共 84 项通过。
旧基线 `2c7b53f` 的 cube 实际抬升约 8 cm，但节点视觉验证失败。
真实工业工具、咖啡杯、水果的 unseen-object 抓取能力尚未验收。

以下保留原仿真/视觉工程说明。

机械臂仿真工程：Isaac Sim 桌面 SO-101 + cuMotion follow-target，接入 Florence-2 FIND 与 YOLOE / OpenCV TRACK，并提供桌面顶视与腕部两路 RGB-D 相机。

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

### 2. 独立视觉依赖（CLI / WebUI）

```bat
powershell -ExecutionPolicy Bypass -File scripts\setup_vision.ps1
```

安装到本地 `_envs/vision` overlay，复用 Isaac 的 CUDA torch，不修改原环境。
固定版本权重下载到 `_models/florence2/large` 与 `_models/yoloe`，不进 Git。
命令、接口、离线测试与能力边界见 [本地视觉支持](docs/LOCAL_VISION_SETUP.md)。

## 启动

FineGrasp 的架构、Isaac 闭环 demo、失败排查和真机前置条件见
[`docs/FINE_GRASP.md`](docs/FINE_GRASP.md)；模型实测与选型证据见
[`docs/MODEL_EVALUATION.md`](docs/MODEL_EVALUATION.md)。
当前开发基线与后续主动观察版本的边界见
[`docs/FINE_GRASP_BASELINE.md`](docs/FINE_GRASP_BASELINE.md)；尚未通过真实物体泛化验收。

| 做什么 | 命令 |
|---|---|
| 腕部 RGB-D 精细抓取闭环（默认 GraspGenX + geometric fallback） | `scripts\run_fine_grasp_demo.bat` |
| 拖绿立方体，臂跟随（一期） | `scripts\run_follow_target.bat` |
| 场景相机 → FIND/TRACK → 立方体 → 臂 | `scripts\run_vision_follow.bat` |
| 只加载桌 + SO-101 | `run_hello_world.bat` |
| Kit GUI + 工程扩展 | `launch_isaac_sim.bat` |
| 独立视觉 WebUI（不开仿真，默认 :7860） | `scripts\run_yoloe_webui.bat` |
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

Play 之后拖动 `/World/TargetCube`，SO-101 用 cuMotion RMPflow 跟随。Stop/Play 会重置控制器。

独立视觉 CLI：

```bat
scripts\run_vision.bat --source samples\bus.jpg --prompt "bus"
scripts\run_vision.bat --webui --host 127.0.0.1 --port 7860
```

`--prompt` 多个目标用分号分隔；`--fast yoloe|cv`。

## 资产与配置

- 桌子：`Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd`
- 机械臂 USD：`Isaac/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd`
- 规划：`assets/robots/so101/robot.urdf` + `robot.xrdf`
- 基座通过 `/World/SO101Mount` Fixed Joint 焊在桌面
- 目标：`/World/TargetCube`（`configs/scene.yaml`）

| 文件 | 内容 |
|---|---|
| `configs/scene.yaml` | 桌、臂、灯光、TargetCube |
| `configs/robot_so101.yaml` | 关节名 / 初始姿态 |
| `configs/motion.yaml` | physics dt、device、RMPflow |
| `configs/cameras.yaml` | 场景相机 / 腕部相机 |

相机 rig 默认启用：`/World/Cameras/TableTopRGB` 位于桌面正上方并输出 RGB 与米制深度（保留旧 prim 路径兼容已有消费者）；
`/World/SO101/gripper/WristRGBD` 挂载在夹爪节点下，随机械臂运动并输出 RGB 与米制深度。
