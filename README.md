# MR Liu + YOLOE

机械臂仿真工程：Isaac Sim 桌面 SO-101 + cuMotion follow-target，接入共享 Florence-2 FIND 与双路独立 YOLOE / OpenCV TRACK，并提供桌面顶视 RGB 与腕部 RGBD 相机。

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

| 做什么 | 命令 |
|---|---|
| 拖绿立方体，臂跟随（一期） | `scripts\run_follow_target.bat` |
| 场景相机 → FIND/TRACK → 立方体 → 臂 | `scripts\run_vision_follow.bat` |
| 只加载桌 + SO-101 | `run_hello_world.bat` |
| Kit GUI + 工程扩展 | `launch_isaac_sim.bat` |
| 独立视觉 WebUI（不开仿真，默认 :7860） | `scripts\run_yoloe_webui.bat` |
| 关节名自检（无 GUI） | `scripts\run_tests.bat` |

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

`--slow-interval 0` 表示只在首次检测、目标丢失、提示词变化或手动请求时
运行 Florence；设置为正数则按相应帧数周期重新 FIND。`--no-follow` 只关闭
检测框到机械臂目标的映射，物理仿真、双相机和推理仍正常运行。

### 接入 BusAgent 语音控制

先保持上述 `run_vision_follow.py` 运行，再启动 `BusAgent/backend` 和 Web 前端。BusAgent 会把“找红色方块”“跟踪电钻”“停一下”“现在什么状态”等语音指令转换为结构化技能，并通过 `/api/command` 下发。当前机械臂侧只实现目标识别与跟随；抓取和放置会明确返回未实现，详见 [BusAgent 说明](BusAgent/README.md)。

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

相机 rig 默认启用：`/World/Cameras/TableTopRGB` 位于桌面正上方并输出 RGB；
`/World/SO101/gripper/WristRGBD` 挂载在夹爪节点下，随机械臂运动并输出 RGB 与米制深度。
