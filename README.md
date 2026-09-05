# MR Liu + YOLOE

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

### 2. 视觉依赖（WebUI / vision follow）

```bat
D:\isaac\env_isaacsim60\python.exe -m pip install -r requirements-vision.txt
```

视觉权重不进 Git。把 `yoloe-26x-seg.pt` 放到仓库根目录。没有权重时，仿真仍可手动拖绿立方体。

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
D:\isaac\env_isaacsim60\python.exe vision_main.py --source samples\bus.jpg --prompt "bus"
D:\isaac\env_isaacsim60\python.exe vision_main.py --webui --host 127.0.0.1 --port 7860
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
