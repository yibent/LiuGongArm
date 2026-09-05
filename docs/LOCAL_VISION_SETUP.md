# 本地 Florence / YOLOE / CV 支持

这一层提供 **RGB 目标查找与 2D 跟踪**，不是新的抓取策略。没有修改 FineGrasp 的
默认感知、手眼标定、碰撞检查或机械臂控制。也不代表工业工具、杯子、水果的泛化验收通过。

## 当前分工

| 组件 | 做什么 | 本地依赖 |
|---|---|---|
| Florence-2-large | 英文语义提示 → 目标框；低频 FIND / 丢失后重定位 | 约 1.55 GB safetensors，CUDA 或 CPU |
| YOLOE-26x-seg | 将 Florence 框编码为视觉提示，随后检测、跟踪 | 约 172 MB checkpoint；本次配置视觉提示模式 |
| OpenCV CV | 给定初始框，MIL + LK 光流 + 模板/ORB 跟踪 | CPU，无 YOLOE 权重依赖 |

CV 自己不理解“扳手/杯子”等语义；CLI 的 `--fast cv` 仍由 Florence 初始化。
如果上游已给框，可以直接使用 `CvFeatureTracker`，不加载 Florence 或 YOLOE。
YOLOE 的 text-prompt 模式还需额外文本编码器，本次没有安装它；视觉提示模式无需该编码器。
底层 YOLOE 有 segmentation 输出，但目前项目 `Detection` 对外仍只返回框，没有导出 mask。

模型来源：[Florence 原生 Transformers 转换权重](https://huggingface.co/florence-community/Florence-2-large)、
[Ultralytics YOLOE 文档](https://docs.ultralytics.com/models/yoloe)。Florence 模型卡标注 MIT；
Ultralytics 项目使用 AGPL-3.0，其商业部署许可需要在产品集成时单独核实，不能将所有组件视为 MIT。

## 环境与安装

本机建立 `_envs/vision` **overlay venv**，通过 `--system-site-packages` 只读复用
`D:\isaac\env_isaacsim60` 的 Python 3.12、CUDA PyTorch。
基础环境的 torchvision 实际是 CPU 版本，因此 overlay 单独安装 `torchvision==0.26.0+cu128`，
修正 Florence / YOLOE 同进程运行时触发的 CUDA NMS 缺失，并实际调用 CUDA NMS 自检。
新增 pip 包装在 overlay 中，不向 Isaac 或 `_envs/graspgenx` 安装。
因此这不是可拷贝到任意机器的独立环境；删除/升级基础环境后需重建并重新验证。

在仓库根目录 PowerShell 中运行（首次需联网下载）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_vision.ps1
```

其他机器可以指定 `-BasePython 'D:\path\to\python.exe'`，此本地安装器要求基础解释器
事先装好 `torch==2.11.0+cu128`，否则拒绝隐式替换 torch。
版本组合依据 [PyTorch 官方安装列表](https://pytorch.org/get-started/previous-versions/)。仅装依赖可加 `-SkipModels`。
固定依赖版本见 `requirements-vision.txt`；不要把它直接装进抓取模型环境。

权重位置：

- `_models/florence2/large/`：固定 HF revision `4271c66b88cdbc05735372ec13b2360108de5317`。
- `_models/yoloe/yoloe-26x-seg.pt`：Ultralytics assets `v8.4.0`。
- `scripts/download_vision_models.py` 下载并校验两个大权重的 SHA-256，支持 `--verify-only`。
- 环境、权重、缓存、运行结果不进入 Git。

本机 `pip check` 原有的 `isaacsim-kernel` / `psutil` 版本冲突同样会被 overlay 看见；
本任务不修改这一已有基础环境问题。以实际测试与版本记录为准，不能声称整个 Isaac 环境依赖完全无冲突。

## 一条命令运行

```powershell
# WebUI，默认 http://127.0.0.1:7860
.\scripts\run_yoloe_webui.bat --prompt "coffee cup" --fast yoloe

# 同一个 WebUI 使用 CV 跟踪（不要求 YOLOE checkpoint 存在）
.\scripts\run_yoloe_webui.bat --prompt "apple" --fast cv

# 图片 / 视频；输出带标注的文件
.\scripts\run_vision.bat --source samples\bus.jpg --prompt "bus" --fast yoloe --output output\vision_bus.jpg
.\scripts\run_vision.bat --source samples\bus_short.mp4 --prompt "bus" --fast cv --interval 0 --output output\vision_cv.mp4
```

启动器会设置 HF 离线模式并禁用 YOLO 自动安装；模型运行使用本地数据。
Florence 原生 Transformers 实现不启用 `trust_remote_code`。
`--interval 0` 只在初始化、提示/后端变化或丢失时 FIND，其他帧运行 TRACK；
默认间隔 45 帧也会刷新目标。多个英文目标用分号分隔，如 `"wrench; coffee cup; apple"`。
这只是可输入的提示示例，不是这些类别已经测试成功的承诺。

`--florence`、`--weights` 可覆盖模型位置；或设置
`BUSAGENT_FLORENCE_MODEL` / `BUSAGENT_YOLOE_WEIGHTS`。
CLI 和 WebUI 都使用同一套模型位置与后端参数。
`--device cpu` 可强制 CPU，但 Florence 和 YOLOE 的 CPU 性能未做验收。

## Python 接入

使用 `_envs\vision\Scripts\python.exe`，并将仓库 `source` 加到 `PYTHONPATH`：

```python
from find_and_track import FindTrackPipeline, RuntimeConfig

pipe = FindTrackPipeline()  # 共享本地默认模型路径
cfg = RuntimeConfig(prompt="coffee cup", fast_backend="yoloe", slow_interval=45)
pipe.load(cfg.fast_backend)

# frame_bgr: 当前帧 HxWx3 uint8 BGR，不是 RGB；每帧更新一次。
result = pipe.process_frame(frame_bgr, cfg)
for det in result.fast:
    print(det.label, det.xyxy, det.track_id, det.score)
```

纯 CV，不加载神经网络：

```python
import numpy as np
from find_and_track.cv_tracker import CvFeatureTracker
from find_and_track.types import Detection

tracker = CvFeatureTracker()
tracker.init(first_bgr, [Detection(np.array([x1, y1, x2, y2]), "target")])
detections = tracker.update(next_bgr)
```

**边界**：这些框没有深度、时间同步和机器人坐标系；不能直接当作抓取位姿。
`score` 不是统一校准概率（Florence 当前为 1.0，CV 是启发式分数）。
同类别多实例的类别标签不等于持久 object identity。CV 的遮挡、漂移和丢失判定仍有限。
后续接入 FineGrasp 时，必须结合当前帧 RGB-D、实例选择、几何一致性与手眼变换，并保留过期观测拒绝机制。
旧 `run_vision_follow` 是 2D→桌面坐标启发式演示，不是经过标定的精细抓取控制；本次没有重新运行它。

## 自检

### 本机验证结果（2026-09-05）

RTX 4070 Laptop 8 GB；`bus_long.mp4` 前 24 帧，提示 `bus`，禁止 TCP 连接；
同一进程顺序测试 YOLOE → CV，因此 CV 的 Florence 是热模型，首次耗时不可直接横比。

| 项目 | Florence → YOLOE | Florence → CV |
|---|---:|---:|
| 首帧 FIND | 1 个 bus 框 | 1 个 bus 框 |
| 后续 TRACK 有输出 | 23/23 帧 | 23/23 帧 |
| 快环整帧墙钟耗时中位数 | 88.3 ms | 43.9 ms |
| 快环整帧墙钟耗时 P95 | 95.5 ms | 45.6 ms |

两个模型合计加载约 4.65 s；冷启动 FIND 约 2.40 s，热 FIND 约 0.43 s。
整个测试 PyTorch 显存峰值 allocated 2934 MiB、reserved 3076 MiB（不含系统/Isaac 显存）。
这些是当前样例、图像尺寸、图形界面占用下的单次性能记录，不是吞吐承诺。

原始证据在本机 `output/vision_validation/validated/report.json`；标注视频同目录。
无头 Edge WebUI 的真实 Florence + CV 图片推理通过，证据在 `output/vision_webui/report.json` 和 `result.png`。
原 Isaac Python 的全部 **136 项单测通过**，其中此次新增 10 项视觉回归测试。
初次联调的空框和 CUDA NMS 失败记录仍保留在此前 `local_setup*` 输出目录，没有用成功结果覆盖失败记录。

### 复现命令

```powershell
# 原环境的无仿真单测（也可使用 vision 解释器）
& D:\isaac\env_isaacsim60\python.exe -m unittest discover -s tests -v

# 实际 Florence + 两个 TRACK 后端；阻止 TCP 连接后加载模型和推理
& .\_envs\vision\Scripts\python.exe scripts\verify_vision.py --frames 24
```

输出 `output/vision_validation/<时间>/report.json`、两个标注视频与首帧图片。
报告包含版本、输入哈希、逐帧框、加载耗时、FIND/快环耗时以及 PyTorch 显存峰值。
验收条件要求首帧 FIND 成功，且后续真实 TRACK 帧都有输出，避免把首帧 Florence 框复制当作跟踪测试通过。
这是本地样例集成测试，不是检测准确率、unseen-object 泛化或真实抓取成功率 benchmark。

可选 WebUI 回归：本机另装了 `playwright==1.62.0`（仅测试工具，不是推理依赖），
使用已安装的 Edge 无头运行。在一个终端启动
`scripts\run_vision.bat --webui --fast cv --prompt bus --port 7863`，另一个终端执行
`_envs\vision\Scripts\python.exe scripts\verify_vision_webui.py`。
报告和截图保存在 `output/vision_webui/`；测试后关闭临时服务器。

## 此次修正

- CV 模式不再被 YOLOE 权重检查阻断。
- 接受 OpenCV MIL 初始化成功时的 `None` 返回值，保留显式 `False` 失败处理。
- WebUI 继承 CLI 的 Florence 路径、提示、后端和参数，界面显示与后台一致。
- 切换 CV/YOLOE 时重新 FIND，用最新帧初始化对应后端。
- YOLOE 路径不再无谓初始化 CPU MIL；视频生成器中断时释放 capture。
- 固定依赖、权重版本与校验和，保留可复现安装/离线验证入口。
- Florence 解码不在特殊坐标 token 之间插入空格，兼容原生 checkpoint 的 slow BART tokenizer，防止生成了坐标却被解析成空框。
