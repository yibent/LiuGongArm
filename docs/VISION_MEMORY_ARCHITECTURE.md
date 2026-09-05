# Florence + YOLOE 记忆视觉方案

本文说明 `feature/yoloe-florence-memory` 分支中视觉记忆方案的结构、线程边界、启动方式和接口。方案面向 Isaac Sim 中的 SO-101 双相机机械臂，也可以脱离仿真运行独立视觉 WebUI。

## 1. 运行目标

系统把目标定位分为慢环和快环：

1. YOLOE 先使用文本标签库尝试检测目标。
2. 最高置信度低于 `max(conf, 0.55)`，或快环已经丢失目标时，调用 Florence 完成目标圈定。
3. Florence 的框同时初始化 YOLOE visual prompt 和 OpenCV 特征跟踪器。
4. 后续帧默认走 YOLOE TRACK 或 OpenCV TRACK，不重复运行 Florence。
5. CV 丢失达到 `lost_patience` 后，重新执行 Florence -> YOLOE 重捕获 -> CV 的恢复链路。
6. 用户要求记住目标时，把顶部相机和腕部相机的当前裁剪图保存为记忆。回放记忆时直接用裁剪图建立 YOLOE visual prompt。

## 2. 模块结构

```text
Isaac Sim / SO-101
  |
  +-- scene RGB-D camera -----------+
  |                                  |
  +-- wrist RGB-D camera -----------+-- MultiViewFindTrackPipeline
                                     |     |
                                     |     +-- shared FlorenceFinder (slow FIND)
                                     |     +-- scene YoloeVisualTracker
                                     |     +-- wrist YoloeVisualTracker
                                     |     +-- one CvFeatureTracker per view
                                     |
                                     +-- VisionRuntimeControl / HTTP :7861
                                           |
                                           +-- ObjectMemoryStore
                                           +-- MotionCommands
                                           +-- target / arm follow
```

主要文件：

| 文件 | 职责 |
| --- | --- |
| `source/find_and_track/pipeline.py` | 单相机 YOLOE/Florence/CV 状态机 |
| `source/find_and_track/multiview.py` | 共享 Florence，维护独立的双视角快环 |
| `source/find_and_track/florence_finder.py` | Florence 目标检测和框解析 |
| `source/find_and_track/yoloe_tracker.py` | YOLOE 文本提示、图片提示、检测和跟踪 |
| `source/find_and_track/cv_tracker.py` | MIL、光流、模板和 ORB 跟踪 |
| `source/find_and_track/memory.py` | JSON 元数据和多视角 JPEG 记忆库 |
| `source/mr_liu/vision/control.py` | 运行时控制、记忆 API、状态页 |
| `source/mr_liu/app/run_vision_follow.py` | Isaac Sim 双相机采集和机械臂目标跟随 |

## 3. 单帧状态转换

```text
无目标 / 提示变化
       |
       +-- 有 memory_id --> 读取记忆图片 --> YOLOE visual prompt
       |
       +-- 无 memory_id --> YOLOE text prompt
                               |
                 高置信度 ----+---- 低置信度或失败
                    |                         |
                 YOLOE detect             Florence FIND
                    |                         |
                    +-----------+-------------+
                                |
                       初始化 YOLOE/CV TRACK
                                |
                         持续快环跟踪
                                |
              丢失达到 lost_patience？
                         |
                       Florence 重定位
                         |
                  YOLOE 重捕获并恢复跟踪
```

`MultiViewFindTrackPipeline` 对两个视角分别维护 `_has_target`、丢失计数、YOLOE tracker 和 CV tracker。Florence 模型对象共享，但相机坐标、YOLOE 跟踪器和跟踪状态不共享。

## 4. 记忆格式和采集流程

记忆目录为：

```text
E:\LiuGongArm\runs\memories\
  <memory_id>.json
  <memory_id>\
    scene-<timestamp>-<random>.jpg
    wrist-<timestamp>-<random>.jpg
```

JSON 中保存：

- `memory_id`：记忆唯一 ID。
- `label`：目标标签。
- `created_at`：创建时间。
- `aliases`：可选别名。
- `views`：相机视角、裁剪图片路径、原始框和置信度。
- `metadata`：上游可附加的任务或位姿信息。

采集是快照式的。视觉循环先发布两个相机的最新原图和检测框，调用 `remember` 后保存当前快照。若要覆盖多个机械臂角度：

1. 目标识别稳定后调用一次 `remember`，得到 `memory_id`。
2. 通过已有运动接口转动机械臂或腕部，等待双相机画面稳定。
3. 再次调用 `remember` 并传入同一个 `memory_id`，追加新的视角样本。

回放时，两个视角会优先选择各自对应的记忆图片；指定 `view` 时优先使用该视角。记忆图片无法读取、YOLOE 无法重捕获或后续跟踪丢失时，系统仍会回退到 Florence。

## 5. Isaac Sim 启动

先确认仓库根目录存在 `yoloe-26x-seg.pt`，然后启动双相机机械臂视觉循环：

```bat
.\scripts\run_vision_follow.bat --headless --device cuda --prompt "red cube"
```

只测试感知，不让检测结果驱动机械臂：

```bat
.\scripts\run_vision_follow.bat --headless --no-follow --device cuda --prompt "red cube"
```

运行后控制页和 JSON API 默认位于：

```text
http://127.0.0.1:7861
```

主要启动参数：

| 参数 | 含义 |
| --- | --- |
| `--prompt` | 标签或描述；多个目标用分号分隔 |
| `--fast yoloe` | YOLOE 快环 |
| `--fast cv` | OpenCV 快环；丢失恢复仍会临时使用 YOLOE |
| `--slow-interval 0` | 仅首次、丢失、提示变化或手动 FIND 时运行 Florence |
| `--control-port` | 修改运行时控制端口 |

## 6. 记忆和控制 API

### 查询状态和记忆

```http
GET /api/status
GET /api/memories
GET /api/frame/scene.jpg
GET /api/frame/wrist.jpg
```

### 采集记忆

```http
POST /api/command
Content-Type: application/json

{"skill":"remember","params":{"label":"red mug","aliases":["mug"]}}
```

也可以直接调用 `POST /api/memory`。返回值中的 `memory.memory_id` 用于后续追加和回放。

### 回放记忆

```http
POST /api/command
Content-Type: application/json

{"skill":"recall","params":{"memory_id":"<memory_id>","view":"wrist"}}
```

也可以调用 `POST /api/recall`。回放会更新当前 prompt 和视觉状态，下一视觉周期直接建立图片提示。

### 删除记忆

```http
POST /api/command
Content-Type: application/json

{"skill":"forget","params":{"memory_id":"<memory_id>"}}
```

也可以调用 `POST /api/forget`。控制层还提供 `GET /api/capabilities`，其中包含 `remember`、`recall` 和 `forget`。

## 7. 独立视觉 WebUI

不启动 Isaac Sim 时，可以运行：

```bat
.\scripts\run_florence_local.bat
```

默认地址为 [http://127.0.0.1:7860](http://127.0.0.1:7860)。独立 WebUI 适合验证图片/视频上的 Florence、YOLOE 文本提示和跟踪；机械臂双相机记忆采集应使用 7861 控制服务。

当前本地视觉环境使用 `E:\LiuGongArm\.venv\python.exe`。第一次启用 YOLOE 文本提示时，Ultralytics 会准备 MobileCLIP 文本编码器 `mobileclip2_b.ts`；该文件已经被 `.gitignore` 忽略。GPU 可用时建议使用 `--device cuda`，否则会回退到 CPU。

## 8. 线程和数据安全

- Isaac Sim 的关节读取、目标更新和关节写入只发生在仿真线程。
- `VisionWorker` 只允许一个推理任务在途，避免无限堆积旧帧。
- 相机帧、深度图和相机解投影参数在提交推理前复制或冻结，推理完成后不会用新相机位姿解释旧帧。
- HTTP 控制层只修改线程安全的 `RuntimeConfig`、记忆库和运动命令队列。
- 记忆元数据使用临时 JSON 后 `os.replace` 写入，避免留下半个 JSON 文件。

## 9. 验证结果和已知问题

已执行：

- `python -m compileall -q source tests vision_main.py`：通过。
- 记忆回读、双视角、控制 API 和 YOLOE 图片提示测试：通过。
- YOLOE 真实权重文本提示 `bus`：1 个检测，最高置信度约 `0.917`。
- Florence + YOLOE 真实图片提示链路：已在 `samples\\bus.jpg` 上验证。
- 完整测试：修正根目录 README 的历史 `BusAgent/README.md` 链接后，122 项全部通过。
