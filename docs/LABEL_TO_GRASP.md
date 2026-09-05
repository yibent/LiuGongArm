# 标签 → 光流粗接近 → 腕部 FineGrasp

日期：2026-09-05。开发分支：`codex/label-to-grasp`。

这是原有精细抓取模块之前新增的**可选入口**，不是替换 GraspGenX，也不是重新训练 VLA。
本地权重、环境安装见 [LOCAL_VISION_SETUP.md](LOCAL_VISION_SETUP.md)；
原抓取与主动恢复见 [FINE_GRASP_CURRENT_HANDOFF.md](FINE_GRASP_CURRENT_HANDOFF.md)。

已合并 `origin/master` 的 `f9b0c65`（合并提交 `3ed87b7`），保留其视觉记忆、YOLOE 优先检测及 Web 双视角/控制接口。
**硬件和相机配置不按主分支优先：腕部安装位置、角度和焦距已恢复原成功方案，并在合并代码上实际复测。**

## 一条命令

在仓库根目录 `D:\liuGong` 的 PowerShell 中运行：

```powershell
.\scripts\run_label_grasp.ps1 -Label 'red cube' -RecordVideo
```

默认无头 Isaac Sim、Florence + YOLOE、GraspGenX/GraspMoE、原有 active recovery、`fine_grasp` 腕部安装位。
GraspGenX 服务不可用时明确失败，不再自动退回几何 backend；显式 `-Backend geometric` 仍可用于诊断。
输出目录自动按时间创建在 `output/label_grasp/`；指定 `-Output` 时必须使用空目录，避免覆盖证据。
脚本启动并回收自己创建的本地模型服务，不关闭原已存在的共享服务。

```powershell
# 仅测试定位、机械臂移动和腕部交接，不闭爪、不抬升
.\scripts\run_label_grasp.ps1 -Label 'red cube' -Backend geometric -CoarseOnly

# 不用 YOLOE 做框细化，仍然使用同一套 LK 光流跟踪
.\scripts\run_label_grasp.ps1 -Label 'red cube' -LocalizationMode florence

# 咖啡杯测试场景：CaseJson 只负责构造物理场景和提供物品属性
.\scripts\run_label_grasp.ps1 -Label 'coffee cup' -CaseJson configs/eval/fine_grasp_v2/development/011_coffee_mug_0.json

# 显示 Isaac 界面（不要与另一个 Isaac 测试抢同一 GPU）
.\scripts\run_label_grasp.ps1 -Label 'red cube' -NoHeadless

# 显式仿真扰动：第二段粗接近开始时将物体沿世界 X 移动 2.5 cm
.\scripts\run_label_grasp.ps1 -Label 'red cube' -TestCoarseShiftM 0.025 -RecordVideo
```

`-Label` 不会在场景中生成对应物体；默认场景始终是红方块。测试其他物体须提供相应 `-CaseJson`。
标签建议先使用简单英文短语。尚未验收中文标签、复杂空间指代和多同类实例选择。
不带 `--label` 的原 `run_fine_grasp_demo` 保留原行为。

## 执行和相机分工

```text
标签
  → 固定场景 RGB-D：Florence 开放词汇检测 → 可选 YOLOE 视觉提示细化框
  → 框内有效深度 / 桌面剔除 / 机器人自身剔除 → 米制目标位置
  → Shi–Tomasi 特征 + 金字塔 Lucas–Kanade 前后向光流
  → 小段、较高速粗接近；移动期间持续读 RGB-D，失配先停
  → 明确停止粗接近控制，获取新的腕部 RGB-D 并核对目标身份
  → 原 GeneralGraspNode / RecoveringGraspNode
  → GraspGenX 候选 / IK 碰撞筛选 / 腕部小步伺服 / 闭爪 / 抬升 / 验证
```

- 两个**感知相机**：固定场景 RGB-D 与 wrist RGB-D。此入口默认把固定相机设为斜俯视，以减轻手臂遮挡。
- 视频额外有一个 overview 展示机位，不参加任何感知或决策。
- 粗接近使用固定相机的同帧 `RGB + depth + K + T_base_camera` 反投影，不使用经验像素到桌面映射。
- FineGrasp 保持 eye-in-hand：每次新观测按 `T_base_ee @ T_ee_camera` 更新视觉几何，不固定相机位姿。
- 算法不读取物体 USD 位姿来定位或补救。场景构造、显式扰动器、失败诊断及最终物理抬升统计才读取仿真真值；机器人自遮挡掩码使用机器人实例 ID，不使用目标实例 ID。
- 物品 fragile/thin/reflective 等属性仍可由上游提供，位置从标签 RGB-D 观测得到。

### 腕部安装选择与“画面倒置”

`configs/cameras.yaml` 的全局默认和 demo 默认均为原成功安装位；`SceneCamera` 的 profile 按副本应用，不改写共享配置。

| 参数（相对 gripper） | 默认 `fine_grasp` | 仅显式对照 `tabletop_wide` |
|---|---|---|
| 位移 m | `[0.060, 0, -0.030]` | `[0.150, 0, -0.100]` |
| 安装四元数 wxyz | `[0.53895, 0, 0.84234, 0]` | `[0, 0, 1, 0]` |
| 焦距（Isaac API 参数） | `1.8` | `1.0` |
| 同一方块交接时光轴向下俯角 | 约 67.1° | 约 2.5° |
| 合并后完整抓取结果 | 成功，物理抬升 6.15 cm | 目标裁切，腕部身份检查失败，未抓取 |

两种安装角相差约 65.2°。宽视野方案在该精抓姿态下近乎水平看桌面，背景看起来倒置；
它不是 RGB 被显示程序翻转，也不是把图像旋转 180° 就能解决的覆盖问题。
以上俯角来自 `12_merged_cube` / `13_wide_camera` 的 `initial_rgbd.npz` 中同帧 `T_base_camera`：
OpenCV 光轴为旋转矩阵第 3 列，基坐标 Z 朝上时俯角为 `asin(-T_base_camera[2, 2])`。
安装配置采用 Isaac 的 +X forward 约定，运行时视觉外参采用 OpenCV +Z forward，不能混用。
不对模型 RGB 单独翻转；RGB、深度、K、手眼变换必须配套一致。

```powershell
# 仅用于复现对照；不作失败时的自动相机 fallback
.\scripts\run_label_grasp.ps1 -Label 'red cube' -WristCameraProfile tabletop_wide -CoarseOnly
```

当前仍为两路 15 Hz、120 Hz 物理步及合并后的 +90° Z 机器人基座方向，**不是整套仿真环境原样回退**。
它们已随近距离相机在 `12_merged_cube` 通过一次完整回归；单次相机 A/B 不能证明所有物体的最优安装位。
独立 demo 不加载主线默认的六个松散道具，保持显式单对象开发场景；Web 主场景默认多物体行为保留。
若改真机安装或基座方向，必须重新标定/验收，不可直接套用这些仿真四元数。

## 快慢环与安全门

FIND 在独立 `_envs/vision` 进程执行。默认端口 5570，仅绑定 `127.0.0.1`。
每次 FIND 使用离线 Florence-2-large 和可选 YOLOE-26x-seg，结束后释放模型权重，给 Isaac 与 GraspGenX 留显存。
响应带原始 `sequence`；模型推理结束后必须成功取得更新的 RGB-D/光流观测，旧检测帧不能直接驱动机器人。
YOLOE 在这里是 **Florence 框的视觉提示细化器**，不是独立语义判别器，也不负责后续 TRACK。

TRACK 明确使用 LK，不使用旧 WebUI 中以 MIL 为主的 CV 跟踪器。
要求至少 3 个可靠特征、前后向误差 ≤1 px、位移一致性和光度误差通过。
短时间内采用稳健平移，不假装解决大旋转、缩放或长期遮挡。LK 原理参考
[OpenCV 官方光流教程](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)。

粗接近默认最多 4 cm/段、速度比例 0.65；这是 SO-101 仿真参数，不是真机速度认证。
默认开放夹爪的指尖中心在观测顶面上方 6 cm。
若不可达，可尝试最低 5 cm、朝初始机械臂位置的另一闭合轴方向，以及 15°/30° 的有限倾斜观察候选，缓解非对称夹爪 EE 偏置及五轴约束带来的可达性限制。
粗接近可请求 IK 返回其实际达到的附近观察姿态作为新候选，最多调整三次，
位置变化 ≤3 mm、相对请求轴变化 ≤15°、相对竖直方向 ≤30°；仍需对新姿态完整重新规划检查。
这不放宽执行器 5° 轴误差验收标准，也不改动 FineGrasp 的抓取姿态求解参数。
这些都是新候选，仍须完整通过 IK、机器人/世界碰撞与观测点云上的指尖扫掠检查。
它不是任意远距离、任意障碍场景的全局路径规划器；受阻时返回失败。

执行期间每 6 个控制 tick 读取一次 RGB-D；目标变化 >12 mm 或光流失效时停止。
规划后再观察，变化 >8 mm 就丢弃旧路径；首次 FIND 加最多两次重新定位，共 3 次，粗接近总预算 100 秒 / 24 段。
跨相机交接要求腕部采集时间晚于粗接近停止时间栅栏、年龄 ≤350 ms、序列未复用、
有 ≥80 个有效目标像素、米制中心差 ≤3 cm、颜色差通过。
这些是保守的关联阈值，不代表最终抓取精度；最后误差由原腕部 FineGrasp 继续消除。

## BusAgent 接口

新代码按职责拆分：

| 文件 | 职责 |
|---|---|
| `source/mr_liu/perception/semantic_target.py` | 本地标签请求、RGB-D 几何、跨帧/跨视角身份检查 |
| `source/mr_liu/perception/optical_flow.py` | 独立纯 LK 跟踪器，失败返回 None |
| `source/mr_liu/control/coarse_approach.py` | 有限步粗接近、移动中 guard、返回新的 TargetSpec |
| `scripts/serve_semantic_locator.py` | 离线模型服务，不包含机器人控制 |
| `scripts/run_label_grasp.ps1` | 本地环境/服务生命周期和完整 demo 入口 |

装配逻辑对应 demo 中的 label 分支：

```python
if not gripper.open(0.075, speed_mps=0.04):
    raise RuntimeError("Open gripper precondition failed")
tracker = SemanticFlowTarget(scene_camera, LocalSemanticLocator(), target,
                             table_height_m=table_height, trace=trace)
updated_target = CoarseApproach(tracker, motion, trace=trace).run()
# 粗接近已 stop 并清除路径缓存，以下步骤不能省略。
obs = wrist_camera.capture(updated_target)
mask = seed_segmenter.segment(obs, updated_target)
tracker.validate_wrist_handoff(obs, mask)
result = fine_node.execute(FineGraspRequest(target=updated_target))
```

这是装配片段：camera、motion、gripper、segmenter、fine_node 需由 BusAgent 注入。
当前指尖扫掠与 EE 偏置建模假设**夹爪已成功张开 75 mm**；换夹爪须替换宽度与几何标定，不能直接套用。
两个相机须使用同一个单调时钟，`capture()` 必须返回新采集的同步 RGB-D 与外参，不可给缓存图像重打时间戳。
整个流程必须持有机器人运动独占权，不能同时运行旧 follow-target 或另一个控制器。
`motion` 除原机器人状态和运动接口外，还需实现 `clear_grasp_plan()`、`ee_pose_for_grasp()`、
`is_observation_path_safe()`（检查实际执行路径的世界/自身碰撞与观测表面扫掠），
以及 `execution_guard` 回调契约：
在实际运动期间周期调用；返回 False 后停止并报告运动未完成；不能只在运动结束检查。
遇到 `TargetObservationError` 必须终止此轮交接，不得以旧粗位置继续抓取。
完整可运行装配在 `scripts/run_fine_grasp_demo.py`，原 FineGraspRequest/Result 接口未改变。

## 实测证据与限制

本轮结果记录在 `output/label_grasp/`，所有尝试（包括失败）保留。
这些是开发场景连通性测试，**不是冻结 unseen-object 泛化验收**，没有真机测试。

| 运行 | 结果 |
|---|---|
| `01_coarse` | 方块：模型/光流通过，额外 12 cm 高度路径不可达，安全停止 |
| `02_coarse` | 方块：4 段粗移动，23 次光流观测，6 cm 腕部交接通过；未闭爪 |
| `03_cube_full` | `red cube`：完整成功；实物理抬升 8.15 cm；20 次伺服、1 次模型调用；录有 `demo.mp4` |
| `04_cup_full` / `05_cup_full` | 咖啡杯定位/光流通过，6 cm / 5 cm 固定方向交接不可达，安全停止 |
| `06_cup_full` / `08_cup_full` | 咖啡杯：闭合轴及倾角候选仍未通过原 IK 轴误差门；诊断显示位置误差很小，轴误差仍为约 7–14° |
| `07_cube_moving` | 方块在粗接近时移开 2.5 cm：真实触发 guard 停止，更新位置后完成抓取，抬升 8.17 cm；21 次 FineGrasp 伺服；录有视频 |
| `09_fruit_full` | `red ball`、球形水果代理：完整成功，物理抬升 7.90 cm，28 次伺服；腕部单独验证未通过，由场景视觉验证确认；不是实物苹果测试 |
| `10_cup_full` | 尝试提高粗 IK 轴任务权重仍未通过；该权重试验已撤回，保留失败证据 |
| `11_absent_label` | 方块场景传入不存在的 `banana`：返回 `semantic_target_not_found`，未进入粗移动或抓取 |
| `12_merged_cube` | 合并后、原近距离安装位：完整成功，实际抬升 6.15 cm；6 段粗移动、13 次精细伺服、1 次模型调用约 687 ms；计划 8 cm 抬升终点未完全到达，但腕部和场景验证均确认已抓起 |
| `13_wide_camera` | 同场景对照主线宽视野安装位：粗接近完成；目标被画面上沿裁切，`wrist_handoff_identity_mismatch`，不执行抓取 |
| `14_merged_cup` | 咖啡杯识别、7 段粗接近及腕部交接通过；FineGrasp 24 次伺服/4 次模型调用后仍有 3.75 cm 平移误差，`servo_timeout`，未闭爪；恢复退让路径未验证，安全停止，抬升 0 |
| `15_merged_moving` | 当前默认近距离相机：方块在粗接近中移动 2.5 cm，实际触发 `target_moved_during_transit` 停止并更新目标；7 段粗移动、35 次光流观测、15 次精细伺服后成功，物理抬升 5.61 cm，腕部与场景均确认；计划 8 cm 抬升终点仍未完全到达；录有视频 |
| `16_camera_baseline_final` | 修复录像扰动标注并保存完整手眼链后的重复回归：2.5 cm 移位再次触发暂停/更新；7 段粗移动、15 次精细伺服，实际抬升 5.56 cm，两路验证通过；8 cm 抬升终点仍未完全到达；完整录像 `demo.mp4` |

合并前 `03_cube_full` 方块完整录像运行中，FIND（含模型加载）14.33 s，标签到交接 22.06 s，FineGrasp 21.84 s。
这些不包括 Isaac/服务启动，且含录像和诊断开销；不能据此宣称工业实时吞吐率。
所选 grasp 的 backend 为 GraspGenX，但实际是 **OBB 候选经模型评分**，不是纯扩散候选成功。

诊断文件：

- `coarse_report.json`：检测框、光流统计、实际命令、guard、交接、失败原因。
- `coarse/debug/`：RGB、深度、mask、同帧内外参、轨迹事件；可回放检查漂移。
- `initial_wrist.png` / `initial_mask.png`：交接时腕部视野。
- `initial_rgbd.npz`：同帧 RGB-D、K、相机外参；新运行还保存 `T_base_ee` / `T_ee_camera`，可检查变换链。
- `report.json`：FineGrasp 结果、候选来源、伺服误差、抓取/抬升验证和仿真真值对照。
- `demo.mp4` / `demo.video.json`：可选展示录像及时间元数据。
- `07_cube_moving` / `15_merged_moving` 的旧录像页脚误标“无故障注入”，实际扰动以 `coarse_report.json` 的 `test_coarse_shift_applied` 和 guard 事件为准；已修复新录像的粗接近扰动标注，原证据不覆盖。
- `console.log` / `locator_*.log` / `error.txt`：进程、模型和异常诊断。

不能忽略的边界：

1. Florence 会幻觉定位；实测不存在的 `banana` 也可能返回框。YOLOE 视觉提示不能独立纠正语义错误。几何、机器人掩码、外观一致性能够拒绝部分错误，但不保证拒绝所有错标签。
2. 只支持已标定的固定场景相机、已知水平支撑桌面、当前本地工作区。当前前端以桌面以上 4 mm 的有效深度提取目标，不覆盖贴桌薄件、透明件、严重反光深度孔洞。
3. 同标签多个合格实例默认报歧义；还未实现复杂指代/任意杂乱场景的开放集身份验证。
4. 光流无可靠特征、遮挡、视图变化过大时返回失败/重定位；不承诺纯色、金属和薄件都能跟踪。
5. 真机还需实际标定、时间同步、机器人自遮挡模型、障碍地图、限速/急停与力控认证。此入口没有接管真机控制。
6. 咖啡杯尚未完整抓起；工业工具、薄件、多姿态/光照/随机种子的冻结测试集也未完成本入口验收，不能由方块或球形代理成功推导泛化能力。
7. 合入的 Web/CLI YOLOE 文本优先分支需要额外 CLIP tokenizer / MobileCLIP 文本编码权重，当前本地尚未配齐，会回退 Florence；本入口已实跑的是 Florence → YOLOE **视觉提示**，不要混淆两条分支的验证范围。

单测运行：

```powershell
D:\isaac\env_isaacsim60\python.exe -m unittest discover -s tests -v
```

覆盖真实 LK 平移、无纹理拒绝、坐标变换、深度缺失、机器人排除、检测延迟、帧新鲜度、歧义目标、腕部身份门、运动中停止与物理扰动器。
本次合并及相机取舍后的完整单测为 **242 项通过**；包括默认安装朝向夹持区域、profile 互不污染、正确的扰动录像标注，以及 IK 邻近姿态必须重新经过完整安全检查。
