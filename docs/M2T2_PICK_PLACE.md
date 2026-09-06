# M2T2 抓取与放置实验线

分支：`codex/m2t2-pick-place`。本实验不修改 BusAgent 上层抓取请求接口。

## 当前状态

2026-09-06：已在官方 Franka Panda 上完成一次严格的全流程成功，证据为 `output/franka/22_m2t2_release_retreat`。抓取、持续夹持、放置、释放、撤离及 5 帧释放后稳定验证全部通过；释放前视觉 XY 误差 2.744 mm、间隙偏差 -1.145 mm、持物偏移 3.122 mm，释放后位置扩散 0.121 mm。单次开发成功不能代表稳定成功率。

Franka 使用 `configs/robot_franka.yaml`，腕部相机安装在 `panda_hand`，另启用独立 `PlacementRGB` 固定相机。粗接近/恢复相机不再被放置阶段重新定向。`scripts/verify_cameras.py --robot franka` 已验证三路 RGB-D：scene、placement、wrist 均有连续 acquisition、有效深度与合法旋转矩阵。M2T2 上游 decoder 的 `torch.cross` 已显式指定最后一维，修复 PyTorch 2.x 三候选时生成非正交旋转的问题，网络权重及预测轴定义未改变。

新增复测 `output/franka/25_m2t2_tray_repeat`：标签交接和抓取通过；托盘内部最高模型置信度 0.05686，低于上游 0.4，零个可用放置候选，因此保持夹持、未释放。整个命令约 134.9 s。这次失败不能计为完整成功，也没有降低模型阈值。

`output/franka/batch_20260906_a` 的第一批冻结试验显示：seed1 方块在旧跨视角身份阈值处以颜色误差 0.1813 拒绝，第二个 AnyPlace 方块以 20/20 个旋转候选拒绝，第三个 M2T2 托盘在夹爪闭合阶段失败。交接阈值已单独收窄调整到 0.20，并在新指纹批次复测；这只处理相机光照差异，不放宽目标位置/深度/释放检查。

冻结批次入口：`scripts/benchmark_franka_pick_place.py --manifest configs/eval_franka_pick_place_v1.json --output output/franka/<新目录>`。覆盖不同 seed、方块/加大方块/圆柱/球体、目标坐标、托盘、障碍物，以及 GraspGenX + AnyPlace 对照。每个结果绑定物体 JSON、源文件/相机/机器人配置指纹、命令、日志和完整结果；`--resume` 拒绝在代码或配置改变后继续同一批次。完整成功必须同时满足抓取成功、放置成功、释放、验证和进程正常退出。该清单是开发对照，未作为独立验收集。

稳定底面先验现在只对仿真中确定竖直创建的 cube/cylinder 声明；sphere 仍不能由平面控制器宣称稳定放置。新增 `socket/insert` 与 `rack/hang` 场景和碰撞几何，现有执行器明确返回 `placement_relation_not_executable`，尚不代表实现了物理插入或悬挂。

已在本机 RTX 4070 Laptop 8 GB 上加载官方通用 `m2t2.pth`，实际运行抓取和放置推理；没有针对测试物体训练模型。**模型推理通过不等于物理抓放通过，目前不能声称通用抓放验收完成。**

- 离线 RGB-D 回放：首次抓取推理约 6.62 s，随后放置约 1.49 s；模型峰值 allocated tensor memory 约 363 MiB，不含 Isaac、Florence、SAM 和 CUDA 上下文。
- Isaac 同时运行时，已测抓取推理约 8.40 s、10.49 s。当前使用便携 PointNet++ 算子，不是上游原生 CUDA 性能。
- `output/m2t2/01_closed_loop`：旧固定接近姿态下目标不在腕部视野，未调用模型。
- `output/m2t2/02_label_closed_loop`：Florence 标签定位、光流、粗接近及腕部交接通过；M2T2 实际生成候选并进入视觉伺服，随后执行失败并重新估计，最终候选未通过 IK/夹持中心约束。未闭合、未抬升、未放置。
- `output/m2t2/03_aperture_diversity`：加入观测开口下限与候选去重后，仍在接近/IK 阶段失败。
- `output/m2t2/04_calibrated_aperture`：沿用既有夹爪的 10 mm 开口余量，M2T2 抓取成功，15 次腕部伺服、一次模型调用，视觉测得抬升约 53.9 mm；仿真真值仅用于独立验收（54.7 mm）。放置候选未通过区域/姿态检查，`released=false`。这是单次方块抓取通过，不是拿放通过或泛化成功率。
- `output/m2t2/05_region_first`：再次抓起，但指定区域没有超过上游 0.4 阈值的放置候选。仍未释放。
- `output/m2t2/07_support_preprocessed`：固定 Torch 随机种子后的新观测没有抓取候选，恢复安全退让也未通过。未闭合、未进入放置；说明前两次抓起不代表稳定泛化。
- `output/m2t2/08_hybrid_place`：GraspGenX 抓起，M2T2 放置有区域候选但未通过支撑检查，未释放。
- `output/m2t2/09_hybrid_local_context`：混合方案采用 20 cm 模型上下文，M2T2 选点并实际完成两步放置对齐；随后姿态变化使边缘选点的支撑余量不足，安全停止、未释放。后续改为预留任意 yaw / 允许倾斜下的物体投影包络，仍用原始测量进行最终支撑检查。
- `output/m2t2/11_hybrid_robust_footprint`：Florence 加载时出现 Windows commit/pagefile 不足（1455），识别失败，未抓取。不是 GPU OOM。后续 M2T2 服务改为首次请求才加载权重，避免在 Florence 冷启动阶段叠加内存峰值；不修改系统 pagefile。
- `output/m2t2/12_lazy_hybrid`：混合方案抓起并完成 23 轮放置伺服，视觉估计 XY 误差 0.395 mm、底面间隙 2.942 mm；最终撤离预检出现 `observed_surface_finger_collision`，未释放，完整成功为 false。上述精度是视觉估计，不是独立真值精度。
- `output/m2t2/13_all_m2t2_robust`：再次用 M2T2 抓起，并用同模型生成放置目标、完成对齐；下降过程中手指扫掠检查失败，未释放。明确说明 M2T2 不是完全不能抓，但完整拿放仍未成功。
- `output/m2t2/14_escape_corridor`：混合方案再次到达释放预检；轴向及 +tool X 3/6/10 mm 的撤离候选均未通过手指扫掠检查，未释放。该改动没有被记成成功修复，仍需定位具体碰撞样本及执行几何误差。
- `output/m2t2/15_contact_evidence`：全 M2T2 在抓取阶段失败；恢复后 96 个候选中 56 个开口超限、38 个 IK 不可达、2 个桌面间隙不足，未闭合、未进入放置。最终主失败分类是 `gripper_width_infeasible`，不代表只有开口一个原因。
- `output/m2t2/16_contact_hybrid`：混合方案抓起并到达释放复检；首次 10 mm 侧向撤离通过，新帧复检全部失败，`release_space_changed`、未释放。记录的被拒绝 FK 样本位于路径第 8/17 个采样点，命中均是持物边缘点；不能归因为起始帧坐标偏差，也不是外部支撑面碰撞。后续加入 15/20 mm 的受检侧向候选，未放宽任何接触阈值。
- 后续试验结果以对应 `report.json`、`place_report.json` 和 trace 为准。无 `place_report.json` 不能解释成放置成功。

## 本机运行

在 `D:\liuGong` 的 PowerShell 中运行：

```powershell
.\scripts\run_m2t2_pick_place.ps1 -Robot franka -ContactMaterial profile -Label 'red cube' -Destination 'blue square' -RecordVideo
```

命令启动本地 M2T2 与 Florence 服务、Isaac headless 场景，并尝试完整标签抓取和放置。`-NoHeadless` 打开界面；默认输出为 `output/m2t2/YYYYMMDD_HHMMSS`，可用 `-Output` 指定。视频为该目录的 `demo.mp4`。输出目录必须为空或不存在，避免覆盖证据。退出码非零表示未完成完整流程。

`run_summary.json` 汇总抓取/放置状态；`model_requests/*.json` 保存模型实际收到的点云、位姿、序号及返回值。初始 RGB-D 回放与实际模型请求不一定是同一帧，诊断时优先重放精确请求：

```powershell
.\_envs\m2t2\Scripts\python.exe scripts\replay_m2t2_request.py <记录文件> --output output\m2t2\exact_replay.json
```

M2T2 服务默认延迟加载，`/health` 的 `ready` 表示接口就绪，`model_loaded` 才表示权重已加载；`--eager` 可在启动时加载。首次请求包含加载耗时，控制器必须重新观察，不能拿旧帧直接动作。长时间复用的服务可能仍持有模型内存，不能据此保证下一轮 Florence 冷启动不会遇到提交内存不足。

`scripts/diagnose_place_contact.py <运行目录>` 可只读重放最终帧的接触诊断，输出框体命中、持物归属及估计侵入深度；默认使用外观关联而非运行时 SAM2，不能据此直接放宽碰撞阈值。后续撤离策略增加 +tool X 的 3/6/10 mm 候选偏移，均需通过原来的实际 FK、世界和接触扫掠检查，不豁免新的碰撞。

新增的运行时 `place_path_rejected.profile` 保存被拒绝的实际 FK 起止姿态、关节值，以及 `contact_start/end` 的原始手指框命中点（最多 64 个）、持物归属、初始/当前侵入深度与参考末端姿态。诊断不改变安全判定；`raw_finger_hits` 包含允许的旧接触，不能直接当成最终被拒绝点数。16 轮后候选扩展为 0/3/6/10/15/20 mm；每一个都须在最新观测上通过复检。

环境依赖：

- Isaac：`D:\isaac\env_isaacsim60\python.exe`。
- Florence：`_envs\vision\Scripts\python.exe` 及原有本地权重。
- M2T2：`_envs\m2t2\Scripts\python.exe`，通过 `--system-site-packages` 继承已有 `_envs\graspgenx` 的 Torch 2.6 / CUDA 12.4 等依赖；没有升级原环境。不是独立、完全锁定的可分发环境。
- SAM2：已有 `_vendor\sam2` 和 `_models\sam2\sam2.1_hiera_tiny.pt`。在 M2T2 服务内按需加载，用于持物轮廓，不需要另行加载 GraspGenX 神经网络。
- M2T2 源码：`_vendor\M2T2`，上游提交 `be2e5f6feae2f961615ab0f7b643f0f98b0c3a50`。
- 权重：`_models\m2t2\m2t2.pth`，137562712 bytes；SHA256 `e35c3cb11e06f46c5d406bdfc756bc06f48b256dd6d638408d9a5ff13deb97fb`。

`_models`、`_envs`、`_vendor`、`output` 不入 Git。新机器必须另行部署依赖与权重，不能只 clone 本仓库就运行。

只跑模型回放，不移动机械臂：

```powershell
# 终端 1
.\_envs\m2t2\Scripts\python.exe scripts\serve_m2t2.py
# 终端 2：需要已有这次 RGB-D 记录
.\_envs\m2t2\Scripts\python.exe scripts\benchmark_m2t2.py --run output\fine_place\20_sequential_models
```

回放 JSON 明确标记 `offline_model_inference_only`，不计算抓放成功率。

显式混合对照（不算 M2T2 全流程成功）：

```powershell
.\scripts\run_m2t2_pick_place.ps1 -GraspBackend graspgenx -RecordVideo
```

`-M2T2SurfaceRangeM 0` 关闭模型输入的支撑面归一化；默认 `.02` 与上游示例的 surface range 一致。仅归一化网络特征中的邻近平面高度；输出接触点、支撑、障碍、释放高度一律保留原始测量。这个开关不是允许穿过 2 cm 障碍。

`-M2T2SceneRadiusM` 控制放置模型上下文半径，当前 demo 默认 `.2` m；安全检查仍使用完整的当前局部观测。抓取模型保持原始几何输入，未因为放置消融有效就同步启用归一化。抓取归一化仅有离线实验选项，尚无物理收益证据。

预处理消融：`output/m2t2/06_*.json`。对同一已记录场景，原始输入的目标区域最大置信度约 0.28；支撑面归一化后约 0.68，出现通过几何筛选的候选。单独把上下文半径 0.4 m 缩到 0.2 m 无效；二者结合在两次记录上都得到可用候选。仍然只是离线候选结果，不是放置成功。

## 模块和接口

- `grasp/backends/m2t2.py`：Torch-free HTTP 客户端、`GraspBackend.generate()` 适配器。配置 `backend: m2t2`，可通过 `backends.m2t2.url` 修改服务地址。
- `scripts/serve_m2t2.py`：同一份官方权重处理 `pick` 和 `place`。只绑定 `127.0.0.1:5580`，没有机械臂动作接口。`GET /health`、`POST /infer`、`POST /segment`。
- `grasp/backends/m2t2_ops.py`：Windows 无 CUDA 编译工具时的真实采样、分组和插值算子；神经网络层和权重仍使用上游实现。推理专用，不支持训练。保留原生 FPS 的原点排除规则，CPU/GPU 平局选择和浮点边界可能不同；尚未完成原生 CUDA 算子逐层数值等价验收。需要原生算子时使用 `--ops native` 并先安装上游扩展。
- `place/m2t2.py`：模型提出放置候选，再按语义区域、观测支撑、持物尺寸和现有姿态能力筛选。模型分数不构成稳定性或碰撞证明。
- `GeneralPlaceNode(..., backend=...)`：新增可选后端，未传入时保留原有几何方案。

抓取输入来自当前腕部 RGB-D，包含周围场景点云，而不只输入物体 mask。候选绑定观测序号，通过当前 `T_base_ee @ T_ee_camera` 转换。M2T2 默认抓取原点在夹持中心后方 103.4 mm，客户端先恢复夹持中心，再由原有 SO-101 适配器转换到实际末端。候选按目标接触点关联；最小开口以同帧观测到的手指通道截面为下限，位姿不因开口修正而重写。

慢模型完成后，原有 FineGrasp 会重新观察目标，拒绝发生变化的旧候选，再执行小步伺服。保留 IK、真实夹爪约束、碰撞检查、夹爪闭合、抬升和视觉验证；没有几何回退伪装成 M2T2 成功。

16 轮后修正候选去重：不再将相差 180° 的两指分配当成重复项。对于对称夹爪它们可能几何等价，但 SO-101 固定指/动指不对称，两者映射到不同末端位姿及 IK。只保留模型实际输出的两个方向，不人为旋转补造候选；收益须以之后的物理测试为准。

## 必须保留的限制

1. 放置目前只选择与当前末端朝向一致的模型候选。其它 yaw 候选明确拒绝，不把旋转静默丢弃。尚未实现任意 6D 持物跟踪与旋转放置。
2. 沿用稳定底面、紧凑物体和近水平放置的既有限制。当前 Isaac 稳定底面先验只对竖直创建的 cube/cylinder fixture 声明；不是模型自动证明各种杯子、工具和水果稳定。
3. 放置主要使用外部 RGB-D 观察目的地，腕部优先跟踪持物。最终释放与稳定性验证仍由现有闭环完成。
4. 当前缺少新增未知物体冻结测试集验收、真机测试，以及与 GraspGenX 在相同配置下的完整成功率对照。不能宣称 M2T2 已经优于原方案。
5. 神经网络没有解决 SO-101 的自由度、夹爪尺寸、深度缺失、支撑面遮挡和接触力限制。不可行时必须停住，不得为演示绕过检查。
6. 上游 FPS 排除半径约 31.6 mm 内的中心点；当前小方块居中物体云全部落在该范围，采样中心退化为重复点（邻域分组仍运行）。该行为保留用于兼容，但小物体形状表征存在风险；诊断返回 `object_fps_valid_points`，不能宣称已经解决小薄件泛化。
7. 早期 01–05 只固定 NumPy 采样种子，未固定 Torch 候选抽样；07 起补齐 Torch seed，服务返回 seed、权重、源码和算子指纹。不能把早期偶发成功当成冻结测试集结果。

## 测试与证据

```powershell
& D:\isaac\env_isaacsim60\python.exe -m unittest discover -s tests -q
```

使用 Isaac Python 运行完整测试集；M2T2 环境与其它模块混合导入时发现 OpenMP runtime 冲突，未用 `KMP_DUPLICATE_LIB_OK` 绕过。

新增测试覆盖夹持中心转换、开口约束、重复候选、目标关联、观测序号、非法位姿、放置区域/旋转约束、模型失败不松手、姿态包络、受检撤离候选和五个便携算子语义用例。2026-09-06 完整回归为 **326 tests passed**，包含矩阵严格统计、复杂关系拒绝和孔座碰撞几何测试；这是软件回归，不是 326 次物理抓放。算子小用例不等同于原生 CUDA 逐层数值对照。

运行证据：`initial_rgbd.npz`、`initial_mask.png`、`debug/trace.jsonl`、`report.json`、`place/trace.json`、`place_report.json`、`demo.mp4`。抓取成功与放置成功是两个独立结果；完整成功要求抓取 `success`，且放置 `success && released && verified`。

上游：[代码](https://github.com/NVlabs/M2T2)、[权重](https://huggingface.co/wentao-yuan/m2t2)。代码/权重可获得不等于训练集完整开放；遵守上游各自条款。
