# FineGrasp 通用抓取模型评估与决策

本文记录已实际运行的 GraspGenX 与 VGN，并说明运行时架构决策。完整原始记录分别位于：

- `output/model_bench/graspgenx/REPORT.md`
- `output/model_bench/vgn/SUMMARY.md`

这里的“primary”是工程选型，不等同于真机成功率。GraspGenX 已接入 backend factory、一键 wrapper 和 Isaac runner，完成实际 ZMQ adapter smoke，并保存一次由 GraspGenX 候选直接驱动、无 fallback 的 Isaac cube 物理仿真闭环成功 run。VGN 尚未接入运行时插件。

## 评估原则

模型只回答“从当前腕部局部几何看，哪些 6-DoF 双指抓取值得尝试”。最终可执行性仍由夹爪几何、桌面间隙、IK、碰撞、路径连续性、视觉伺服、夹爪限制和抬升验证决定。

本轮优先验证以下工程事实，而不是只引用论文指标：

- 官方代码和权重能否取得、固定版本、完全本地运行；
- 对分割点云或 RGB-D + intrinsics + mask 的接口是否适合 eye-in-hand；
- 热推理时延、首帧/加载开销、GPU/CPU 内存；
- 输出是否为合法 SE(3)，随机采样是否跨帧跳变；
- 局部/少视角、深度孔洞、薄物体和小物体的已观察失败模式；
- 官方仿真环境中的实际抓取/抬升，而非只看网络 score。

没有完成的测试会明确标记，不使用论文数字填补本地证据空白。

## 结论

部署方向为：

1. **GraspGenX 作为低频 primary proposal backend**：object-points 模式把最新腕部点云转到重力对齐的基座系推理，再把候选转回当前相机系；另支持 camera-frame depth+intrinsics+instance-mask。本地 8 GB RTX 4070 可运行，支持 gripper sweep-volume 条件，代码和权重许可证来源明确，并已有一次无 fallback Isaac cube 闭环成功。
2. **Geometric antipodal 作为当前内置 runtime fallback**：无权重、依赖小、输出可解释；用于 GraspGenX 服务不可用或空输出，也支撑当前 Isaac 闭环开发。但它不是已训练的 unseen-object 网络，复杂凹形/遮挡几何能力有限。
3. **VGN 保留为超快的可选 TSDF fallback/ensemble 候选**：速度和显存优秀，但本地 20 次官方 held-out 仿真只达到 45% 总体 lift success，40³ 体素使 0.30 m 工作区的平移量化为 7.5 mm，单/少视角 top grasp 不稳定，且权重没有独立许可证文件。当前不作为唯一 primary，也尚未集成到 BusAgent runtime。
4. **视觉伺服保持高频且模型无关**：GraspGenX 100 候选热 p50 约 0.69 s，不适合每帧重跑；VGN 即使很快也不能替代毫米级对齐。模型在入口、抓取失效或目标明显变化时刷新，最后几厘米每帧使用新腕部观测修正。

## 对比表

| 维度 | GraspGenX | VGN | Geometric antipodal |
|---|---|---|---|
| 代码/权重 | 官方代码与 checkpoint 已下载并校验 | 官方代码与 README 链接权重已运行 | 本仓库源码，无权重 |
| 代码许可证 | Apache-2.0 | BSD-3-Clause | 仓库目前未见根/包级 LICENSE，发布前必须补明确项目许可证 |
| 权重许可证 | NVIDIA Open Model License | archive 无独立 weight license/model card；打包前需复核 | 不适用 |
| 输入 | 最新 wrist object cloud 转 base/gravity frame；或 camera-frame depth+K+instance mask | target-centric TSDF | 当前相机系目标点云 |
| 输出 | 多个 6-DoF 抓取与 score | 40³ voxel grasp field | 少量 PCA antipodal 候选 |
| RTX 4070 热时延 | Robotiq 100：p50 689.22 ms，p95 707.37 ms | p50 8.38 ms，p95 8.77 ms | 未做独立稳定计时；单次仿真记录常在仿真 clock 精度下显示 0.0 ms |
| GPU 峰值 | 100 候选约 683.59 MiB allocated / 812 MiB reserved | 12.0 MB allocated / 27.3 MB reserved | 无神经网络 GPU 要求 |
| 已测抓取结果 | ZMQ adapter smoke；无 fallback Isaac cube 1/1 成功，不能外推 unseen 成功率 | 官方 held-out 仿真 9/20 lift（45%） | sphere smoke 失败；反光圆柱仅 dry-run 可行；可作为 runtime fallback |
| 主要风险 | 多模态 raw top-1 大幅跳变；独立 server 运维；广泛 unseen/真机统计仍不足 | 7.5 mm 量化、少视角敏感、空候选/高分失败、权重再分发不清 | 单视角 PCA、凹形/遮挡、接触质量估计有限 |
| 当前角色 | 选定 primary，已接入 factory/runner，低频 | 可选未来 fallback/ensemble，未接入 | 已集成 runtime fallback/基线 |

## GraspGenX 实测

### 版本与完整性

- 官方代码：`NVlabs/GraspGenX`，提交 `b9429097728cb1c430dd78b92edf17ba318aad03`。
- 官方模型：`adithyamurali/GraspGenXModel`，revision `7c834043c11a11417e31d6d5ea9355801e40a2c1`。
- generator：1,210,918,342 bytes，SHA-256 `8B55F31CDB8340A573B4DF27B027C15CFF326BD6DEBCB389BF631D2AAAB7AC44`。
- discriminator：483,889,478 bytes，SHA-256 `CBF3F3BDB2E4C03FCA8486ED24DE0E6A8A859E6BD22BCE2F1434A610335ABD3E`。
- 两个文件大小和 hash 均与官方 Git LFS pointer 一致。
- 参数量 141,156,167。

代码根许可证和 SPDX header 是 Apache-2.0。README 明确把 checkpoint 置于 NVIDIA Open Model License；模型仓库 front matter 也链接同一许可证。`pyproject.toml` 只写了较模糊的 `NVIDIA Corporation`，发布第三方清单时应保留这个元数据差异。

本次 Robotiq 2F-85 gripper description 来自官方 companion repository。model card 标注 MIT，资产清单把该 Robotiq 资源归因为 ros-industrial BSD-2-Clause；不要把整个 companion package 简化成单一许可证标签。

### 环境和可部署性

实测环境是 Windows、Python 3.11.15、PyTorch 2.6.0+cu124、RTX 4070 Laptop 8,187.5 MiB。模型加载用时 2.22--4.46 s；加载后 CUDA 约 545.32 MiB allocated、566 MiB reserved。

官方 `uv pip install -e ".[serve]"` 在该 Windows 主机上会因未使用的 `SharedArray==3.2.4` 需要 MSVC 14.0 而失败。审计确认 inference 路径不导入 `SharedArray`/`urdfpy` 后，安装其余依赖并 `uv pip install -e . --no-deps` 才得到可运行环境。该 workaround 是本机已验证事实，不应理解为上游官方支持的安装路径。

PointNet++ 可选 CUDA extension 因本机无 `nvcc` 未构建；发布的 `ptv3vanilla` generator/discriminator 使用纯 PyTorch 并已成功执行。官方安装 smoke 2/2 通过，随机权重路径产生 100 个合法 4x4 transform。

### 时延与显存

在官方 bundled real-world segmented object 上，3 次 warmup 后执行 10 次热推理：

| Gripper | 候选数 | 首次推理 | hot p50 | hot p95 | hot mean | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| procedural parallel 2F | 100 | 1,823.23 ms | 674.15 ms | 690.88 ms | 673.99 ms | 683.59 MiB | 812 MiB |
| procedural parallel 2F | 200 | 1,424.58 ms | 813.36 ms | 838.64 ms | 817.49 ms | 683.60 MiB | 806 MiB |
| official Robotiq 2F-85 | 100 | 1,722.61 ms | 689.22 ms | 707.37 ms | 683.14 ms | 683.59 MiB | 812 MiB |

每次都返回请求数量，矩阵有限且通过齐次底行、旋转正交和 det=+1 检查。200 降到 100 候选使热时延下降约 17%，是当前首选速度旋钮。

官方支持的可选 TensorRT FP16 diffusion-denoiser 路径在本机未安装，`tensorrt_available()` 为 false。FP32 已低于 1 GiB reserved，当前没有因显存需要而启用 TensorRT 的证据。

### Eye-in-hand wire API

官方 ZMQ `infer_scene_depth` 已用 720×1280 float-meters depth、3×3 intrinsics 和两实例 integer mask 实际调用。Robotiq 2F-85 sweep geometry 的新 server 加载 3,407.54 ms；完整冷 RPC 2,270.96 ms，两个实例各返回 20 个有限相机系抓取，请求期 peak allocated 621.79 MiB。

BusAgent 侧 `ZmqGraspGenXTransport` 不导入 torch，只依赖 NumPy、msgpack、msgpack-numpy、pyzmq。支持 `object_points` 和 `scene_depth` 两种请求模式；服务超时、不可用、协议错误和推理失败会映射为稳定 `FailureCode`。

本仓库 adapter 还用当前 SO-101 sweep-volume 做了实际 client/server smoke：object-points 和 scene-depth 各返回 64 个有限 camera-frame 候选；object-points 冷 round trip 2,567.8 ms，随后两次 977.4/768.6 ms，scene-depth 热 round trip 666.5 ms。证据在 `output/model_bench/graspgenx/adapter_zmq_smoke.json`。

当前集成配置使用 `planner: graspmoe`：合并 learned diffusion 与 geometry-seeded OBB proposals，再由发布的 discriminator 评分。MoE 分支使用输入坐标的竖直/重力轴生成候选族。eye-in-hand camera 会随腕部旋转，不能把 camera `+Z` 当作固定重力轴。当前 `object_points` 默认 `inference_frame: base`，用同一观测的 `T_base_camera` 把点从 camera 转到 base；服务返回的 base-frame pose 再转回 `T_camera_grasp`，保持 BusAgent 稳定契约。为控制上游 N×N outlier filter 的内存，46,466 个分割点会确定性降采样到配置上限 4,096。`scene_depth` 因依赖图像像素和内参，配置校验要求 camera inference frame。

Isaac demo 和 batch runner 已接受 `--backend graspgenx` 并通过 factory 构造后端。端口未占用时，`scripts/run_fine_grasp_demo.bat`/`.ps1` 会按本机路径启动官方 server、等待 TCP、同步等待 warmup JSON `status=ready`，再运行 Isaac，并在结束时回收它创建的进程。ready 已返回但子进程带非零 Windows shutdown code 时只发 warning，因为结构化响应才是就绪契约。已有端口会被直接复用并跳过 wrapper warmup。默认使用按秒时间戳输出目录；常规串行运行不会覆盖，若同一 backend 同秒并发则应显式传不同的 `-Output`。

### Isaac primary 闭环实证

一键 wrapper 保存的 `output/fine_grasp_runs/20260905_072102_graspgenx_cube/report.json` 记录了默认 cube 的完整闭环：

| 指标 | 结果 |
|---|---:|
| 实际 selected backend | `graspgenx` |
| Fallback | 无（`fallback_reason` 不存在） |
| Selected GraspMoE branch | `obb`；属于 GraspGenX MoE 候选集，不是 geometric fallback |
| 模型时延 | 687 ms |
| 候选 | 128；67 个 table-clearance、11 个 IK 不可达被过滤 |
| 腕部观测 / servo steps | 22 / 19 |
| 最终平移 / 姿态误差 | 3.39 mm / 0.15° |
| 腕部视觉验证 lift | 82.05 mm |
| Isaac 刚体 ground-truth lift | 82.06 mm |
| FineGrasp 节点周期（不含 server/Isaac 启动） | 17.875 s |

该 run 的 `selected_grasp.metadata.model_frame=base`，源点 46,466、模型输入点 4,096；节点和刚体抬升双条件均成功。这证明 primary 的坐标转换、GraspMoE 候选/评分、可行性过滤、pre-grasp、最新腕部帧伺服、夹爪、抬升和验证能在该 case 串通。历史 report 中 `model_pose_convention` 字符串仍误写为 `T_camera_gripper_base`，但 `model_frame=base` 与实际输入/输出变换为 base；当前源码已把该标签更正为 `T_model_gripper_base`。它只是一种 cube/seed 的 1 次结果，不能作为水果、钥匙、薄件、反光件或真机的总体成功率。

### 跨帧稳定性

在同一个 `00/obj_1` 点云上，用真实 Robotiq geometry 独立采样 12 次、每次 100 候选：

- top score 范围 0.8010--0.9103；
- raw top-1 位置两两距离 p50 249.2 mm、p95 387.8 mm；
- 考虑双指 180° 对称后的旋转距离 p50 88.0°、p95 175.5°。

该物体本身约 298×50×129 mm，多个面都可能存在有效抓取，不能仅凭 top-1 距离断言候选质量差；但这足以证明不能每次采样都无条件替换 active grasp。

当前 backend 因而在基座系执行候选聚类、cluster consensus bonus、相对上次同 object ID 的 continuity bonus 和 switch hysteresis，并按双指对称旋转距离关联候选。随后仍需 collision、IK 和 gripper filters。

### 实证边界

- 已保存的 primary Isaac 成功目前只有默认 cube 单次结果；bundled sample 与单 case 不能覆盖水果、钥匙、薄片、反光工业件的最终效果。
- 尚未执行真机抓取，不能把 Isaac 的力、摩擦、深度和接触验证结果直接外推到真机。
- 未完成 TensorRT、长期 server 恢复、并发/背压和 network partition 压测。
- 上游 Windows 交互 demo 在 inference 完成后因 `select.select([sys.stdin])` 触发 WinError 10038；这是 GUI/键盘控制缺陷，不是模型推理失败。

## VGN 实测

### 版本与许可证

- 官方代码：`ethz-asl/vgn`，提交 `d7af0622433f52ae88ebe81533f12b46b33e951a`。
- 代码许可证：BSD-3-Clause。
- 权重：1,255,822 bytes，313,238 参数，SHA-256 `ba3391d0805e9c9b178cd18106866313cee808ff2b654f689663e92a814cec4b`。
- README 指向官方 Google Drive archive；archive 内没有独立权重许可证或 model card。内部使用前仍应保留来源，若要随产品重新分发则必须先复核授权。

### 时延和物理仿真

- RTX 4070：首次 end-to-end 141.0 ms；100 次热推理 p50 8.38 ms、p95 8.77 ms；peak allocated 12.0 MB、reserved 27.3 MB。
- CPU 8 threads：首次 103.4 ms；40 次热推理 p50 55.4 ms、p95 57.0 ms。
- 官方 held-out `pile/test`、单物体、六视角、seed 1--20：14/20 产生候选，9/20 成功抬升；总体 45%，有候选条件下 64.3%。这是小型本地工程样本，不是论文级估计。
- 另一个 seed-42 spray-can trial 产生 3 个有限候选，并以 score 0.9753 成功抓取/抬升。

若干 donut、薄包装棒、hammer、pliers trial 没有候选；soccer-ball、boot、can、hammer 存在高分但 miss/slip。网络 score 不能当作执行成功概率。

### Eye-in-hand 局限

官方输出是 40³。0.30 m workspace 下一个 voxel 为 7.5 mm，因此 VGN 不能独立完成当前 4 mm 终止容差。seed-42 spray-can 的 1/2/3 views top grasp 相对六视角最佳位置相差 97.8/71.5/88.1 mm，四视角才下降到一个 voxel。随机 depth holes 往往保留候选，但会把 global argmax 切到另一抓取族。

若后续接入，必须使用 target-masked rolling TSDF、multi-view coverage gate、候选族关联/滞回，并让独立视觉伺服完成最后毫米；不能把单帧 VGN argmax 直接发给机器人。

## Geometric fallback 的定位

`GeometricAntipodalBackend` 对当前相机系目标点云估计 PCA frame，排除与 view/approach ray 近似平行的 closing axis，再生成至多四个双指候选。对支持面上的单视角遮挡，它可用已知桌面高度补全物体垂直中心。它输出 `support_surface_completion` 和 `fallback` metadata，方便 selector 与 debug 识别。

优点是确定、无权重、无外部服务、可完整调试。限制是接触质量来自粗几何，不理解复杂凹形、可变形材质和遮挡后的完整形状；因此只作为可用性 fallback 和闭环基础线，不承担 GraspGenX 的泛化结论。

## 可复现实验

复现本项目 GraspGenX Isaac 闭环并自动保存唯一 run：

```powershell
scripts\run_fine_grasp_demo.bat
```

wrapper 会在控制台打印 run 目录；已验证证据保存在 `output/fine_grasp_runs/20260905_072102_graspgenx_cube/`。整个 `output/` 由 `.gitignore` 排除，该目录不进入分支，但作为本机实测产物保留；迁移工作站时应连同 report 和图像另行归档。

GraspGenX 环境和权重不进入 Git。完整复现命令、依赖 freeze、hash 与 JSON 指标在 `output/model_bench/graspgenx/REPORT.md`；关键命令如下：

```powershell
$env:GRASPGENX_CHECKPOINT_DIR='D:\liuGong\_models\graspgenx\checkpoints'
$env:GRASPGENX_GRIPPER_CFG_DIR='D:\liuGong\_vendor\GraspGenX\ext\gripper_descriptions'
$env:PYTHONDONTWRITEBYTECODE='1'

_envs/graspgenx/python.exe output/model_bench/graspgenx/benchmark_realworld.py `
  --checkpoint-root _models/graspgenx/checkpoints/release `
  --sample-data-dir _vendor/GraspGenX/assets/sample_data/real_world `
  --assets-dir _vendor/GraspGenX/assets `
  --gripper robotiq_2f_85 --num-grasps 100 --warmup 3 --runs 10 `
  --output output/model_bench/graspgenx/result_robotiq_100_rerun.json
```

VGN 的完整 driver、环境清单和逐 trial 结果保存在 `output/model_bench/vgn/`。解释结果时保留 case 数、视角数、seed、候选率和 lift 条件，不从 20 次小样本外推总体 unseen 成功率。

## 后续验收门

在把单次 Isaac 成功扩展为长期生产能力前，继续补齐：

1. cube、sphere、cylinder、thin、reflective 多 seed 的节点+刚体抬升双条件统计；
2. 持续记录实际 selected backend、configured backend、fallback reason、候选过滤、模型时延、完整周期、显存和重规划次数；
3. 真机完成手眼、camera-joint 同步、pinch-center、深度、力/夹爪和碰撞模型标定后，低速有人值守测试；
4. 单列透明/强反光深度缺失、桌面薄片、指尖不可达、滑落和错误验证；
5. 对 server 长时间运行、并发/背压、断连恢复和可选 TensorRT 路径做压力回归。

当前准确表述是：“GraspGenX 已在目标 GPU 上验证可部署，已作为 primary 接入稳定 backend/runner，并由其候选完成一次保存完备的 Isaac cube 闭环抓取；尚无广泛 unseen 统计或真机成功率。Geometric 是可归因的 runtime fallback，VGN 是已独立实测但未接入的备选。”
