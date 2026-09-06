# LiuGong · Arena / Panda 抓放适配

本分支基于 `codex/official-franka-progress` 的 `0576d673fa396d1373d9bcd298d45a93bc582904`，增加独立的 Arena 执行入口。服务器部署目录为 `/root/gpufree-data/arena-panda`；原 `/root/gpufree-data/LiuGongArm` 保留。

## 已实现的责任边界

```mermaid
flowchart TD
  U[用户文字 / 语音] --> B[BusAgent：目标、目的地、任务模式]
  B --> R{auto / basic / enhanced}
  R -->|普通任务| S[NVIDIA PickPlaceController 快速阶段控制]
  R -->|陌生 / 杂乱 / 精确任务| G[GraspGenX：目标点云 → 多个 6DoF 抓取]
  G --> A[Arena Panda IK：抓取、抬升、重新观测]
  A --> P[AnyPlace：物体与支撑区域点云 → 多个放置变换]
  P --> E[Arena Panda IK：搬运、放置、松爪、退离]
  S --> E
  S -->|未抓住：重新观察| G
  S -->|已抓住：仅升级放置| P
  E --> V[PandaTask：物理状态评测]
  V --> B
```

- 场景、官方 Franka Panda、相机和动作管理来自 IsaacLab-Arena。机械臂由官方 `DifferentialInverseKinematicsActionCfg` 求解，快速路径复用原仓库使用的 NVIDIA `PickPlaceController` 底层阶段控制器，将其笛卡尔目标交给 Arena IK；模型路径组织模型生成的完整姿态。没有另外构造 RMPFlow 机械臂或复制官方插值算法。
- `auto` 在 `unfamiliar / cluttered / precise` 为真时启用增强路径。`auto` 普通任务与 `basic` 均先执行官方快速抓放，成功时不调用 GraspGenX / AnyPlace（视觉模型仍负责识别）；实际抓放失败时自动升级一次。未抓住则张开夹爪、退离、重新观测并调用 GraspGenX；仍成功持物则保留抓取，直接调用 AnyPlace 放置。`enhanced` 直接进入模型路径。停止请求不会触发升级，模型失败不会无限重试。
- BusAgent 下发一次完整 `pick_place` 事务，含目标和目的地标签，不生成 XYZ、机械臂关节角或抓放姿态。
- GraspGenX 与 AnyPlace 分别运行在 Python 3.11 模型环境。Isaac/Arena 使用独立 Python 3.12 环境，通过 ZMQ / HTTP 通信。
- 相机、物理状态和 IK 只在仿真主线程访问。模型推理在工作线程等待时，仿真持续步进。命令去重账本禁止同一 `command_id` 重复执行；重启后未完成任务标记结果未知，不重放。

## 已删掉的重复要求

已删除 BusAgent 每任务 64 条事件的路由上限，避免长任务进度消息耗尽计数后丢失最终成功/失败事件。Panda 路径不再使用旧机械臂的抓取准备确认、恢复确认提示、假定相机已就绪的状态机，以及候选姿态的俯视角度、工作空间、重复点数和位移门槛。BusAgent 只校验语义任务字段，Arena 承担实际执行与结果评测。语言模型暂时不可用时，已解析完整的文字任务继续执行。保留命令去重、停止和必要的消息格式检查，避免网络重试重复驱动同一机械臂。

执行路径已删除配置名称、类别和别名匹配：不再有 `resolve_entity` 或配置抓取／放置目录。`entities` 仅创建物理场景，字段为内部编号、几何、质量、材质和动态属性，默认场景不保存语义类别或别名。目标文字直接交给图像模型，实际掩码与同帧深度生成抓放点云；此前观察到的实例可用实例引用再次定位。

仿真评测在模型完成分割后，将模型掩码与同帧渲染实例 ID 对照，自动取得真实刚体的接触／抬升见证，并核对多个视角是否指向同一实例。这个对应关系不提供类别、不生成模型掩码、不裁剪规划点云，也不将真值姿态送进姿态模型。实例 ID 图保存在独立文件中，不进入视觉模型输入或语言模型上下文。刚体与接触传感器由实际 Arena 场景注册，包含桌面；没有“仅指定物体能抓、仅指定区域能放”的名单。该桥接属于仿真观测关联和评测基础设施，实际场景几何与碰撞体仍须由 Arena 加载，不是实机的物体身份方案。

多视角错误召回曾把绿色物体与蓝色垫子的点云合并。现按每个视角分别关联实例，大面积错误区域不能凭像素更多覆盖另一个视角；冲突触发一次 SAM3 新观测，重新生成模型掩码和点云，仍不一致则报告观测不确定。已持物体的几何观测也保持同一实例。这个回退属于视觉环，不自动要求使用慢速运动模型。

是否能稳定放置由执行后的实际接触与稳定性判断，不要求方块底面完全包含在支撑面的轮廓内，也不使用配置中的初始位置评测已移动的支撑对象。

## 跨命令持物放置

`grasp` 完成后保留物体实例、识别标签、抓取命令编号、初始抬升参考与相对夹爪的实测位置。`place_held` 只需新的目的地，不重复抓取。普通放置继续使用 NVIDIA `PickPlaceController` 的搬运、释放、退离阶段；前面的抓取阶段只在脱离实际机械臂的适配器上初始化，不步进仿真、不再次闭爪。复杂放置直接用 AnyPlace，姿态仍交给 Arena IK。

目的地识别失败保留夹持和上下文；停止同样保留持物。再次放置先核对夹爪开口与物体相对位置，避免把已掉落／已释放的物体当成仍在手中。任务失败会保留实际结果，放置入口不会偷偷重新抓取。状态接口提供 `holding` 及最初的 `grasp_command_id`。这是运行进程内的跨命令状态；重启重建场景后不恢复旧夹持。

```json
{"command_id":"pickup-1","skill":"grasp","params":{"target":{"category":"red cube"}}}
```

等前一命令确认完成后，可在另一轮调用：

```json
{"command_id":"place-2","skill":"place_held","params":{"destination":{"label":"blue pad"},"mode":"auto"}}
```

这些接口通过同一 Arena 任务队列执行，支持命令去重、停止和物理评测。按用户调整的顺序，BusAgent 的多步规划及新入口的语言接线留到后续阶段，本轮没有改动 BusAgent。

## 坐标约定

所有几何采用米，传递完整 SE(3) 齐次变换。IsaacLab 3 的四元数顺序为 **XYZW**。

GraspGenX 的 Panda 描述为 80 mm 开口、103.4 mm 指尖深度。模型夹持轴 X 与 Panda 手部夹持轴 Y 相差绕 Z 的 +90°；执行 TCP 为 `panda_hand` 局部 Z 上 103.4 mm。利用平行夹爪绕接近轴 180° 的对称性选择等效腕部朝向。

AnyPlace 输出的是作用于输入物体点云的变换，并非末端绝对姿态：

```text
T_world_object_final = relative @ T_world_object_input
T_world_tcp_goal = T_world_object_final @ inverse(T_tcp_object)
```

输入物体参考系与持物变换在抬升后的 RGB-D 观测上建立，持物阶段优先使用腕部近距图像，腕部识别失败才回到固定视角，避免将侧面误分割的夹爪混进物体点云。腕部相机显式启用 `update_latest_camera_pose=True`，使每帧深度使用更新后的相机外参。候选保留完整旋转，按腕部转动幅度、平移距离和模型抓取分数排序；没有手写工作空间、俯视角度、放置高度或区域边界的硬拦截。区域和支撑关系由任务执行后的物理评测判断。

## 速度与相机布置

保留 120 Hz 物理与 60 Hz IK 动作更新，将三路 RTX 渲染从 60 Hz 降为 **15 Hz（仿真时间）**，默认使用 Isaac Lab `performance` 渲染预设。官方阶段控制器使用约 400 个动作步，加最终稳定性观察；相机低频更新前等待新帧，腕部外参随相机实时更新。执行耗时从 Arena 开始执行算到物理评测完成，包含点云观测，**不含启动仿真或 BusAgent 的语言理解等待**。

相机参数集中在 `configs/arena_panda.json`。这些位置是根据当前桌面、机械臂遮挡与实际点云观测选择的工程配置，官方没有要求这一组固定坐标。

| 相机 | 位置（米） | 朝向/作用 |
| --- | --- | --- |
| 工作区 | 世界 `[1.00, -0.60, 0.80]` | 看向 `[0.48, 0, 0.055]`，覆盖抓取与放置区域 |
| 对侧 | 世界 `[0.90, 0.62, 0.66]` | 看向同一点，从另一侧补足夹爪、手臂遮挡 |
| 腕部 | `panda_hand` 局部 `[0.055, 0, 0.02]` | 看向局部夹持中心 `[0, 0, 0.1034]`，偏离夹爪中线获得近距持物几何 |

依照 [Isaac Lab 相机接口](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/api/lab/isaaclab.sensors.html) 使用 ROS 光学坐标（前方 +Z、上方 −Y）、相机内参和 Z 深度反投影；分辨率为 640×480。抓取前融合两台固定相机，持物后优先使用腕部视角。保留真实观测点数量、相机外参和每视角点云包围范围，便于发现遮挡或错位；AnyPlace 的固定输入长度适配在模型边界完成，诊断仍报告原始观测数量。新视角下红方块初始观测融合约 1857 点，黄色圆柱约 1304 点，两个固定视角的世界坐标范围一致。

## 运行

已部署机器的依赖位于 `_envs`、`_vendor` 和 `_models`；精确源码版本见 [版本清单](arena/versions.json)，`docs/arena/*-environment.txt` 是已部署环境快照（含继承包与本地 editable 路径），用于排查版本差异，并非跨机器直接安装的锁文件。BusAgent 的 `.env` 仅在服务器保存，使用独立 `busagent_arena` 数据库，连接本机 MariaDB 3307。

```bash
cd /root/gpufree-data/arena-panda
python3 ops/arena_stack.py start
python3 ops/arena_stack.py status
# 重启会重新生成仿真场景。
python3 ops/arena_stack.py stop arena
python3 ops/arena_stack.py start arena
# 不含类别／别名的新物体测试场景：
python3 ops/arena_stack.py stop arena
ARENA_PANDA_CONFIG="$PWD/configs/arena_panda_generalization.json" python3 ops/arena_stack.py start arena
# 需要场景空闲且夹爪为空；会实际执行仿真动作：
python3 scripts/check_arena_holding.py --target 'green object' --destination 'yellow cylinder'
```

管理脚本仅管理其启动的进程组。日志与 PID 记录在 `output/services/`。数据库和 Nginx 由系统独立管理。

直接运行单项回归：

```bash
bash scripts/run_arena_panda.sh --smoke --mode basic --viz kit
bash scripts/run_arena_panda.sh --smoke --mode enhanced --viz kit
# 以下仅用于故障回归：分别制造一次抓空、成功抬升后触发升级。
bash scripts/run_arena_panda.sh --smoke --mode auto --smoke-fault miss_grasp --viz kit
bash scripts/run_arena_panda.sh --smoke --mode auto --smoke-fault after_lift --viz kit
```

服务器的 Isaac Sim 为 **6.0.1-rc.7**。已处理版本兼容问题：Lab 接管 PhysX 前关闭 Sim 的重复初始化回调；用生产资产库的 `FrankaRobotics/FrankaPanda/franka.usd` 替换已经失效的 staging Panda 路径。接触传感器使用 `patches/isaaclab-sim6-contact-import.patch` 修正 Sim 6 的模块路径（在 IsaacLab 源码目录运行 `git apply --unidiff-zero` 应用）。下降放置使用 Lab 的接触传感器判断物体接触指定支撑面后松爪，避免继续压向模型目标。该机器的无界面模式无法返回深度，当前使用已有 X11 `:20` 的 `--viz kit`。这是实际验证过的运行模式。

## 网页与接口

| 端口 | 服务 |
| --- | --- |
| 8991 | BusAgent Panda 控制台，`/v1/` 代理后端，`/preview/` 打开观察台 |
| 8993 | 三路实时 RGB-D 相机观察台，仅代理只读状态和图像 |
| 8999 | 预留，未占用 |
| 127.0.0.1:7861 | Arena 任务接口 |
| 127.0.0.1:5556 | GraspGenX ZMQ |
| 127.0.0.1:5590 | AnyPlace HTTP |
| 127.0.0.1:5570 | 本地视觉 HTTP，图像通过本地观测文件传递 |
| 127.0.0.1:3100 | BusAgent 后端 |

Nginx 配置在 `ops/arena-nginx.conf`。云平台以 HTTPS 域名代理这些端口，直接 IP:8991 不可达。已验证的公网入口：

- [BusAgent 控制台](https://8byr7v21-8qne20ga-8991.zj02restapi.gpufree.cn:8443/)
- [Isaac 三路相机观察台](https://8byr7v21-8qne20ga-8993.zj02restapi.gpufree.cn:8443/)

当前电脑也保留 SSH 隧道 `http://127.0.0.1:18991/` 和 `http://127.0.0.1:18993/`。

```bash
curl http://127.0.0.1:7861/api/status
curl -X POST http://127.0.0.1:7861/api/command \
  -H 'Content-Type: application/json' \
  -d '{"command_id":"demo-001","skill":"pick_place","params":{"target":{"category":"red block"},"destination":{"label":"blue pad"},"mode":"enhanced"}}'
curl http://127.0.0.1:7861/api/commands/demo-001
```

HTTP 202 仅表示接受任务。最终 `completed` 必须来自物理评测；失败或停止时返回 `failed / cancelled`。评测检查实际抬升、夹爪张开、末端退离、与指定支撑对象的向上接触力、物体位于支撑对象上方、释放后位移及两个物体的线速度。支撑对象的位置来自当前仿真状态；不依赖配置尺寸或底面包围盒间距。每次任务保存 JSON 及三路图像到 `output/arena/`。

每个可抓物体使用一个 Isaac Lab 接触传感器，按其他场景实体过滤接触力，以区分“放在指定圆柱上”和“掉到旁边的垫子上”。接口依据为 [Isaac Lab 接触传感器](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.sensors.html)，采用单物体对多个指定物体的过滤矩阵。

## 当前验证范围

删除配置标签执行目录后的验证见 [观测执行与持物续放记录](arena/observed-execution-validation.json)：新增绿色长方体未登记任何类别或别名，`green object` 也不在场景盘点词汇中；从图像定位后成功抓取，并以独立命令放到黄色圆柱上。快速抓取约 9 秒，续放约 13.2 秒。红方块快速续放约 12 秒；完整抓放到此前不在放置目录中的桌面约 21.3 秒。绿色物体的独立 AnyPlace 放置约 23.7 秒，保留原抓取，物理评测通过。空夹爪、目的地不存在、空闲停止后继续，以及搬运中途停止后续放均记录真实结果。最后一项在运输阶段取消，再用同一次抓取约 12.5 秒完成放置；补测失败和多视角错误召回的修复证据一并保留。这些是有限场景验证，不能外推为任意工业物体成功率。

2026-09-06 支撑目标回归通过 BusAgent 的文字消息入口连续执行，没有重置中间场景：先把黄色圆柱放到蓝色区域，再发送用户原句“把红色方块放到黄色柱子上”。第二步绑定 `yellow cylinder`，通过图像点云观测移动后的圆柱，官方快速路径 19.73 秒完成，未调用 GraspGenX / AnyPlace。红方块与圆柱中心的水平距离为 3.17 mm，支撑接触力 Z 分量为 0.581 N，松爪、退离和稳定性评测通过；从发送文字到收到完成回复为 22.36 秒。红方块宽 45 mm、圆柱直径 40 mm，覆盖了原有“底面完整包含”规则必然拒绝的情况。完整参数与事件摘要见 [堆放验证记录](arena/stacking-validation.json)。这是一次实际场景回归，不是随机场景成功率或精度保证。

这是分层架构的第一版仿真集成，当前场景是红色方块、黄色圆柱与蓝色支撑垫。本轮快速路径与相机优化的物理回归如下（均通过抬升、支撑、释放、退离、稳定性评测）：

下表是接入图像模型之前使用仿真语义掩码的历史基线。当前图像链路结果单独记录在 `docs/arena/vision-validation.json`。

| 回归 | Arena 执行耗时 | 结果 |
| --- | --- | --- |
| 红方块快速抓放 | 18.45 秒（旧路径 48.45 秒） | 无模型调用，耗时减少 62%，落点距中心 4.9 mm |
| 公网网页红方块快速抓放 | 17.28 秒（旧网页路径 48.45 秒） | 耗时减少 64%，页面显示真实路线，聊天收到完成结果 |
| 黄色圆柱快速抓放 | 17.69 秒 | 无模型调用，落点距中心 4.0 mm |
| 注入一次抓空，自动升级 | 47.43 秒 | 快速尝试 8.22 秒后真实未抬升，重新观察，GraspGenX + AnyPlace 接手成功 |
| 成功抬起圆柱后注入故障 | 32.80 秒 | 保留夹持，仅 AnyPlace 接手，无第二次抓取，落点距中心 2.3 mm |

两项恢复测试使用明确的故障注入，**不计作自然失败率或工业场景成功率**。模型、GPU和场景均已加载，未计入语言理解、排队与启动耗时。前端显示本次执行路线、是否升级及真实耗时。公网红方块回归中，从用户消息到聊天完成回复约 20 秒（页面时间精确到秒）。

以下为此前集成阶段的网页记录：普通抓放已通过公网网页下发的完整事务，红方块距区域中心约 1.2 mm；GraspGenX + AnyPlace 红方块增强流程也已完成真实抓起、接触放置、释放、退离和稳定性验证。黄色圆柱增强抓放也通过公网网页端到端验证：抬升 16.0 cm，落点距区域中心 3.9 mm，聊天收到真实完成结果。BusAgent 195 项测试、Arena 16 项测试和前端构建通过。开发回归记录见 `docs/arena/validation.json`，其中保留了修复前的失败记录。

以下能力尚不能宣称已完成：

- **VLA 策略尚未加载。** 普通路径使用 NVIDIA 官方阶段控制器 + Arena IK，接口明确返回 `vla_loaded=false`。
- **当前分割已使用 YOLOE / SAM2 Tiny / SAM3，Florence 提供通用视觉任务。** 详细链路、图像与总线边界见 [视觉链路](ARENA_VISION_PIPELINE.md)。执行不再要求配置语义目录；跨同类多实例的选择、实机身份关联和任意工业场景鲁棒性仍需专项验证。
- **当前评测支持支撑面放置和物体顶面堆放，悬挂、插入尚未接通。** 默认差分 IK 是局部控制器，模型候选可能到达关节极限；复杂场景仍需接入成熟运动规划器和相应 Arena 任务。这里没有自研关节控制器或宣称已经实现完整机械臂避障。
- **AnyPlace 没有学习成功概率或碰撞保证。** 只将其作为姿态提议器，以执行后物理评测确认结果。
- **尚无陌生工业物体数据集的成功率。** 示例测试不能外推为工业鲁棒性；需要继续接入工业 USD 资产、杂乱场景、随机姿态、多目标整理和遮挡回归。
- 快速路径失败时会自动升级；增强路径失败或用户停止时可能保持持物并报告结果。重启会重建场景并保留旧命令的未知结果记录。

## 参考实现

- [NVIDIA Franka Pick and Place 示例](https://docs.isaacsim.omniverse.nvidia.com/latest/examples/manipulation_franka_pick_place.html)
- [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena)
- [GraspGenX](https://github.com/NVlabs/GraspGenX)
- [AnyPlace](https://github.com/ac-rad/anyplace)
- [AnyPlace 官方权重](https://huggingface.co/datasets/yuchiallanzhao/anyplace)

AnyPlace multitask 权重 SHA-256：`d3d33f0a279633c25f252960a208d4b4447a756f0cff8e94be0faadc20dc5be5`。模型源码、权重和 Python 环境不提交到仓库。

## 工业工作台场景

`configs/arena_panda_industrial.json` 加入七个可移动工业零件、两个敞口零件托盘及带定位销／挂钩的工装。资产是原创 USD 组合刚体，带孔零件采用多个凸碰撞体保留孔洞，详见 [资产说明](../assets/scenes/industrial/README.md)。配置中的实体仍只有物理属性，不登记类别或别名。场景盘点词汇是视觉查询提示，不限制抓放指令。

```bash
python3 ops/arena_stack.py stop arena
python3 ops/arena_stack.py start arena --config configs/arena_panda_industrial.json
```

场景选择记录在 `output/services/arena-scene.json`，后续常规重启沿用该场景。切回基础场景时用 `--config configs/arena_panda.json`。工业场景两路固定相机从对角上方观察工作区，分辨率 800×600，腕部相机保留抓持后的局部观测。挂钩和定位销目前是几何测试对象，尚不表示已支持悬挂／插入动作。

工业零件测试发现持物遮挡使螺母仅有 406 个可见点。旧 AnyPlace 适配器把 1024 个网络输入样本误当成 1024 个独立观测点；现对非空稀疏观测重复采样，保留原始点数和采样信息。该适配不补全不可见表面，也不保证放置成功。原始 406 点回放已通过官方模型推理。

在一次夹持中，成功获得的持物点云保存在夹爪坐标系，后续放置使用当前 TCP 位姿更新它，避免遮挡后每次重新定位。释放、新抓取或夹持失效不再沿用旧几何。事件区分 `held_geometry_recorded / held_geometry_reused`，并保留原始观测编号；目标区域仍使用新视觉观测。

工业场景首轮结果：绿色支架至蓝色托盘约 21.8 秒，金色螺栓至橙色托盘约 22.6 秒，两次均走快速控制器并通过物理评测。五个带描述的查询可定位；首次泛类别盘点漏报多数零件，Florence 描述较泛。螺母增强测试出现过点数拦截、夹持遮挡和一次物理抬升失败，不能宣称稳定成功。原始结果及修复回放见 [绿色目标歧义与工业场景验证](arena/green-ambiguity-industrial-validation.json)。

支架补测已验证中断后复用夹爪坐标系中的旧观测几何，且 AnyPlace 返回候选；选中的增强放置姿态随后因 Arena IK 未到达而失败（位置误差 66.7 mm，姿态误差 9.19°）。模型返回候选不代表动作可执行，后续需改进候选可达性处理。本轮保留该失败，未放宽物理评测。

增强搬运失败后的夹持复核显示相对位置已改变，未再沿用该夹持执行放置。验证后重置工业场景，供后续从原始零件位置继续测试。
