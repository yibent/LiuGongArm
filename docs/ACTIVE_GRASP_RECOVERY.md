# 实验开发线：双 RGB-D 主动抓取恢复（左顾右盼）

分支：`codex/active-grasp-recovery`；基于已发布的 `2c6bfbe`。
本页描述新的实验能力；原始基线和接入契约见 [BusAgent README](BUSAGENT_README.md)。
没有宣称工业工具、真实杯子/水果或真机泛化已通过验收。
本文“头顶”泛指固定外部 RGB-D，可选垂直俯视 `overhead` 或斜上方 `oblique`。

本轮结论：受控空抓后的完整恢复已在 Isaac 跑通；同场景基础 0/1、增强 1/1。
斜上方相机的无扰动回归中水果/杯子代理成功，锤子失败并保守停止。
这是可以继续开发的实验节点，不是成熟通用抓取发布版。证据和限制见下文。

## 已实现的机制

`RecoveringGraspNode.execute(FineGraspRequest) -> FineGraspResult` 包装现有
`GeneralGraspNode`，仍支持 `cancel()`。通过 `attempt_factory(index, failed_grasps, target)`
注入每次全新的节点；不绑定神经网络后端。

1. 首次使用腕部 RGB-D 闭环接近；固定头顶 RGB-D 建立辅助目标观测。
2. 发生可恢复的感知、伺服、空抓或验证失败时停止运动，检查夹爪是否可能持物。
3. 只有两次新鲜头顶观测确认目标位置稳定、且不存在持物疑虑时，规划安全退让。
4. 清除旧规划；按实际关节轨迹的 FK 插值检查全局碰撞和局部手指扫掠体后，低速退让。
5. 用新的头顶 RGB-D 位置重新初始化目标；下一次尝试启用腕部主动多视角。
6. 腕部到两个安全侧向视点采集 RGB-D，按各帧实际 `T_base_ee @ T_ee_camera`
   融合目标局部表面；低频模型重新提出候选，高频观测/小步伺服继续对齐。
7. 每个候选分别做局部接近碰撞筛选；几何发生重置时停止并重新推理。
8. 闭合后使用腕部视觉和夹爪状态进行小幅试抬升；头顶观测若有可靠目标对应，
   独立确认目标跟随或否决“目标仍在桌上”的视觉假成功。

上游粗位置只用于初始化。头顶相机使用固定的 `T_base_world @ T_world_camera`，
不会伪装成腕部变换链。腕部仍负责抓取姿态与最后接近；头顶不会直接下发抓取位姿。
当前融合是局部稀疏表面/自由空间检查，不是完整物体重建，不保证获得物体所有背面。

## 安全边界与失败语义

- 默认最多两次尝试、总预算 110 秒、退让 4 cm、速度比例 0.25；单次还有原节点时限。
- 闭爪后宽度非空/未知、检测到接触/堵转或目标是薄件时，不主动松爪、不横向扫视；
  `result.metrics['recovery_stop_reason']` 返回 `payload_uncertain_do_not_open`，
  交给上游安全处理；`result.failure` 保留该次原始失败，不是新增的 FailureCode。
- 头顶目标不确定/多目标歧义或路径未验证时禁止恢复退让和重试。
  首次头顶没有目标、闭爪前头顶被遮挡或抬升后头顶无证据时，仍保留腕部验证结果，
  不将“头顶未知”自动判为成功或失败。腕部增强分割缺少机器人遮罩时不能继续接近。
- 校准、夹爪、模型以及明确碰撞/IK 等错误不会被无条件重试掩盖。
- 恢复节点不自行打开夹爪。只有下一次节点选好可执行候选后才会正常预开爪。
- 入场前提是上游确认空手；持物保护针对本次执行已发出的闭爪命令。
  次数耗尽或其他原因停止时，没有 `payload_uncertain_do_not_open` 标记也不代表空手。
- 整体超时依赖驱动调用能够返回，尚无强制终止阻塞硬件 I/O 的独立实时守护。
- 两指夹爪无法进入的薄件仍应拒绝；主动观察不能创造桌面间隙或机械臂自由度。

Isaac 中机器人遮罩来自已知机器人路径 `/World/SO101` 的 instance-ID 渲染，
**不使用目标的实例标签、真实位姿或网格进行感知/决策**。
真实机器人需用已标定机器人几何投影/深度一致性实现等价遮罩；缺失时增强模式拒绝动作。
已修复 Isaac 6 将 annotator 名称 `_fast` 归一化后，旧代码读取不到机器人遮罩的问题。

仿真并非端到端盲抓：场景配置提供初始粗位置、物体属性、桌面高度，
demo 根据已知对象高度调整交接姿态；规划器按已知目标路径将目标从普通全局障碍中排除，
再由局部观测做手指/接近检查。这些场景先验必须与真实 BusAgent 的粗定位和碰撞场景更新对应。

## 运行与公平对比

环境和权重准备沿用 [接入 README](BUSAGENT_README.md)，以下均在仓库根目录运行。
默认仍为 `off`，不会自动启用恢复；新鲜腕部观测和规划时限复核同样适用于基础模式。

```powershell
# 完整主动恢复 demo（正常成功时不会额外扫视）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -Output output/recovery_demo

# 固定斜上方相机，减少手指遮挡；不改变腕部闭环控制职责
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -SceneView oblique -Output output/recovery_oblique_demo

# 显式故障注入：初始 5 次腕部获取返回空，验证恢复路径；不是自然失败成功率
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -DropInitialWristFrames 5 -Output output/recovery_fault_demo

# 同一冻结开发集对比：基础 / 双相机辅助但不重试 / 主动恢复
& D:/isaac/env_isaacsim60/python.exe scripts/run_grasp_recovery_comparison.py --output output/recovery_comparison/pilot --cases cube,apple,hammer,wrench,coffee_mug --seeds 0

& D:/isaac/env_isaacsim60/python.exe -m unittest discover -s tests
```

`off`：原始单次节点；`assisted`：机器人自遮罩和头顶验证，最多一次；
`active`：相同辅助机制，失败后可退让并多视角再尝试。
若额外指定 `-Perception multiview`，第一次也会主动观察；配对脚本固定为 `single`。
用这三个组分离机器人遮罩/验证改善与真正主动恢复带来的增益。
比较脚本冻结相同场景文件及哈希，保存源码提交、脏工作区标记和逐组结果。
开发集不是验收集，当前默认五种对象均为程序生成的几何代理，不是实物扫描模型。

### BusAgent 装配要求

原接入 README 的 `FineGraspSkill` 可以继续包装新节点，因为 `execute/cancel` 契约不变；
但不能只把旧示例的类名替换，恢复功能还需要以下依赖：

- `SceneTargetObserver(scene_rgbd_camera, table_height_m=..., require_robot_mask=True)`；
  固定相机需标定到 base，提供与腕部同一单调时钟域的新鲜 RGB-D。
- `attempt_factory(index, failed_grasps, target)` 中每次构造新 segmenter、新 node、新 verifier；
  `index` 从 0 开始，`index > 0` 同时开启 `fusion.enabled` 和 `active_views.enabled`。
- 分割外包 `RobotMaskedSegmenter(..., require_mask=True)`；
  `SceneAssistedVerifier(observer, motion, target)` 负责辅助验证；
  `GeneralGraspNode(..., failed_grasps=failed_grasps)` 保留失败候选排除。
- `motion.is_observation_path_safe(pose, points_base, FingerGeometry)` 必须验证实际可执行的
  整条关节轨迹及局部手指扫掠体；没有此钩子时拒绝恢复运动。
  `clear_grasp_plan()` 应清除旧抓取分支与缓存，不能删除手眼标定。
- 组装为 `RecoveringGraspNode(attempt_factory=..., observer=..., motion=..., gripper=...)`。

可运行装配示例是 `scripts/run_fine_grasp_demo.py` 的 `attempt_factory` 和其外围代码。
硬件适配器尚未提供这些完整安全钩子；保留实验模式，不直接启动真机执行。

## 当前实测证据（2026-09-05，开发工作区）

| 测试 | 物理结果 | 机制结论 |
|---|---|---|
| `output/active_recovery_dev/normalized_mask_cube` | 实际抬升 83.1 mm，节点成功，14.2 s | 自遮罩修复有效；没有主动扫视，不能归因于恢复 |
| `output/active_recovery_dev/transient_camera_fault` | 初次 `no_observation`；第二次成功；头顶确认抬升 81.0 mm，总计 20.8 s | 实际执行两次侧向观察；仅一次闭爪，是感知恢复而非夹空后重抓 |
| 遮罩修复前两次 cube | 物理抬升但视觉验证失败 | 检测持物疑虑并停止，没有松爪或盲目重试 |

上述测试为 GraspGenX 与几何候选混合后端，不代表全部抓取由神经候选产生。
模型服务来源见 `selected_grasp.backend`，内部几何/生成分支见 `selected_grasp.metadata.branch`
（例如 `obb`）。不能只凭 backend 名称将几何候选成功算作纯神经生成成功。
尚需同版本、多对象、多种子的配对结果，
以及多对象的物理空抓重试验证；下文新增的单场景受控恢复不等于泛化改善。

## 调试产物与尚未完成项

### 冻结开发配对测试

代码 `1f93c77`，五对象（apple、coffee_mug、cube、hammer、wrench）、seed 0、共 15 次
独立 Isaac 进程。场景定义完全相同，15 次 grasp 源码哈希一致。
精简原始数据及报告哈希见 [配对证据](benchmarks/active_recovery_pilot_20260905.json)，
本地完整产物：`output/recovery_comparison/pilot_20260905/`。

| 模式 | 节点与物理同时成功 | 仅物理抬升达标 | 平均节点周期 | 成功重抓 |
|---|---:|---:|---:|---:|
| off | 1/5 | 4/5 | 15.64 s | 0 |
| assisted | 2/5 | 3/5 | 17.82 s | 0 |
| active | 2/5 | 4/5 | 17.74 s | 0 |

三组均无“节点成功但未物理抬升”的假成功；所有最终选择都是 GraspMoE 的 `obb` 分支。
结果不证明主动恢复改善自然失败成功率：本轮主动扫视次数为 0。
苹果持物但验证未知而停止，锤子的额外退让路径未验证通过，扳手闭爪命令失败。
辅助组锤子与另两组的物理结果也不同，说明需要更多种子/重复次数，不能只比较一个总分。

后续安全修正不混入上述冻结结果：堵转即使宽度近零也不视为空手；
固定/腕部相机均拒绝重复或倒退的渲染时间；头顶验证在同帧搜索“跟随抬升”和“仍在桌面”
两种假设，避免抬升后搜索框移走导致漏掉桌面上的目标。
这一阶段为 120 项单测通过。后续物理空抓测试见下一节，不以 mock 测试代替。

可显式用 `-TestTargetShiftM 0.04` 在首次闭爪前将仿真目标沿 base X 移动 4 cm，
上游粗位置、候选和验证结果均不修改；默认 0 关闭。只有报告中的
`test_target_shift_applied=true` 才表示实际执行过该扰动，未到闭爪不能算注入成功。
比较脚本对应参数为 `--test-target-shift-m`，必须与自然场景得分分开报告。

### 第二阶段：真正空抓恢复与相机遮挡

`2f954ca` 的同场景 4 cm 移位对比：基础版夹空失败；主动版实际两次闭爪、
两次侧向观察，第二次将目标抬升 77.7 mm，但视觉验证仍不确定，严格得分为 0/1。
头顶在闭爪前只看见 21 个目标像素，抬升后看见另一小块侧面；不能将这种不完整几何
直接当作完整点云对应，也没有放宽原来的成功阈值。

增加显式 `-SceneView oblique`：固定安装位置 `[0.65, -0.65, 1.75]`，
看向工作区 `[0.35, -0.15, 1.08]`；参数在 `configs/cameras.yaml`，
不根据测试对象名称/尺寸改变视点。仍使用标定后的固定相机变换。
默认保持 `overhead`，实际硬件的安装/手眼标定需要单独完成。

该开发工作区的默认方块 smoke 测试完成了完整恢复：首次夹空、重新观察、第二次闭爪，
实际抬升 81.8 mm，斜上方视觉确认抬升 72.3 mm，跟随误差 9.2 mm，总耗时 39.0 s。
第二次闭爪前/抬升后可见目标像素为 409/558；模型调用两次，主动侧向移动两次。
**这个 smoke 不是原来的冻结 cube 摆放，不能直接作为相机改动的配对增益。**
详见 [受控故障证据](benchmarks/active_recovery_fault_20260905.json)。

新增慢模型/慢 IK 后的腕部新观测复核，即使单视图模式也不能在模型计算期间丢失目标、
目标显著移动或预算耗尽后启动旧 pre-grasp。可观测长轴改变也触发重新推理；
对称形状的 PCA 轴不作为可靠朝向。当前 124 项单测通过。

### 同版本、同冻结场景的恢复复测

`b15ff6a`、干净工作区、相同 seed 0 cube 文件、相同斜上方安装、相同 4 cm
首次闭爪前扰动：基础 `off` 为 **0/1**，增强 `active` 为 **1/1**。
增强版首次真实夹空，安全退让后实际移动到两个侧向视点，再闭爪成功；
最终物理抬升 **81.9 mm**、头顶视觉抬升 71.5 mm、跟随误差 9.7 mm。
总周期 **39.19 s**，两次模型调用累计 1.42 s，实际两次闭爪。
两次试验均无物理假成功；证据见 [同场景配对结果](benchmarks/active_recovery_oblique_paired_20260905.json)。

这证明“受控空抓后的完整增强恢复链路”可执行并成功，不证明 unseen-object 泛化。
也没有单独消融“只重试但不多视角”的模式，因此不能把全部增益归因于扫视本身。
新相机位置没有修改腕部控制职责或放宽原验证阈值。

同一 `b15ff6a` 随后运行无扰动族群回归：apple 实际抬升 **80.9 mm**、
coffee_mug **51.6 mm**，均由腕部控制并得到固定外部相机确认；hammer 没有实际抬起，
夹爪堵转/持物状态不明，返回 `payload_uncertain_do_not_open`，未盲目再扫视。
结果 **2/3**，三者均仅一次闭爪；没有把此组算作重抓增益。
见 [族群回归证据](benchmarks/active_recovery_oblique_family_20260905.json)。
这些仍是几何代理，只有一个种子；真实工具/杯子/水果、多光照、反光深度退化、杂乱场景、
冻结验收集和真机测试均未完成。锤子/扳手的接触、夹爪状态与可抓部位质量仍是下一阶段重点。

每次尝试保存到 `attempt_N/debug/`：RGB、深度、掩码、自遮罩、相机变换、候选与事件。
头顶观测在 `overhead/`；总 `report.json` 包含 `attempt_results`、`attempt_physics`、
`recovery_stop_reason`、实际闭爪次数、主动视点次数和累计模型/伺服耗时。
`attempt_physics` 是仿真评价仪表，只写报告，不参与控制。
比较脚本输出 `comparison.json` 和 `COMPARISON.md`，分开统计首次成功、最终成功、
恢复成功、真正多次闭爪成功和物理假成功。

当前头顶识别仍是颜色/连通域/三维几何一致性，不是已验证的开放词汇语义融合；
同色杂乱场景、强反光/透明物体、薄工具、旋转突变和完全遮挡仍可能返回不确定。
主动视点是有界侧向候选加安全筛选，尚非学习的下一最佳视点策略。
手指扫掠体是几何近似，不等于完整机械手接触认证。没有真机验证，不能自动开启真机运动。
