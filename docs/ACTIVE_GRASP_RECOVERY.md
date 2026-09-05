# 实验开发线：双 RGB-D 主动抓取恢复（左顾右盼）

分支：`codex/active-grasp-recovery`；基于已发布的 `2c6bfbe`。
本页描述新的实验能力；原始基线和接入契约见 [BusAgent README](BUSAGENT_README.md)。
没有宣称工业工具、真实杯子/水果或真机泛化已通过验收。

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
- 闭爪后宽度非空/未知、检测到接触或目标是薄件时，不主动松爪、不横向扫视；
  返回 `payload_uncertain_do_not_open`，交给上游安全处理。
- 头顶目标不确定/多目标歧义、深度不足、机器人遮罩缺失、路径未验证时停止。
- 校准、夹爪、模型以及明确碰撞/IK 等错误不会被无条件重试掩盖。
- 恢复节点不自行打开夹爪。只有下一次节点选好可执行候选后才会正常预开爪。
- 整体超时依赖驱动调用能够返回，尚无强制终止阻塞硬件 I/O 的独立实时守护。
- 两指夹爪无法进入的薄件仍应拒绝；主动观察不能创造桌面间隙或机械臂自由度。

Isaac 中机器人遮罩来自已知机器人路径 `/World/SO101` 的 instance-ID 渲染，
**不使用目标的实例标签、真实位姿或网格进行感知/决策**。
真实机器人需用已标定机器人几何投影/深度一致性实现等价遮罩；缺失时增强模式拒绝动作。
已修复 Isaac 6 将 annotator 名称 `_fast` 归一化后，旧代码读取不到机器人遮罩的问题。

## 运行与公平对比

环境和权重准备沿用 [接入 README](BUSAGENT_README.md)，以下均在仓库根目录运行。
默认仍为 `off`，新开发线不静默改变基础版行为。

```powershell
# 完整主动恢复 demo（正常成功时不会额外扫视）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -Output output/recovery_demo

# 显式故障注入：初始 5 次腕部获取返回空，验证恢复路径；不是自然失败成功率
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_fine_grasp_demo.ps1 -Backend graspgenx -Recovery active -DropInitialWristFrames 5 -Output output/recovery_fault_demo

# 同一冻结开发集对比：基础 / 双相机辅助但不重试 / 主动恢复
& D:/isaac/env_isaacsim60/python.exe scripts/run_grasp_recovery_comparison.py --output output/recovery_comparison/pilot --cases cube,apple,hammer,wrench,coffee_mug --seeds 0

& D:/isaac/env_isaacsim60/python.exe -m unittest discover -s tests
```

`off`：原始单次节点；`assisted`：机器人自遮罩和头顶验证，最多一次；
`active`：相同辅助机制，失败后可退让并多视角再尝试。
用这三个组分离机器人遮罩/验证改善与真正主动恢复带来的增益。
比较脚本冻结相同场景文件及哈希，保存源码提交、脏工作区标记和逐组结果。
开发集不是验收集，当前默认五种对象均为程序生成的几何代理，不是实物扫描模型。

## 当前实测证据（2026-09-05，开发工作区）

| 测试 | 物理结果 | 机制结论 |
|---|---|---|
| `output/active_recovery_dev/normalized_mask_cube` | 实际抬升 83.1 mm，节点成功，14.2 s | 自遮罩修复有效；没有主动扫视，不能归因于恢复 |
| `output/active_recovery_dev/transient_camera_fault` | 初次 `no_observation`；第二次成功；头顶确认抬升 81.0 mm，总计 20.8 s | 实际执行两次侧向观察；仅一次闭爪，是感知恢复而非夹空后重抓 |
| 遮罩修复前两次 cube | 物理抬升但视觉验证失败 | 检测持物疑虑并停止，没有松爪或盲目重试 |

上述测试为 GraspGenX 与几何候选混合后端，不代表全部抓取由神经候选产生。
候选实际来源保存在 `selected_grasp.backend`。尚需同版本、多对象、多种子的配对结果，
以及真实空抓后的物理重抓测试；不能根据方块演示宣称泛化改善。

## 调试产物与尚未完成项

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
