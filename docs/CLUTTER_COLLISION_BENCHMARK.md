# 杂乱桌面与复杂碰撞测试

本工程新增了一套可复现的 Isaac Sim 场景，用来验证“标签定位 → YOLOE/LK 快速跟踪 → 腕部交接 → FineGrasp 碰撞检查 → 有界恢复”的泛化边界。场景定义在 [`source/mr_liu/sim/clutter.py`](../source/mr_liu/sim/clutter.py)，运行入口是 [`scripts/run_clutter_grasp_benchmark.py`](../scripts/run_clutter_grasp_benchmark.py)。

场景生成器把干扰物放在 `/World/ClutterProps`，把挡板、导轨和运动故障物放在 `/World/ClutterObstacles`。这些路径没有加入 `configs/motion.yaml` 的排除列表，因此 `bind_world()` 会将它们作为 PhysX 碰撞体提供给 cuMotion。目标仍使用 `/World/FineGraspTarget`，由 `IsaacCumotionExecutor(target_object_paths=[...])` 单独排除，避免规划器把待抓目标当成静态障碍。

当前包含五类布局：

| 场景 | 覆盖内容 | 期望行为 |
|---|---|---|
| `mixed_tools` | 六个不同形状、颜色和工业标签的桌面干扰物 | 目标可见时完成接近；目标不可见时安全停止并记录 `target_not_visible` |
| `occluded_target` | 近距离低挡板和侧向挡板 | 重新观察；无法确认时安全拒绝 |
| `narrow_corridor` | 两侧高导轨和后挡板 | 夹爪/路径碰撞阻断时拒绝，不强行穿越 |
| `duplicate_label` | 两个相同“red cube”干扰物 | 返回 `ambiguous_label_instances`，不执行闭爪 |
| `moving_blocker` | 动态挡板和后方障碍 | 运动中守卫触发时停止并重新定位；失败则终止 |

## 运行

需要 Isaac Sim 6.0 Python 和已启动的本地语义定位服务（默认端口 5570）。先确认视觉服务可用，再运行全部场景：

```powershell
$env:ISAAC_PYTHON = 'D:\isaacsim\python.bat'
& $env:ISAAC_PYTHON scripts\run_clutter_grasp_benchmark.py `
  --output output\clutter_benchmark_$(Get-Date -Format yyyyMMdd_HHmmss) `
  --label 'red cube' `
  --localization-mode florence_yoloe `
  --perception multiview `
  --recovery active
```

只检查“杂乱场景定位、粗接近、腕部交接”和障碍路径时，可以不进入闭爪：

```powershell
& $env:ISAAC_PYTHON scripts\run_clutter_grasp_benchmark.py `
  --scenes mixed_tools,occluded_target,narrow_corridor,duplicate_label `
  --coarse-only `
  --output output\clutter_handoff_$(Get-Date -Format yyyyMMdd_HHmmss)
```

单独复现一个场景：

```powershell
& $env:ISAAC_PYTHON scripts\run_clutter_grasp_benchmark.py `
  --scenes narrow_corridor `
  --output output\clutter_narrow_corridor_$(Get-Date -Format yyyyMMdd_HHmmss)
```

每个子目录保存 `clutter_scene.json`、`target_case.json`、实际执行命令、Isaac 标准输出/错误、`coarse_report.json` 和 `report.json`。根目录的 `summary.json` 与 `SUMMARY.md` 记录每个场景的期望结果、返回码、失败原因、障碍物数量和通过状态。

## 结果解释

基准脚本使用以下状态：

- `PASS`：FineGrasp 节点完成请求；若运行完整抓取，还需结合 `report.json` 的实际刚体抬升判断物理成功。
- `HANDOFF_PASS`：使用 `--coarse-only` 时，语义定位、LK 跟踪和腕部交接通过。
- `SAFE_REJECT`：场景预期系统拒绝，且报告的失败原因与场景匹配，例如 `ambiguous_label_instances` 或 `obstacle_collision`。
- `UNEXPECTED`：仿真进程运行了，但没有得到场景声明的拒绝原因；应检查 `stdout.log`、`stderr.log` 和调试帧。
- `FAIL`：预期可执行的场景没有完成请求，可能是遮挡后重定位失败、IK 不可达、碰撞路径阻断或 FineGrasp 物理验证失败。

系统在这些场景中保持 fail-closed：目标不唯一、深度不足、路径碰撞、运动中目标变化、腕部交接身份不一致或抬升未验证时，会停止当前动作并清除旧路径。恢复最多使用配置的有限次数；恢复仍不能得到新鲜且一致的 RGB-D 证据时，返回失败，不会盲目闭爪或打开不确定的夹持。

## 验证层级

1. `tests/test_clutter_scenes.py` 验证场景数据、JSON 冻结和必需覆盖面，不需要启动 Isaac。
2. `--coarse-only` 验证 Florence/YOLOE、LK 光流、粗接近路径和双相机交接。
3. 完整运行验证 FineGrasp 的候选筛选、IK、cuMotion world collision、指尖扫掠碰撞、闭爪和实际抬升。仅节点返回成功而刚体没有抬升时，仍按物理失败记录。

仿真中的目标真实位姿只用于构造场景和最后的物理抬升核验，不会传给视觉控制器。

## 本机实测记录（2026-09-05）

在 Isaac Sim 6.0、RTX 4070 Laptop、当前 `feature/yoloe-florence-memory` 分支上，`mixed_tools` 的无标签 dry-run 已完成场景初始化和 cuMotion 世界绑定，6 个干扰物均进入规划世界；腕部 RGB-D 无法确认目标时返回 `target_not_visible`，记录为 `SAFE_REJECT`。证据见 [`output/clutter_mixed_nolabel_20260905_r2/summary.json`](../output/clutter_mixed_nolabel_20260905_r2/summary.json)。

`narrow_corridor` 的 4 个障碍物也完成加载；高导轨使夹爪在观察开度阶段发生阻断，系统返回 `gripper_open_blocked`，没有继续执行抓取，记录为 `SAFE_REJECT`。证据见 [`output/clutter_narrow_20260905_r2/summary.json`](../output/clutter_narrow_20260905_r2/summary.json)。这证明了场景/碰撞体接入和 fail-closed 行为；它们不是“物理抓取成功率”结论。Florence/YOLOE 语义服务在本轮与 7860 WebUI 共用 GPU 时发生连接重置，已保留为模型服务不可用证据；应在空闲 GPU 或单独推理进程下重跑带 `--label` 的语义验收。
