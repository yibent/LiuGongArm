# 泛化能力验证记录

更新时间：2026-09-05

本记录对应分支 `feature/yoloe-florence-memory`，验证基线为最新 `origin/master` 合并提交 `0197e13`。测试在 Isaac Sim 6.0、RTX 4070 Laptop、640x480 RGB-D、SO-101 和本地离线模型上完成。

## 验证范围

系统按三层判定：

1. **语义定位**：Florence-2 根据输入标签产生候选框；`florence_yoloe` 模式再用 YOLOE 视觉提示重新确认框。
2. **持续跟踪和交接**：场景相机使用带前后向误差检查的 LK 光流和深度重建；机械臂移动后，在腕部相机重新采集并检查时间、深度、颜色和米制中心一致性。
3. **物理抓取**：FineGrasp 继续执行候选生成、IK/碰撞检查、闭爪、抬升和视觉/物理验证。前两层通过不能代替物理抓取成功。

本轮默认使用 `geometric` FineGrasp 后端做可重复的感知/运动接近验证。GraspGenX 本机环境和服务权重不存在，因此没有把 GraspGenX 缺失误报为模型泛化结果。

## 模型和本地配置

当前实际可用权重：

| 组件 | 本机位置 | 用途 |
|---|---|---|
| Florence-2-large | `.cache/huggingface/Florence-2-large` | 慢环标签定位 |
| YOLOE-26x-seg | `yoloe-26x-seg.pt` | 快环文本/视觉提示检测与跟踪 |
| MobileCLIP | `mobileclip2_b.ts` | 已下载；当前标签抓取入口使用 Florence 框的 YOLOE 视觉提示 |

标签抓取入口会优先使用 `_envs\vision`，不存在时由 `scripts/run_label_grasp.ps1` 回退到仓库 `.venv`。Isaac 入口通过 `ISAAC_PYTHON` 或当前本机 `D:\isaacsim\python.bat` 解析。

## 桌面物体结果

所有测试均为单个目标、头顶场景相机到腕部相机的粗接近，参数为 `Recovery=active`、`SceneView=oblique`、`CoarseOnly`。`find` 是 Florence/YOLOE 慢环调用次数，`flow` 是成功的光流观测数，`handoff` 是两相机米制中心差。

| 目标和模式 | 语义定位 | 光流 | 粗移动 | 腕部交接 | 结果 |
|---|---:|---:|---:|---:|---|
| `bottle` + Florence/YOLOE | 1 | 39 | 8 | 16.1 mm | 通过 |
| `hammer` + Florence | 1 | 35 | 6 | 9.6 mm | 通过 |
| `wrench` + Florence/YOLOE | 1 | 35 | 6 | 3.7 mm | 通过 |
| `screwdriver` + Florence/YOLOE | 1 | 34 | 6 | 5.7 mm | 通过 |
| `coffee mug` + Florence/YOLOE | 1 | 36 | 7 | 10.9 mm | 通过 |
| `key` + Florence/YOLOE | 1 或 2 次重定位 | 35 | 7 | 未通过 | 反光薄件深度不足 |

证据目录为 `output/generalization_label/`。每个通过案例至少包含 `coarse_report.json`、`coarse/debug/`、`initial_wrist.png` 和 `initial_rgbd.npz`。`key` 的最终重测目录为 `output/generalization_label/key_florence_yoloe_thin_gate02/`。

### key 失败解释

Florence 和 YOLOE 都找到了钥匙框，且场景相机完成了 35 次光流和 7 段受 guard 保护的移动。腕部 RGB-D 中该 3 mm 反光目标的重建高度约为桌面上方 0 mm，无法通过 `insufficient_target_depth` 交接门。系统因此没有进入闭爪阶段，这是预期的安全停止。

`SemanticFlowTarget` 现在允许 `ObjectProperties(thin=True)` 使用更低的 `0.2 mm` 支撑面门限；普通物体仍使用保守的 `4 mm` 门限。该调整改善了薄件候选保留，但不能凭空恢复反光表面的有效深度，也不会放宽最终抓取的桌面间隙约束。

## 记忆采集和回放

`ObjectMemoryStore` 每个视角保存：

- 带 padding 的 JPEG crop，供人工检查和旧记忆兼容；
- 原始参考帧 `reference_image`；
- 原始 `bbox`、置信度、可选位姿和视角名。

回放优先用 `reference_image + bbox` 调用 YOLOE `set_visual_prompt`，旧 JSON 没有参考帧时自动使用 crop 的 `set_image_prompt`。这解决了仅使用小 crop 在全场景尺度变化下无法稳定重捕获的问题。

复现瓶子记忆测试：

```powershell
& .\.venv\python.exe scripts\verify_memory_recall.py `
  --scene-image output\generalization_label\bottle_florence_yoloe\coarse\debug\obs_0001_rgb.png `
  --scene-bbox 273,217,304,276 `
  --wrist-image output\generalization_label\bottle_florence_yoloe\initial_wrist.png `
  --wrist-bbox 93,97,281,280 `
  --label bottle `
  --output output\generalization_memory\bottle_recall_reference
```

报告 `output/generalization_memory/bottle_recall_reference/report.json` 的结果：双视角记忆写入成功，`memory_ready=true`，`path=fast`，YOLOE 输出 1 个目标框，`florence_guard=not_called`。这证明该次回放没有重新调用 Florence。

## 物理抓取现状

已有独立回归证据显示 `red cube` 和苹果代理完成闭爪、腕部交接和抬升；本轮新增的 `bottle` 完整抓取尝试在标签接近后于 FineGrasp 阶段返回 `ik_unreachable`，没有闭爪，不能计为抓取成功。杯子本轮只验证到腕部交接，未计为物理抓取成功。

因此目前可以确认的是：

- Florence 慢环能在多个工业/日常目标上给出可用框；
- YOLOE 视觉提示和 LK 快环能在这些案例上连续跟踪并安全交接；
- 记忆可以跨视角保存，并在不调用 Florence 的情况下作为 YOLOE 快速重捕获提示；
- 反光薄件仍受 RGB-D 深度和桌面间隙限制；
- 多实例消歧、严重遮挡、透明件、随机光照和真机时钟/手眼标定尚未完成验收。

## 回归命令

```powershell
# 视觉/记忆专项测试
& .\.venv\python.exe -m unittest discover -s tests -p 'test_label_grasp.py' -v
& .\.venv\python.exe -m unittest discover -s tests -p 'test_object_memory.py' -v

# 全量 Isaac/项目测试
& D:\isaacsim\python.bat -m unittest discover -s tests -v

# 本地 Florence + YOLOE 离线烟测
& .\.venv\python.exe scripts\verify_vision.py --frames 24
```

本次修改后的专项测试为 `20 + 5` 项通过；当前代码全量测试为 `243 tests OK`，`compileall` 通过。视觉烟测报告 `output/generalization_vision/report.json` 为 `passed=true`：YOLOE 24/24 帧、CV 24/24 帧均有跟踪输出。
