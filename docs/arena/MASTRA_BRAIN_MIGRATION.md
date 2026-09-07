# Mastra 决策运行时迁移（2026-09-08）

## 已落地的边界

Mastra `Agent.generate` 负责模型工具循环、工具并发、步骤预算与原生工作记忆。删除原来的手写规划循环和单独的首次意图分类模型调用。BusAgent 保留事件总线、结构化队列、命令账本、事务 outbox 和串行机械臂执行；这些组件保证命令只执行一次，不进行自然语言决策。

模型可调用状态、执行队列、历史、观察集合、物体几何、同帧图像分析与目标绑定等工具，然后一次提交多个动作。提交后立即结束该轮推理，不再让模型解释刚刚提交的计划。正常动作之间不调用规划模型。阶段边界、执行异常、局部检查不确定和显式最终验收才再次进入 Mastra。

Mastra 在首次规划中同时判断复杂度并提交动作，不增加分类模型调用。目标明确、可一次给出完整有限序列的任务按 simple 处理；“拿起、放到桌面空处、回位”有多个动作也仍可简单。此类任务通常 complete + final_review=false，普通动作 review_after=false，依靠控制器和指定的局部监督完成；执行失败或证据不确定仍返回 Mastra。旧提交省略 mode 时默认 simple，避免凭空增加阶段或最终模型复核。用户明确要求最终复核时保留 final_review=true。

未知数量循环、需要判断装满、姿态纠正或执行结果决定下一批对象的任务，由 Mastra 选择 complex；仅有新证据依赖才用 stage。复杂任务也尽量批量提交已确定的动作。任务复杂度与执行快慢环独立：简单任务可以允许慢环回退，复杂任务内也可选择纯快环技能。

`submit_plan` 可 append 或 replace_pending；已经执行和正在执行的命令保持原有账本。动作的 title 仅用于显示，执行只使用 skill、params 与 execution。抓取与持物放置是独立技能，不需要再次抓取。

```json
{
  "mode": "simple",
  "outcome": "continue",
  "plan_scope": "complete",
  "queue_update": "append",
  "final_review": false,
  "actions": [{
    "title": "将持物放到桌面空处",
    "skill": "place_held",
    "params": {"destination": {"label": "table", "selection": "free_space"}},
    "execution": {
      "loop": "fast_only",
      "max_attempts": 2,
      "supervision": {"kind": "physical", "wait": false}
    }
  }]
}
```

## 快慢环与监督

- `fast_only`：快视觉与基础运动，不进入慢模型；已确认失败可按 max_attempts 重试，超出后返回 Mastra。
- `fast_then_slow`：技能内部快环优先，再调用已有慢环能力；耗尽本动作尝试后返回 Mastra。
- `slow`：直接使用慢视觉/增强运动能力。
- 已释放或物理执行结果未知时，不自动重放转移动作；仍持物不重抓，已确认失败且仍可靠持物的 place_held 可按指定次数重试。

异步监督任务写入同一持久队列，每个完成命令最多产生一个检查。重启后恢复未完成检查；检查关联动作版本和执行后冻结图像，不能拿后续场景冒充之前动作的结果。默认可与下一动作并行，wait=true 才等待检查。

physical 检查控制器反馈；florence 只把指定相机的执行后 ROI 交给本地 Florence，结合物理释放/持物/成功反馈得出 passed/failed/uncertain。Florence 检出物体不能独立证明接触稳定、闭口端朝上或插入成功；证据不足返回 uncertain，由 Mastra 决定补充观察。当前 `upright` 不能仅靠 Florence 自动判定成功。

## 记忆、上下文与开销

Mastra Memory + LibSQL 保存最近消息和结构化工作记忆，工作记忆资源按会话、目标及智能体角色隔离。使用资源级工作记忆规避当前 SDK 线程 metadata 保存覆盖的问题；已有跨两轮调用和目标隔离回归测试。使用原生 rotateResponseMessageId 按工具步骤划分消息，避免整个工具循环聚合成一条消息后无法裁剪。不开启自动标题生成、语义向量检索或额外常驻摘要模型。

TokenLimiter 限制输入消息，预算同时考虑工具 schema；用户原始目标始终保留。完整观察保存在证据库，模型按 ref 分页读取，读取证据根对象也必须经过预算预览；省略路径指向原证据，避免反复归档嵌套预览。同一轮相同只读观察合并，独立工具允许并行。图像字节只进入短视觉调用，结果和引用回到 Mastra，不把图像反复带入长期上下文。

初始任务、实时报告、会话和其他队列先合并，再按一个总预算生成可检索预览，避免每块分别限长后总和仍超限。原生输入处理器在每步替换同一个带标签的控制状态消息，保留精简的 holding、失败步骤及 pending 列表；TokenLimiter 裁剪历史时不会丢掉这些当前事实。read_state/read_execution_queue 会刷新该状态，不创建额外模型调用。恢复需要先修复失败再执行旧后续步骤时，用 replace_pending 重排剩余序列；append 仍严格表示排到现有 pending 之后。

沿用现有 Gemini 双渠道，网络适配器只转换消息与工具协议，不实现推理循环。Mastra/AI SDK 隐式重试关闭；接口失败进入冷却，避免每一步反复等待失效渠道。对话模型和语音协议未改变。

## 验证记录

简单任务分流与恢复补充（当前完整后端回归 352 项通过，1 项 MySQL 测试按环境跳过）：

- `20260908-mastra-simple-batch`：普通指令“拿起一个金属圆柱，放到桌面的空处，然后机械臂回位。”，没有额外指定复杂度或模型策略。双渠道分别超时 8 秒和 30 秒，0 次成功模型响应、0 个动作，结果 blocked。
- `20260908-mastra-simple-batch-2`：同样指令，首次有效模型响应直接选择 simple + complete + final_review=false，提交 pick_place、home。首选超时 8 秒，备用响应 21 秒，无独立分类、观察或二次规划前置调用。执行时 IK 接近放置点失败（位置误差 6.4 mm），AnyPlace 没有可用候选；恢复暴露历史裁剪丢失持物与初始消息总量超限，导致重提抓放、错误 append 后再被控制器拒绝。保留 138.77 秒 blocked 结果，不能算顺利完成。
- `20260908-mastra-simple-batch-recovery`：补上总预算和控制状态保留后，直接恢复同一个目标，未重置场景或更改原指令。Mastra 用 replace_pending 提交 place_held、home，保留已完成动作；物理放置与释放确认成功，异步 physical 检查 0.275 秒通过。恢复 38.10 秒完成，新增有效模型响应 2 次，均在动作下发之前；放置与回位之间以及最终完成时为 0 次。控制器放置 20.934 秒、回位 0.474 秒。此时队列任务完成、机械臂 idle、空夹爪。

这些记录验证简单任务首次批量规划和失败续接，不构成完整工业验收；接口延迟和放置 IK/候选问题仍有实测失败证据。

本地完整后端回归：347 项通过，1 项 MySQL 集成测试按环境跳过（另在服务器直接调用部署的 QueueStore 通过）；Python 控制器/视觉针对性回归：36 项通过。新增回归覆盖 Mastra 实际工具循环、并发与合并、提交后停止、原生工作记忆持续性和隔离、执行策略、有限重试、重复完成事件、物理检查、13 次模型步骤的上下文裁剪和大型执行报告进入恢复。

云端自然语言简单任务记录：`output/industrial-acceptance/20260908-mastra-simple-2`。只在任务开始前重置夹具，未向模型注入仿真资产名或答案。Mastra 连续提交 grasp、place_held，两步均纯快环、无额外监督和慢抓放回退，最终完成。总观察时间约 76 秒，两次物理动作合计 31.514 秒；返回成功的决策模型调用共 4 次（首次两次、最终两次），两动作之间为 0 次。最初首选渠道超时 8 秒。该结果证明简单动作续接，不能代替完整工业装箱验收。

此前第一次简单任务双渠道超时，未产生物理动作；证据保留在 `20260908-mastra-simple`。完整工业任务结果另行记录，不把接口测试或单个动作成功计为整体验收。

工业试跑发现并修复的底层断点：

- 并发队列事务以 INSERT IGNORE 获取共享锁后，再升级 FOR UPDATE，会产生死锁。观察已完成却因记录失败被返回为工具失败。改为首次 upsert 即获取独占锁；独立 InnoDB 并发对照旧写法 80 次仅 12 次提交，新写法 80/80。随后直接调用部署的 QueueStore，在临时独立数据库执行 80 次并发变更，状态 revision=80、outbox=80、无丢失。测试数据库已删除。可选集成回归 `BUSAGENT_TEST_MYSQL_SOCKET=... pnpm exec vitest run test/queue-store.mysql.spec.ts`。
- 实际技能名现在从控制器能力目录注入 submit 工具 schema 的 enum，不再只放进可被压缩的初始状态，避免模型反复猜测 pick、pick_and_place。
- 稳定 cell_id 由 RGB-D 几何产生，却被原来的动作校验当成非法 obs ref。现在从观测几何记忆解析到最新格位引用，不依赖配置资产。初始任务状态、失败执行报告和证据根读取均经过预算预览，完整数据仍可按原引用查阅。
- Florence 正负 ROI 对照：冷加载 8.29 秒，热调用 0.30–0.35 秒；原框选接口会对空桌面输出近乎整个 ROI 的框。已将这类整体框判为 uncertain，不能冒充检测成功。局部节点仍只提供可见性证据，不能从一对样本外推可靠成功率。

工业整体验收四轮均未通过，不能宣布 PDF 场景完成。第一轮暴露数据库死锁、技能名称与格位误用；第二、三轮暴露 Mastra 整轮消息聚合和不受预算约束的证据读取；第四轮在首批观察阶段两个 Gemini 渠道的图像调用分别超时（8 秒和 30 秒），未下发动作。

`20260908-mastra-async` 额外验证异步调度：Mastra 一次提交抓取、放置；抓取成功后的 physical 检查 0.233 秒完成，直接续接下一命令，无中间 LLM。纯快环未识别桌面，放置失败且保持持物；失败报告过大导致恢复入口被限长器阻断，该输入现已纳入同一证据预算。此条测试保留失败结果，不计为完整抓放成功。

`20260908-mastra-held-recovery` 在同一持物现场以新的自然语言指令续接，未重置场景或指定物体资产：Mastra 选择 fast_then_slow 的 place_held，放置成功、释放确认、异步 physical 检查通过（0.125 秒），最终返回完成。共 51.04 秒，其中控制器命令 18.118 秒，决策模型成功调用 2 次（下发前、最终复核）。该测试证明持物恢复与最终验收链路，不能代替工业装箱测试。机器可读结果见 [MASTRA_VALIDATION_20260908.json](MASTRA_VALIDATION_20260908.json)。

部署前备份位于服务器 `output/backups/mastra-20260908/`，保留旧代码、队列快照和权限受限的配置副本。本地界面映射为 http://127.0.0.1:18991/ ，预览为 http://127.0.0.1:18993/。

API 依据：[Mastra agents](https://mastra.ai/docs/agents/overview)、[tools](https://mastra.ai/docs/agents/tools)、[working memory](https://mastra.ai/docs/memory/working-memory)、[processors](https://mastra.ai/docs/agents/processors)。依赖锁定 @mastra/core 1.64.0；最低 Node 22.13。
