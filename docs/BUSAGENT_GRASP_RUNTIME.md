# 现有场景的 BusAgent 抓取接入

`scripts/run_vision_follow.py` 现在提供完整 `grasp` 控制命令，不另开 demo、不重建场景、不瞬移机械臂。
这是开发接入；本地回归不代表 Isaac GPU 抓取验收已通过。

## 默认双相机与对话重试

Web 不提供模式选择或独立重试按钮。默认装配机器人自遮罩、腕部闭环与固定 RGB-D 辅助验证。
第一次失败后保持当前位置；只有夹持、目标身份和退让路径允许恢复时，返回 `retry_available=true`。
聊天模型据此简短反馈、询问；操作人明确说“再试一次”后，语义模型生成 `grasp` / `retry_last:true`。
控制器恢复同一次失败上下文，重新检查现场与夹持状态后退让、侧向观察并进行第二次尝试。
不会重新走未经检查的“接近并张爪”，也不由控制器自动重试。持物不确定时不提供重试。
等待上下文保留 120 秒；停止、其他机械臂动作或超时会使其失效。每个上下文最多恢复一次。
GraspGenX 服务不可用时不会回退几何抓取。Web 显示实际阶段、尝试次数和结果，不播放相机画面。
本次接入仅经本地测试；服务器部署与 Isaac GPU 实抓仍需单独验证。

## 启动与依赖

用现有 Isaac Python 启动（替换为服务器的实际解释器路径）：

```bash
<isaac-python> scripts/run_vision_follow.py --no-follow --grasp-backend graspgenx
```

沿用原有 `--headless`、`--control-host`、`--control-port` 参数。
默认 GraspGenX 使用 `configs/fine_grasp.yaml` 中的本机 ZMQ 服务；模型环境及权重准备见
[模块交接文档](BUSAGENT_README.md)。Isaac 客户端依赖见
[`requirements-graspgenx-client.txt`](../requirements-graspgenx-client.txt)。
不会自动下载模型、启动模型服务或改变已验证的关节阻尼。

- `--grasp-backend geometric`：显式使用无神经模型的几何基线。
- `--grasp-backend disabled`：不注册抓取能力。
- 默认配置及 BusAgent 在线会话禁止几何回退；模型超时、断开或无候选时明确失败。
  geometric 仅保留为手动选择的离线对照基线，不用于在线服务降级。
- Linux 模型服务使用独立 `_envs/graspgenx` 环境，运行
  `bash scripts/run_graspgenx_server.sh`；只监听本机 5556，不修改 Isaac 的 Torch。
  模型使用官方 GraspMoE（扩散候选与 OBB 候选经神经判别器评分），
  这与项目的无模型 geometric 回退是不同的流程。

BusAgent 是独立的嵌套仓库：其 `backend/` 也必须构建并重新启动，配置仍指向现有
`controller_url`（默认 `http://127.0.0.1:7861`）。前端继续经 Web 语音交互，
增加执行阶段显示，不传输实时相机画面；Isaac 画面仍在服务器桌面观察。

## 指令与执行链

例如：“拿起绿色方块”“抓起最左边的黄色螺栓”。目标是否可抓由当次观测和规划决定。

1. NLU 生成 `pick`；计划只包含一个完整 `grasp` 步骤，目标仍是类别、颜色、选择器。
2. 与归位、关节运动共用物理队列。轮到抓取后，控制器独占机械臂并重新取得顶视 RGB-D。
3. 顶视类别/颜色核对与实例分割关联到桌面实体；位置来自深度，不来自 USD 预设坐标。
4. 用手眼变换求目标上方的腕部观察姿态，并经现有 cuMotion IK/碰撞路径检查接近。
5. 新建腕部追踪器，执行观察、候选生成、预抓取、视觉微调、闭爪、抬升和独立验证。
6. `execution.progress` 汇报阶段，仍由对话 LLM 组织短句；只有验证成功返回 completed。

## 控制接口

提交 `POST /api/command`：

```json
{
  "command_id": "grasp-example-001",
  "skill": "grasp",
  "params": {
    "target": {
      "category": "block",
      "attributes": {"color": "green"},
      "spatial_ref": null,
      "quantity": 1
    }
  }
}
```

HTTP 202/accepted 只表示接收。通过 `GET /api/commands/grasp-example-001` 查询
`state`、`phase`、`message` 和最终 `result`。相同命令编号在保留窗口内不会重复执行。
`GET /api/capabilities` 的 `skills` 包含 `grasp` 才说明当前实例已注册。

`hold` / `stop` 继续走现有接口：HTTP 线程只设置取消标记，Kit 主线程停止驱动并清理；
绝不从 HTTP 线程调用 Isaac SDK。相机等待、模型返回和每步运动边界都会检查中断。
阻塞中的模型 RPC/规划不能保证硬实时取消，因此不把它描述为硬件急停。
中断后不自动松爪、不自动重抓、不用 resume 续接半截抓取；重新下达完整抓取指令。

## 约束与待验证项

- 本次只支持单个桌面动态实体。多目标有歧义时拒绝；仅支持最左/最右选择器。
- 用于拖拽跟随的绿色 TargetCube 是控制标记，不是可抓取物体，接口会明确拒绝。
- 放置/搬运未接入。复合“抓取并放置”不会偷偷只执行抓取前半段。
- 缺深度、实例不明确、不可达、碰撞或验证失败均返回失败，不自动重试，不放宽原验证阈值。
- 保留现有防抖配置；不使用 demo 的 800/20 增益覆盖当前运行配置。
- 规划只排除所选目标的碰撞体以允许接触，不排除桌面与其他物体；指尖/扫掠碰撞仍是近似。
- 总调度预算 90 秒，精细抓取节点预算 45 秒；模型/规划阻塞可能影响截止时间与画面流畅度。
- 调试记录每次使用新的 `output/busagent_grasp/<uuid>/`；不以旧一次成功替代本次验证。
- 待 GPU 实测：当前防抖增益下的粗接近可达性、腕部视野、物理抬升及停止响应。
  原抓取基线存在视觉抬升验证失败记录，本次没有用仿真真值掩盖它。
