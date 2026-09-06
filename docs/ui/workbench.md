# 刘工智能工作台

UI 位于独立主仓工作树 `LiuGongArm-ui`、独立 BusAgent 工作树 `LiuGongArm-ui/BusAgent`，两者分支均为 `codex/liugong-ui`。起点分别是 `bbff7ee` 与 `09ba9fb`。不会覆盖机械臂调试工作树里的未提交修改。

## 已实现

- React + shadcn/Radix + Tailwind 深色工作台；场景、仿真两页。先选择场景才可进入仿真。
- 功能面板采用左侧纯图标栏，悬停显示名称，支持上下方向键切换。提供上下文属性、对象位置/旋转、Panda 状态与执行参数、当前对话；上下/左右面板可拖动分隔线调整，功能面板最多占上半区约 75%。
- 内容根据面板自身宽度适配，内容区达到 560px 时，物体列表与编辑区、机械臂状态与参数区、节点信息与事件区左右排布；较窄时上下排布。很窄的物体编辑区将 X/Y/Z 输入改为三行。拖动面板不会重置正在编辑的数值。布局可继续容纳后续机械臂十字键与物体微调操作区，目前不新增这些控制接口。
- 同页工作区、侧视、腕部三路实时预览；工作区占主画面，两路小画面等大；单路/三路、完整/半/四分之一采样、全屏。右侧预览被挤窄时，工具栏换行，工作区画面置顶、两路小画面并排置底，中断和重置保持可用。
- 四条时间轨道、拖动浏览、缩放、选择节点、因果端点配色、快绿慢蓝、运行中宽度增长。时间轴是只读记录，不修改执行顺序。按调用顺序使用非等比显示：默认最小卡片宽 140px，运行一分钟约增加 40px，真实耗时仍在详情中；顶部用调用编号标记顺序。刻度、轨道和卡片共用滚动区域，左侧名称与顶部刻度固定，四轨按容器高度展开。
- 悬浮语音球、悬停或键盘聚焦后的文字入口、渐隐回复。球旁仅提供文字输入；对话留在左侧功能面板。顶部工作区面包屑条已移除。图片继续仅由预览与视觉服务处理，不进入语言模型上下文。
- 中断直接调用已有 stop 接口。重置支持物体、机械臂、全部；不清空时间轴。时间轴在当前页面内存中保留，提供清空，无归档/持久化历史。
- 删除被新界面替代的四个旧页面、旧图标/抓取面板和旧样式，避免保留两套前端。

## 部署边界

独立预览：https://8byr7v21-8qne20ga-8999.zj02restapi.gpufree.cn:8443/ 。2026-09-06 部署验证 HTTP 200、三路真实 800×600 画面、只读状态问答及 BusAgent 事件显示。nginx 平滑重载前后 Arena / BusAgent 的 PID 保持 289722 / 222504。

同日布局修复已更新到 8999：图标侧栏、面板宽度自适应、非等比时间轴和文字入口。仅替换静态资源，未重启后端；当时运行的 Arena / BusAgent PID 为 292953 / 222504。公网再次验证新 CSS/JS 均 HTTP 200、三路真实画面、浏览器无异常且未发送控制指令。部署前静态资源备份位于服务器 `/root/gpufree-data/liugong-ui-preview-before-layout-fixes.tar.gz`。

独立 nginx 配置 `ops/arena-ui-preview-nginx.conf` 使用 8999，静态目录 `/root/gpufree-data/liugong-ui-preview`，只复用当前 7861 画面/命令、3100 BusAgent 接口。8991/8993 保持原服务。部署本预览不上传或重启 Arena/BusAgent。

运行中的旧后端没有 `/api/workspace` 时，UI 自动读取实际能力信息，允许选择“当前工作台”。三种预设布局仍可查看，但不会谎称已载入；编辑和场景重置显示待接入。旧 BusAgent 只有结果事件时显示事件时刻，仅明确的 execution.started/terminal 显示持续时间，不虚构模型起止时间或因果关系。

主仓新增 `source/mr_liu/arena/workspace.py`，在 `service.py` 的同一个命令队列中处理场景切换/编辑/重置。所有 Isaac 操作在仿真线程执行；HTTP 线程只读快照及排队。编辑通过真实 rigid_objects 的对象 ID 寻址，包括未写入配置的物体，和抓放目标的语义匹配无关。预设是现有物体的可重复布局，不是额外的 USD 场景。尺寸与颜色目前只读。

BusAgent 增加非业务执行追踪：`node.started/updated/completed/failed`，`source_span_id`、`causation_id`；后台模型/动作的异步任务延长 span 生命周期，但不延长交付队列占用时间。Qwen 请求标记慢环；未标注的调用保持灰色。`session.welcome/subscribe` 允许重连继续接收当前会话事件，不重新发送用户任务。断线时未结束的记录标记结果待确认。

合并阶段需要分别合并 BusAgent 和主仓变更，然后与机械臂分支整合 `service.py` 的小范围接口改动。不要拿本工作树的旧 runtime/config 覆盖正在调试的新版资产发现与机械臂控制。部署两个后端需要选择不影响机械臂调试的时机；重置/物体写入与节点遥测尚未在生产进程中启用。

正式切换到 8991 时，除前端 dist 外，需要为 nginx 增加准确的 `location = /api/command` 代理，保留 8993 的只读预览：

```nginx
location = /api/command {
    proxy_pass http://127.0.0.1:7861;
    proxy_read_timeout 15s;
    client_max_body_size 64k;
}
```

## 验证与运行

前端 `pnpm build`、`pnpm test`。后端 `pnpm build`；定向验证使用 `pnpm exec vitest run test/execution-span.spec.ts test/di-graph.spec.ts`。后端基线的全量 typecheck 另有 3 处旧测试类型错误（arena-panda、vision-node、visual-context），与 UI 变更无关；构建及运行测试通过。

Python：`python -m unittest discover -s tests -p 'test_arena_workspace.py'` 及 `test_arena_service.py`。模拟 scene 验证真实注册表发现、初始位置恢复、三种重置范围、预设复现、无效坐标不部分写入、编辑无关物体保留持物上下文。尚未用这一套新接口重置实际云端场景。

浏览器：安装 Python Playwright 后运行 `python scripts/check_ui_workbench.py --url http://127.0.0.1:5176 --browser /path/to/chromium`。它拦截全部 `/api/command` 请求与 WebSocket，不触发真实动作/模型，画面与状态读取已有服务。18 项交互检查及截图输出到 `output/ui-validation`，含 `timeline-fixture.png`（仅测试协议样本，非真实任务证据）。

本地前端：

```sh
cd BusAgent/frontend
BUSAGENT_PROXY_TARGET=http://127.0.0.1:3100 ARENA_PROXY_TARGET=http://127.0.0.1:7861 pnpm dev --host 127.0.0.1 --port 5176
```

预览精度在浏览器 canvas 上实际降采样，不改变模型输入，也不宣称降低服务器渲染负担或网络流量。

补充布局回归：`scripts/check_ui_timeline_layout.py` 使用拦截的 124 个事件、16 个真正并行的 span、四种窗口尺寸及面板拖动，检查最小卡片宽度、轨道/名称对齐、容器填满、固定刻度和文字按钮不裁切。截图与结果同样输出在 `output/ui-validation`。

功能面板回归：`scripts/check_ui_inspector_layout.py` 验证图标侧栏、提示文字与方向键、75% 拖动边界、同一窗口下的左右/上下切换、编辑草稿保留，以及 1024px 窗口内预览工具不溢出。使用拦截的物体与机器人状态，仅画面读取实际服务；浏览和拖动未发出任何控制指令。
