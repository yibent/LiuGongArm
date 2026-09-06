# 刘工智能工作台

新版 UI 原先在独立工作树 `LiuGongArm-ui` 开发，现已合并到主开发分支 `codex/arena-panda-grasp-place`。保留视觉实例执行、持物续接、工业场景与新版手动控制，BusAgent 子模块固定对应集成版本。

## 界面与品牌

- React + shadcn/Radix + Tailwind 深色工作台，场景、仿真两页。先选择场景，再进入仿真。
- 使用用户提供的刘工智能 Logo 原始透明 PNG，未重绘。替换页眉标记、网页图标、移动端图标、语音球、回答头像、关于对话框、README 与独立仿真预览页。原图位于 `assets/brand/liugong-logo.png`，前端公开资源位于 `BusAgent/frontend/public/brand/`。
- 功能面板采用左侧纯图标栏，悬停显示名称，支持上下方向键切换。面板最多占上半区约 75%；根据面板自身宽度切换上下/左右布局。物体列表有独立滚动，窄面板优先显示机械臂操作区。
- 同页显示工作区、侧视、腕部三路实时预览；工作区最大，两路小画面等大。支持单路/三路、完整/半/四分之一显示采样、全屏。预览变窄时工具栏换行、两路小画面移到主画面下方。显示采样不改变视觉模型输入或服务器渲染精度。
- 四条 BusAgent 执行轨道：信息、视觉、动作、其他。时间轴只读，按调用顺序非等比显示，默认最小卡片宽 140px，单个运行节点一分钟约增长 40px，真实耗时在详情显示。共享滚动容器，轨道名称与刻度固定；显示真实并行、快绿慢蓝和明确的因果端点关系。
- 悬浮语音球仅保留文字输入快捷入口，回答气泡渐隐。对话在功能面板中查看。图片不进入语言模型上下文。
- 时间轴仅保留在本页内存，提供手动清空，没有归档。重置场景不会清空时间轴。

## 已接入真实 Isaac 的操作

工作台编辑不再依赖只读能力列表，`GET /api/workspace` 从场景中的真实 rigid body 注册表提供物体句柄、实时位姿、控制参数和机械臂状态。新增物体会自动进入编辑列表。物体 ID 仅用于用户指定的编辑操作，不是自然语言抓放任务的名称白名单。

- **物体**：位置/旋转数值编辑、单击十字键按步长微调、长按与平面摇杆连续移动。尺寸/颜色保持只读。
- **机械臂**：末端位置/姿态编辑，世界或工具坐标系下的平移和旋转，XY/XZ/YZ 平面选择，剩余轴单独增减，十字键、摇杆，夹爪开合，回到待机位姿。
- **重置**：仅物体、仅机械臂、全部。恢复所选布局的初始位置和机械臂初始关节状态；与“回到待机位姿”是不同操作。重置先中断当前执行，并等待队列中的动作结束。
- **场景**：初始布局、桌面整理、精确摆放，均为当前场景刚体的可重复布局。固定支撑在预设重排时保持原位。
- **参数**：位置容差、旋转容差即时应用。实时刷新不会覆盖尚未应用的物体/末端/参数输入；提供读取当前位置的入口。

手动操作直接进入仿真控制接口，不绕行大模型。机械臂沿用 Arena 的 IK action 和已有 `runtime.move/tick`；没有增加第二套 IK 求解器。所有物理写入都在仿真线程执行，HTTP 线程只更新输入或排队。

`workspace` 命令支持 `object`、`robot`、`jog`、`teleop`、`gripper`、`controller`、`reset`、`scene`。持续移动使用带命令 ID 的 `POST /api/teleop` 最新输入邮箱；旧输入不排队，松手、窗口失焦、切换功能或 600ms 未续约都会停止。停止只作用于所属手势，迟到的数据包不能重新启动已停止的手势。无法跟随的 IK 目标会停止并报告原因；不伪报到位。

Isaac Lab 3 的状态通过 `numpy_data` 读取 Warp 数组，写入采用实际安装版本的 `write_root_pose_to_sim_index`、`write_root_velocity_to_sim_index`、`write_joint_state_to_sim_mask` 和 `set_joint_position_target_index`。采用 XYZW 四元数。接口依据安装源码核对，并参考[官方 3.0 迁移说明](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/docs/source/migration/migrating_to_isaaclab_3-0.rst)。布局修改会失效旧的几何观测缓存，但保留模型记忆；编辑无关物体不会清除当前持物信息。

## 部署

新版工作台主入口：https://8byr7v21-8qne20ga-8991.zj02restapi.gpufree.cn:8443/ 。8999 同步同一版本，8993 保持独立观察页。

8999 的 nginx 配置为 `ops/arena-ui-preview-nginx.conf`，静态目录 `/root/gpufree-data/liugong-ui-preview`，复用 7861 仿真 HTTP 和 3100 BusAgent WebSocket。8993 保持只读预览。

2026-09-06 已在工业场景运行版本上部署 `service.py`、`workspace.py`、`teleop.py`，核对原始源码 SHA 后仅重启 Arena 载入接口，保留当前工业场景配置。未覆盖运行中的 runtime、模型、视觉服务或 BusAgent 后端。原后端文件及状态备份在服务器 `output/ui-control-release/backup-20260906-232704`；部署前静态资源备份为 `/root/gpufree-data/liugong-ui-before-controls-20260906.tar.gz`。

2026-09-07 已部署合并后的 BusAgent 后端，启用 `node.started/updated/completed/failed` 的实际调用追踪和独立持物放置。手动编辑仍直接调用仿真接口。机械臂面板显示持物标签、放置结果和距所选空位的误差；时间轴接收真实语言、视觉、执行事件。

真实网页联调使用 `scripts/check_ui_held_placement.py`，不拦截 WebSocket、不直接调用抓放 API、不重置中间场景。验证“拿起齿轮”后单独“在桌子上随便找个地方放下”，以及再次拿起后“把手里的齿轮放到蓝色托盘”；两次续放均保留原抓取，并通过接触、松爪、退离和稳定性评测。具体命令、耗时、成功及失败记录见 [持物续放网页验证](../arena/workbench-held-placement-validation.json)。

联调时云端 Qwen 返回 HTTP 400 `Arrearage`（欠费停用），TTS 同时不可用。已补齐常见中文物体/颜色提示的本地解析回退，未知标签不会映射成配置资产；任意复杂语义理解仍依赖恢复模型服务，不能把本地回退宣称为完整大模型能力。

## 验证

- 前端 `pnpm build`、`pnpm test`：11 项时间轴单测通过。
- `python -m unittest discover -s tests -p 'test_arena_workspace.py'`：14 项，通过 Warp 数组兼容、真实注册表发现、不同重置范围、工具坐标变换、增量移动、持物缓存、停止和输入租约。
- `test_arena_service.py`：5 项命令队列测试通过，覆盖中断、幂等、进程重启与并发互斥。
- `scripts/check_ui_workbench.py`：18 项页面交互检查；`check_ui_timeline_layout.py` 覆盖 124 个事件、16 个并行节点和四种窗口；`check_ui_inspector_layout.py` 验证图标侧栏与宽窄布局；`check_ui_manual_controls.py` 覆盖点按、长按、摇杆、松手、失焦、键盘、旋转/工具坐标、输入保留。以上脚本拦截所有控制写入。
- `scripts/check_arena_workspace.py` **执行真实仿真写入**：14 项物理检查通过，包含物体平移/旋转/摇杆，机器人 IK 移动、摇杆松手与断连停止、夹爪、参数、三类重置和三种布局。机械臂摇杆松手后的实测漂移约 0.17mm。结束恢复初始布局。证据在 `output/ui-control-validation/`。
- `scripts/check_ui_live_controls.py` **从真实网页触发动作**：公网的 Logo/三路画面、物体十字键、Panda 十字键/旋转/摇杆、夹爪实际开度及最终场景恢复均通过。切换物体导致控件重复的组件标识问题已修正，并加入双物体切换回归。可提交的测量摘要见 `docs/ui/control-validation.json`。完整证据与截图在 `output/ui-validation/live-controls.json`、`live-object-controls.png`、`live-robot-controls.png`。

本地前端：

```sh
cd BusAgent/frontend
BUSAGENT_PROXY_TARGET=http://127.0.0.1:3100 ARENA_PROXY_TARGET=http://127.0.0.1:7861 pnpm dev --host 127.0.0.1 --port 5176
```

浏览器脚本使用 Python Playwright，可指定 `--url` 与 `--browser /path/to/chromium`。请勿把真实操作脚本当作只读检查运行。
