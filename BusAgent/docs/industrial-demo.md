# 工业金属柱取放演示

该模式保留 BusAgent 的真实语音识别、对话、语义决策和任务总线，把底层抓取替换为固定动作。默认在篮子外摆放 **48 根金属柱**，另有 **96 个动态散料**增加工业场景密度。金属柱按固定顺序逐根放入蓝色篮子，每层 24 根，默认码放两层。

这是一种明确的演示执行器：取放柱体使用运动学刚体，机械臂由固定路点的逆运动学解和关节插值驱动，持物通过位置绑定模拟。没有调用 GraspGenX、接触抓取或视觉成功验证；完成状态表示脚本播放完成。散料参与真实物理计算。其他启动方式仍使用原有抓取执行器。

## 启动演示

在已安装 Isaac Sim 的 Windows 机器上，从仓库根目录运行：

```bat
scripts\run_industrial_demo.bat --demo-autostart
```

该命令会自动展示全部取放动作。默认每个阶段 1 秒、每根 8 个阶段，48 根约需 6.4 分钟仿真时间。可用 `--demo-phase-seconds 0.6` 加速播放，或 `--demo-count 24` 缩短展示。数量支持 1–96，位置和篮内层数由固定布局生成。停止/暂停仿真不会继续播放动作。重新启动场景可完整复位。

保留真实语音与决策时，不加 `--demo-autostart`，在 `BusAgent/backend/.env` 中设置：

```dotenv
BUSAGENT_INDUSTRIAL_DEMO=1
```

按原有方式启动后端与前端。控制器仍使用 `127.0.0.1:7861`。说“把金属柱放入篮子”会触发剩余全部柱体的固定码放；“抓取金属柱”播放单根取放；“停止”中断并保持当前位置。该模式的单根抓取也包含放入篮子。当前不支持指定数量、指定某根、其他目标或其他目的地。若中途停止，再次下达取放指令会从未完成柱体继续；完整场景复位请重启。后端关闭演示开关后恢复原有放置限制。

无需 Agent 的手动触发也可通过原控制 API：

```json
POST /api/command
{"skill":"grasp","params":{"target":{"category":"metal cylinder"},"demo_stack":true}}
```

状态由 `/api/status` 的 `grasp` 字段和 `/api/commands/<command_id>` 返回；`demo: true` 标记演示结果。

## 负载与性能

`--demo-clutter N` 独立调整动态散料数量（0–5000），不改变 48 根演示柱体的动作。大量散料会形成堆积；超大数量是压力场景，物体可能滚落桌面。`--demo-no-vision` 关闭视觉模型，适合隔离渲染与物理负载，语音和语义决策仍可使用。默认保留视觉推理。

```bat
scripts\run_industrial_demo.bat --demo-autostart --demo-clutter 500 --test-frames 1800 --demo-no-vision
```

退出后写入 `output/industrial_demo/latest.json`：应用更新频率、更新耗时 P95、采样数、散料数量、视觉开关、物理步长、已码放数量及执行状态。更新频率是应用循环指标，不等同 GPU 渲染 FPS，也不是抓取成功率；固定路点在启动时预求解，统计跳过前 30 次应用更新；中途取消后重新取放可能触发额外路点求解。`--test-frames` 较短时任务可能尚未完成，这是正常的采样结束。

使用普通 Python 可批量启动不同负载，保留各轮日志和独立结果：

```bat
python scripts\benchmark_industrial_demo.py --isaac-python C:\isaac-sim\python.bat --clutter-levels 0 96 240 500 1000 --frames 1800 --no-vision
```

需要先关闭正在运行的控制器以释放 7861 端口。在相同机器、相同窗口/相机/渲染设置下对比，然后去掉 `--no-vision` 再跑一轮。根据实测报告选择演示负载；仓库不预填任何性能数值。
