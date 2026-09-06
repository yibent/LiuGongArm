# Franka 官方抓放方案进展（2026-09-06）

## 目的

本阶段把机械臂切换为 NVIDIA Isaac Sim 官方 Franka Panda，并先验证官方抓放闭环的物理效果，再把结果作为后续 M2T2 和 AnyPlace 集成的基线。官方控制器路径不改写控制器内部状态机，也不叠加项目自定义的释放门控。

## 当前实现

运行入口是 scripts/run_official_franka_pick_place.py（PowerShell 封装为 scripts/run_official_franka_pick_place.ps1）。它直接使用：

- Isaac Sim 官方 FrankaPickPlace 状态机；
- 官方 Franka RMPFlowController；
- 仿真物理中的夹爪闭合、搬运和释放；
- 结束后的物体实际位姿、目标误差和稳定性记录。

项目侧增加了可配置 Franka profile（configs/robot_franka.yaml）和本地 URDF/XRDF/RMPFlow 资产（assets/robots/franka/）。切换机械臂后，场景、工具坐标系和相机也同步调整：

- Franka 基座：/World/Franka；
- 末端抓取坐标系：grasp_center；
- 放置观察相机：/World/Cameras/PlacementRGB，位于 Franka 工作区前上方；
- 腕部 RGB-D 相机：挂载到 /World/Franka/panda_hand/WristRGBD；
- 旋转工件的偏航角作为显式输入传给官方控制器包装层。

## 已验证结果

配置文件 configs/eval_official_franka_pick_place_v1.json 定义了 12 个基础和扰动场景，覆盖：

- 对称、镜像、跨工作区和短距离搬运；
- 小、中、大、扁平和高矩形工件；
- 45° 旋转工件；
- 抬高平台放置；
- 120 g 较重工件；
- 工作区边缘位置。

在最终物体中心误差不超过 10 mm 的判定下，批次 output/franka/official_reliability_20260906_b 和 output/franka/official_reliability_20260906_c 合计完成 **12/12** 次物理抓起、搬运和放下。最终中心误差范围为 1.5–6.2 mm；旋转工件在显式传入 45° 末端偏航后误差为 1.72 mm。未传入偏航时，同一类工件曾出现约 39.8 mm 的放置偏差，说明姿态必须成为任务输入。

## 工业场景和视频

工业多物体场景在执行目标旁保留了 6 个额外动态工件，包括多个矩形件和圆柱。已录制并检查以下视频：

- 标准抓放：output/franka/official_videos_20260906_a/01_standard_b/official_pick_place.mp4
- 45° 旋转矩形件：output/franka/official_videos_20260906_a/02_rotated_block/official_pick_place.mp4
- 工业多物体场景：output/franka/official_videos_20260906_a/03_industrial/official_pick_place.mp4
- 接近阶段移动目标：output/franka/official_videos_20260906_a/04_target_moved_during_approach/official_pick_place.mp4
- 闭爪前移动目标：output/franka/official_videos_20260906_a/05_target_moved_before_close_b/official_pick_place.mp4

output/ 按仓库规则被忽略，因此视频和逐帧报告保留在本地实验目录，不随代码提交。复现实验后可在相同目录生成视频。

## 目标被挪动时的行为

故障注入把立方体移动 80 mm 并停止其速度：

- **接近阶段移动**：官方控制器每周期读取当前目标位置，能够跟随移动后的目标并完成放置；一次记录的最终误差为 2.74 mm。
- **闭爪前短暂移动**：官方状态机已经锁定了抓取 XY，仍会完成十个阶段，但夹爪落在旧位置，物体未被抓起，物理成功判定为失败。

因此，状态机完成事件不能单独当作成功信号。后续工业层需要持续检查目标身份和位移；若目标在接近后丢失或移动，应重新观测并重启官方控制器。

## 形状边界

紧凑的动态圆柱（直径 50 mm、高 50 mm）已成功完成抓放，记录误差 3.02 mm，证据目录为 output/franka/official_shapes_20260906_b/cylinder_compact。自由球体在释放后会滚动，过高圆柱也未能稳定捕获；这两类物体需要专用夹取姿态、支撑夹具或形状化目标高度配置，不能直接沿用立方体参数。

## 可复现命令

基础运行：

    .\scripts\run_official_franka_pick_place.ps1 -Output output\franka\official_pick_place_local

录制视频：

    .\scripts\run_official_franka_pick_place.ps1 -Output output\franka\official_videos_local -RecordVideo

批量验证：

    .\_envs\anyplace\Scripts\python.exe scripts\benchmark_official_franka_pick_place.py --manifest configs\eval_official_franka_pick_place_v1.json --output output\franka\official_reliability_local

## 验证和限制

本阶段对官方脚本执行 Python 编译检查，并运行与 Franka profile、相机配置、几何和放置契约相关的单元测试。Isaac Sim 运行依赖本机安装的 Isaac Sim 6.0 和对应资产；没有在文档提交中包含引擎安装、模型权重或大型视频文件。

下一阶段继续沿用同一物理成功判定，分别增加：

1. M2T2 全程夹住并放下的重复实验；
2. AnyPlace 的平面放置、插入、悬挂/挂钩任务；
3. 更多目标坐标、遮挡、杂物和目标移动扰动；
4. 目标重新识别、重规划和失败恢复；
5. 在算力不足时把独立批次分配到开发机执行，汇总报告后再比较策略。

当前结论是：官方 Franka 基线已经在矩形工件矩阵上达到 12/12 的物理成功，但工业级通用性、自由曲面物体和移动目标恢复仍需继续实验。
