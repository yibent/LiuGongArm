# FineGrasp 双 RGB-D 开发基线

状态：开发检查点，不是已通过泛化验收的发布版本。创建于 2026-09-05，
基于 `e402a79`，保留既有提交历史和实验代码，不回滚或改写历史。

## 本次边界

- 仿真头顶相机由 RGB 改为 RGB-D，保留 `/World/Cameras/TableTopRGB` 路径。
- 两路深度均为 `distance_to_image_plane`，单位米，不是相机到点的斜距。
- `SceneCamera.rgbd_frame()` 从同一个渲染采集回调复制 RGB、深度、渲染位姿和时间；
  位姿是 `T_world_camera`，采用 OpenCV 光学坐标系，不能在真机上直接当作标定结果。
- 默认 `fusion.enabled=false`、`active_views.enabled=false`。FineGrasp 仍以腕部相机
  为控制输入，持续重新观察和小步伺服；头顶深度尚不参与抓取决策。
- 历史提交中已有 `--perception multiview` 原型；保留为显式实验选项，
  不将它称为已完成的“失败后左顾右盼＋双相机协作”版本。

## 运行

本机依赖已安装时，在仓库根目录运行基础抓取 demo：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_fine_grasp_demo.ps1 -Backend graspgenx -Perception single -Segmenter depth
```

只验证两路 RGB-D，不执行抓取：

```powershell
& D:\isaac\env_isaacsim60\python.exe scripts\verify_cameras.py
```

这些路径对应当前开发电脑。真机 RGB 摄像头不能通过修改 YAML 变成 RGB-D 硬件；
真实部署需确认头顶传感器有深度输出、完成外参标定并接入其驱动。

## 本次实际验证

- `python -m unittest discover -s tests`：82 项通过。包含 RGB-D 同回调采集、
  SDK 缓冲区复制、Isaac 6 ReferenceTime 兼容，以及基础模式不启用主动观察。
- Isaac Sim 双相机验证通过，退出码 0：两路均为 640×480；头顶有效深度
  307,200 像素，当前腕部视角有效深度 144,000 像素。两路采集标识及时间均更新。
  头顶中心深度为 1.25212 m，相机世界高度为 2.30000 m，与桌面位置一致。
- 同一提交前代码运行基础 GraspGenX cube demo：**节点失败，不能算完整抓取成功**。
  独立仿真刚体读数显示抬升 80.24 mm；模型推理 641 ms，20 次伺服，节点耗时
  13.281 s。腕部视觉估计抬升 106.01 mm、随动误差 27.06 mm，超过 25 mm
  验证阈值，返回 `lift_verification_failed / target_did_not_follow_wrist`。
  本次没有放宽阈值，也没有用仿真真值替代节点判断。
- 修正 Isaac 6 快速关闭时掩盖失败退出码的问题；上述 demo 正确返回退出码 2。
  这些单次检查不构成泛化或双相机性能增益证据。

本机原始报告（`output/` 不进入 Git）：

- `output/camera_verify/baseline_dual_rgbd_20260905_v2/report.json`，
  SHA256 `9bf8c3b4b149ac1499ed9dd5b707118656263fd29fee62a8eac1a9bfa3fd388d`。
- `output/baseline_dual_rgbd/cube_20260905/report.json`，
  SHA256 `d7ea0523ede0b55663940fbe43d2f1121acdd1dc6d9e87134f30b215fff7fcc6`。
- 首轮相机验证因 Isaac 6 将整数帧号改为 ReferenceTime 字典而失败，记录保留在
  `output/camera_verify/baseline_dual_rgbd_20260905/failure.json`；修复后才有上述通过结果。

RGB-D NPZ 在 Isaac 6 中保存 `render_reference=[numerator, denominator]`，
`rendering_frame=-1` 表示不存在旧版整数帧号；旧 SDK 则相反。
两个摄像头各自同步采集不代表它们彼此同时曝光，后续融合仍需时间门控。

## 后续优化单独提交

1. 先区分空抓、滑落、目标移动和因遮挡而无法确认抓取，避免误松爪。
2. 安全退让；头顶相机重新定位目标，腕部选择能补充缺失几何的安全视点。
3. 检查两路时间、外参和身份一致性，目标移动后废弃旧局部几何。
4. 重新生成抓取候选并排除已失败方案，回到最新腕部观测下的闭环执行。
5. 限制重试次数/总时长，分别报告首次成功率、重试后成功率、误报和耗时；
   在相同冻结对象/姿态/种子上比较基础与优化版本。

目前仍有接触几何近似、局部跟踪和遮挡验证问题；此前的代理物体成绩不能代替
工业工具、咖啡杯、水果等真实 unseen-object 测试。先保存本地基线提交；
GitHub 推送仍遵守用户此前要求的泛化证据门槛。
