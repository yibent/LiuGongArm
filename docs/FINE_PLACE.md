# Florence + RGB-D 非快环放置：开发说明

分支：`codex/florence-fine-place`，基于已提交的 `codex/label-to-grasp`。

## 状态与范围

这是开发中的 FinePlace 节点，不是已经验收的通用桌面整理系统。当前核心状态机、安全失败单元测试已运行；Isaac 端正在联调。没有真机放置测试，也没有工业工具、水果、杯子放置泛化成功率。

不改已有抓取默认入口，不自动启用 Web `/api/command` 的 `place`；该接口继续报告放置 unsupported，直到 live assembly 和验收完成。调用方可以直接注入相机、规划器、夹爪适配器使用 `FinePlaceSkill`。

本节点不做远距离运输。上游需要先抓稳物体、将末端安全移动到目的地附近；请求终态末端位置与进入节点时末端位置的三维欧氏距离超过 12 cm 返回 `requires_fast_loop_handoff`。中间步骤也不能离开以初始末端为中心的 12 cm 球；不是允许任意长度的运输路径。演示为了获得真实持物，先执行已验证的标签粗接近与 FineGrasp，再进入放置。

## 一条命令（当前电脑）

在 `D:\liuGong` 的 PowerShell 中：

```powershell
.\scripts\run_fine_place.ps1 -Destination 'blue square' -RecordVideo
```

这会启动本地 Florence 服务、GraspGenX 服务和无头 Isaac，创建一个红色方块与蓝色放置区，实际抓取验证后调用放置节点。不是凭空生成任意标签对应物体。可选 `-NoHeadless`。依赖现有 `_envs/vision`、`_envs/graspgenx`、`_models` 和 `D:\isaac\env_isaacsim60`；参考 `LOCAL_VISION_SETUP.md` 与 `GRASPGENX_SERVER.md`。权重、环境、视频与输出不进 Git。

容器联调命令（尚不代表验收通过）：

```powershell
.\scripts\run_fine_place.ps1 -Destination 'blue tray' -Fixture tray -Relation inside
```

每次输出到新的 `output/fine_place/<timestamp>`。指定 `-Output` 时必须使用空/新目录。`report.json` 是前置抓取报告，`place_report.json` 是独立放置结果；抓取成功不等于放置成功。`place/` 保存实际 RGB-D、每帧坐标链和事件。`console.log`、`console_stderr.log` 与 `locator_*` 保存运行日志。非零退出表示运行错误或任务未成功。启动器仅清理自己启动的服务。

## 感知与执行

- Florence：使用同一原生本地权重，新增 describe（brief/detailed/more/regions）和 REGION_TO_SEGMENTATION。现有检测接口保持兼容。服务 `/locate` 的 `segment=true, describe=true, mode=florence` 返回检测框、区域多边形和描述，携带输入帧序号；描述不是运动命令。
- 固定 RGB-D：先定位唯一目的地实例；多实例不擅自选择。慢模型完成后必须获取新帧，通过真实 LK 光流和新深度更新，禁止直接执行模型旧帧。
- 几何：只接受实际观测到的水平支撑面。采样整个物体占地加边界余量，深度缺失不当作空位。inside 当前采用四个方向可见抬高边界的保守托盘检查，不支持任意形状/隐藏容器内部。
- 腕部 RGB-D：优先跟踪持物的局部点云和夹持偏移；不可观测时尝试固定相机。每帧使用自身 `T_base_camera`，腕部由 `T_base_ee @ T_ee_camera` 得到；不得翻转 RGB 而不调整深度/标定。
- 精放：持物核验 → 目的地识别 → 支撑位置选择 → 上方对齐 → 每步最多 6 mm 闭环下降 → 开爪与退出路径安全检查（尚未释放） → 再观察 → 慢开爪 → 分步退出 → 外部视觉多帧稳定验证。
- 默认释放底面间隙目标 3 mm、容差 2 mm；这是低落差释放，不是有六维力传感器的接触/力控放置。传感器不确定性和末端偏差仍可能导致失败。

## BusAgent 插件接口

```python
from mr_liu.place import GeneralPlaceNode, PlaceRequest, HeldObject
from mr_liu.control.skills.fine_place import FinePlaceSkill

node = GeneralPlaceNode(perception=perception, motion=motion, gripper=gripper)
skill = FinePlaceSkill(node)
request = PlaceRequest(destination_label="blue tray", relation="inside")
result = skill.execute(request, held_object)
```

`held_object` 必须是 `HeldObject`，包括物体 ID、经过核验的 `T_ee_object`、物体坐标系下半尺寸/参考点云、外观锚点、最近持物核验时间和不确定性。`stable_base_verified` 默认 false：不能凭“看起来像杯子”或 caption 猜它有稳定支撑。没有依据就拒绝执行。

`T_A_B` 是把 B 中坐标变换到 A 的 4×4 齐次矩阵，平移单位米。当前几何要求 base 的 +Z 与重力反向，支撑面近似水平；Isaac 中 base 实际沿用世界米制坐标，不是未经转换的机器人根坐标。物体原点为其保守包围盒中心，轴和半尺寸一致；三维半尺寸与至少 32 个参考点都必需。外观锚点是 RGB 各通道除以总和后的三维中位数。

`verified_at_s` 使用与节点一致的单调时钟秒数，进入节点时年龄不得超过 3 秒；每个相机观测不得超过 350 ms 且序号必须递增。`uncertainty_m` 是米制几何误差余量，当前允许 (0,5 mm]。稳定底面由上游基于可观测支撑几何/已验证物体模型提供，初期 demo 的方块声明是测试夹具先验，不能推广到未观察到物体底面的真机输入。

`PlaceRequest.relation` 支持 `on / inside / relative`。relative 要给 `offset_base_m=(dx,dy)`，单位米，采用已标定 base 坐标系、参照物观测中心到放置物中心的距离；不是腕部图像左右，也不是物体边缘间距。上游负责解释用户方向并处理歧义。relative 若指定点不满足支撑条件，不能偷偷换到旁边。

`PlacePerception`：`destination(request)` 返回同帧几何/目的地区域；`payload(held, expected, released=...)` 必须返回实际视觉证据，而不是原样返回预测值。`PlaceMotion`：实现 `robot_state / move_checked / opening_safe / stop`，需要检查整条实际路径、持物几何、松爪和退出空间。纯 IK 成功不符合这个接口。协议见 `source/mr_liu/place/contracts.py`。

失败前未开爪时保持夹持，不自动回家/重新抓取。`released=true` 表示开爪可能已开始，包括开爪执行失败；不能据此认定物体仍在手里。只有 `success=true, verified=true` 才表示本次视觉验收通过。`dry_run` 返回 phase=planned、success=false，不授予运动许可。cancel 是锁存状态；取消后重试创建新节点，重新确认持物与观测。

释放后失败时，上游应将物体标记为状态未确认，重新观察物体、夹爪和环境；不能携带旧 HeldObject 直接重试。当前成功结果同时具有 released=true 和 verified=true，失败不返回 verified=true。视觉验收默认至少 5 帧、跨越至少 0.35 秒，平移漂移不超过 4 mm，物体中心 XY 距离目标不超过 10 mm，底面到支撑面的误差不超过 6 mm；这不等价于六维姿态精度验收。

## 当前限制，不能隐藏

1. Isaac 适配器只允许小型、稳定底面的物体处于受限指尖空间，暂不支持长工具的完整附着物自碰撞建模。
2. 当前持物视觉配准是平移配准，不是完整可靠的 6D 姿态跟踪；因此不能宣称可精准直立放杯子或定向插槽。演示仅对方块 fixture 显式声明稳定底面，真实环境需要上游提供依据。
3. 已测支撑点可在同一固定相机目标身份和可见支撑一致时保留，以处理持物遮挡；被完全遮挡的变化无法保证检测到。需要继续完善占据图、主动观察和隐藏空间保守处理。
4. 当前手指模型是已有标定盒近似；SO-101 是旋转夹爪，完整扫掠包络还需进一步对照 USD/真机校准。
5. 目的地丢失或明显移动返回失败，重新定位通过新请求进行；本版不会持物横向主动扫描或盲目重试。
6. 相对位置和宽口容器有核心几何支持，但未完成实际成功率验收。桌面整理规则、任务编排、live Web 接入不属于本次已完成能力。

## 开发验证

```powershell
& 'D:\isaac\env_isaacsim60\python.exe' -m unittest discover -s tests -p test_fine_place.py -v
& 'D:\isaac\env_isaacsim60\python.exe' -m unittest discover -s tests -v
```

新增测试覆盖：闭环状态转换、多帧验证、观测过期/重复、目的地移动、持物滑移、无支撑/深度缺失、开爪碰撞、开爪执行失败、松爪后位置错误、取消与参数校验。它们使用可控测试适配器，不是物理抓放成功率。

2026-09-05 当前测试结果：新增 23 项、全套 265 项通过，日志在本地 `output/fine_place/unit_tests.log`。目前尚无成功放置验收记录；03 已能建立场景，但旧固定关节的抓取起点在合并后的机器人根位姿下未看到目标，未闭爪。启动器现改用已通过的标签粗接近进行前置抓取。

初始 Isaac 调试：01 因非均匀缩放碰撞盒被 cuMotion 拒绝；02 因网格缺少显式 scale 属性被 SDK 拒绝。均发生在建立规划器时，没有执行放置。修复使用尺寸已烘焙的网格与显式单位缩放；保留碰撞体及失败记录。
