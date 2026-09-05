# Florence + RGB-D 非快环放置：开发说明

当前分支：`codex/anyplace-placement`；原始开发线为 `codex/florence-fine-place`。默认放置保持几何后端，AnyPlace 是显式选择的实验后端，见 [AnyPlace 记录](ANYPLACE_PLACEMENT.md)。

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

RGB-D 文件使用最多 4 个待写帧的后台队列，退出前等待保存完成；队列拥塞不会刷新观测时间戳。写盘错误记入 `recording_errors`，不改写已经发生的松爪状态。固定相机没有 EE 外参的字段保存为空数组，不需要 `allow_pickle=True`。

## 感知与执行

- Florence：使用同一原生本地权重，新增 describe（brief/detailed/more/regions）和 REGION_TO_SEGMENTATION。现有检测接口保持兼容。服务 `/locate` 的 `segment=true, describe=true, mode=florence` 返回检测框、区域多边形和描述，携带输入帧序号；描述不是运动命令。
- 固定 RGB-D：先定位唯一目的地实例；多实例不擅自选择。慢模型完成后必须获取新帧，通过真实 LK 光流和新深度更新，禁止直接执行模型旧帧。
- 几何：只接受实际观测到的水平支撑面。采样整个物体占地加边界余量，深度缺失不当作空位。inside 当前采用四个方向可见抬高边界的保守托盘检查，不支持任意形状/隐藏容器内部。
- 腕部 RGB-D：优先跟踪持物的局部点云和夹持偏移；不可观测时尝试固定相机。每帧使用自身 `T_base_camera`，腕部由 `T_base_ee @ T_ee_camera` 得到；不得翻转 RGB 而不调整深度/标定。
- 持物轮廓辅助：Isaac demo 现复用本地 SAM2.1 Hiera Tiny（`_models/sam2/sam2.1_hiera_tiny.pt`），通过已有 GraspGenX 进程的分割 RPC 补全颜色种子遗漏的轮廓。Florence 仍负责目的地；SAM2 不产生位姿或动作。核心 `IsaacPlaceMotion` 可以不注入 refiner，但颜色版曾在真实仿真联调中连续遭遇边缘误报，不能认为两者验收等价。掩码必须对应同一帧、覆盖种子、满足范围/实例/分数检查，不能吞掉支撑面；预热图像的掩码被丢弃，不用于控制。
- 精放：持物核验 → 目的地识别 → 支撑位置选择 → 上方对齐 → 每步最多 6 mm 闭环下降 → 开爪与退出路径安全检查（尚未释放） → 再观察 → 慢开爪 → 分步退出 → 外部视觉多帧稳定验证。
- 小内存主机按顺序加载模型：Florence 目的地查询结束释放权重，SAM2 延迟到首次路径预检加载。冷预检的旧帧不授予运动许可，之后仍重新采集并在 350 ms 门限内检查。模型加载/服务错误返回 `destination_model_unavailable`，不能退化为使用硬编码目标坐标。
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

`PlaceRequest.relation` 支持 `on / inside / relative`。relative 要给 `offset_base_m=(dx,dy)`，单位米，表示沿已标定 base X、Y 轴的带符号位移分量：`放置中心_xy = 参照物观测中心_xy + (dx,dy)`；不是腕部图像左右，也不是物体边缘间距。上游负责解释用户方向并处理歧义。relative 若指定点不满足支撑条件，不能偷偷换到旁边。

`PlacePerception`：`destination(request)` 返回同帧几何/目的地区域；`payload(held, expected, released=...)` 必须返回实际视觉证据，而不是原样返回预测值。`PlaceMotion`：实现 `robot_state / move_checked / opening_safe / retreat_target / stop`；`retreat_target()` 返回最近一次成功开爪预检所保留的退出目标，不能临时改成未经检查的方向。需要检查整条实际路径、持物几何、松爪和退出空间。纯 IK 成功不符合这个接口。协议见 `source/mr_liu/place/contracts.py`。

失败前未开爪时保持夹持，不自动回家/重新抓取。`released=true` 表示开爪可能已开始，包括开爪执行失败；不能据此认定物体仍在手里。只有 `success=true, verified=true` 才表示本次视觉验收通过。`dry_run` 返回 phase=planned、success=false，不授予运动许可。cancel 是锁存状态；取消后重试创建新节点，重新确认持物与观测。

释放后失败时，上游应将物体标记为状态未确认，重新观察物体、夹爪和环境；不能携带旧 HeldObject 直接重试。当前成功结果同时具有 released=true 和 verified=true，失败不返回 verified=true。视觉验收默认至少 5 帧、跨越至少 0.35 秒，窗口中各帧物体中心相对最后一帧的三维距离最大值不超过 4 mm，物体中心 XY 距离目标不超过 10 mm，底面到支撑面的误差不超过 6 mm；这不等价于六维姿态精度验收。

## 当前限制，不能隐藏

1. Isaac 适配器只允许小型、稳定底面的物体处于受限指尖空间，暂不支持长工具的完整附着物自碰撞建模。
2. 当前持物视觉配准是平移配准，不是完整可靠的 6D 姿态跟踪；因此不能宣称可精准直立放杯子或定向插槽。演示仅对方块 fixture 显式声明稳定底面，真实环境需要上游提供依据。
3. 已测支撑点可在同一固定相机目标身份和可见支撑一致时保留，以处理持物遮挡；被完全遮挡的变化无法保证检测到。需要继续完善占据图、主动观察和隐藏空间保守处理。
4. 当前手指模型是已有标定盒近似；SO-101 是旋转夹爪，完整扫掠包络还需进一步对照 USD/真机校准。
5. 目的地丢失或明显移动返回失败，重新定位通过新请求进行；本版不会持物横向主动扫描或盲目重试。
6. 相对位置和宽口容器有核心几何支持，但未完成实际成功率验收。桌面整理规则、任务编排、live Web 接入不属于本次已完成能力。
7. 当前颜色锚点仅用于提供持物分割种子，不能保证多色、同色近邻或严重遮挡物体身份可靠；SAM2 分数不是抓放成功概率。
8. 外部点云碰撞是观测表面的补充检查，不是对全部未知空间的安全证明。深度一致性过滤要求 3 个相邻有效样本（8 mm 内），不插值；单像素/薄结构不能据此得到可放认证。机器人轮廓有 1 像素的注册余量，实际真机需重新标定。官方也说明了深度传感器的[抗锯齿/重投影离群点处理](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.sensors.experimental.rtx/docs/api.html)，但本实现是本地显式置信度过滤，不冒充 NVIDIA 的硬件深度传感器仿真。
9. Isaac 的自屏蔽使用已知机器人资产的 instance mask，不读取目标/目的地的实例真值。真机需要关节状态与机器人模型投影或经过标定的等效自屏蔽实现。

## 开发验证

```powershell
& 'D:\isaac\env_isaacsim60\python.exe' -m unittest discover -s tests -p test_fine_place.py -v
& 'D:\isaac\env_isaacsim60\python.exe' -m unittest discover -s tests -v
```

新增测试覆盖：闭环状态转换、多帧验证、观测过期/重复、目的地移动、持物滑移、无支撑/深度缺失、开爪碰撞、开爪执行失败、松爪后位置错误、取消与参数校验。它们使用可控测试适配器，不是物理抓放成功率。

历史基线（2026-09-05 原 Florence 放置线）：全套 288 项通过，其中放置专项 46 项；AnyPlace 接入及几何修复提交 `6b6bb37` 为全套 301 项通过。尚无成功放置验收记录；不能把下面的逐轮修复统计成成功率。

| 实际运行目录 | 观察到的结果 | 修正 |
|---|---|---|
| `01_region` / `02_region` | cuMotion 建模失败，未抓放 | 网格烘焙尺寸、显式单位 scale，保留碰撞体 |
| `03_region` | 旧固定抓取起点看不到目标，未闭爪 | 使用原标签粗接近流程准备真实持物 |
| `04_region` | 抓取成功，放置持物交接失败 | 已转换到 EE 的抓取位姿不能再次套用夹爪偏移 |
| `05_region` | Florence 识别/分割和支撑估计通过；等待时物体掉落 | 旧 `stop()` 把夹爪堵转位置重新设为目标，消除预紧力；现在只停止臂关节，保持独立夹爪目标 |
| `06_hold_preload` | 物体保持悬空；首步支撑判定失败 | 选位置和执行检查统一使用旋转后的物体占地 |
| `07_rotated_footprint` / `08_payload_mask` | 可达 IK 通过，观测碰撞拒绝，未释放 | 分析发现初始单视角参考云/单一色度会遗漏持物新可见面或背光面；正在验证空间与实例约束下的明暗稳健关联 |
| `09_shading` | 色相关联仍留下混合边缘像素，未移动或释放 | 色度/色相联合关联；只允许 2 像素、5 mm 三维邻接的边缘关联，保护支撑面；这是需要真机标定的误差模型，不是任意邻近空间都可排除 |
| `10_registered_edges` | 前置抓取 IK/路径筛选耗尽 102 s，未闭爪，未进入放置 | 保留为端到端失败；不计入放置成功或修改抓取安全条件 |
| `11_frozen_repeat` / `12_self_boundary` | 实际持物小步移动 2 次 / 6 次后遭观测碰撞拒绝，未释放 | 12 中移动约 3.4 cm；自屏蔽边界与持物轮廓仍需区分 |
| `13_measured_attachment` | 改用夹稳后、抬升前的实测 EE 姿态交接，仍被轮廓点拒绝 | 不再用模型提出的 EE 姿态冒充实测姿态；离线 SAM2 掩码覆盖遗漏像素，回放端点碰撞数为 0（不是物理成功） |
| `14_sam2_contour` | 旧参考图预热结果被误用为当前掩码验收，交接拒绝，未释放 | 预热和实时分割分开；旧掩码彻底丢弃 |
| `15_live_sam2` | 实时分割与碰撞检查通过，但新观测在检查结束时过期，未移动/释放 | SAM2 约 62–94 ms；保持 350 ms 时效门限，继续剖析其余延迟 |
| `16_profile_path` | 新观测检查结束时年龄 391 ms，未移动/释放 | 检查 266 ms、检查前同步保存/处理约 125 ms。改用有界异步保存、精确包围球初筛及相邻姿态缓存；初筛与原全量碰撞计算有一致性测试 |
| `17_realtime_path` | 21 次新观测迭代完成对齐和下降，松爪前的退出路径检查拒绝；仍未释放 | 视觉估计 XY 误差 0.45 mm、底面间隙 3.10 mm；独立仿真真值 XY 偏差约 3.19 mm，不能把视觉残差当作物理精度。检查发现两个持物点仍与固定手指表面接触 |
| `18_release_contact` | 再次 21 次迭代到位，退出预检仍拒绝，未释放 | 已有接触豁免不足以允许后续压入。离线回放发现略倾斜手指按世界竖直退回会刮入；工具轴退回无直线体积冲突，继续真实 FK 路径测试 |
| `19_tool_axis_retreat` | 前置抓取成功；Florence 重新加载遇到 Windows error 1455（内存提交不足），尚未进入放置运动 | 去掉 Florence 之前的 SAM2 预加载，改成串行加载；不修改系统页面文件，也不杀无关用户进程 |
| `20_sequential_models` / `21_baseline_recheck` | 抓取成功，移动中支撑范围检查失败，未释放 | 选点只考虑初始占地，少量姿态变化就使边界余量不足；现在预留所有平面偏航及允许倾角的完整占地 |
| `22_reserved_footprint` | 通过原支撑问题，SAM2 掩码深度超出持物 ROI 时停止，未释放 | 改为只移除 ROI 内已关联持物点，外部深度点全部保留为环境障碍；不扩张 ROI，不把背景当持物清除 |

`23_mask_depth_separation`：23 次闭环迭代完成对齐和下降，视觉 XY 残差 0.42 mm、底面间隙 3.01 mm；退出预检发现一个已关联持物接触点在近似手指模型内 1.034 mm，超过原固定 1 mm 门限，未开爪。独立仿真真值仍有约 4.4 mm XY 误差，视觉残差不能替代物理精度。

接触几何误差现使用 HeldObject 声明的不确定性（最多 3 mm），仍只限已有持物接触、禁止继续压入和豁免新障碍。

- `24_contact_uncertainty`：抓取成功；第 8 次放置迭代被一个持物边缘点拦截，未释放，尚未检验到退出。回放确认机器人 1 像素边界屏蔽过早删掉了颜色种子；调整为先建立持物身份、再应用同样的 1 像素屏蔽，同一帧剩余持物包围盒碰撞点从 1 变成 0。SAM2 轮廓与有效种子取并集，不丢掉已经通过几何/实例检查的种子。
- `25_preserve_identity_seed`：20 次迭代到达释放位置，第一次退出预检通过；新观测下侧面接近点导致第二次检查失败，未开爪。这不是放置成功。
- `26_attachment_escape`：新增基于实际持物偏移的多方向退出搜索与同帧几何缓存后的独立物理复测，结果待补充。

每段实际 FK 路径还受局部距离/旋转范围限制：不能只因终点距离 6 mm 就接受绕远的大 IK 分支。退出空间预检按“已张开”的夹爪几何进行，开爪扫掠则单独检查；不会拿夹持时闭合的手指体积模拟已经松开的退出动作。当前标定的退出沿 EE 的 -Z（手指轴线）45 mm，要求其向上分量至少为 0.9，避免世界竖直方向刮向略倾斜的手指；执行必须沿同一个预检目标分步退出。

退出接触检查只对同帧视觉关联的持物点开放：初始点距近似手指表面外侧不超过 1 mm、内侧不超过持物接口声明的 `uncertainty_m`（上限 3 mm）；路径中模型穿入深度不能比初始增加超过 0.2 mm，也不能超过这个不确定性。新障碍物、深度穿入和非持物点不豁免；不是忽略整个被放物体。释放后仅前 15 mm 退出保留这个接触上下文，物体参考位置固定在世界中，不跟着腕部走。它是几何误差容许量，不是真实允许压入物体 3 mm 的接触策略；真机需重新标定，不可随意增大。

若纯工具轴退出被拒绝，还会测试横向偏移 3/6/10/15/20 mm 的退出目标：工具 +X、远离实测持物中心的方向、向外的工具 Y 方向。每条候选仍须通过完整实际 FK、世界/自身碰撞与上述接触约束，全部失败则不开爪。新观测先重新检查此前胜出的方向，不复用其安全结论。不是把模型旋转强行下发。`scripts/inspect_place_contact.py <运行目录>` 可离线查看已记录末端的接触点；它不发送控制命令，也不替代真实路径检查。

同一 RGB-D 快照、同一实测末端姿态及持物参数下，重复碰撞检查复用点云分类结果，避免一次开爪预检重复解投影三次。新快照或机器人姿态变化立即失效；不缓存跨帧碰撞许可，350 ms 时效门限不变。

重复的新深度路径检查会缓存固定机器人基座下、完全相同关节值的 FK；基座变化立即失效。世界、自碰撞和新点云碰撞仍逐样本重新检查，缓存不构成运动许可。

Florence 在这些图像中确实找到了蓝色区域和多边形，但 detailed caption 曾把场景描述成“3D printer”。所以 caption 只进入调试报告，不用于推断容器可放性、稳定底面或开爪动作。几何与运动安全不能交给图片描述。

RGB-D 支撑几何可以无 GPU 重放（不启动 Isaac 渲染或 Florence）：

```powershell
& 'D:\isaac\env_isaacsim60\python.exe' scripts/replay_fine_place_geometry.py output/fine_place/06_hold_preload
```
