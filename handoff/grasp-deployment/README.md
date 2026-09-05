# 抓取部署差异排查交接包

目的：排查“作者机器可以抓、另一台机器总失败”，不是替换抓取算法或保证成功率。本包只新增文档、快照和只读检查器，不修改运行代码，不自动覆盖配置。

审计日期：2026-09-05。项目快照基线：`master` 的 `d63e2d014e4653453c35867ec5878898cc6db146`。本地被忽略的模型 YAML 另行采集，来源见 [manifest.json](manifest.json)。不包含正在开发的 M2T2 改动。

**当前结论：项目核心 YAML 没有被漏提交；模型目录中的必要 YAML 确实被忽略。另外存在启动入口、环境路径、外部资产和运行参数差异。尚未拿到对方的启动命令、SHA 和失败日志，不能认定这些就是对方抓取失败的根因。**

## 先让对方做什么

1. 不要先覆盖配置或重装环境。在对方现有项目目录记录 `git rev-parse HEAD`、启动命令和失败阶段。
2. 获取本文件夹，在任意 Python 3.9+ 下运行下面的只读检查。
3. 将 JSON 报告和 [PEER_REPORT.md](PEER_REPORT.md) 填好的信息发回。公开前检查报告中的用户名、绝对路径、分支名和文件名；不要上传密码、完整进程环境或未脱敏日志。
4. 先比较同一个案例、同一个 seed、同一个入口，再判断是缺依赖、感知、坐标、控制还是接触问题。

可以在另外一个目录获取排查包，不切换正在运行的项目分支：

```powershell
git clone --depth 1 --branch codex/grasp-deployment-audit https://github.com/yibent/LiuGongArm.git LiuGongArm-audit
```

Windows 示例：将 `--repo` 改成**对方实际项目根目录**，不要指向只用来读文档的 audit 克隆。

```powershell
python -B .\LiuGongArm-audit\handoff\grasp-deployment\check_deployment.py --repo D:\LiuGongArm
```

Linux 示例：

```bash
python3 -B ./LiuGongArm-audit/handoff/grasp-deployment/check_deployment.py --repo /path/to/LiuGongArm
```

可选：检查两份 GraspGenX 权重的完整 SHA-256，并探测**可信的本地模型 Python**的包元数据和安装位置：

```powershell
python -B .\LiuGongArm-audit\handoff\grasp-deployment\check_deployment.py --repo D:\LiuGongArm --hash-weights --model-python D:\LiuGongArm\_envs\graspgenx\python.exe
```

`--hash-weights` 会顺序读取约 1.7 GB，但不会加载模型/GPU。`--model-python` 只查询包元数据，不导入 torch/GraspGenX、不推理。脚本默认只打印 JSON；要保存可自行追加 `| Out-File -Encoding utf8 deployment-report.json`（这一步会创建报告文件）。

解释报告：

- 退出码 0 只表示检查完成，**不表示依赖齐全或抓取成功**；无效仓库参数返回 2。
- `exists: false`：缺文件；视觉默认路径缺失也可能是用了自定义路径/旧缓存，不能单凭这一项判失败。
- `matches_reference: false`：和这次快照不同，可能是合理的新版本；先 diff，不能直接判定配置错。
- `snapshot_integrity: false`：本包内快照与清单不一致，应先重新获取排查包。
- `hash_status: not_checked`：没有计算权重哈希；同样大小不代表同样权重。
- `tracked` / `ignore_rule`：解释文件是否由 Git 交付。
- `entrypoint_paths` 是审计基线的路径提示，不是自动解析对方最新脚本，更不是跨平台安装要求。

## 已确认事实与待验证原因

| 项目 | 已确认事实 | 对方应核对什么 |
|---|---|---|
| 项目配置 | `configs/cameras.yaml`、`fine_grasp.yaml`、`robot_so101.yaml`、`motion.yaml`、`scene.yaml` 和 URDF/XRDF 均被跟踪 | 是否相同提交、是否手工改过，是否加载了另一套配置 |
| 模型配置 | 根目录 `/_models/`、`/_vendor/`、`/_envs/` 被忽略；`gen/config.yaml` 和 `dis/config.yaml` 不随 clone 到达 | 不能只复制 `.pth`；两份 YAML、权重、模型源码及服务依赖都要完整 |
| Demo / 在线控制 | Demo 覆盖五个臂关节的增益；在线会话保留 live gains | 对比实际 `Runtime joint gains`、末端偏差、settle 超时，不能只比较 YAML |
| Isaac 启动器 | `isaac_env.bat` 用 `D:\isaacsim\python.bat`；抓取 PS1 直接用 `D:\isaac\env_isaacsim60\python.exe` | 改 BAT 不会同时修复 PS1；确认实际运行的 Isaac/Python/扩展版本 |
| 可编辑模型安装 | 本机 editable install 指向 `D:\liuGong\_vendor\GraspGenX\graspgenx` | 把虚拟环境复制到另一目录后，可能导入旧目录或找不到包；看 `graspgenx_origin` |
| 现有模型服务 | PS1 发现端口 5556 打开就复用，不执行自己启动分支的依赖检查和预热 | 是否复用了旧服务/不同权重？端口可连接不等于协议和推理正确 |
| 外部 USD 资产 | SO101 USD、桌子、部分物体来自 Isaac 资产库，不在本 Git 中 | Isaac 版本、资产根、USD 内容/缓存是否一致；导入的刚度等属性可能不同 |
| 标签识别链 | Florence/YOLOE 权重、vision 环境同样被忽略 | 若从标签开始失败，先核对识别服务、权重、RGB-D mask、光流跟踪，而非先改抓取模型 |

缺模型/错误路径往往导致启动或服务失败；末端偏差、碰撞和滑落也可能来自控制/几何差异。两类问题不要混为一谈。

### 1. 模型侧真正缺失的配置

项目 `.gitignore` 中的 `/_models/` 会连同 YAML 一起忽略，而不只是忽略大权重。需要的布局为：

```text
_models/graspgenx/checkpoints/release/
  gen/config.yaml
  gen/epoch_736.pth
  dis/config.yaml
  dis/epoch_1056.pth
_vendor/GraspGenX/
  graspgenx/serving/zmq_server.py
  ext/gripper_descriptions/
_envs/graspgenx/                  # 在目标机建立，不建议直接搬虚拟环境
```

本包提供两份小 YAML 的参考快照，**不提供权重**。上游 checkpoint/配置应以完整、配套的固定 revision 获取：

- [GraspGenX 源码](https://github.com/NVlabs/GraspGenX/tree/b9429097728cb1c430dd78b92edf17ba318aad03)：`b9429097728cb1c430dd78b92edf17ba318aad03`。
- [模型文件](https://huggingface.co/adithyamurali/GraspGenXModel/tree/7c834043c11a11417e31d6d5ea9355801e40a2c1)：`7c834043c11a11417e31d6d5ea9355801e40a2c1`。
- [夹爪描述资产](https://huggingface.co/datasets/adithyamurali/gripper_descriptions)。SO101 推理用的 sweep-volume 参数在项目 `fine_grasp.yaml` 中，不要臆造另一个私有 SO101 模型配置。

文件长度、哈希和来源在清单中。两份 YAML 不应交叉使用，也不应随意把训练字段改成自己的路径。只有确认用的是同一 release、目标确实缺失、并保留原文件之后，才考虑手工恢复；检查器不执行恢复。

Windows 模型环境曾用 Python 3.11 / Torch 2.6.0+cu124 跑通；不能把它安装进 Isaac 的 Python 环境。Windows 上官方 `[serve]` 安装可能卡在 `SharedArray` 编译，已有说明见 [模型评估](../../docs/MODEL_EVALUATION.md)。不要仅凭 `--no-deps` 安装成功就认为推理依赖完整。

### 2. 不能忽略 Demo 与在线入口的控制差异

审计基线的 `scripts/run_fine_grasp_demo.py::_configure_arm_drives` 在初始化后给五个机械臂关节设置：

```yaml
arm_drive_stiffness_nm_per_rad: 800.0
arm_drive_damping_nms_per_rad: 20.0
arm_drive_max_effort_nm: 10.0
```

而 `source/mr_liu/grasp/isaac_session.py` 明确保留 live calibrated gains；`source/mr_liu/robot/so101.py::configure_drives` 保留导入刚度，只按 `robot_so101.yaml` 设置阻尼。例如 shoulder_lift 为 3.75、wrist_flex 为 1.95，并非 20。

因此，“同一套模型、同一 YAML”不保证 Demo 与 Web/BusAgent 的实际驱动相同。这是值得优先测量的差异，**不是已证实的对方失败根因，也不是让你直接改成 800/20 的修复建议**。这些数值仅用于特定 Isaac demo，严禁直接套到真机控制器。

### 3. 相机与坐标系

本次 GitHub master 已包含之前的近距离 `fine_grasp` 腕部相机：相对 `/World/SO101/gripper`，平移 `[0.060, 0, -0.030]`，四元数 **wxyz** `[0.53895, 0, 0.84234, 0]`，焦距 1.8。不能说“最新 master 仍然是旧相机”。

`tabletop_wide` 是对比用 profile，不是自动 fallback；基线注释记录其在 close-range handoff 会裁掉目标。核对**实际传入的 profile**，不要仅看默认配置。不能独立旋转 RGB 而不调整 depth、K 和坐标变换。

真机应重新核对本机手眼/夹爪标定，不能直接把仿真相机位姿当标定：每帧用同步的 `T_base_ee @ T_ee_camera`，再转换当前相机中的物体/抓取位姿。另核对深度单位、TCP、夹爪厚度与桌面高度。快照不是跨机器通用的标定文件。

### 4. 标签链、入口和版本

`run_label_grasp.ps1` 默认是 Florence+YOLOE、`Recovery=active`、`fine_grasp` 腕部 profile、斜视场景相机；直接 `run_fine_grasp_demo.ps1` 默认 recovery 关闭、场景相机俯视。两者不能当作完全相同实验。

Florence/YOLOE 路径、revision/hash 在快照 `source/find_and_track/settings.py` 中；此文件只是参考副本，不应从 snapshots 目录导入。`BUSAGENT_FLORENCE_MODEL` / `BUSAGENT_YOLOE_WEIGHTS` 以及旧缓存可能改变实际模型路径。检查器只报告这两个覆盖变量是否设置，不输出值。

`scripts/setup_vision.ps1` 针对特定 Torch 2.11.0+cu128 基础环境，并装匹配 CUDA torchvision；不要在别人的不同基础环境直接照搬执行。先参见 [标签抓取说明](../../docs/LABEL_TO_GRASP.md)。本次排查不安装或改动这些环境。

旧交接文档可能要求 clone `codex/fine-grasp-loop` 或 `codex/active-grasp-recovery`。务必比对具体 SHA；不要把旧文档的参数、当前主分支配置和实验分支代码混搭。本包当前基线不包含实验性 M2T2。

### 5. 文档与证据边界

`docs/MODEL_EVALUATION.md` 保留了早期 geometric fallback 的历史描述；本次基线 `fine_grasp.yaml` 已是 `fallback_backend: none`、`fallback_on_empty: false`，在线会话也禁用这类降级。故应检查**实际 selected backend / failure reason**，不能只引用旧报告说服务失败会自动换几何模型。

`output/`、日志和历史 benchmark 被忽略，Git clone 不包含原始成功录像或报告。历史单个成功不等于另一台机器已经复现，更不证明复杂工具/反光物体的总体成功率。本交接包也没有执行新的物理抓取验证。

## 文件清单与安全说明

- `snapshots/project/`：5 份项目 YAML、客户端/视觉 requirements、视觉模型路径设置。
- `snapshots/model/graspgenx/{gen,dis}/config.yaml`：本地被忽略的模型配置参考。
- `manifest.json`：来源、固定版本、文本哈希、权重预期大小/哈希。
- `check_deployment.py`：标准库只读检查，跨 Windows/Linux；不发网络请求、不执行 Git fetch/checkout、不启动服务或仿真。
- `test_check_deployment.py`：检查器自身单元测试，不是抓取测试。
- `VALIDATION.md`：单元测试、本机与干净克隆的实际排查结果。
- `PEER_REPORT.md`：对方填写的最小复现表。
- `THIRD_PARTY.md`：第三方配置来源和授权指引。

文本哈希规范化 CRLF 为 LF，避免 Windows/Linux 换行差异误报；这不是 YAML 语义比较，注释变化也会报告差异。权重使用原始二进制 SHA-256。

本包不含 SSH 密码、API token、完整环境、模型权重、虚拟环境、Isaac 安装或原始私有日志。请勿为了“完整打包”把 `_envs/`、`.env` 或进程环境加入 Git。

## 交接阅读检查

未做独立读者验收。对方读完后应能回答：哪些配置确实被忽略？为什么不能直接覆盖？如何针对现有仓库只读检查？为什么 Demo 成功不证明在线入口成功？失败应归类到哪一阶段？请把不清楚的地方连同复现表反馈，下一步再做定点修复。
