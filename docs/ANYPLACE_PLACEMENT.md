# AnyPlace 放置实验线

分支 `codex/anyplace-placement`，从接入 M2T2 之前的 `5a1d81d` 开始。
按用户要求停止 M2T2 开发并删除本地 `codex/m2t2-pick-place`；删除前保存于提交 `2c8962c`，短期可用 reflog / 提交号恢复。没有删除模型缓存、实验输出，也没有删除远端分支。

## 保留什么、替换什么

- **抓取保持 GraspGenX**，保留 Florence/YOLOE → LK 光流 → 粗接近 → 腕部 FineGrasp → 抬升验证。
- AnyPlace 仅作为新放置模型候选；输出是作用在输入物体点云上的相对 SE(3) 变换，不是直接的末端控制指令。
- 当前已完成官方权重的真实离线推理入口，**尚未接入 Isaac 物理闭环放置或真机**。原 `run_fine_place.ps1` 仍是原有 Florence/RGB-D 几何放置，不会暗中调用 AnyPlace。
- 后续映射必须使用 `T_base_object_final = relative @ T_base_object_input`，再通过实际持物变换得到 `T_base_ee_final = T_base_object_final @ inverse(T_ee_object)`。不能只取模型 XY 而静默丢弃朝向。

## 本机模型测试

官方源码 `_vendor/anyplace`，提交 `3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e`。
通用多任务权重 `_models/anyplace/anyplace_ckpts/anyplace_multitask/model.pth`，81,354,343 bytes，SHA256 `d3d33f0a279633c25f252960a208d4b4447a756f0cff8e94be0faadc20dc5be5`。
没有采用堆叠/悬挂/插入专用权重，也没有针对测试对象训练。

```powershell
.\_envs\anyplace\Scripts\python.exe scripts\benchmark_anyplace.py `
  --checkpoint _models/anyplace/anyplace_ckpts/anyplace_multitask/model.pth `
  --parent _models/anyplace/anyplace_eval/task_gpp_stacking/stacking/pod_0_on_flatplate_0/base_obj_pointcloud.ply `
  --child _models/anyplace/anyplace_eval/task_gpp_stacking/stacking/pod_0_on_flatplate_0/target_obj_pointcloud.ply `
  --output output/anyplace/new_stacking.json
```

默认 20 个初始候选、50 轮扩散，固定随机种子 0。输出不得覆盖已有文件。
`parent` 为目的地几何，`child` 为待放物体几何；两者同一坐标系、单位米。当前入口要求 parent ≥8192 点、child ≥1024 点（上游代码还无条件计算一个未使用的 8192 点分类输入），不通过补点伪造完整几何。

实测 `output/anyplace/01_official_stacking.json`：20 个有限 4×4 变换，推理约 **12.55 s**，CUDA 峰值 allocated tensor memory **1,503,347,712 bytes**（约 1.40 GiB）。不含加载、Isaac 或 CUDA 上下文。使用官方测试点云，不是本机机器人实拍成功，也尚未审计与训练集的重叠。

实测 `output/anyplace/02_official_hanging.json`：同一多任务权重用于杯子/挂架点云，20 个候选、50 轮，约 **13.10 s**，峰值 allocated tensor memory **1,503,697,920 bytes**。两组均未运行物理执行，不给出放置成功率。

完整现有软件回归 **293 tests passed**，包括 4 个新增 AnyPlace 输入契约测试；使用 `D:\isaac\env_isaacsim60\python.exe -m unittest discover -s tests -q`。这是退回接入 M2T2 前基线后的测试集，不能与已删除分支的测试数直接比较。

所有结果标记 `physical_success: null`、`collision_checked: false`。上游此配置没有学习到的成功率评分器，候选的统一分数不能当成成功概率。

## 环境与下载

`_envs/anyplace` 是继承已有 GraspGenX Torch 2.6 / CUDA 12.4 的独立 venv，新增依赖只写入本 venv，没有升级原抓取环境。不是完全锁定的可分发环境。

```powershell
.\_envs\graspgenx\python.exe -m venv --system-site-packages _envs/anyplace
.\_envs\anyplace\Scripts\python.exe -m pip install torch-cluster torch-scatter --no-index -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
.\_envs\anyplace\Scripts\python.exe -m pip install 'numpy==1.26.4' 'plyfile==1.0.3' meshcat yacs einops 'astropy-healpix<2' colorlog rtree
git clone --depth 1 https://github.com/ac-rad/anyplace.git _vendor/anyplace
git clone --depth 1 --branch panda-2f140 https://github.com/Improbable-AI/airobot.git _vendor/airobot
git -C _vendor/anyplace apply --unidiff-zero ../../patches/anyplace-lazy-knn.patch
.\_envs\anyplace\Scripts\python.exe scripts/fetch_anyplace_assets.py anyplace_ckpts.zip --member anyplace_ckpts/anyplace_multitask/model.pth
```

克隆命令只用于尚未存在的目录，已有本机环境无需重做。Airobot 验证提交为 `7ae04feb07d554835bf6d917063def26c8b7e1d1`。

`fetch_anyplace_assets.py anyplace_eval.zip` 列出官方压缩包条目；加 `--member` 精确下载所需点云。脚本检查 HTTP Range、解压路径及 ZIP CRC，拒绝覆盖已有文件。不下载 35 GB 训练集。完整网络下载曾中断，保留 `.cache` 中的未完成文件，不将其当可用权重。

兼容修改仅包括：

1. 未使用的 `Group` 的 KNN_CUDA 依赖改为构造时加载；实际推理使用原生 `torch-cluster` FPS，不替换采样为几何抓取。
2. HEALPix 球面采样使用有 Windows wheel 的 `astropy-healpix`，保留上游 RING 顺序与 Euler 组合方式；尚未逐点对照 healpy。
3. 关闭 MeshCat 输出，不修改官方扩散网络；严格加载 checkpoint 参数。

## 下一阶段验收

不能把官方干净点云推理通过当成腕部 RGB-D 泛化通过。接入实际闭环之前，需要：

- 从多视角测量提供足够完整的持物点云，明确遮挡、深度缺失和置信度，不能用仿真真值代替部署输入。
- Florence/SAM 给出目的地局部区域；AnyPlace 输入保留容器边缘、孔位等三维结构。
- 模型提出多种完整姿态，随后检查 SO-101 自由度、IK、附着物/手指扫掠、支撑和夹爪释放可行性。
- 重模型完成后重读两路 RGB-D、验证目标未移动，执行小步闭环接近、释放、撤离、稳定性检查。
- 分别记录模型推理、几何可行、物理释放与最终成功，使用未针对性训练的多对象冻结测试集。

官方来源：[AnyPlace 代码](https://github.com/ac-rad/anyplace)、[权重及测试数据](https://huggingface.co/datasets/yuchiallanzhao/anyplace)。本实验不使用其 AnyGrasp 抓取步骤，抓取保留 BusAgent 原后端。
