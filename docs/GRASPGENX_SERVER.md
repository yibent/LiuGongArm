# GPU 服务器上的 GraspGenX 服务

2026-09-05 部署：`/root/gpufree-data/LiuGongArm`。
模型进程独立于 Isaac，通过本机 ZMQ 5556 接收物体点云并返回候选抓取姿态。
Isaac 保留自己的 Python/Torch；只在项目 `.venv` 增加 msgpack、msgpack-numpy、pyzmq 客户端。

## 文件与版本

- 模型代码：`_vendor/GraspGenX`，官方提交 `b9429097728cb1c430dd78b92edf17ba318aad03`。
- 权重：`adithyamurali/GraspGenXModel`，版本 `7c834043c11a11417e31d6d5ea9355801e40a2c1`，
  保存到 `_models/graspgenx/checkpoints/release`。
- `gen/epoch_736.pth`（1,210,918,342 字节）：
  SHA-256 `8b55f31cdb8340a573b4df27b027c15cff326bd6debcb389bf631d2aaab7ac44`。
- `dis/epoch_1056.pth`（483,889,478 字节）：
  SHA-256 `cbf3f3bdb2e4c03fca8486ed24de0e6a8a859e6bd22bce2f1434a610335abd3e`。
- 两个目录各有原版 `config.yaml`。通过 HF 镜像下载，以上哈希已核验。
- 独立 `_envs/graspgenx`：Python 3.11.16、Torch 2.6.0/CUDA 12.4、
  torchvision 0.21.0、NumPy 1.26.4，安装官方项目的 `[serve]` 依赖，
  另加 `opencv-python-headless==4.11.0.86` 供项目预热客户端使用。
- Gripper 配置位于 `_vendor/GraspGenX/ext/gripper_descriptions`。
  当前点云/sweep-volume 推理不需要上游演示媒体或网格资产；未下载可选 LFS 演示媒体。

## 启动与检查

在项目目录执行（先确认本机 5556 没有已有实例，避免重复启动）：

```bash
bash scripts/run_graspgenx_server.sh
```

本次后台进程日志：`logs/graspgenx-server.log`。进程不会因 SSH 断开退出，
但尚未配置服务器重启自启动。启动脚本设置权重和 gripper 路径，
并显式加入独立环境的 CUDA NVRTC 库路径，避免 `libnvrtc-builtins.so.12.4` 加载失败。

另一个终端进行真实模型推理检查（不会移动机械臂）：

```bash
_envs/graspgenx/bin/python scripts/warm_graspgenx_server.py --timeout-ms 45000
```

预热必须返回非空且评分有限的候选；仅端口打开不算通过。
本次 RTX 4090 上首次成功实测约 987 ms，128 个候选，最高分 0.95056。
评分不是抓取成功概率；此测试不证明物理抓取、可达性或抬升验证成功。

## 不允许自动几何回退

`configs/fine_grasp.yaml` 默认 `backend: graspgenx`，
`fallback_backend: none`、`fallback_on_empty: false`。
BusAgent 在线会话再次强制关闭这两项降级路径。模型故障、超时或无候选时报告失败。

保留官方 GraspMoE：扩散模型生成候选，OBB 分支也提供候选，统一由神经判别器评分。
这不是本项目的无模型 `GeometricAntipodalBackend` 回退。
后者只保留作显式选择的离线基线，不会在在线抓取失败后自动启用。

本次切换前的代码与私有进程环境备份：
`/root/gpufree-data/deploy-backups/grasp-ehRVvB2r`。
其中进程环境含私有配置，不应公开或提交到 Git。
