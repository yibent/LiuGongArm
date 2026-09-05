# M2T2 抓取到放置

当前分支把官方 NVIDIA M2T2 接到现有 FineGrasp 闭环。M2T2 只负责从 RGB-D 点云提出抓取和放置姿态，Florence/YOLOE 负责标签定位，腕部分割与跟踪负责目标点云，原有夹爪宽度、桌面间隙、碰撞、IK、闭爪验证和抬升验证继续生效。

## 结构

```text
Florence/YOLOE 标签定位
        ↓
腕部 RGB-D + 目标分割/跟踪
        ↓
M2T2Backend（ZMQ，抓取候选）
        ↓
FineGrasp：宽度/碰撞/IK/伺服/闭爪/抬升验证
        ↓
M2T2PlacementPlanner（ZMQ，放置候选）
        ↓
搬运 → 下降 → 打开夹爪
        ↓
固定相机重新定位 + 区域内/夹爪卸载验证
```

Isaac 进程不导入 Torch。`source/mr_liu/grasp/backends/m2t2.py` 是轻量客户端；`scripts/serve_m2t2.py` 在独立 Python 环境中加载 `_models/m2t2/m2t2.pth`。这样 M2T2 的 CUDA/PointNet++ 依赖不会替换 Isaac 的运行时。

M2T2 输出的候选首先按接触点与 Florence/YOLOE 已确认的目标点云做关联，然后才交给 FineGrasp 的安全筛选。候选为空、服务不可用、协议错误或推理失败都会返回明确的 `model_unavailable`、`model_timeout`、`model_protocol_error` 或 `model_inference_failed`，不会闭爪，也不会盲目释放。

## 安装与权重

官方代码来源：[NVlabs/M2T2](https://github.com/NVlabs/M2T2)，权重来源：[wentao-yuan/m2t2](https://huggingface.co/wentao-yuan/m2t2)。

```powershell
.\scripts\setup_m2t2.ps1 -Python C:\path\to\m2t2\python.exe
```

该环境需要 CUDA 版 PyTorch 和可用的 `nvcc`，因为官方 `pointnet2_ops` 是 CUDA 扩展；当前仓库的 Isaac/视觉环境不作为 M2T2 服务环境使用。

脚本会准备 `_vendor/M2T2`、安装官方 `pointnet2_ops` 和依赖，并下载：

```text
_models/m2t2/m2t2.pth             # 通用抓取 + 放置模型
_models/m2t2/m2t2_language.pth    # 语言条件模型（可选）
```

本次开发机已下载并校验：`m2t2.pth` SHA-256 `E35C3CB11E06F46C5D406BDFC756BC06F48B256DD6D638408D9A5FF13DEB97FB`；`m2t2_language.pth` SHA-256 `8CC96DAF6A50CDC3E52BB27BB163DCF78CDEB64C0972D9E7D51B24B70E1CF5FA`。

权重和 vendor 目录被 `.gitignore` 忽略。官方仓库代码带 NVIDIA 非商业研究/评估限制；Hugging Face 模型卡标注 Apache-2.0。产品化或重新分发前应分别复核两项许可。

## 启动

独立启动服务：

```powershell
python scripts/serve_m2t2.py --device cuda --port 5562
```

也可以由演示 wrapper 自动启动（需要设置 `M2T2_PYTHON`，或准备 `_envs/m2t2/python.exe`；不会复用 Isaac 的 Python）：

```powershell
.\scripts\run_fine_grasp_demo.ps1 `
  -Backend m2t2 -M2T2Port 5562 -Label "red block" `
  -LocalizationMode florence_yoloe -Recovery active -NoHeadless
```

该 wrapper 在默认配置下同时启动原有 GraspGenX 服务（端口 5556），作为
M2T2 无候选或服务异常时的继承路径；将 `backends.m2t2.fallback_backend` 改为
`none` 后可进行严格的 M2T2 试验，但仍可手动启动 wrapper 使用同一套入口。

`configs/fine_grasp.yaml` 的 `backends.m2t2` 控制服务地址、点数、置信度、目标接触关联半径和 fallback。当前 M2T2 分支默认 `fallback_backend: graspgenx`；严格 M2T2 试验可设为 `none`。fallback 只用于生成候选，不能绕过 FineGrasp 安全门。

## 放置区域

`configs/fine_grasp.yaml` 中的 `place` 使用机器人 base 坐标：

```yaml
place:
  enabled: true
  target_center_base_m: [0.68, 0.00, 1.05]
  target_size_xy_m: [0.14, 0.14]
  surface_z_m: 1.05
```

当后端为 `m2t2` 且抓取、闭爪、抬升均通过后，系统使用固定相机的持物点云调用 M2T2 放置推理。放置候选还要通过可达性和碰撞检查。打开夹爪后，固定相机必须重新观测到目标中心位于区域内，并且夹爪不再报告 contact/stalled，才返回 `success: true`。释放后目标丢失、落点越界或夹爪仍有载荷都会返回 `place_verification_failed`。

运行结果的 `report-*.json` 同时包含：

```json
{
  "success": true,
  "message": "抓取、搬运、放置及落点验证成功",
  "metrics": {"place_success": 1, "inside_place_region": 1},
  "place": {"success": true, "phase": "succeeded"}
}
```

`success` 只代表完整链路成功；仅闭爪或仅抬升成功不会被标记为放置成功。

## 常见故障

- `model_unavailable`：M2T2 客户端依赖、服务进程或权重缺失。
- `model_timeout`：服务超时；保持当前夹爪状态，不能据此宣称抓取/放置成功。
- `no_grasp_candidates`：模型没有与目标点云关联的接触点，或所有候选未通过宽度、桌面间隙、碰撞、IK 筛选。
- `place_unreachable`：放置候选都不可达或碰撞。
- `place_verification_failed`：释放后没有足够证据证明目标在区域内且已脱离夹爪。

视频、RGB-D 帧和日志保存在 `output/`，不进入 Git。实际部署前仍应在目标硬件上重新做工具坐标标定，并分别统计抓取成功率、放置成功率和物理刚体位移。
