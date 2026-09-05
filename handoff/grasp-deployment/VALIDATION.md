# 排查包自身验证

日期：2026-09-05。这里验证的是交接包和检查器，**没有启动模型推理或 Isaac，也没有执行新的抓取**。

## 已执行

```powershell
python -B handoff/grasp-deployment/test_check_deployment.py
```

5 个单元测试通过：CRLF/LF 等价哈希、10 个快照完整性、缺失/改动配置与错误权重的报告、缺命令的结构化错误、元数据探测不导入模型。

另外实际针对两种目录运行检查器：

| 被检查目录 | 结果 |
|---|---|
| 独立的 GitHub master 克隆（`d63e2d014e4653453c35867ec5878898cc6db146` 加本交接包） | 项目参考文件均匹配；两份模型 YAML 与两份权重均被正确报告缺失 |
| 现有开发机仓库（检查时 HEAD 为 `a30cbea20267793fbd07f6a6706ee1cdb7b59364`） | 5 份核心 YAML、GraspGenX 客户端 requirements、两份模型 YAML 匹配；两份模型权重的完整 SHA-256 均匹配 |

现有开发机与 master 有两处被正确检出的视觉文件差异：

- master 的 `requirements-vision.txt` 多一条 `requests>=2.32`。
- master 的 Florence 路径选择还支持固定 revision 的 hub cache，现有开发分支没有该候选路径。

这里只记录差异，不将其直接判定为抓取失败原因。工作区由主开发任务继续更新，因此此报告不是“当前机器永久状态”；快照来源仍固定为清单中的 commit。

## 本机模型环境元数据探测

- Python 3.11.15。
- Torch 2.6.0+cu124，torchvision 0.21.0+cu124，NumPy 1.26.4。
- GraspGenX 1.0.0，pyzmq 27.2.0，msgpack 1.2.2，msgpack-numpy 0.4.8。
- `graspgenx_origin` 指向本机项目 `_vendor/GraspGenX/graspgenx/__init__.py`，不是可随意搬目录的独立安装。

以上由包元数据和 `find_spec` 取得，没有加载神经网络；不代表 CUDA kernel/服务协议/物理抓取健康检查通过。完整虚拟环境 freeze 和私有日志没有加入本包。

## 未验证

- 对方机器的真实运行状态和故障根因。
- Linux 上脚本的实际执行（实现仅用 Python 标准库和 Git；需要对方运行验证）。
- 本次复制快照之后的任何控制效果、真机安全性或泛化成功率。
- 自动恢复/覆盖配置：本包刻意不提供这种操作。
