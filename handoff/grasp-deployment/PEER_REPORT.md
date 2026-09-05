# 对方机器最小复现信息

请填到本地副本，脱敏后发送；无需提供密码、访问令牌、完整环境变量或完整私有日志。

## 运行身份

- OS / GPU / 驱动：
- 项目 `git rev-parse HEAD`：
- 分支及是否有本地修改：
- 原样启动命令（先移除密钥）：
- 入口：standalone demo / label demo / Web / BusAgent / 真机：
- 实际 Isaac / Python / Torch / CUDA 版本及解释器位置：
- GraspGenX vendor SHA / 权重 SHA 是否匹配：
- 5556 服务是本次启动还是复用？由谁启动、什么配置？（不要提供完整进程环境）：

## 场景与执行

- 标签、对象、case JSON、seed：
- 腕部 profile / 场景相机视角 / recovery / perception / segmenter：
- `Runtime joint gains` 或等价实际控制增益日志：
- 最终使用的 backend / fallback / failure code：
- 首次失败阶段：启动 / 标签定位 / 跟踪 / 深度与点云 / 模型候选 / 碰撞或 IK / pregrasp / servo / 合爪 / 抬升 / 验证：
- 目标是否正确？腕部图像是否看见目标？深度是否有效？：
- 末端目标与实际误差、超时信息、是否滑落：
- 同 case 重复成功次数 / 总次数：

## 附件

- `check_deployment.py` 的 JSON 输出（先脱敏绝对路径）：
- 首个错误附近的少量日志：
- 腕部 RGB、对齐深度/掩码、抓取叠图、执行前后画面（移除无关人员/私有内容）：
- `report.json` 中 failure/selected backend/servo/lift 等相关字段：

请不要先通过放宽碰撞/力限制来“让它动起来”。如果是真机，先停止失败动作并由操作者核对标定和安全边界。
