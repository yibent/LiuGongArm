# 云服务器自启

当前部署：`/root/gpufree-data/arena-panda`。此实例以容器方式运行，PID 1 是 tini，云平台的 Go supervisord 读取 `/opt/supervisord.yaml`，其 include 已包含 `/.gpufree/supervisord/conf.d/*.conf`。

已安装 `/.gpufree/supervisord/conf.d/liugong-stack.conf`，链接到项目中的 `ops/arena-cloud-boot.conf`。容器启动时，平台自动运行本项目专用的 Python Supervisor；不修改平台的 SSH、桌面或只读主启动配置。Nginx 与 Xorg 已由平台桌面启动程序提供，8991/8993/8999 配置继续有效。

专用 Supervisor 管理 MariaDB 3307、vision 5570、GraspGenX 5556、AnyPlace 5590、Arena 7861、BusAgent 3100，配置在 `ops/arena-supervisord.conf`。GPU 模块等待 NVIDIA 驱动；Arena 等待显示服务和模型接口；BusAgent 等待数据库及 Arena。进程异常退出自动重启，手动 `stop` 则保持停止，日志自动轮转。管理接口仅使用权限 0700 的本机 Unix socket。

Arena 从 `output/services/arena-scene.json` 读取上次选择的场景配置，目前为 `configs/arena_panda_industrial.json`。重启会重新建立这个场景的初始布局，**不恢复重启前的抓持或物体实时位置，不重放旧动作**。数据库继续使用现有 `/var/lib/mysql`，不执行初始化或重建。

常用操作：

```sh
cd /root/gpufree-data/arena-panda
python3 ops/arena_stack.py status
python3 ops/arena_stack.py stop arena
python3 ops/arena_stack.py start arena
supervisorctl -c ops/arena-supervisord.conf status
supervisorctl -c ops/arena-supervisord.conf restart busagent
```

`arena_stack.py` 检测到专用 Supervisor 后会转交控制，避免并行启动重复 GPU 进程。完整停止或启动（含数据库）可使用 `supervisorctl -c ops/arena-supervisord.conf stop all` / `start all`。直接启动专用管理器的命令为 `/usr/bin/supervisord -c ops/arena-supervisord.conf`，仅在管理器未运行时使用。

部署时已停止原独立进程，用同一份自启配置完成整套服务冷启动，约一分钟恢复。随后用 SIGKILL 模拟 BusAgent 进程异常退出，验证自动生成新 PID、WebSocket 恢复，Arena 保持原 PID。HTTP 与场景读取检查见 [验证记录](autostart-validation.json)。未对云实例执行整机重启，也未改动云平台的开关机策略；此配置针对现有实例启动，不负责新实例创建或镜像迁移。

配置语义参考 [Supervisor 官方配置文档](https://supervisord.org/configuration.html)；平台管理器的 include 和程序启动能力参考其 [官方仓库](https://github.com/ochinchina/supervisord)。本机原版 Supervisor 和平台 Go 版本使用不同配置与控制 socket。
