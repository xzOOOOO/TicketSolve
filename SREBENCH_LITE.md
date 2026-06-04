# SREBench Lite Mini Lab

这个项目现在包含一个小型真实故障靶场，目录在 `lab/`。

它的目标是让工单 Agent 从“模拟诊断”升级为“真实观察、真实审批、真实白名单修复、可评测”：

1. 用 Docker Compose 启动真实服务。
2. 注入可重复的故障。
3. MCP 工具读取真实容器、HTTP、数据库和 Redis 信号。
4. LangGraph workflow 诊断并生成修复方案。
5. Executor 在 `docker_lab` 模式下执行白名单修复命令。
6. eval 输出诊断和修复结果表。

## 快速启动

```bash
python lab/chaos.py up
```

检查靶场：

```bash
curl http://localhost:18080/health
```

注入故障：

```bash
python lab/chaos.py inject DB_CONN_FAIL
```

恢复故障：

```bash
python lab/chaos.py recover DB_CONN_FAIL
```

## 启用真实执行

默认执行器仍是 mock，不会真实修复靶场。要让 Agent 真实执行白名单修复命令，需要设置：

```bash
EXECUTOR_MODE=docker_lab
```

Windows PowerShell 示例：

```powershell
$env:EXECUTOR_MODE="docker_lab"
cd src
python server.py
```

`docker_lab` 模式只允许固定白名单命令，例如：

- `python lab/chaos.py recover DB_CONN_FAIL`
- `python lab/chaos.py recover APP_PROCESS_DOWN`
- `python lab/chaos.py recover REDIS_DOWN`
- `python lab/chaos.py recover NGINX_BAD_ROUTE`
- `python lab/chaos.py recover DB_SLOW_QUERY`
- `docker start srebench-postgres`
- `docker start srebench-app`
- `docker start srebench-redis`
- `docker restart srebench-nginx`
- `docker exec srebench-postgres psql -U labuser -d labdb -c "create index if not exists idx_orders_status_created_at on orders (status, created_at desc);"`

修复步骤也可以使用结构化 Action DSL。Executor 会根据 `action_type + target`
编译成上述白名单命令，存在结构化动作时不依赖 LLM 生成的自由文本 `command`：

```json
{
  "action_type": "RECOVER_FAULT",
  "target": "APP_PROCESS_DOWN",
  "command": "python lab/chaos.py recover APP_PROCESS_DOWN"
}
```

```json
{
  "action_type": "START_CONTAINER",
  "target": "srebench-app"
}
```

不在白名单里的命令会被拒绝，返回 `exit_code=126`。

## 评测

先启动主工单 API：

```bash
cd src
python server.py
```

然后运行指定 case：

```bash
python eval/run_eval.py --cases DB_CONN_FAIL APP_PROCESS_DOWN REDIS_DOWN --json-out eval/reports/latest.json
```

评测脚本会先记录 `fixed_by_agent`，再做兜底恢复，避免下一条 case 被污染。

## 故障 Case

初始 case 集在 `eval/cases.yaml`：

- `DB_CONN_FAIL`
- `APP_PROCESS_DOWN`
- `REDIS_DOWN`
- `NGINX_BAD_ROUTE`
- `DB_SLOW_QUERY`

靶场故意保持很小。重点不是服务数量，而是提供可重复证据：Agent 能观察、诊断、审批、执行和评测。
