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

执行完成后，工作流会进入 Verify 节点，依次探测：

- `curl http://localhost:18080/health`
- `curl http://localhost:18080/cache/ping`
- `curl http://localhost:18080/orders/pending`

验证结果会写入 `execution_result`：

```json
{
  "verified": true,
  "verification_probe": [],
  "recovered_at": "2026-06-04T00:00:00+00:00"
}
```

如果 Executor 执行失败，会先进入 Replanner/Critic 节点。Replanner 会读取
`stderr`、`stdout` 和 `execution_trace`，判断失败类型并选择下一步：

- `retry`：环境临时不可用、连接拒绝、超时等，回到 `execute`
- `re-diagnose`：命令不在白名单、诊断目标不匹配、缺少工具上下文，回到诊断链路
- `rollback`：Executor 已执行回滚，停止继续修复并保存当前状态
- `escalate`：重规划预算耗尽或权限/高风险失败，保存并交由人工处理

工作流入口前还有 CaseMemory 节点，会从 `eval/case_library.json` 检索相似历史案例，
并将压缩后的 `case_context` 提供给 Supervisor 和 FixAgent。案例字段包括：

- `symptoms`
- `tool_evidence`
- `root_cause`
- `successful_repair_action`
- `verification`

如果本次工单最终 `verified=true`，Save 节点会把它沉淀回案例库，后续新工单可复用。

## 标准化 Trace

系统仍保留原有 `audit_logs` 作为人工审计日志，同时新增 `trace_events` 作为标准事件流，方便 eval 分析每一步成功/失败。事件会在保存工单时写入 `execution_result.trace_events`，`/api/tickets/{ticket_id}/agent-flow` 也会返回 `standard_trace`。

标准事件名固定为：

```text
agent_started
tool_called
observation_received
diagnosis_generated
handoff_requested
plan_generated
policy_checked
approval_received
action_executed
verification_passed
```

每条事件使用统一字段：

```json
{
  "schema_version": "trace.v1",
  "event_type": "action_executed",
  "ticket_id": "TKT-001",
  "agent_name": "executor",
  "status": "success",
  "timestamp": "2026-06-04T00:00:00+00:00",
  "input": {},
  "output": {},
  "error": null,
  "metadata": {}
}
```

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
