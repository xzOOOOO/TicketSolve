# 智能工单分诊与自动化处置系统

基于 LangGraph 多 Agent 协作的智能工单处理系统，实现工单自动分类、智能诊断、Agent 间动态协作、修复方案生成、人工审批和自动执行的全流程自动化。

## 项目简介

本项目是一个企业级 AI 工单处理系统，利用大语言模型（LLM）和 LangGraph 工作流编排技术，构建了一个多 Agent 协作的智能诊断与修复平台。系统采用 Supervisor 调度架构，支持并行派发多个诊断 Agent，Agent 间通过通信总线动态协作，并使用 MCP（Model Context Protocol）标准化工具调用接口。

### 核心特性

- **Supervisor 智能调度**：Supervisor Agent 分析故障症状，智能决定派发哪些诊断 Agent，支持并行派发
- **多 Agent 并行诊断**：DB Agent、Net Agent、App Agent 三大专业诊断 Agent 并行执行，通过 asyncio.gather 实现高效并发
- **Agent 间动态协作**：Agent 通过 CommunicationBus 发布 hypothesis、evidence_request、evidence_response 等结构化协议消息，DynamicCheck 节点自动追加派发或补充证据响应，实现跨领域协作诊断
- **ReAct 推理循环**：诊断 Agent 采用 Think → Act → Observe 循环，多轮调用工具直至信息充足
- **MCP 工具集成**：基于 FastMCP 实现 MCP Server，通过 langchain-mcp-adapters 自动适配 LangChain 工具链，工具按类别分组注入各 Agent
- **Structured Output**：所有 Agent 使用 LLM with_structured_output，通过 Pydantic schema 约束输出格式，替代手动 JSON 解析
- **聚合推理**：Aggregate 节点综合多个 Agent 的诊断结果，加权判断给出最终诊断结论
- **修复方案生成**：基于聚合诊断结果自动生成包含步骤、命令、回滚方案的完整修复计划
- **人工审批机制**：利用 LangGraph 的 `interrupt` 特性实现安全的人工审批中断点
- **LLM 限流器**：令牌桶算法控制并发请求数和 RPM，通过 LangChain 回调机制自动拦截
- **审计日志**：全流程操作轨迹记录，支持工单处理流程可追溯
- **异步架构**：全链路异步设计，支持高并发工单处理
- **持久化存储**：PostgreSQL 数据库实现工单全生命周期管理
- **RESTful API**：基于 FastAPI 提供标准化接口，支持 Swagger 文档

## 简历展示 Demo

推荐用 `NGINX_BAD_ROUTE` 作为面试展示用例，因为它能同时体现“多 Agent 协作、证据覆盖调度、安全执行、恢复验证”：

```text
注入 NGINX_BAD_ROUTE
→ Supervisor 派发 app_agent / net_agent
→ app_agent 发现应用直连健康，发布 hypothesis
→ app_agent 请求 net_agent 提供 nginx route status / direct app health
→ DynamicCheck 发现旧 net_agent 结果只包含 ping，不覆盖证据请求
→ DynamicCheck 触发 targeted re-dispatch，强制 net_agent 重跑 route 工具
→ net_agent 返回 check_network_http_route 结构化证据
→ Aggregate 根据 hypothesis_scores 选择 NGINX_BAD_ROUTE
→ FixAgent 生成 Action DSL
→ Guardrail 校验通过
→ Executor 在 Docker Lab 中恢复 Nginx 配置
→ Verify 探测 /health 返回 200
```

生成一份不依赖 LLM/Docker 的展示轨迹：

```bash
python eval/demo_trace.py --json-out eval/demo_trace_nginx_bad_route.json
```

样例文件：[eval/demo_trace_nginx_bad_route.json](eval/demo_trace_nginx_bad_route.json)

重点字段：

```json
{
  "event_type": "handoff_requested",
  "agent_name": "dynamic_check",
  "metadata": {
    "coverage": false,
    "forced_redispatch": true,
    "required_evidence": ["nginx route status", "direct app health"],
    "suggested_tools": ["check_network_http_route"]
  }
}
```

这条 trace 用来展示系统“不是缓存里有 Agent 结果就糊一个响应”，而是先做 evidence coverage 判定；旧证据不覆盖请求时，会定向重跑目标 Agent。

## 项目质量检查

提交简历或演示前建议先跑以下命令：

```bash
python -m pytest -q
python -m compileall -q src tests eval
cd frontend
npm run build
```

当前已覆盖 Action DSL、Guardrail、Agent 协作协议、Trace 事件、案例库和 Replanner 等核心模块；完整链路演示需要额外准备 LLM API Key、PostgreSQL 和可选的 SREBench Lite Docker 靶场。

## 沙盒与安全边界

本项目没有让 LLM 直接执行任意 shell。修复执行链路被限制在以下边界内：

- LLM 只能生成结构化 Action DSL，例如 `{"action_type": "RECOVER_FAULT", "target": "NGINX_BAD_ROUTE"}`
- `RepairPlanner` 将修复计划规范化为平铺 Action DSL
- `Guardrail` 对 Action DSL 做确定性规则校验，拒绝危险动作和未知目标
- `Executor` 在 `docker_lab` 模式下只执行白名单命令
- 命令作用域限制在 `srebench-*` 容器、`lab/chaos.py` 故障注入/恢复和健康探测 URL
- 自由文本 `command` 只用于展示和日志；存在 `action_type + target` 时，以本地编译后的安全命令为准

这套边界适合简历项目的 SRE 靶场：既能展示真实闭环，又不会把工程量拖进完整虚拟化沙盒。

## 技术栈

### 核心框架

| 技术 | 用途 |
| --- | --- |
| Python 3.10+ | 开发语言 |
| LangGraph | 工作流编排与状态机 |
| LangChain | LLM 抽象层与工具链 |
| langchain-openai | OpenAI 兼容接口集成 |
| langchain-mcp-adapters | MCP 工具自动适配 LangChain |
| FastMCP | MCP Server 实现 |
| FastAPI | 异步 Web 框架 |
| SQLAlchemy 2.0 | 异步 ORM |
| Pydantic | 数据验证与 Structured Output |

### 基础设施

| 技术 | 用途 |
| --- | --- |
| PostgreSQL | 工单数据持久化 + 审计日志 |
| asyncpg | PostgreSQL 异步驱动 |
| Uvicorn | ASGI 服务器 |
| python-dotenv | 环境变量管理 |

### AI 模型

- 支持 OpenAI 兼容接口（如通义千问、DeepSeek 等）
- 默认配置：通义千问 qwen3.5-flash

## 系统架构

### 整体架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          FastAPI Server                              │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐  │
│  │ POST /tickets  │  │ POST /tickets │  │ GET /tickets/{id}      │  │
│  │   创建工单     │  │  /{id}/approve│  │   查询工单详情          │  │
│  └───────┬───────┘  └───────┬───────┘  └───────────┬─────────────┘  │
│  ┌───────┴───────┐  ┌───────┴───────┐  ┌───────────┴─────────────┐  │
│  │ GET /tickets/ │  │ GET /rate-    │  │                           │  │
│  │ {id}/agent-   │  │ limiter/stats │  │                           │  │
│  │ flow          │  │               │  │                           │  │
│  └───────┬───────┘  └───────┬───────┘  └───────────┬─────────────┘  │
└──────────┼──────────────────┼───────────────────────┼────────────────┘
           │                  │                       │
           ▼                  ▼                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       LangGraph Workflow                             │
│                                                                      │
│  ┌──────────────┐                                                   │
│  │  Supervisor   │ 分析症状，决定派发哪些 Agent                       │
│  │  调度主管     │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│    dispatched_agents?                                               │
│    ┌────┴────┐                                                      │
│    ▼         ▼                                                      │
│  有Agent   无Agent                                                  │
│    │         │                                                      │
│    ▼         ▼                                                      │
│ ┌──────────┐ ┌──────────────┐                                      │
│ │ Dispatch │ │Other Handler │──────▶ END                            │
│ │并行派发  │ │ 归档处理     │                                      │
│ └────┬─────┘ └──────────────┘                                      │
│      │                                                              │
│      ▼                                                              │
│ ┌──────────────┐    有协作请求    ┌──────────┐                      │
│ │DynamicCheck  │──────────────▶ │ Dispatch │ (追加派发，循环)       │
│ │ 动态检查     │──── 无请求 ────▶│          │                      │
│ └──────┬───────┘                 └──────────┘                      │
│        │ (无协作请求)                                              │
│        ▼                                                           │
│ ┌──────────────┐                                                   │
│ │  Aggregate   │ 综合多个 Agent 诊断结果                            │
│ │  聚合推理    │                                                   │
│ └──────┬───────┘                                                   │
│        ▼                                                           │
│ ┌──────────────┐                                                   │
│ │  Fix Agent   │ 生成修复方案                                      │
│ └──────┬───────┘                                                   │
│        ▼                                                           │
│ ┌──────────────┐                                                   │
│ │Human Approval│◀── interrupt() 中断点                             │
│ └──────┬───────┘                                                   │
│        │                                                           │
│   approved?                                                        │
│   ┌────┴────┐                                                      │
│   ▼         ▼                                                      │
│ Execute    END                                                      │
│ 执行修复                                                             │
│   │                                                                 │
│   ▼                                                                 │
│  END                                                                │
└──────────────────────────────────────────────────────────────────────┘
           │                  │
           ▼                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          MCP Server (FastMCP)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │ DB 诊断工具      │  │ Net 诊断工具     │  │ App 诊断工具    │     │
│  │ check_db_conn   │  │ check_net_ping  │  │ check_app_proc  │     │
│  │ check_db_slow   │  │ check_net_dns   │  │ check_app_port  │     │
│  │ check_db_dead   │  │                 │  │                 │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│  通过 langchain-mcp-adapters 自动转换为 LangChain BaseTool          │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          PostgreSQL                                  │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  tickets 表                                                 │     │
│  │  - 工单基本信息 (ticket_id, symptom)                        │     │
│  │  - 诊断结果 (diagnosis_type, diagnosis_result)              │     │
│  │  - 修复方案 (fix_plan)                                      │     │
│  │  - 审批信息 (approval_status, approver_comments)            │     │
│  │  - 执行结果 (execution_result)                              │     │
│  └────────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  ticket_audit_logs 表                                       │     │
│  │  - Agent 操作审计日志，记录每个 Agent 的完整操作轨迹          │     │
│  │  - 支持按 ticket_id 查询，还原工单处理流程                   │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

### 工作流状态模型

```
SystemState
├── 输入信息
│   ├── ticket_id: 工单ID
│   └── symptom: 故障现象描述
├── Supervisor 决策
│   ├── diagnosis_type: 诊断类型 (db/net/app/other)
│   ├── urgency: 紧急程度 (low/medium/high/critical)
│   ├── supervisor_decision: Supervisor 派发决策详情
│   └── dispatched_agents: 被派发的 Agent 列表
├── Agent 诊断结果
│   ├── db_agent_result: 数据库诊断结果
│   ├── net_agent_result: 网络诊断结果
│   └── app_agent_result: 应用诊断结果
├── 聚合诊断
│   └── aggregated_diagnosis: 综合诊断结果
├── 动态调度
│   ├── dispatch_round: 当前调度轮次
│   └── max_dispatch_rounds: 最大动态调度轮次 (默认3)
├── Agent 间通信
│   └── agent_messages: Agent 间通信消息 (追加式)
├── 修复方案
│   └── fix_plan: 包含步骤、命令、回滚方案
├── 人工审批
│   ├── approval_status: 审批状态 (pending/approved/rejected)
│   └── approver_comments: 审批意见
├── 执行结果
│   └── execution_result: 执行步骤与结果
└── 审计日志
    └── audit_logs: Agent 操作审计日志 (追加式)
```

## 快速开始

### 环境要求

- Python 3.10+
- PostgreSQL 12+
- 有效的 LLM API Key（支持 OpenAI 兼容接口）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/xzOOOOO/TicketSolve.git
cd TicketSolve
```

#### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# LLM 配置（支持 OpenAI 兼容接口）
LLM_API_KEY=your-api-key-here
LLM_MODEL=your-model-here
LLM_BASE_URL=your-url-here

# LLM 限流配置
LLM_MAX_CONCURRENT=5  # 最大并发数
LLM_RPM_LIMIT=60  # 每分钟请求数

# LLM 重试配置
LLM_MAX_RETRIES=3  # 最大重试次数
LLM_RETRY_EXPONENTIAL_JITTER=true  # 指数退避抖动

# 数据库配置
DB_USER=postgres
DB_PASSWORD=your-password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tickets
DB_ECHO=true

# 服务配置
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

#### 5. 初始化数据库

系统启动时会自动创建数据库表，确保 PostgreSQL 服务已运行且数据库已创建：

```sql
-- 在 PostgreSQL 中创建数据库
CREATE DATABASE tickets;
```

#### 6. 启动服务

```bash
cd src
python server.py
```

或使用 uvicorn 直接启动：

```bash
cd src
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：

- API 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

## API 接口文档

### 1. 创建工单

**接口**：`POST /api/tickets`

**请求体**：

```json
{
  "ticket_id": "TKT-001",
  "symptom": "数据库连接超时，应用无法访问"
}
```

**响应示例**：

```json
{
  "code": 200,
  "message": "工单已提交，等待审批",
  "data": {
    "ticket_id": "TKT-001",
    "status": "pending_approval",
    "next_step": "请调用 /api/tickets/{ticket_id}/approve 进行审批"
  }
}
```

### 2. 审批工单

**接口**：`POST /api/tickets/{ticket_id}/approve`

**请求体**：

```json
{
  "approved": true,
  "comments": "同意执行，请在业务低峰期操作"
}
```

**响应示例**：

```json
{
  "code": 200,
  "message": "审批完成",
  "data": {
    "ticket_id": "TKT-001",
    "approved": true,
    "final_result": {
      "execution_result": {
        "plan_id": "PLAN-001",
        "executed_steps": [...],
        "overall_status": "success"
      }
    }
  }
}
```

### 3. 查询工单详情

**接口**：`GET /api/tickets/{ticket_id}`

**响应示例**：

```json
{
  "code": 200,
  "message": "查询成功",
  "data": {
    "id": "uuid-string",
    "ticket_id": "TKT-001",
    "symptom": "数据库连接超时，应用无法访问",
    "diagnosis_type": "db",
    "urgency": "high",
    "status": "completed",
    "diagnosis_result": {
      "diagnosis": "数据库连接池耗尽",
      "possible_causes": ["连接数配置过小", "存在慢查询占用连接"]
    },
    "fix_plan": {
      "plan_id": "PLAN-001",
      "description": "重启连接池并优化慢查询",
      "steps": [...]
    },
    "approval_status": "approved",
    "execution_result": {
      "overall_status": "success"
    },
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:05:00"
  }
}
```

### 4. 查询工单 Agent 执行流程

**接口**：`GET /api/tickets/{ticket_id}/agent-flow`

返回工单的 Agent 执行流程，包含每个 Agent 的操作轨迹、调度轮次等信息，用于追溯和可视化。

**响应示例**：

```json
{
  "code": 200,
  "message": "查询成功",
  "data": {
    "ticket_id": "TKT-001",
    "diagnosis_type": "db",
    "urgency": "high",
    "status": "completed",
    "dispatched_agents": ["supervisor", "db_agent", "fix_agent"],
    "agent_summary": {
      "supervisor": {
        "actions": ["dispatch"],
        "dispatch_rounds": [0]
      },
      "db_agent": {
        "actions": ["tool_call", "diagnosis"],
        "dispatch_rounds": [1]
      },
      "fix_agent": {
        "actions": ["fix_plan"],
        "dispatch_rounds": [1]
      }
    },
    "flow_steps": [...],
    "total_steps": 5
  }
}
```

### 5. 健康检查

**接口**：`GET /health`

**响应示例**：

```json
{
  "status": "ok",
  "message": "AI工单系统运行中"
}
```

### 6. 查询限流器状态

**接口**：`GET /api/rate-limiter/stats`

**响应示例**：

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "max_concurrent": 5,
    "rpm_limit": 60,
    "current_rpm": 12,
    "available_capacity": 48
  }
}
```

## 核心模块说明

### 项目结构

```
TicketSolve/
├── src/
│   ├── agents/                # Agent 模块
│   │   ├── __init__.py        # Agent 导出
│   │   ├── base.py            # Agent 抽象基类（含 ReAct 循环）
│   │   ├── supervisor.py      # Supervisor 调度主管 Agent
│   │   ├── db.py              # 数据库诊断 Agent
│   │   ├── net.py             # 网络诊断 Agent
│   │   ├── app.py             # 应用诊断 Agent
│   │   ├── fix.py             # 修复方案生成 Agent
│   │   └── communication.py   # Agent 间通信总线
│   ├── agent_protocol/        # 多 Agent 证据协作协议
│   │   ├── messages.py        # hypothesis/evidence_request/response 构造
│   │   ├── coordination.py    # pending request、自动响应
│   │   ├── coverage.py        # evidence coverage 判定与定向重派发依据
│   │   ├── context.py         # 协议上下文构造
│   │   └── scoring.py         # hypothesis_scores 可解释裁决
│   ├── evidence/              # Typed Evidence 证据模型与清洗/转换/组装
│   ├── api.py                 # FastAPI 路由定义
│   ├── config.py              # 配置管理（含重试配置）
│   ├── database.py            # 数据库模型与操作（含审计日志）
│   ├── llm_rate_limiter.py    # LLM 请求限流器
│   ├── logger.py              # 日志配置
│   ├── main.py                # 命令行入口
│   ├── mcp_server.py          # MCP Server（FastMCP 诊断工具）
│   ├── nodes.py               # 工作流节点（dispatch/aggregate/approval/executor）
│   ├── prompts.py             # Prompt 模板定义
│   ├── schemas.py             # Pydantic 数据模型 + Structured Output 模型
│   ├── server.py              # 服务器启动入口
│   ├── state.py               # 工作流状态定义
│   ├── utils.py               # 工具调用执行辅助
│   └── workflow.py            # 工作流编排（含 MCP Client 初始化）
├── eval/
│   ├── cases.yaml             # SREBench Lite 评测用例
│   ├── demo_trace.py          # 生成 NGINX_BAD_ROUTE 展示轨迹
│   └── demo_trace_nginx_bad_route.json
├── .env.example               # 环境变量模板
├── .gitignore
├── LICENSE
└── requirements.txt           # Python 依赖
```

### 模块详解

#### 1. 工作流编排 (`workflow.py`)

使用 LangGraph 的 `StateGraph` 构建有向图工作流：

- **Supervisor 调度**：替代原 Router，支持并行派发多个 Agent
- **Dispatch 并行执行**：根据 dispatched_agents 列表，asyncio.gather 并行调用
- **DynamicCheck 动态协作**：扫描 evidence_request 消息，先做 evidence coverage 判定；证据覆盖则自动生成 evidence_response，证据不足则定向重派发目标 Agent（最多 3 轮）
- **Aggregate 聚合推理**：综合多个 Agent 诊断结果，使用 Structured Output
- **审批分支**：根据审批结果决定执行或终止
- **MCP Client 初始化**：工作流创建时一次性初始化 MCP 连接，获取所有工具并按类别分组注入各 Agent
- **检查点机制**：使用 `MemorySaver` + `JsonPlusSerializer` 保存工作流状态，支持中断恢复

#### 2. Agent 架构 (`agents/`)

所有 Agent 继承自 `BaseAgent`，核心设计：

| Agent | 职责 | 工具来源 | Structured Output |
| --- | --- | --- | --- |
| SupervisorAgent | 分析症状，决定派发哪些 Agent | 无 | SupervisorDecisionOutput |
| DBAgent | 数据库故障诊断 | MCP: check_db_* | DiagnosisOutput |
| NetAgent | 网络故障诊断 | MCP: check_network_* | DiagnosisOutput |
| AppAgent | 应用故障诊断 | MCP: check_app_* | DiagnosisOutput |
| FixAgent | 生成修复方案（含步骤、命令、回滚） | 无 | FixPlanOutput |

**BaseAgent 核心能力**：
- `react_loop()`：ReAct 推理循环（Think → Act → Observe），最多 3 轮工具调用
- `run()`：抽象方法，子类实现具体诊断/修复逻辑

**CommunicationBus 通信机制**：
- `send()`：向指定 Agent 发送消息
- `broadcast()`：广播消息给所有 Agent
- `publish_hypothesis()`：发布可验证故障假设
- `request_evidence()`：请求指定 Agent 补充证据
- `respond_evidence()`：响应证据请求
- `receive()`：获取发给指定 Agent 的消息（含广播）
- 消息类型：hypothesis / evidence_request / evidence_response / support / challenge / diagnosis

#### 3. MCP Server (`mcp_server.py`)

基于 FastMCP 实现的独立工具服务进程：

| 工具类别 | 工具名 | 功能 |
| --- | --- | --- |
| 数据库 | check_db_connection | 检查数据库连接状态 |
| 数据库 | check_db_slow_query | 检查慢查询 |
| 数据库 | check_db_deadlock | 检查死锁 |
| 网络 | check_network_ping | 检查网络连通性 |
| 网络 | check_network_dns | 检查 DNS 解析 |
| 应用 | check_app_process | 检查应用进程状态 |
| 应用 | check_app_port | 检查应用端口状态 |

通过 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 以 stdio 模式启动 MCP Server 子进程，自动将 MCP 工具转换为 LangChain `BaseTool`，按工具名前缀分组注入各诊断 Agent。

#### 4. 状态管理 (`state.py`)

定义完整的工单状态模型，包含：

- 输入信息、Supervisor 决策
- 各 Agent 诊断结果、聚合诊断
- 动态调度信息（dispatch_round、max_dispatch_rounds）
- Agent 间通信消息（追加式 operator.add）
- 修复方案、审批状态
- 执行结果、审计日志（追加式 operator.add）

#### 5. 数据库 (`database.py`)

- 使用 SQLAlchemy 2.0 异步 ORM
- **Ticket 模型**：工单全生命周期字段
- **TicketAuditLog 模型**：Agent 操作审计日志，支持按 ticket_id 查询完整处理流程
- 支持工单创建、更新、查询操作
- `save_ticket()` 同时保存审计日志到 ticket_audit_logs 表

#### 6. LLM 限流器 (`llm_rate_limiter.py`)

- **LLMRateLimiter**：令牌桶算法，控制并发请求数和 RPM
- **RateLimitCallback**：LangChain AsyncCallbackHandler，挂载到 ChatOpenAI 的 callbacks 参数上
- 每次 LLM 调用前自动 acquire，调用后自动 release
- with_structured_output / bind_tools 产生的内部调用也会被拦截

#### 7. API 接口 (`api.py`)

- FastAPI 异步路由
- lifespan 管理应用生命周期（初始化数据库、LLM、限流器、工作流）
- 统一响应格式（`APIResponse`）
- 新增 `/api/tickets/{id}/agent-flow` 查询 Agent 执行流程
- 新增 `/api/rate-limiter/stats` 查询限流器状态

## 使用示例

### 生成展示 Trace

用于面试/README 截图的固定轨迹，不依赖 LLM、Docker 或数据库：

```bash
python eval/demo_trace.py --json-out eval/demo_trace_nginx_bad_route.json
```

查看 coverage 调度关键事件：

```bash
python eval/demo_trace.py | findstr /C:"forced_redispatch" /C:"hypothesis_scores"
```

### 命令行模式

```bash
cd src
python main.py
```

### API 调用示例

使用 curl：

```bash
# 1. 创建工单
curl -X POST http://localhost:8000/api/tickets \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT-001",
    "symptom": "数据库连接超时，应用无法访问"
  }'

# 2. 审批工单
curl -X POST http://localhost:8000/api/tickets/TKT-001/approve \
  -H "Content-Type: application/json" \
  -d '{
    "approved": true,
    "comments": "同意执行"
  }'

# 3. 查询工单
curl http://localhost:8000/api/tickets/TKT-001

# 4. 查询 Agent 执行流程
curl http://localhost:8000/api/tickets/TKT-001/agent-flow

# 5. 查询限流器状态
curl http://localhost:8000/api/rate-limiter/stats
```

使用 Python requests：

```python
import requests

BASE_URL = "http://localhost:8000"

# 创建工单
response = requests.post(f"{BASE_URL}/api/tickets", json={
    "ticket_id": "TKT-001",
    "symptom": "数据库连接超时"
})
print(response.json())

# 审批工单
response = requests.post(f"{BASE_URL}/api/tickets/TKT-001/approve", json={
    "approved": True,
    "comments": "同意执行"
})
print(response.json())

# 查询工单
response = requests.get(f"{BASE_URL}/api/tickets/TKT-001")
print(response.json())

# 查询 Agent 执行流程
response = requests.get(f"{BASE_URL}/api/tickets/TKT-001/agent-flow")
print(response.json())
```

## 故障诊断类型

系统支持以下故障类型的自动诊断：

### 数据库问题 (db)

- 连接超时/连接池耗尽
- 慢查询
- 死锁
- 索引缺失

### 网络问题 (net)

- 网络连通性
- DNS 解析
- 延迟/丢包
- 防火墙拦截

### 应用问题 (app)

- 进程异常（CPU/内存过高）
- 端口占用
- 服务崩溃
- 线程阻塞

### 其他问题 (other)

- 配置错误
- 权限问题
- 第三方服务异常

## 扩展开发

### 添加新的 MCP 诊断工具

在 `mcp_server.py` 中使用 `@mcp.tool()` 装饰器定义新工具：

```python
@mcp.tool()
def check_db_replication_lag() -> str:
    """检查数据库复制延迟"""
    result = {
        "replication_lag": "5s",
        "status": "warning",
        "possible_issue": "主从复制延迟过高"
    }
    return json.dumps(result, ensure_ascii=False)
```

工具会自动被 MCP Client 加载，按名称前缀分组注入对应 Agent。

### 添加新的诊断 Agent

1. 在 `agents/` 目录创建新的 Agent 类，继承 `BaseAgent`
2. 实现 `run()` 方法，使用 `react_loop()` 进行工具调用
3. 在 `agents/__init__.py` 中导出
4. 在 `workflow.py` 中注册 Agent runner 和工作流节点

### 替换为真实工具

将 MCP Server 中的 Mock 工具替换为实际执行逻辑：

```python
@mcp.tool()
def check_db_connection() -> str:
    """检查数据库连接状态"""
    import subprocess
    result = subprocess.run(["pg_isready"], capture_output=True, text=True)
    return json.dumps({
        "status": "ok" if result.returncode == 0 else "error",
        "output": result.stdout
    })
```

### 添加新的 Structured Output 模型

在 `schemas.py` 中定义 Pydantic 模型，然后在 Agent 中使用 `llm.with_structured_output()` 约束输出格式。

## 性能优化建议

1. **LLM 调用优化**
   - 已集成限流器（令牌桶算法），控制并发和 RPM
   - 已使用 with_retry 配置重试策略（指数退避抖动）
   - 可添加缓存层避免重复调用
2. **数据库优化**
   - 为常用查询字段添加索引
   - 使用连接池管理数据库连接
   - 定期归档历史工单和审计日志
3. **并发处理**
   - 使用 Redis 替代 MemorySaver 实现分布式状态保存
   - 多 worker 部署 Uvicorn
   - 添加消息队列处理异步任务
4. **MCP 扩展**
   - 支持 SSE 传输模式实现远程工具调用
   - 添加更多专业诊断工具

## 常见问题

### Q: LLM 返回的 JSON 解析失败怎么办？

系统已全面使用 `with_structured_output` 替代手动 JSON 解析。LLM 通过 function calling 机制直接返回符合 Pydantic schema 的结构化数据，无需手动解析 JSON 字符串。同时内置了兜底处理，当 Structured Output 返回 None 时使用默认值。

### Q: 如何更换 LLM 提供商？

修改 `.env` 文件中的 LLM 配置即可，系统支持任何 OpenAI 兼容接口：

```env
LLM_API_KEY=your-key
LLM_MODEL=your-model
LLM_BASE_URL=https://your-provider/v1
```

### Q: 审批流程可以跳过吗？

可以。修改 `workflow.py` 中的路由逻辑，将 `human_approval` 节点直接连接到 `execute` 节点即可实现自动执行。

### Q: 如何持久化工作流状态？

当前使用 `MemorySaver`，生产环境建议切换到 PostgreSQL checkpointer：

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver(connection_string="postgresql://...")
app = workflow.compile(checkpointer=checkpointer)
```

### Q: Agent 间如何协作？

Agent 通过 CommunicationBus 通信总线协作（证据协作协议 v1）：
1. 诊断 Agent 发布 `hypothesis`，声明当前故障假设和证据
   - hypothesis：一句话可验证假设，例如"Postgres 容器停止导致数据库连接失败"
   - evidence：支持该假设的结构化证据列表，包含 source_agent、tool_name、target、status、observed、expected、supports_hypothesis、confidence
2. 需要跨域确认时，Agent 发送 `evidence_request`
   - 指定 target_agent（找谁帮忙）、required_evidence（需要什么证据）、reason（为什么需要）、suggested_tools（建议用什么工具查）
3. DynamicCheck 节点扫描未响应的证据请求，自动追加派发目标 Agent
   - 如果目标 Agent 尚未执行 → 追加派发
   - 如果目标 Agent 已有结果且证据覆盖 → 自动生成 evidence_response
   - 如果目标 Agent 已有结果但证据不足 → 定向重跑
4. 同一个 evidence_request 最多触发一次定向重跑，避免多 Agent 协作进入无限循环
5. Aggregate 节点根据 hypothesis、evidence_response、support/challenge 进行冲突裁决
   - 输出 protocol_summary，包含 winning_hypothesis_id、hypothesis_scores（support_score/tool_evidence_score/confidence_score/conflict_score/final_score）、conflicts

### Q: 如何查看工单的完整处理流程？

调用 `GET /api/tickets/{ticket_id}/agent-flow` 接口，返回包含所有 Agent 操作轨迹的审计日志，可还原完整的工单处理流程。

### Q: LLM 请求频率过高怎么办？

系统已内置 LLMRateLimiter 限流器，通过 `.env` 配置：
- `LLM_MAX_CONCURRENT`：最大并发请求数
- `LLM_RPM_LIMIT`：每分钟请求数上限

限流器通过 LangChain 回调机制自动拦截所有 LLM 调用（包括 with_structured_output / bind_tools 产生的内部调用）。

## 开发计划

- [x] 添加核心单元测试与离线 demo trace
- [x] 完成基础前端工作台
- [ ] 扩展真实诊断工具（SQL 查询、网络 ping 等）
- [ ] 添加 JWT 认证与角色权限
- [ ] 集成 Prometheus 监控指标
- [ ] 补充 API 层集成测试
- [ ] Docker 容器化部署
- [ ] Webhook 通知（邮件/钉钉/企业微信）
- [ ] 工单统计分析面板
- [ ] MCP Server 支持 SSE 传输模式
- [ ] Redis 分布式检查点替代 MemorySaver

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

[MIT License](LICENSE)
