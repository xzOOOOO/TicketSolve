# 智能工单分诊与自动化处置系统

基于 LangGraph 多 Agent 协作的智能工单处理系统，实现工单自动分类、智能诊断、修复方案生成、人工审批和自动执行的全流程自动化。

## 项目简介

本项目是一个企业级 AI 工单处理系统，利用大语言模型（LLM）和 LangGraph 工作流编排技术，构建了一个多 Agent 协作的智能诊断与修复平台。系统能够自动分析故障现象、分类诊断类型、调用专业工具进行深度诊断、生成可执行的修复方案，并通过人工审批机制确保操作安全性。

### 核心特性

- **智能路由分类**：自动分析工单症状，精准分类为数据库/网络/应用/其他故障类型
- **多 Agent 协作**：DB Agent、Net Agent、App Agent 三大专业诊断 Agent 协同工作
- **工具调用诊断**：每个 Agent 配备专业工具集，实现深度故障诊断
- **修复方案生成**：基于诊断结果自动生成包含步骤、命令、回滚方案的完整修复计划
- **人工审批机制**：利用 LangGraph 的 `interrupt` 特性实现安全的人工审批中断点
- **异步架构**：全链路异步设计，支持高并发工单处理
- **持久化存储**：PostgreSQL 数据库实现工单全生命周期管理
- **RESTful API**：基于 FastAPI 提供标准化接口，支持 Swagger 文档

## 技术栈

### 核心框架

| 技术         | 版本      | 用途          |
| ---------- | ------- | ----------- |
| Python     | 3.10+   | 开发语言        |
| LangGraph  | 1.1.6   | 工作流编排与状态机   |
| LangChain  | 1.2.15  | LLM 抽象层与工具链 |
| FastAPI    | 0.136.0 | 异步 Web 框架   |
| SQLAlchemy | 2.0.49  | 异步 ORM      |
| Pydantic   | 2.12.5  | 数据验证与序列化    |

### 基础设施

| 技术            | 用途              |
| ------------- | --------------- |
| PostgreSQL    | 工单数据持久化         |
| asyncpg       | PostgreSQL 异步驱动 |
| Uvicorn       | ASGI 服务器        |
| python-dotenv | 环境变量管理          |

### AI 模型

- 支持 OpenAI 兼容接口（如通义千问、DeepSeek 等）
- 默认配置：通义千问 qwen3.5-flash

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ POST /tickets │  │ POST /tickets│  │ GET /tickets/{id}    │  │
│  │   创建工单    │  │  /{id}/approve│  │   查询工单详情       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
└─────────┼─────────────────┼──────────────────────┼──────────────┘
          │                 │                      │
          ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Workflow                         │
│                                                                 │
│  ┌─────────┐     ┌──────────────────────────┐                  │
│  │  Router  │────▶│   条件路由                │                  │
│  │  路由节点 │     │  diagnosis_type 判断      │                  │
│  └─────────┘     └────────┬─────────────────┘                  │
│                           │                                    │
│              ┌────────────┼────────────┐                       │
│              ▼            ▼            ▼                       │
│        ┌──────────┐ ┌──────────┐ ┌──────────┐                 │
│        │ DB Agent │ │ Net Agent│ │ App Agent│                 │
│        │ 数据库诊断│ │ 网络诊断  │ │ 应用诊断  │                 │
│        └────┬─────┘ └────┬─────┘ └────┬─────┘                 │
│             │            │            │                        │
│             └────────────┼────────────┘                        │
│                          ▼                                    │
│                   ┌──────────────┐                            │
│                   │  Fix Agent   │                            │
│                   │ 修复方案生成  │                            │
│                   └──────┬───────┘                            │
│                          │                                    │
│                          ▼                                    │
│                   ┌──────────────┐                            │
│                   │ Human Approval│◀── interrupt() 中断点      │
│                   │  人工审批节点  │                            │
│                   └──────┬───────┘                            │
│                          │                                    │
│                    approved?                                  │
│                   ┌──────┴──────┐                             │
│                   ▼             ▼                             │
│            ┌──────────┐    ┌────────┐                        │
│            │ Execute  │    │  END   │                        │
│            │ 执行修复  │    │  结束   │                        │
│            └────┬─────┘    └────────┘                        │
│                 │                                            │
│                 ▼                                            │
│            ┌──────────┐                                     │
│            │   END    │                                     │
│            └──────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PostgreSQL                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  tickets 表                                               │  │
│  │  - 工单基本信息 (ticket_id, symptom)                      │  │
│  │  - 诊断结果 (diagnosis_type, diagnosis_result)            │  │
│  │  - 修复方案 (fix_plan)                                    │  │
│  │  - 审批信息 (approval_status, approver_comments)          │  │
│  │  - 执行结果 (execution_result)                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 工作流状态模型

```
SystemState
├── 输入信息
│   ├── ticket_id: 工单ID
│   └── symptom: 故障现象描述
├── 路由决策
│   ├── diagnosis_type: 诊断类型 (db/net/app/other)
│   └── urgency: 紧急程度 (low/medium/high/critical)
├── Agent 诊断结果
│   ├── db_agent_result: 数据库诊断结果
│   ├── net_agent_result: 网络诊断结果
│   └── app_agent_result: 应用诊断结果
├── 修复方案
│   └── fix_plan: 包含步骤、命令、回滚方案
├── 人工审批
│   ├── approval_status: 审批状态 (pending/approved/rejected)
│   └── approver_comments: 审批意见
└── 执行结果
    └── execution_result: 执行步骤与结果
```

## 快速开始

### 环境要求

- Python 3.10+
- PostgreSQL 12+
- 有效的 LLM API Key（支持 OpenAI 兼容接口）

### 安装步骤

#### 1. 克隆项目

```bash
git clone <your-repo-url>
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
LLM_MODEL=qwen3.5-flash
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# LLM 限流配置
LLM_MAX_CONCURRENT=5  # 最大并发数
LLM_RPM_LIMIT=60  # 每分钟请求数

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

### 4. 健康检查

**接口**：`GET /health`

**响应示例**：

```json
{
  "status": "ok",
  "message": "AI工单系统运行中"
}
```

## 核心模块说明

### 项目结构

```
TicketSolve/
├── src/
│   ├── api.py              # FastAPI 路由定义
│   ├── config.py           # 配置管理
│   ├── database.py         # 数据库模型与操作
│   ├── logger.py           # 日志配置
│   ├── main.py             # 命令行入口
│   ├── nodes.py            # 工作流节点定义
│   ├── schemas.py          # Pydantic 数据模型
│   ├── server.py           # 服务器启动
│   ├── state.py            # 工作流状态定义
│   └── workflow.py         # 工作流编排
├── .env.example            # 环境变量模板
├── .gitignore
└── requirements.txt        # Python 依赖
```

### 模块详解

#### 1. 工作流编排 (`workflow.py`)

使用 LangGraph 的 `StateGraph` 构建有向图工作流：

- **条件路由**：根据 `diagnosis_type` 动态路由到对应 Agent
- **审批分支**：根据审批结果决定执行或终止
- **检查点机制**：使用 `MemorySaver` 保存工作流状态，支持中断恢复

#### 2. 节点定义 (`nodes.py`)

| 节点             | 功能                 | 工具集                                                                |
| -------------- | ------------------ | ------------------------------------------------------------------ |
| Router         | 分析故障症状，分类诊断类型和紧急程度 | 无                                                                  |
| DB Agent       | 数据库故障诊断            | check\_db\_connection, check\_db\_slow\_query, check\_db\_deadlock |
| Net Agent      | 网络故障诊断             | check\_network\_ping, check\_network\_dns                          |
| App Agent      | 应用故障诊断             | check\_app\_process, check\_app\_port                              |
| Fix Agent      | 生成修复方案（含步骤、命令、回滚）  | 无                                                                  |
| Human Approval | 人工审批中断点            | interrupt()                                                        |
| Executor       | 执行修复方案并保存工单        | 无                                                                  |

#### 3. 状态管理 (`state.py`)

定义完整的工单状态模型，包含：

- 输入信息、路由决策
- 各 Agent 诊断结果
- 修复方案、审批状态
- 执行结果、消息记录

#### 4. 数据库 (`database.py`)

- 使用 SQLAlchemy 2.0 异步 ORM
- Ticket 模型包含工单全生命周期字段
- 支持工单创建、更新、查询操作

#### 5. API 接口 (`api.py`)

- FastAPI 异步路由
- lifespan 管理应用生命周期
- 统一响应格式（`APIResponse`）

## 使用示例

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

### 添加新的诊断工具

在 `nodes.py` 中使用 `@tool` 装饰器定义新工具：

```python
@tool
def check_custom_metric(metric_name: str) -> str:
    """检查自定义指标"""
    # 实现你的检测逻辑
    return json.dumps({"metric": metric_name, "value": "xxx"})
```

然后在对应 Agent 的工具列表中添加：

```python
tools = [check_db_connection, check_db_slow_query, check_custom_metric]
```

### 添加新的诊断类型

1. 在 `state.py` 的 `DiagnosisType` 枚举中添加新类型
2. 创建新的 Agent 节点函数
3. 在 `workflow.py` 中注册节点和路由规则

### 替换为真实工具

将 Mock 工具替换为实际执行逻辑：

```python
@tool
def check_db_connection() -> str:
    """检查数据库连接状态"""
    import subprocess
    result = subprocess.run(["pg_isready"], capture_output=True, text=True)
    return json.dumps({
        "status": "ok" if result.returncode == 0 else "error",
        "output": result.stdout
    })
```

## 性能优化建议

1. **LLM 调用优化**
   - 使用流式输出减少等待时间
   - 添加缓存层避免重复调用
   - 设置合理的超时和重试策略
2. **数据库优化**
   - 为常用查询字段添加索引
   - 使用连接池管理数据库连接
   - 定期归档历史工单
3. **并发处理**
   - 使用 Redis 替代 MemorySaver 实现分布式状态保存
   - 多 worker 部署 Uvicorn
   - 添加消息队列处理异步任务

## 常见问题

### Q: LLM 返回的 JSON 解析失败怎么办？

系统内置了 `parse_json_content` 函数，支持多种 JSON 格式：

- 标准 JSON
- Markdown 代码块包裹的 JSON（`json ... `  ）
- 包含额外文本的 JSON

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

## 开发计划

- [ ] 实现真实诊断工具（SQL 查询、网络 ping 等）
- [ ] 添加 JWT 认证与角色权限
- [ ] 集成 Prometheus 监控指标
- [ ] 添加单元测试与集成测试
- [ ] Docker 容器化部署
- [ ] Webhook 通知（邮件/钉钉/企业微信）
- [ ] 前端管理界面
- [ ] 工单统计分析面板

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目 Issues：[GitHub Issues](your-repo-url/issues)
- 邮箱：<your-email@example.com>

***

**注意**：本系统目前使用的诊断工具为 Mock 实现，主要用于演示工作流和架构设计。生产环境需要替换为真实的诊断工具实现。
