"""
LangGraph 工作流定义 - Multi-Agent 架构

工作流结构:
    CaseMemory → Supervisor → Dispatch(并行派发) → DynamicCheck → [有协作请求?]
                                                    ├─ 是 → Dispatch(追加派发) → DynamicCheck → ...
                                                    └─ 否 → Aggregate(聚合推理) → Fix → RepairPlanner → Guardrail → [通过?]
                                                                                              ├─ 是 → Human Approval → Executor(闭环) → Replanner → Verify/Retry/Re-diagnose/Save
                                                                                              └─ 否 → END(方案被拦截)
                    ↓ (other/无Agent)
                Other Handler → END

核心改造:
- Supervisor 替代原 Router，支持并行派发多个 Agent
- CaseMemory 检索历史相似案例，供 Supervisor/FixAgent 复用
- Dispatch 节点并行执行被派发的 Agent
- DynamicCheck 节点扫描 Agent 间 evidence_request 消息，动态追加派发或自动补证据响应
- Aggregate 节点综合多个 Agent 的诊断结果
- Fix Agent 优先使用聚合诊断结果
- RepairPlanner 规范化 Action DSL 并生成可审计命令
- Agent 间通过 CommunicationBus 通信
- Guardrail 确定性安全护栏：用代码规则约束 LLM 输出边界
- Executor 闭环执行器：Observe → Decide → Act 循环
- Replanner/Critic：执行失败后读取 trace 并选择 retry/re-diagnose/rollback/escalate
- Verify 恢复验证节点：探测 /health、/cache/ping、/orders/pending
- Save 统一归档节点：执行和验证完成后保存工单

MCP集成说明:
- 使用 langchain-mcp-adapters 的 MultiServerMCPClient
- 工作流创建时一次性初始化 MCP 连接，获取所有工具
- 按类别分组传递给各 Agent 节点，节点内部不再管理连接

技术栈:
- LangGraph: 状态图编排
- langchain-mcp-adapters: MCP工具自动适配LangChain
- FastMCP: MCP Server实现
"""

# os：用于文件路径操作，获取 MCP Server 脚本路径
import os
# sys：用于获取当前 Python 解释器路径，启动 MCP Server 子进程
import sys

# StateGraph：LangGraph 状态图构建器，用于定义工作流节点和边
# END：LangGraph 特殊节点，表示工作流结束
from langgraph.graph import StateGraph, END
# MemorySaver：内存检查点保存器，用于保存/恢复工作流状态（支持人机交互断点续跑）
from langgraph.checkpoint.memory import MemorySaver
# JsonPlusSerializer：扩展 JSON 序列化器，支持 Pydantic 模型和自定义类型的序列化
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
# MultiServerMCPClient：langchain-mcp-adapters 提供的 MCP 客户端，支持多服务器连接
from langchain_mcp_adapters.client import MultiServerMCPClient

# SystemState：工作流全局状态模型；ApprovalStatus：审批状态枚举
from state import SystemState, ApprovalStatus

# SupervisorAgent：调度中心，决定派发哪些 Agent
# DBAgent/NetAgent/AppAgent：专业诊断 Agent
# FixAgent：修复方案生成 Agent
# CommunicationBus：Agent 间通信总线
from agents import (
    SupervisorAgent,
    DBAgent,
    NetAgent,
    AppAgent,
    FixAgent,
    CommunicationBus,
)

# 各工作流节点的工厂函数，每个函数返回一个异步节点函数
from workflow_nodes import (
    create_case_memory_node,      # 案例记忆检索节点
    create_dispatch_node,         # 并行派发节点
    create_dynamic_check_node,    # 动态检查节点（证据协作协议调度器）
    create_aggregate_node,        # 聚合诊断节点
    create_repair_planner_node,   # 修复规划节点（规范化 Action DSL）
    create_guardrail_node,        # 安全护栏节点
    create_human_approval_node,   # 人工审批节点
    create_executor_node,         # 闭环执行器节点
    create_replanner_node,        # 重规划节点（Critic）
    create_verify_node,           # 恢复验证节点
    create_save_node,             # 统一归档节点
    create_other_handler_node,    # 其他处理节点（非技术工单）
)

# logger：项目统一日志记录器
from logger import logger


def route_after_supervisor(state: SystemState) -> str:
    """Supervisor 后路由：有派发Agent则进入dispatch，否则走other_handler"""
    if state.dispatched_agents:
        return "dispatch"
    return "other_handler"


def route_after_dynamic_check(state: SystemState) -> str:
    """DynamicCheck 后路由：有新派发Agent则循环回dispatch，否则进入aggregate"""
    if state.dispatched_agents:
        return "dispatch"
    return "aggregate"


def route_by_approval(state: SystemState) -> str:
    """审批后路由：批准则执行，拒绝则保存审批结果"""
    if state.approval_status == ApprovalStatus.APPROVED:
        return "execute"
    return "save"


def route_after_guardrail(state: SystemState) -> str:
    """护栏后路由：通过则进入人工审批，未通过则结束（方案被拦截）"""
    if state.guardrail_result and state.guardrail_result.get("passed", False):
        return "human_approval"
    return END


def route_after_replanner(state: SystemState) -> str:
    """Replanner 后路由：根据 Critic 决策进入验证、重试、重诊断或保存。"""
    decision = (state.replanner_result or {}).get("decision", "escalate")
    if decision == "verify":
        return "verify"
    if decision == "retry":
        return "execute"
    if decision == "re-diagnose":
        return "dispatch"
    return "save"


def _get_mcp_server_path() -> str:
    """获取 MCP Server 脚本绝对路径

    MCP Server 是一个独立进程，通过 stdio 与主程序通信。
    它暴露了一系列诊断工具（如 check_db_connection、check_network_ping 等），
    供各 Agent 调用。

    返回：
        mcp_server.py 的绝对路径字符串
    """
    # current_dir：当前文件（workflow.py）所在目录，即 src/ 目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接得到 mcp_server.py 的完整路径
    return os.path.join(current_dir, "mcp_server.py")


def _classify_tools(all_tools):
    """按工具名前缀分类

    MCP Server 返回的工具名有统一前缀：
    - check_db_xxx：数据库诊断工具
    - check_network_xxx：网络诊断工具
    - check_app_xxx：应用诊断工具

    参数：
        all_tools: MCP 返回的所有工具列表

    返回：
        (db_tools, net_tools, app_tools) 三元组
    """
    # db_tools：数据库相关工具，如 check_db_connection、check_db_slow_queries
    db_tools = [t for t in all_tools if t.name.startswith("check_db_")]
    # net_tools：网络相关工具，如 check_network_ping、check_network_http_route
    net_tools = [t for t in all_tools if t.name.startswith("check_network_")]
    # app_tools：应用相关工具，如 check_app_process、check_app_memory
    app_tools = [t for t in all_tools if t.name.startswith("check_app_")]
    return db_tools, net_tools, app_tools


async def create_async_workflow(llm, checkpointer=None):
    """
    创建 Multi-Agent 异步工作流

    流程:
    1. CaseMemory 检索历史相似案例
    2. Supervisor 分析症状，决定派发哪些 Agent
    3. Dispatch 并行执行被派发的 Agent
    4. Aggregate 综合各 Agent 诊断结果
    5. Fix Agent 生成修复方案
    6. RepairPlanner 规范化修复计划
    7. Human Approval 人工审批
    8. Executor 执行修复
    9. Replanner/Critic 判定执行结果并决策下一步
    10. Verify 验证恢复
    11. Save 保存工单

    MCP Client 在此处一次性初始化:
    1. 启动 MCP Server 子进程 (stdio)
    2. 获取所有工具并按类别分组
    3. 将分组工具注入各 Agent 节点
    4. 节点内部只使用工具，不管理连接
    """
    # ═══════════════════════════════════════════════
    # 初始化 MCP Client（一次性）
    # ═══════════════════════════════════════════════
    # mcp_server_path：MCP Server 脚本路径，用于启动子进程
    mcp_server_path = _get_mcp_server_path()
    # mcp_client：MCP 客户端，通过 stdio 与 MCP Server 子进程通信
    # transport="stdio" 表示使用标准输入输出进行进程间通信
    mcp_client = MultiServerMCPClient(
        {
            "diagnosis": {
                "transport": "stdio",           # 通信方式：标准输入输出
                "command": sys.executable,      # 使用当前 Python 解释器启动
                "args": [mcp_server_path],      # 传入 mcp_server.py 路径作为参数
            }
        }
    )

    # 获取所有 MCP 工具（自动转换为 LangChain BaseTool 对象）
    # 每个工具都有 name、description、args_schema 等属性
    all_tools = await mcp_client.get_tools()
    logger.info(f"MCP工具加载完成，共 {len(all_tools)} 个: {[t.name for t in all_tools]}")

    # 按类别分组：把工具按 db/network/app 分类，分别注入对应 Agent
    db_tools, net_tools, app_tools = _classify_tools(all_tools)
    logger.info(f"工具分组 - DB: {len(db_tools)}, Net: {len(net_tools)}, App: {len(app_tools)}")

    # communication_bus：Agent 间通信总线，用于传递 evidence_request/evidence_response 消息
    communication_bus = CommunicationBus()

    # ═══════════════════════════════════════════════
    # 初始化各 Agent
    # ═══════════════════════════════════════════════
    # supervisor_agent：调度中心，只依赖 LLM，不需要工具
    supervisor_agent = SupervisorAgent(llm)
    # db_agent：数据库诊断 Agent，注入数据库工具和通信总线
    db_agent = DBAgent(llm, db_tools, communication_bus)
    # net_agent：网络诊断 Agent，注入网络工具和通信总线
    net_agent = NetAgent(llm, net_tools, communication_bus)
    # app_agent：应用诊断 Agent，注入应用工具和通信总线
    app_agent = AppAgent(llm, app_tools, communication_bus)
    # fix_agent：修复方案生成 Agent，只依赖 LLM，不需要工具
    fix_agent = FixAgent(llm)

    # 构建 Agent runner 映射（供 dispatch 节点并行调用）
    # key 是 Agent 名称，value 是 Agent 的 run 方法（异步函数）
    agent_runners = {
        "db_agent": db_agent.run,
        "net_agent": net_agent.run,
        "app_agent": app_agent.run,
    }

    # ═══════════════════════════════════════════════
    # 创建工作流节点实例
    # ═══════════════════════════════════════════════
    case_memory_node = create_case_memory_node()           # 案例记忆检索
    dispatch_node = create_dispatch_node(agent_runners)    # 并行派发（传入 runner 映射）
    dynamic_check_node = create_dynamic_check_node()       # 动态检查
    aggregate_node = create_aggregate_node(llm, communication_bus)  # 聚合诊断
    repair_planner_node = create_repair_planner_node()     # 修复规划
    guardrail_node = create_guardrail_node()               # 安全护栏
    human_approval_node = create_human_approval_node()     # 人工审批
    # executor_node：闭环执行器，传入 LLM 用于执行过程中的错误分析
    executor_node = create_executor_node(llm)
    replanner_node = create_replanner_node()               # 重规划（Critic）
    verify_node = create_verify_node()                     # 恢复验证
    save_node = create_save_node()                         # 统一归档
    other_handler_node = create_other_handler_node()       # 其他处理

    # 构建状态图
    workflow = StateGraph(SystemState)

    # 添加节点
    workflow.add_node("case_memory", case_memory_node)
    workflow.add_node("supervisor", supervisor_agent.run)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("dynamic_check", dynamic_check_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("fix_agent", fix_agent.run)
    workflow.add_node("repair_planner", repair_planner_node)
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute", executor_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("save", save_node)
    workflow.add_node("other_handler", other_handler_node)

    # 设置入口
    workflow.set_entry_point("case_memory")

    # CaseMemory → Supervisor
    workflow.add_edge("case_memory", "supervisor")

    # Supervisor → 有Agent派发则走dispatch，否则走other_handler
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"dispatch": "dispatch", "other_handler": "other_handler"},
    )

    # Dispatch → DynamicCheck（检查是否需要追加派发）
    workflow.add_edge("dispatch", "dynamic_check")

    # DynamicCheck → 有协作请求则循环回dispatch，否则进入aggregate
    workflow.add_conditional_edges(
        "dynamic_check",
        route_after_dynamic_check,
        {"dispatch": "dispatch", "aggregate": "aggregate"},
    )

    # Aggregate → Fix → RepairPlanner → Guardrail（确定性安全检查）
    workflow.add_edge("aggregate", "fix_agent")
    workflow.add_edge("fix_agent", "repair_planner")
    workflow.add_edge("repair_planner", "guardrail")

    # Guardrail → 通过则进入人工审批，未通过则结束（方案被拦截）
    workflow.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"human_approval": "human_approval", END: END},
    )

    # 审批后路由：批准则执行，拒绝则保存审批结果
    workflow.add_conditional_edges(
        "human_approval",
        route_by_approval,
        {"execute": "execute", "save": "save"},
    )

    # 执行完成 → Replanner 判定 → 验证/重试/重诊断/保存
    workflow.add_edge("execute", "replanner")
    workflow.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "verify": "verify",
            "execute": "execute",
            "dispatch": "dispatch",
            "save": "save",
        },
    )
    workflow.add_edge("verify", "save")
    workflow.add_edge("save", END)
    workflow.add_edge("other_handler", END)

    # ═══════════════════════════════════════════════
    # 编译工作流（带检查点）
    # ═══════════════════════════════════════════════
    # 配置序列化器以支持自定义类型（如 state.DiagnosisType、state.FixPlan）
    # 默认的 JSON 序列化器无法处理 Pydantic 模型和 Enum，需要显式声明允许的模块
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("state", "FixPlan"),         # 修复方案模型
            ("state", "FixStep"),         # 修复步骤模型
            ("state", "ApprovalStatus"),  # 审批状态枚举
            ("state", "DiagnosisType"),   # 诊断类型枚举
            ("state", "Urgency"),         # 紧急程度枚举
        ]
    )

    # checkpointer：检查点保存器，用于保存/恢复工作流状态
    # 如果调用方没有传入，则使用默认的 MemorySaver（内存保存，进程重启后丢失）
    if checkpointer is None:
        checkpointer = MemorySaver(serde=serde)

    # 编译工作流：将状态图转换为可执行的应用对象
    # checkpointer 参数启用断点续跑功能，支持 human_approval 等中断节点
    app = workflow.compile(checkpointer=checkpointer)

    # 返回编译后的工作流应用对象，供外部调用 invoke/ainvoke
    return app
