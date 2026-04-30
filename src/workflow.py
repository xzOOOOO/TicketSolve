"""
LangGraph 工作流定义 - Multi-Agent 架构

工作流结构:
    Supervisor → Dispatch(并行派发) → DynamicCheck → [有协作请求?]
                                                    ├─ 是 → Dispatch(追加派发) → DynamicCheck → ...
                                                    └─ 否 → Aggregate(聚合推理) → Fix → Human Approval → Executor
                    ↓ (other/无Agent)
                Other Handler → END

核心改造:
- Supervisor 替代原 Router，支持并行派发多个 Agent
- Dispatch 节点并行执行被派发的 Agent
- DynamicCheck 节点扫描 Agent 间 request_help 消息，动态追加派发
- Aggregate 节点综合多个 Agent 的诊断结果
- Fix Agent 优先使用聚合诊断结果
- Agent 间通过 CommunicationBus 通信

MCP集成说明:
- 使用 langchain-mcp-adapters 的 MultiServerMCPClient
- 工作流创建时一次性初始化 MCP 连接，获取所有工具
- 按类别分组传递给各 Agent 节点，节点内部不再管理连接

技术栈:
- LangGraph: 状态图编排
- langchain-mcp-adapters: MCP工具自动适配LangChain
- FastMCP: MCP Server实现
"""

import os
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import SystemState, ApprovalStatus
from agents import (
    SupervisorAgent,
    DBAgent,
    NetAgent,
    AppAgent,
    FixAgent,
    CommunicationBus,
)
from nodes import (
    create_dispatch_node,
    create_dynamic_check_node,
    create_aggregate_node,
    create_human_approval_node,
    create_executor_node,
    create_other_handler_node,
)
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
    """审批后路由：批准则执行，否则结束"""
    if state.approval_status == ApprovalStatus.APPROVED:
        return "execute"
    return END


def _get_mcp_server_path() -> str:
    """获取MCP Server脚本绝对路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "mcp_server.py")


def _classify_tools(all_tools):
    """按工具名前缀分类"""
    db_tools = [t for t in all_tools if t.name.startswith("check_db_")]
    net_tools = [t for t in all_tools if t.name.startswith("check_network_")]
    app_tools = [t for t in all_tools if t.name.startswith("check_app_")]
    return db_tools, net_tools, app_tools


async def create_async_workflow(llm, checkpointer=None):
    """
    创建 Multi-Agent 异步工作流

    流程:
    1. Supervisor 分析症状，决定派发哪些 Agent
    2. Dispatch 并行执行被派发的 Agent
    3. Aggregate 综合各 Agent 诊断结果
    4. Fix Agent 生成修复方案
    5. Human Approval 人工审批
    6. Executor 执行修复

    MCP Client 在此处一次性初始化:
    1. 启动 MCP Server 子进程 (stdio)
    2. 获取所有工具并按类别分组
    3. 将分组工具注入各 Agent 节点
    4. 节点内部只使用工具，不管理连接
    """
    # 初始化 MCP Client（一次性）
    mcp_server_path = _get_mcp_server_path()
    mcp_client = MultiServerMCPClient(
        {
            "diagnosis": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [mcp_server_path],
            }
        }
    )

    # 获取所有 MCP 工具（自动转换为 LangChain BaseTool）
    all_tools = await mcp_client.get_tools()
    logger.info(f"MCP工具加载完成，共 {len(all_tools)} 个: {[t.name for t in all_tools]}")

    # 按类别分组
    db_tools, net_tools, app_tools = _classify_tools(all_tools)
    logger.info(f"工具分组 - DB: {len(db_tools)}, Net: {len(net_tools)}, App: {len(app_tools)}")

    communication_bus = CommunicationBus()

    supervisor_agent = SupervisorAgent(llm)
    db_agent = DBAgent(llm, db_tools, communication_bus)
    net_agent = NetAgent(llm, net_tools, communication_bus)
    app_agent = AppAgent(llm, app_tools, communication_bus)
    fix_agent = FixAgent(llm)

    # 构建 Agent runner 映射（供 dispatch 节点并行调用）
    agent_runners = {
        "db_agent": db_agent.run,
        "net_agent": net_agent.run,
        "app_agent": app_agent.run,
    }

    # 创建工作流节点
    dispatch_node = create_dispatch_node(agent_runners)
    dynamic_check_node = create_dynamic_check_node()
    aggregate_node = create_aggregate_node(llm, communication_bus)
    human_approval_node = create_human_approval_node()
    executor_node = create_executor_node()
    other_handler_node = create_other_handler_node()

    # 构建状态图
    workflow = StateGraph(SystemState)

    # 添加节点
    workflow.add_node("supervisor", supervisor_agent.run)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("dynamic_check", dynamic_check_node)
    workflow.add_node("aggregate", aggregate_node)
    workflow.add_node("fix_agent", fix_agent.run)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute", executor_node)
    workflow.add_node("other_handler", other_handler_node)

    # 设置入口
    workflow.set_entry_point("supervisor")

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

    # Aggregate → Fix → Human Approval（固定流程）
    workflow.add_edge("aggregate", "fix_agent")
    workflow.add_edge("fix_agent", "human_approval")

    # 审批后路由：批准则执行，否则结束
    workflow.add_conditional_edges(
        "human_approval",
        route_by_approval,
        {"execute": "execute", END: END},
    )

    # 执行完成 → 结束
    workflow.add_edge("execute", END)
    workflow.add_edge("other_handler", END)

    # 编译工作流（带检查点）
    # 配置序列化器以支持自定义类型（如 state.DiagnosisType、state.FixPlan）
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("state",)])

    if checkpointer:
        checkpointer = checkpointer.with_serde(serde)
        app = workflow.compile(checkpointer=checkpointer)
    else:
        memory = MemorySaver(serde=serde)
        app = workflow.compile(checkpointer=memory)

    return app
