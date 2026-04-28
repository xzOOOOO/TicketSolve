"""
LangGraph 工作流定义

MCP集成说明:
- 使用 langchain-mcp-adapters 的 MultiServerMCPClient
- 工作流创建时一次性初始化 MCP 连接，获取所有工具
- 按类别分组传递给各 Agent 节点，节点内部不再管理连接

Multi-Agent 改造说明:
- 原 nodes.py 函数式节点已重构为 agents/ 目录下的 Agent 类
- 每个 Agent 拥有独立身份（name, role），通过 run() 方法执行
- 返回格式与原节点完全一致，确保兼容
- human_approval / executor / other_handler 暂保留函数式实现

技术栈:
- LangGraph: 状态图编排
- langchain-mcp-adapters: MCP工具自动适配LangChain
- FastMCP: MCP Server实现
"""

import os
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from state import SystemState, DiagnosisType, ApprovalStatus
from agents import RouterAgent, DBAgent, NetAgent, AppAgent, FixAgent
from nodes import (
    create_human_approval_node,
    create_executor_node,
    create_other_handler_node
)
from database import AsyncSessionLocal
from logger import logger


def route_by_diagnosis(state: SystemState) -> str:
    diagnosis_type = state.diagnosis_type
    if diagnosis_type == DiagnosisType.DB:
        return "db_agent"
    elif diagnosis_type == DiagnosisType.NET:
        return "net_agent"
    elif diagnosis_type == DiagnosisType.APP:
        return "app_agent"
    else:
        return "other_handler"


def route_by_approval(state: SystemState) -> str:
    if state.approval_status == ApprovalStatus.APPROVED:
        return "execute"
    else:
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
    创建异步工作流

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

    # 创建 Agent 实例（注入对应工具）
    router_agent = RouterAgent(llm)
    db_agent = DBAgent(llm, db_tools)
    net_agent = NetAgent(llm, net_tools)
    app_agent = AppAgent(llm, app_tools)
    fix_agent = FixAgent(llm)
    human_approval_node = create_human_approval_node()
    executor_node = create_executor_node()
    other_handler_node = create_other_handler_node()

    # 构建状态图
    workflow = StateGraph(SystemState)

    workflow.add_node("router", router_agent.run)
    workflow.add_node("db_agent", db_agent.run)
    workflow.add_node("net_agent", net_agent.run)
    workflow.add_node("app_agent", app_agent.run)
    workflow.add_node("fix_agent", fix_agent.run)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute", executor_node)
    workflow.add_node("other_handler", other_handler_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router", route_by_diagnosis,
        {"db_agent": "db_agent", "net_agent": "net_agent", "app_agent": "app_agent", "other_handler": "other_handler"}
    )

    workflow.add_edge("db_agent", "fix_agent")
    workflow.add_edge("net_agent", "fix_agent")
    workflow.add_edge("app_agent", "fix_agent")

    workflow.add_edge("fix_agent", "human_approval")

    workflow.add_conditional_edges(
        "human_approval", route_by_approval,
        {"execute": "execute", END: END}
    )

    workflow.add_edge("execute", END)
    workflow.add_edge("other_handler", END)

    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

    return app
