from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import SystemState, DiagnosisType, ApprovalStatus
from nodes import (
    create_router_node,
    create_db_agent_node,
    create_net_agent_node,
    create_app_agent_node,
    create_fix_agent_node,
    create_human_approval_node,
    create_executor_node,
    create_other_handler_node
)
from database import AsyncSessionLocal

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

def create_async_workflow(llm, checkpointer=None):
    router_node = create_router_node(llm)
    db_agent_node = create_db_agent_node(llm)
    net_agent_node = create_net_agent_node(llm)
    app_agent_node = create_app_agent_node(llm)
    fix_agent_node = create_fix_agent_node(llm)
    human_approval_node = create_human_approval_node()
    executor_node = create_executor_node()
    other_handler_node = create_other_handler_node()

    workflow = StateGraph(SystemState)

    workflow.add_node("router", router_node)
    workflow.add_node("db_agent", db_agent_node)
    workflow.add_node("net_agent", net_agent_node)
    workflow.add_node("app_agent", app_agent_node)
    workflow.add_node("fix_agent", fix_agent_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute", executor_node)
    workflow.add_node("other_handler", other_handler_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges("router", route_by_diagnosis, {"db_agent": "db_agent", "net_agent": "net_agent", "app_agent": "app_agent", "other_handler": "other_handler"})

    workflow.add_edge("db_agent", "fix_agent")
    workflow.add_edge("net_agent", "fix_agent")
    workflow.add_edge("app_agent", "fix_agent")

    workflow.add_edge("fix_agent", "human_approval")

    workflow.add_conditional_edges("human_approval", route_by_approval, {"execute": "execute", END: END})

    workflow.add_edge("execute", END)
    
    workflow.add_edge("other_handler", END)

    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

    return app
