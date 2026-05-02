from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from database import init_db, get_db, AsyncSessionLocal
from workflow import create_async_workflow
from schemas import TicketCreateRequest, ApprovalRequest, TicketResponse, APIResponse
from logger import logger
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from config import settings
from llm_rate_limiter import LLMRateLimiter, RateLimitCallback
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("工单系统启动")
    logger.info("=" * 50)
    
    await init_db()
    logger.info("数据库初始化完成")
    
    rate_limiter = LLMRateLimiter(
        max_concurrent=settings.LLM_MAX_CONCURRENT,
        rpm_limit=settings.LLM_RPM_LIMIT
    )
    logger.info("LLM限流器初始化完成")
    
    llm_config = settings.get_llm_config()
    llm = ChatOpenAI(
        callbacks=[RateLimitCallback(rate_limiter)],
        **llm_config
    )
    logger.info("LLM实例创建完成（已集成限流回调）")
    
    workflow_app = await create_async_workflow(llm, checkpointer=None)
    logger.info("异步工作流创建完成（MCP工具已加载）")
    
    app_state["llm"] = llm
    app_state["workflow"] = workflow_app
    app_state["rate_limiter"] = rate_limiter
    
    logger.info("工单系统准备就绪")
    
    yield
    
    logger.info("工单系统关闭")

app = FastAPI(
    title="AI工单处理系统",
    description="基于LangGraph的智能工单处理系统",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/api/tickets", response_model=APIResponse)
async def create_ticket(request: TicketCreateRequest):
    try:
        workflow = app_state["workflow"]
        config = {"configurable": {"thread_id": request.ticket_id}}
        
        initial_state = {
            "ticket_id": request.ticket_id,
            "symptom": request.symptom
        }
        
        logger.info(f"收到工单创建请求: {request.ticket_id}")
        
        result = await workflow.ainvoke(initial_state, config=config)
        
        logger.info(f"工单 {request.ticket_id} 处理完成，等待审批")
        
        return APIResponse(
            code=200,
            message="工单已提交，等待审批",
            data={
                "ticket_id": request.ticket_id,
                "status": "pending_approval",
                "next_step": "请调用 /api/tickets/{ticket_id}/approve 进行审批"
            }
        )
    except Exception as e:
        logger.exception(f"创建工单失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tickets/{ticket_id}/approve", response_model=APIResponse)
async def approve_ticket(ticket_id: str, request: ApprovalRequest):
    try:
        workflow = app_state["workflow"]
        config = {"configurable": {"thread_id": ticket_id}}
        
        logger.info(f"收到工单审批请求: {ticket_id}, approved={request.approved}")
        
        result = await workflow.ainvoke(
            Command(resume={"approved": request.approved, "comments": request.comments}),
            config=config
        )
        
        logger.info(f"工单 {ticket_id} 审批完成")
        
        return APIResponse(
            code=200,
            message="审批完成",
            data={
                "ticket_id": ticket_id,
                "approved": request.approved,
                "final_result": result
            }
        )
    except Exception as e:
        logger.exception(f"审批工单失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tickets/{ticket_id}", response_model=APIResponse)
async def get_ticket(ticket_id: str, db: AsyncSession = Depends(get_db)):
    from database import get_ticket_by_id
    
    ticket = await get_ticket_by_id(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在")
    
    return APIResponse(
        code=200,
        message="查询成功",
        data=TicketResponse.model_validate(ticket).model_dump()
    )


@app.get("/api/tickets/{ticket_id}/agent-flow", response_model=APIResponse)
async def get_ticket_agent_flow(ticket_id: str, db: AsyncSession = Depends(get_db)):
    from database import get_ticket_by_id, get_ticket_audit_logs

    ticket = await get_ticket_by_id(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在")

    audit_logs = await get_ticket_audit_logs(db, ticket_id)

    flow = []
    for log in audit_logs:
        step = {
            "agent_name": log.agent_name,
            "action_type": log.action_type,
            "action_detail": log.action_detail,
            "input_context": log.input_context,
            "output_result": log.output_result,
            "dispatch_round": log.dispatch_round,
            "timestamp": log.created_at.isoformat() if log.created_at else None,
        }
        flow.append(step)

    agent_summary = {}
    for step in flow:
        name = step["agent_name"]
        if name not in agent_summary:
            agent_summary[name] = {"actions": [], "dispatch_rounds": set()}
        agent_summary[name]["actions"].append(step["action_type"])
        if step["dispatch_round"]:
            agent_summary[name]["dispatch_rounds"].add(step["dispatch_round"])

    for name in agent_summary:
        agent_summary[name]["dispatch_rounds"] = sorted(agent_summary[name]["dispatch_rounds"])

    return APIResponse(
        code=200,
        message="查询成功",
        data={
            "ticket_id": ticket_id,
            "diagnosis_type": ticket.diagnosis_type,
            "urgency": ticket.urgency,
            "status": ticket.status,
            "dispatched_agents": list(agent_summary.keys()),
            "agent_summary": agent_summary,
            "flow_steps": flow,
            "total_steps": len(flow),
        },
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI工单系统运行中"}

@app.get("/api/rate-limiter/stats")
async def get_rate_limiter_stats():
    rate_limiter = app_state.get("rate_limiter")
    if not rate_limiter:
        raise HTTPException(status_code=503, detail="限流器未初始化")
    
    return {
        "code": 200,
        "message": "success",
        "data": rate_limiter.get_stats()
    }