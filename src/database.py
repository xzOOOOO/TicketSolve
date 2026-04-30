from logger import logger
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from datetime import datetime
import uuid
import os
from config import settings

engine = create_async_engine(settings.get_database_url(), echo=settings.DB_ECHO)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

Base = declarative_base()

class TicketStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"

class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_id = Column(String(50), unique=True, nullable=False, index=True)
    symptom = Column(Text, nullable=False)
    diagnosis_type = Column(String(20))
    urgency = Column(String(20))
    status = Column(String(20), default=TicketStatus.PENDING)
    
    diagnosis_result = Column(JSON)
    fix_plan = Column(JSON)
    execution_result = Column(JSON)
    
    approval_status = Column(String(20))
    approver_comments = Column(Text)
    
    messages = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TicketAuditLog(Base):
    """工单审计日志表

    记录每个 Agent 的完整操作轨迹，支持可追溯性查询。
    按时间顺序排列可还原整个工单处理流程。
    """
    __tablename__ = "ticket_audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticket_id = Column(String(50), nullable=False, index=True)
    agent_name = Column(String(50), nullable=False, index=True)
    action_type = Column(String(50), nullable=False)
    action_detail = Column(JSON)
    input_context = Column(JSON)
    output_result = Column(JSON)
    dispatch_round = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

async def init_db():
    """初始化数据库表"""
    try:
        logger.info("初始化数据库表")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info('数据库初始化成功')
    except OperationalError as e:
        logger.error(f"数据库连接失败: {e}")
        raise e
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise e

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("创建数据库会话")
            yield session
        except SQLAlchemyError as e:
            logger.error(f"数据库会话异常: {e}")
            await session.rollback()
            raise e
        finally:
            logger.debug("关闭数据库会话")
            await session.close()

def serialize_value(value):
    """序列化值，处理 Pydantic 模型和枚举"""
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    elif hasattr(value, 'value'):
        return value.value
    elif isinstance(value, list):
        return [serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    return value

async def save_ticket(db: AsyncSession, state: dict):
    """保存工单到数据库

    同时保存审计日志（audit_logs）到 ticket_audit_logs 表，
    用于后续追溯工单处理流程。
    """
    ticket_id = state.get("ticket_id", "unknown")

    try:
        logger.info(f"开始保存工单: ticket_id={ticket_id}")
        from sqlalchemy import select

        result = await db.execute(select(Ticket).filter(Ticket.ticket_id == state["ticket_id"]))
        ticket = result.scalar_one_or_none()

        diagnosis_result = state.get("db_agent_result") or state.get("net_agent_result") or state.get("app_agent_result")
        fix_plan = serialize_value(state.get("fix_plan"))
        execution_result = serialize_value(state.get("execution_result"))
        messages = serialize_value(state.get("messages", []))

        if ticket:
            logger.debug(f"更新现有工单: ticket_id={ticket_id}")
            ticket.symptom = state.get("symptom", ticket.symptom)
            ticket.diagnosis_type = serialize_value(state.get("diagnosis_type", ticket.diagnosis_type))
            ticket.urgency = serialize_value(state.get("urgency", ticket.urgency))
            ticket.diagnosis_result = serialize_value(diagnosis_result)
            ticket.fix_plan = fix_plan
            ticket.approval_status = serialize_value(state.get("approval_status"))
            ticket.approver_comments = state.get("approver_comments")
            ticket.execution_result = execution_result
            ticket.messages = messages

            if state.get("approval_status") == "approved":
                ticket.status = TicketStatus.APPROVED
                logger.info(f"工单已审批通过: ticket_id={ticket_id}")
            elif state.get("execution_result"):
                ticket.status = TicketStatus.COMPLETED
                logger.info(f"工单执行完成: ticket_id={ticket_id}")
            ticket.updated_at = datetime.utcnow()
        else:
            logger.debug(f"创建新工单: ticket_id={ticket_id}")
            ticket = Ticket(
                ticket_id=state["ticket_id"],
                symptom=state["symptom"],
                diagnosis_type=serialize_value(state.get("diagnosis_type")),
                urgency=serialize_value(state.get("urgency")),
                status=TicketStatus.PENDING,
                diagnosis_result=serialize_value(diagnosis_result),
                fix_plan=fix_plan,
                approval_status=serialize_value(state.get("approval_status")),
                approver_comments=state.get("approver_comments"),
                execution_result=execution_result,
                messages=messages
            )
            db.add(ticket)

        # 保存审计日志（如果有）
        audit_logs = state.get("audit_logs", [])
        if audit_logs:
            logger.info(f"保存 {len(audit_logs)} 条审计日志: ticket_id={ticket_id}")
            for log_entry in audit_logs:
                log_entry["ticket_id"] = ticket_id
                log = TicketAuditLog(
                    ticket_id=log_entry["ticket_id"],
                    agent_name=log_entry["agent_name"],
                    action_type=log_entry["action_type"],
                    action_detail=serialize_value(log_entry.get("action_detail")),
                    input_context=serialize_value(log_entry.get("input_context")),
                    output_result=serialize_value(log_entry.get("output_result")),
                    dispatch_round=str(log_entry.get("dispatch_round", "")),
                )
                db.add(log)

        await db.commit()
        await db.refresh(ticket)

        logger.info(f"工单保存成功: ticket_id={ticket_id}, status={ticket.status}")
        return ticket

    except IntegrityError as e:
        logger.error(f"数据完整性错误: ticket_id={ticket_id}, error={e}")
        await db.rollback()
        raise
    except OperationalError as e:
        logger.error(f"数据库连接错误: ticket_id={ticket_id}, error={e}")
        await db.rollback()
        raise
    except KeyError as e:
        logger.error(f"缺少必要字段: ticket_id={ticket_id}, missing_key={e}")
        await db.rollback()
        raise
    except Exception as e:
        logger.exception(f"保存工单时发生未知错误: ticket_id={ticket_id}")
        await db.rollback()
        raise

async def get_ticket_by_id(db: AsyncSession, ticket_id: str):
    from sqlalchemy import select
    result = await db.execute(select(Ticket).filter(Ticket.ticket_id == ticket_id))
    return result.scalar_one_or_none()

async def get_all_tickets(db: AsyncSession, skip: int = 0, limit: int = 50):
    from sqlalchemy import select
    result = await db.execute(select(Ticket).order_by(Ticket.created_at.desc()).offset(skip).limit(limit))
    return result.scalars().all()


async def save_audit_log(db: AsyncSession, log_entry: dict):
    """保存审计日志

    将 Agent 的操作轨迹记录到 ticket_audit_logs 表，
    用于后续追溯工单处理流程。

    Args:
        db: 数据库会话
        log_entry: 日志条目，格式:
            {
                "ticket_id": "TKT-001",
                "agent_name": "db_agent",
                "action_type": "tool_call",
                "action_detail": {"tools": [...]},
                "input_context": {"symptom": "..."},
                "output_result": {"diagnosis": "..."},
                "dispatch_round": "1"
            }
    """
    try:
        log = TicketAuditLog(
            ticket_id=log_entry["ticket_id"],
            agent_name=log_entry["agent_name"],
            action_type=log_entry["action_type"],
            action_detail=serialize_value(log_entry.get("action_detail")),
            input_context=serialize_value(log_entry.get("input_context")),
            output_result=serialize_value(log_entry.get("output_result")),
            dispatch_round=str(log_entry.get("dispatch_round", "")),
        )
        db.add(log)
        await db.commit()
        logger.debug(f"审计日志已保存: ticket_id={log_entry['ticket_id']}, agent={log_entry['agent_name']}, action={log_entry['action_type']}")
    except Exception as e:
        logger.error(f"保存审计日志失败: {e}")
        await db.rollback()


async def get_ticket_audit_logs(db: AsyncSession, ticket_id: str):
    """查询工单的审计日志

    按时间顺序返回该工单的所有 Agent 操作记录，
    用于还原完整的处理流程。
    """
    from sqlalchemy import select
    result = await db.execute(
        select(TicketAuditLog)
        .filter(TicketAuditLog.ticket_id == ticket_id)
        .order_by(TicketAuditLog.created_at.asc())
    )
    return result.scalars().all()