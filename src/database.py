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
    """保存工单到数据库"""
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