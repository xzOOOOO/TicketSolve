# logger：项目统一日志记录器
from logger import logger
# SQLAlchemy 核心组件：用于定义列类型和表结构
from sqlalchemy import Column, String, Text, DateTime, JSON
# SQLAlchemy 异步组件：异步引擎、异步会话、异步会话工厂
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
# declarative_base：声明式基类，用于定义 ORM 模型
from sqlalchemy.orm import declarative_base
# SQLAlchemy 异常类
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
# datetime：用于记录创建/更新时间
from datetime import datetime
# uuid：用于生成唯一 ID
import uuid
# settings：项目配置对象
from config import settings

# ═══════════════════════════════════════════════
# 数据库引擎和会话配置
# ═══════════════════════════════════════════════
# engine：异步数据库引擎，使用 asyncpg 驱动连接 PostgreSQL
# echo=settings.DB_ECHO：是否打印 SQL 语句（调试用）
engine = create_async_engine(settings.get_database_url(), echo=settings.DB_ECHO)

# AsyncSessionLocal：异步会话工厂，用于创建数据库会话
# autocommit=False：不自动提交，需要手动 commit
# autoflush=False：不自动刷新，避免意外查询
# class_=AsyncSession：指定会话类为异步会话
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

# Base：声明式基类，所有 ORM 模型都继承自它
Base = declarative_base()

# ═══════════════════════════════════════════════
# 工单状态枚举
# ═══════════════════════════════════════════════
class TicketStatus:
    """工单状态常量类"""
    PENDING = "pending"      # 待处理
    APPROVED = "approved"    # 已审批
    REJECTED = "rejected"    # 已拒绝
    COMPLETED = "completed"  # 已完成

# ═══════════════════════════════════════════════
# ORM 模型定义
# ═══════════════════════════════════════════════
class Ticket(Base):
    """工单表

    存储工单的基本信息、诊断结果、修复计划、执行结果等。
    是整个系统的核心数据表。
    """
    __tablename__ = "tickets"

    # id：数据库主键，UUID 格式，自动生成
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # ticket_id：业务工单号，唯一，有索引（用于快速查询）
    ticket_id = Column(String(50), unique=True, nullable=False, index=True)
    # symptom：故障现象描述，非空
    symptom = Column(Text, nullable=False)
    # diagnosis_type：诊断类型（app/db/net/other）
    diagnosis_type = Column(String(20))
    # urgency：紧急程度（low/medium/high/critical）
    urgency = Column(String(20))
    # status：工单状态，默认 pending
    status = Column(String(20), default=TicketStatus.PENDING)

    # diagnosis_result：诊断结果（JSON 格式，存储各 Agent 的诊断结论）
    diagnosis_result = Column(JSON)
    # fix_plan：修复计划（JSON 格式，存储修复步骤）
    fix_plan = Column(JSON)
    # execution_result：执行结果（JSON 格式，存储执行器输出）
    execution_result = Column(JSON)

    # approval_status：审批状态
    approval_status = Column(String(20))
    # approver_comments：审批人备注
    approver_comments = Column(Text)

    # messages：消息历史（JSON 格式，存储 Agent 间通信消息）
    messages = Column(JSON, default=list)
    # created_at：创建时间，默认当前 UTC 时间
    created_at = Column(DateTime, default=datetime.utcnow)
    # updated_at：更新时间，默认当前 UTC 时间，更新时自动刷新
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TicketAuditLog(Base):
    """工单审计日志表

    记录每个 Agent 的完整操作轨迹，支持可追溯性查询。
    按时间顺序排列可还原整个工单处理流程。
    """
    __tablename__ = "ticket_audit_logs"

    # id：数据库主键，UUID 格式
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    # ticket_id：关联的工单号，有索引（用于按工单查询审计日志）
    ticket_id = Column(String(50), nullable=False, index=True)
    # agent_name：执行操作的 Agent 名称，有索引（用于按 Agent 查询）
    agent_name = Column(String(50), nullable=False, index=True)
    # action_type：操作类型（如 dispatch、diagnose、execute 等）
    action_type = Column(String(50), nullable=False)
    # action_detail：操作详情（JSON 格式）
    action_detail = Column(JSON)
    # input_context：输入上下文（JSON 格式，记录操作前的状态）
    input_context = Column(JSON)
    # output_result：输出结果（JSON 格式，记录操作后的结果）
    output_result = Column(JSON)
    # dispatch_round：调度轮次（用于多轮调度场景）
    dispatch_round = Column(String(10))
    # created_at：操作时间，默认当前 UTC 时间，有索引（用于按时间排序）
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

# ═══════════════════════════════════════════════
# 数据库工具函数
# ═══════════════════════════════════════════════
async def init_db():
    """
    初始化数据库表

    根据 ORM 模型定义自动创建数据库表（如果表不存在）。
    通常在应用启动时调用一次。
    """
    try:
        logger.info("初始化数据库表")
        # engine.begin()：开启一个事务块，自动提交/回滚
        async with engine.begin() as conn:
            # run_sync：在异步上下文中运行同步函数
            # Base.metadata.create_all：创建所有继承自 Base 的表
            await conn.run_sync(Base.metadata.create_all)
        logger.info('数据库初始化成功')
    except OperationalError as e:
        # OperationalError：数据库连接失败（如网络不通、密码错误）
        logger.error(f"数据库连接失败: {e}")
        raise e
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise e

async def get_db():
    """
    获取数据库会话（异步生成器）

    使用方式：
        async for session in get_db():
            # 使用 session 进行数据库操作
            ...

    特性：
    - 自动管理会话生命周期（创建、回滚、关闭）
    - 发生异常时自动回滚事务
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("创建数据库会话")
            yield session
        except SQLAlchemyError as e:
            # SQLAlchemyError：数据库操作异常，回滚事务
            logger.error(f"数据库会话异常: {e}")
            await session.rollback()
            raise e
        finally:
            # finally：无论是否异常，都会关闭会话
            logger.debug("关闭数据库会话")
            await session.close()

def serialize_value(value):
    """
    序列化值，处理 Pydantic 模型和枚举

    用途：将 Pydantic 模型、枚举等不可 JSON 序列化的对象转为字典/字符串，
          便于存入数据库 JSON 字段。

    参数：
        value: 任意值

    返回：
        可 JSON 序列化的值
    """
    # Pydantic v2 模型：使用 model_dump() 转为字典
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    # 枚举类型：取 value 属性
    elif hasattr(value, 'value'):
        return value.value
    # 列表：递归序列化每个元素
    elif isinstance(value, list):
        return [serialize_value(v) for v in value]
    # 字典：递归序列化每个值
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    # 其他类型：直接返回
    return value


def _has_execution_payload(execution_result: dict | None) -> bool:
    """判断 execution_result 是否包含真正的执行结果（而非只有 trace_events）。

    关键作用：避免 execution_result 里只有 trace_events 时被误判为 completed。
    例如执行器还没真正跑，只记录了 trace，这时候不应该把工单状态设为已完成。
    """
    if not execution_result:
        return False
    if isinstance(execution_result, dict):
        # 字典里除了 trace_events 还有其他字段，才算有真正的执行结果
        return any(key != "trace_events" for key in execution_result)
    return True


def _resolve_ticket_status(state: dict, execution_result: dict | None) -> str:
    approval_status = serialize_value(state.get("approval_status"))
    if _has_execution_payload(execution_result):
        return TicketStatus.COMPLETED
    if approval_status == "approved":
        return TicketStatus.APPROVED
    if approval_status == "rejected":
        return TicketStatus.REJECTED
    return TicketStatus.PENDING

async def save_ticket(db: AsyncSession, state: dict):
    """保存工单到数据库

    同时保存审计日志（audit_logs）到 ticket_audit_logs 表，
    用于后续追溯工单处理流程。
    """
    ticket_id = state.get("ticket_id", "unknown")

    try:
        logger.info(f"开始保存工单: ticket_id={ticket_id}")
        from sqlalchemy import delete, select

        result = await db.execute(select(Ticket).filter(Ticket.ticket_id == state["ticket_id"]))
        ticket = result.scalar_one_or_none()

        diagnosis_result = state.get("db_agent_result") or state.get("net_agent_result") or state.get("app_agent_result")
        fix_plan = serialize_value(state.get("fix_plan"))
        execution_result = serialize_value(state.get("execution_result"))
        verification_result = serialize_value(state.get("verification_result"))
        if verification_result:
            if not isinstance(execution_result, dict):
                execution_result = {}
            execution_result = {**execution_result, **verification_result}
        # 将标准化 Trace 事件保存到 execution_result.trace_events 中，方便持久化和接口查询
        trace_events = serialize_value(state.get("trace_events", []))
        if trace_events:
            if not isinstance(execution_result, dict):
                execution_result = {}
            execution_result = {**execution_result, "trace_events": trace_events}
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
            ticket.status = _resolve_ticket_status(state, execution_result)
            if ticket.status == TicketStatus.COMPLETED:
                logger.info(f"工单执行完成: ticket_id={ticket_id}")
            elif ticket.status == TicketStatus.APPROVED:
                logger.info(f"工单已审批通过: ticket_id={ticket_id}")
            ticket.updated_at = datetime.utcnow()
        else:
            logger.debug(f"创建新工单: ticket_id={ticket_id}")
            ticket = Ticket(
                ticket_id=state["ticket_id"],
                symptom=state["symptom"],
                diagnosis_type=serialize_value(state.get("diagnosis_type")),
                urgency=serialize_value(state.get("urgency")),
                status=_resolve_ticket_status(state, execution_result),
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
            # delete_stmt：同一工单按状态快照替换审计日志，避免待审批保存和最终归档重复插入
            delete_stmt = delete(TicketAuditLog).where(TicketAuditLog.ticket_id == ticket_id)
            await db.execute(delete_stmt)
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
