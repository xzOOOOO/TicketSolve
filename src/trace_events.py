# ============================================================
# trace_events.py：Trace 事件标准化模块
#
# 作用：
#   统一整个工作流中所有关键节点的日志/追踪事件格式。
#   让评测系统、监控系统、前端展示都能消费同一套结构化数据。
#
# 核心概念：
#   - Trace：分布式追踪，记录请求在系统中的完整流转路径
#   - Event：单个节点上发生的具体事件（如 Agent 开始、工具调用、诊断生成）
#   - Schema 版本：防止外部系统解析时因格式变化而崩溃
#
# 为什么需要标准化？
#   如果没有统一格式，每个模块自己打印日志，格式五花八门，
#   后续做流程分析、故障复盘、自动化评测时根本无法解析。
# ============================================================

# from __future__ import annotations：启用 PEP 563  postponed annotations
# 作用：让类型注解支持前向引用（如 dict[str, Any] 在 Python 3.9+ 可用）
from __future__ import annotations

# datetime/timezone：获取当前 UTC 时间，生成 ISO 8601 格式时间戳
from datetime import datetime, timezone
# typing.Any：表示任意类型，用于 input_data/output_data/metadata 等通用字段
from typing import Any


# ═══════════════════════════════════════════════════════════
# 一、全局常量定义
# ═══════════════════════════════════════════════════════════

# TRACE_SCHEMA_VERSION：标准化 Trace 的 Schema 版本号
# 作用：外部系统（如评测平台、监控大盘）解析 trace 时，先检查版本号，
#       版本不匹配时可以给出明确错误，而不是默默解析失败。
# 升级策略：如果事件结构发生破坏性变更，必须升级版本号。
TRACE_SCHEMA_VERSION = "trace.v1"

# STANDARD_EVENT_TYPES：允许的标准事件类型集合，使用 frozenset（不可变集合）
# 作用：
#   1. 防止手滑写错事件名（如 "agent_start" 写成 "agent_started"）
#   2. 作为枚举值约束，调用方只能传这些值
#   3. 评测系统按这些事件类型统计覆盖率
#
# 事件类型说明：
#   agent_started        → Agent 开始执行（如 db_agent 被调度）
#   tool_called          → Agent 调用诊断工具（如 check_db_connection）
#   observation_received → 收到工具返回的观察结果
#   diagnosis_generated  → Agent 生成诊断结论
#   handoff_requested    → Agent 请求其他 Agent 协作（evidence_request）
#   plan_generated       → FixAgent 生成修复方案
#   policy_checked       → 安全护栏/策略检查完成
#   approval_received    → 收到人工审批结果（通过/驳回）
#   action_executed      → 执行修复动作（如重启容器）
#   verification_passed  → 修复验证通过
STANDARD_EVENT_TYPES = frozenset({
    "agent_started",          # Agent 开始执行
    "tool_called",            # 调用工具
    "observation_received",   # 收到工具返回的观察结果
    "diagnosis_generated",    # 生成诊断结论
    "handoff_requested",      # 请求交接/协作给其他 Agent
    "plan_generated",         # 生成修复方案
    "policy_checked",         # 策略检查（安全护栏等）
    "approval_received",      # 收到人工审批结果
    "action_executed",        # 执行修复动作
    "verification_passed",    # 验证通过
})

# TRACE_STATUSES：允许的状态值集合
# 统一四种状态：
#   success → 成功完成
#   failure → 执行失败（通常伴随 error 字段）
#   pending → 等待中（如等待人工审批）
#   skipped → 跳过（如该步骤不需要执行）
TRACE_STATUSES = frozenset({"success", "failure", "pending", "skipped"})


# ═══════════════════════════════════════════════════════════
# 二、核心函数
# ═══════════════════════════════════════════════════════════

def make_trace_event(
    event_type: str,
    *,
    ticket_id: str,
    agent_name: str,
    status: str = "success",
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """构造一条标准 Trace 事件，用于评测和流程分析。

    这是整个系统追踪的"原子单位"——每个关键节点都调用此函数生成一条事件，
    最终所有事件按时间顺序排列，就能还原出完整的工单处理流程。

    参数说明：
    - event_type: 事件类型，必须在 STANDARD_EVENT_TYPES 中
    - ticket_id: 工单 ID，用于关联同一工单的所有事件
    - agent_name: 产生事件的 Agent 名称（如 "db_agent" / "supervisor"）
    - status: 事件状态，必须在 TRACE_STATUSES 中
    - input_data: 输入数据（可选），如 tool_called 事件的参数
    - output_data: 输出数据（可选），如 diagnosis_generated 事件的诊断结果
    - error: 错误信息（可选，status=failure 时必填）
    - metadata: 附加元数据（可选），如执行耗时、重试次数等
    - timestamp: 时间戳（可选，默认取当前 UTC 时间，ISO 8601 格式）

    返回：
        标准化的 trace 事件字典，可直接 JSON 序列化

    使用示例：
        event = make_trace_event(
            "tool_called",
            ticket_id="TICKET-001",
            agent_name="db_agent",
            status="success",
            input_data={"tool": "check_db_connection", "params": {"host": "localhost"}},
            output_data={"connected": False},
        )
    """
    # 校验事件类型：如果传了不认识的 event_type，立即抛异常
    # 双语错误信息：英文在前方便日志搜索，中文在后方便人读
    if event_type not in STANDARD_EVENT_TYPES:
        raise ValueError(f"Unsupported trace event_type: {event_type}（不支持的标准事件类型）")
    # 校验状态值：只允许 success/failure/pending/skipped 四种
    # 防止调用方传 "ok" / "error" 等非标准值，导致下游解析失败
    if status not in TRACE_STATUSES:
        raise ValueError(f"Unsupported trace status: {status}（不支持的状态值）")

    # 构造标准事件字典，所有字段固定，方便下游解析
    return {
        # schema_version：格式版本号，外部系统解析时先检查
        "schema_version": TRACE_SCHEMA_VERSION,
        # event_type：事件类型，用于分类统计
        "event_type": event_type,
        # ticket_id：工单标识，同一工单的所有事件共享此 ID
        "ticket_id": ticket_id,
        # agent_name：产生事件的组件名称
        "agent_name": agent_name,
        # status：事件执行结果状态
        "status": status,
        # timestamp：ISO 8601 格式 UTC 时间戳，如 "2024-01-15T08:30:00+00:00"
        # 如果调用方没传，取当前系统时间
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        # input：输入数据字典，空字典表示没有输入
        "input": input_data or {},
        # output：输出数据字典，空字典表示没有输出
        "output": output_data or {},
        # error：错误信息，None 表示没有错误
        "error": error,
        # metadata：附加元数据字典，用于存放扩展字段
        "metadata": metadata or {},
    }


def status_from_success(success: bool | None) -> str:
    """根据布尔值 success 转换为标准状态字符串。

    这是一个便捷函数，因为业务代码中通常用 True/False/None 表示结果，
    但 trace 事件要求用 "success"/"failure"/"skipped" 字符串。

    映射规则：
    - True  → "success"  （成功）
    - False → "failure"  （失败）
    - None  → "skipped"  （未执行或跳过）

    参数：
        success：布尔值或 None

    返回：
        对应的标准状态字符串

    使用示例：
        result = run_some_command()
        status = status_from_success(result.success)
        event = make_trace_event("action_executed", ..., status=status)
    """
    # None 表示该步骤未执行或被跳过（如条件分支不满足）
    if success is None:
        return "skipped"
    # True/False 直接映射为 success/failure
    return "success" if success else "failure"
