from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


# 标准化 Trace 的 Schema 版本号，用于外部系统识别格式
TRACE_SCHEMA_VERSION = "trace.v1"

# 允许的标准事件类型集合，防止手滑写错事件名
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

# 允许的状态值集合，统一成功/失败/待定/跳过四种状态
TRACE_STATUSES = frozenset({"success", "failure", "pending", "skipped"})


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

    参数说明：
    - event_type: 事件类型，必须在 STANDARD_EVENT_TYPES 中
    - ticket_id: 工单 ID
    - agent_name: 产生事件的 Agent 名称
    - status: 事件状态，必须在 TRACE_STATUSES 中
    - input_data: 输入数据（可选）
    - output_data: 输出数据（可选）
    - error: 错误信息（可选，失败时填写）
    - metadata: 附加元数据（可选）
    - timestamp: 时间戳（可选，默认取当前 UTC 时间）
    """
    # 校验事件类型：如果传了不认识的 event_type，立即抛异常
    # 双语错误信息：英文在前方便日志搜索，中文在后方便人读
    if event_type not in STANDARD_EVENT_TYPES:
        raise ValueError(f"Unsupported trace event_type: {event_type}（不支持的标准事件类型）")
    # 校验状态值：只允许 success/failure/pending/skipped 四种
    if status not in TRACE_STATUSES:
        raise ValueError(f"Unsupported trace status: {status}（不支持的状态值）")

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "event_type": event_type,
        "ticket_id": ticket_id,
        "agent_name": agent_name,
        "status": status,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "input": input_data or {},
        "output": output_data or {},
        "error": error,
        "metadata": metadata or {},
    }


def status_from_success(success: bool | None) -> str:
    """根据布尔值 success 转换为标准状态字符串。

    - True  -> "success"
    - False -> "failure"
    - None  -> "skipped"（表示该步骤未执行或跳过）
    """
    if success is None:
        return "skipped"
    return "success" if success else "failure"
