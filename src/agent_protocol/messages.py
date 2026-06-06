"""协议消息构造器。"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4
from datetime import datetime, timezone

from evidence import normalize_evidence_items
from state import AgentMessage

from .normalize import normalize_message


def make_hypothesis(
    *,
    sender: str,
    content: str,
    hypothesis: str,
    fault_type: Optional[str] = None,
    confidence: float = 0.0,
    evidence: Optional[list[Any]] = None,
) -> dict[str, Any]:
    """发布一个可验证故障假设。"""
    # msg: 构造的 AgentMessage 对象，代表一条 hypothesis 协议消息
    # sender: 发布假设的 Agent 名称，如 "db_agent"
    # receiver: 固定为 "broadcast"，表示广播给所有 Agent
    # content: 人可读的假设描述文本
    # msg_type: 固定为 "hypothesis"，标识这是一条假设消息
    # confidence: 发布者对假设的置信度，范围 0.0-1.0
    # evidence: 支持该假设的结构化证据列表，通过 normalize_evidence_items 统一格式
    # hypothesis: 结构化的假设文本（一句话可验证）
    # fault_type: 标准化故障类型，如 "DB_CONN_FAIL"，用于后续匹配修复动作
    # status: 固定为 "open"，表示假设刚发布，尚未被响应或关闭
    msg = AgentMessage(
        sender=sender,
        receiver="broadcast",
        content=content,
        msg_type="hypothesis",
        confidence=confidence,
        evidence=normalize_evidence_items(
            evidence or [],
            source_agent=sender,
            supports_hypothesis=True,
            confidence=confidence,
        ),
        hypothesis=hypothesis,
        fault_type=fault_type,
        status="open",
    )
    return msg.model_dump()


def make_evidence_request(
    *,
    sender: str,
    receiver: str,
    hypothesis_message: dict[str, Any],
    required_evidence: list[str],
    reason: str,
    suggested_tools: Optional[list[str]] = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """请求另一个 Agent 对某个假设补充证据。"""
    # hypothesis_message: 规范化后的 hypothesis 消息字典，确保包含 message_id、correlation_id 等字段
    hypothesis_message = normalize_message(hypothesis_message)
    # hypothesis_id: 关联假设的消息 ID，evidence_request 通过它指向要验证的假设
    hypothesis_id = hypothesis_message["message_id"]
    # msg: 构造的 AgentMessage 对象，代表一条 evidence_request 协议消息
    # sender: 请求方 Agent 名称
    # receiver: 被请求方 Agent 名称，如 "net_agent"
    # content: 请求原因/说明，给被请求方 LLM 看的上下文
    # msg_type: 固定为 "evidence_request"，标识这是一条证据请求消息
    # confidence: 请求方当前置信度
    # evidence: 证据请求本身不带证据，固定为空列表
    # correlation_id: 继承 hypothesis 的 correlation_id，若不存在则使用 hypothesis_id，保证同一链路可追溯
    # related_to: 指向 hypothesis_message 的 message_id，表示这条请求是针对哪个假设发起的
    # hypothesis_id: 同 related_to，冗余存储方便查询
    # hypothesis: 继承 hypothesis_message 中的假设文本
    # fault_type: 继承 hypothesis_message 中的故障类型
    # required_evidence: 需要对方提供的证据项名称列表，如 ["ping status", "dns resolution"]
    # suggested_tools: 建议对方使用的工具名列表，降低对方决策成本
    # status: 固定为 "open"，表示请求尚未被响应
    msg = AgentMessage(
        sender=sender,
        receiver=receiver,
        content=reason,
        msg_type="evidence_request",
        confidence=confidence,
        evidence=[],
        correlation_id=hypothesis_message.get("correlation_id") or hypothesis_id,
        related_to=hypothesis_id,
        hypothesis_id=hypothesis_id,
        hypothesis=hypothesis_message.get("hypothesis"),
        fault_type=hypothesis_message.get("fault_type"),
        required_evidence=required_evidence,
        suggested_tools=suggested_tools or [],
        status="open",
    )
    return msg.model_dump()


def make_evidence_response(
    *,
    sender: str,
    receiver: str,
    request_message: dict[str, Any],
    evidence: list[Any],
    supports_hypothesis: bool,
    content: Optional[str] = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """对 evidence_request 返回证据。"""
    # request_message: 规范化后的 evidence_request 消息字典，确保包含 message_id、correlation_id 等字段
    request_message = normalize_message(request_message)
    # msg: 构造的 AgentMessage 对象，代表一条 evidence_response 协议消息
    # sender: 响应方 Agent 名称
    # receiver: 请求方 Agent 名称
    # content: 人可读的响应说明，默认值为 "已根据本 Agent 诊断结果补充证据"
    # msg_type: 固定为 "evidence_response"，标识这是一条证据响应消息
    # confidence: 响应方对证据的置信度
    # evidence: 响应方提供的结构化证据列表，通过 normalize_evidence_items 统一格式
    # correlation_id: 继承 request_message 的 correlation_id，保证同一协作链路可追溯
    # related_to: 指向 request_message 的 message_id，表示这条响应是对哪个请求的回复
    # hypothesis_id: 继承 request_message 的 hypothesis_id，表示响应关联的假设
    # hypothesis: 继承 request_message 中的假设文本
    # fault_type: 继承 request_message 中的故障类型
    # supports_hypothesis: 明确回答 "我找到的证据是否支持你的假设"，True=支持，False=反对
    # status: 固定为 "closed"，表示响应已发出，协作链路关闭
    msg = AgentMessage(
        sender=sender,
        receiver=receiver,
        content=content or "已根据本 Agent 诊断结果补充证据",
        msg_type="evidence_response",
        confidence=confidence,
        evidence=normalize_evidence_items(
            evidence,
            source_agent=sender,
            supports_hypothesis=supports_hypothesis,
            confidence=confidence,
        ),
        correlation_id=request_message.get("correlation_id") or request_message["message_id"],
        related_to=request_message["message_id"],
        hypothesis_id=request_message.get("hypothesis_id") or request_message.get("related_to"),
        hypothesis=request_message.get("hypothesis"),
        fault_type=request_message.get("fault_type"),
        supports_hypothesis=supports_hypothesis,
        status="closed",
    )
    return msg.model_dump()


def make_challenge(
    *,
    sender: str,
    receiver: str,
    hypothesis_message: dict[str, Any],
    reason: str,
    evidence: Optional[list[Any]] = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """反驳某个假设。"""
    # hypothesis_message: 规范化后的 hypothesis 消息字典，确保包含 message_id、correlation_id 等字段
    hypothesis_message = normalize_message(hypothesis_message)
    # msg: 构造的 AgentMessage 对象，代表一条 challenge 协议消息
    # sender: 反驳方 Agent 名称
    # receiver: 被反驳方 Agent 名称（或 broadcast）
    # content: 反驳理由/说明
    # msg_type: 固定为 "challenge"，标识这是一条质疑/反驳消息
    # confidence: 反驳方对质疑的置信度
    # evidence: 支持反驳的结构化证据列表，supports_hypothesis 固定为 False
    # correlation_id: 继承 hypothesis 的 correlation_id，保证同一协作链路可追溯
    # related_to: 指向 hypothesis_message 的 message_id，表示这条反驳针对哪个假设
    # hypothesis_id: 同 related_to，冗余存储方便查询
    # hypothesis: 继承 hypothesis_message 中的假设文本
    # fault_type: 继承 hypothesis_message 中的故障类型
    # supports_hypothesis: 固定为 False，表示这是一条反对/质疑消息
    # status: 固定为 "closed"，表示反驳已发出
    msg = AgentMessage(
        sender=sender,
        receiver=receiver,
        content=reason,
        msg_type="challenge",
        confidence=confidence,
        evidence=normalize_evidence_items(
            evidence or [],
            source_agent=sender,
            supports_hypothesis=False,
            confidence=confidence,
        ),
        correlation_id=hypothesis_message.get("correlation_id") or hypothesis_message["message_id"],
        related_to=hypothesis_message["message_id"],
        hypothesis_id=hypothesis_message["message_id"],
        hypothesis=hypothesis_message.get("hypothesis"),
        fault_type=hypothesis_message.get("fault_type"),
        supports_hypothesis=False,
        status="closed",
    )
    return msg.model_dump()


def make_support(
    *,
    sender: str,
    receiver: str,
    hypothesis_message: dict[str, Any],
    reason: str,
    evidence: Optional[list[Any]] = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """支持某个假设。"""
    # hypothesis_message: 规范化后的 hypothesis 消息字典，确保包含 message_id、correlation_id 等字段
    hypothesis_message = normalize_message(hypothesis_message)
    # msg: 构造的 AgentMessage 对象，代表一条 support 协议消息
    # sender: 支持方 Agent 名称
    # receiver: 被支持方 Agent 名称（或 broadcast）
    # content: 支持理由/说明
    # msg_type: 固定为 "support"，标识这是一条支持消息
    # confidence: 支持方对支持的置信度
    # evidence: 支持该假设的结构化证据列表，supports_hypothesis 固定为 True
    # correlation_id: 继承 hypothesis 的 correlation_id，保证同一协作链路可追溯
    # related_to: 指向 hypothesis_message 的 message_id，表示这条支持针对哪个假设
    # hypothesis_id: 同 related_to，冗余存储方便查询
    # hypothesis: 继承 hypothesis_message 中的假设文本
    # fault_type: 继承 hypothesis_message 中的故障类型
    # supports_hypothesis: 固定为 True，表示这是一条支持消息
    # status: 固定为 "closed"，表示支持已发出
    msg = AgentMessage(
        sender=sender,
        receiver=receiver,
        content=reason,
        msg_type="support",
        confidence=confidence,
        evidence=normalize_evidence_items(
            evidence or [],
            source_agent=sender,
            supports_hypothesis=True,
            confidence=confidence,
        ),
        correlation_id=hypothesis_message.get("correlation_id") or hypothesis_message["message_id"],
        related_to=hypothesis_message["message_id"],
        hypothesis_id=hypothesis_message["message_id"],
        hypothesis=hypothesis_message.get("hypothesis"),
        fault_type=hypothesis_message.get("fault_type"),
        supports_hypothesis=True,
        status="closed",
    )
    return msg.model_dump()


def protocol_timestamp() -> str:
    """返回协议层统一时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def new_message_id(prefix: str = "msg") -> str:
    """生成协议消息 ID。"""
    return f"{prefix}-{uuid4().hex}"
