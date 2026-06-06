"""多 Agent 证据协作协议。

这个包只对外暴露协议层入口，内部按职责拆分为：
- messages: 协议消息构造
- normalize: 消息规范化
- coordination: 未响应请求扫描与自动响应
- context/scoring: 协议上下文和假设裁决
"""

from .collaboration import collaboration_requests_from_result
from .constants import PROTOCOL_MESSAGE_TYPES, VALID_AGENTS
from .context import build_protocol_context, format_protocol_context
from .coordination import (
    auto_response_from_agent_result,
    has_response_for,
    pending_requests_for,
)
from .coverage import agent_result_covers_request, evidence_items_from_agent_result
from .messages import (
    make_challenge,
    make_evidence_request,
    make_evidence_response,
    make_hypothesis,
    make_support,
    new_message_id,
    protocol_timestamp,
)
from .normalize import normalize_message, normalize_messages
from .scoring import (
    build_hypothesis_scores,
    choose_winning_hypothesis,
    score_for_hypothesis,
)


__all__ = [
    "VALID_AGENTS",
    "PROTOCOL_MESSAGE_TYPES",
    "make_hypothesis",
    "make_evidence_request",
    "make_evidence_response",
    "make_challenge",
    "make_support",
    "normalize_message",
    "normalize_messages",
    "pending_requests_for",
    "has_response_for",
    "auto_response_from_agent_result",
    "agent_result_covers_request",
    "evidence_items_from_agent_result",
    "build_protocol_context",
    "format_protocol_context",
    "collaboration_requests_from_result",
    "build_hypothesis_scores",
    "choose_winning_hypothesis",
    "score_for_hypothesis",
    "protocol_timestamp",
    "new_message_id",
]
