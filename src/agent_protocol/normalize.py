"""协议消息规范化工具。"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from evidence import normalize_evidence_items


def normalize_message(message: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    """补齐协议消息默认字段，确保读取时不因缺字段失败。"""
    # msg: 输入消息的副本，避免修改原始字典
    msg = dict(message or {})
    # sender: 发送者名称，默认为 "unknown"（未知发送者）
    msg.setdefault("sender", "unknown")
    # receiver: 接收者名称，默认为 "broadcast"（广播给所有 Agent）
    msg.setdefault("receiver", "broadcast")
    # content: 消息内容文本，默认为空字符串
    msg.setdefault("content", "")
    # msg_type: 消息类型，默认为 "info"（普通信息）
    msg.setdefault("msg_type", "info")
    # confidence: 置信度，默认为 0.0
    msg.setdefault("confidence", 0.0)
    # evidence: 结构化证据列表，通过 normalize_evidence_items 统一格式
    # 使用 sender 作为默认 source_agent，supports_hypothesis 保持原值
    msg["evidence"] = normalize_evidence_items(
        msg.get("evidence") or [],
        source_agent=msg.get("sender"),
        supports_hypothesis=msg.get("supports_hypothesis"),
        confidence=float(msg.get("confidence") or 0.0),
    )
    # required_evidence: 请求对方提供的证据项列表，默认为空列表
    msg.setdefault("required_evidence", [])
    # suggested_tools: 建议对方使用的工具名列表，默认为空列表
    msg.setdefault("suggested_tools", [])
    # status: 消息状态，默认为 "open"（尚未响应）
    msg.setdefault("status", "open")
    # hypothesis: 结构化故障假设文本，默认为 None
    msg.setdefault("hypothesis", None)
    # fault_type: 标准化故障类型，默认为 None
    msg.setdefault("fault_type", None)
    # supports_hypothesis: 证据是否支持假设，默认为 None（未明确）
    msg.setdefault("supports_hypothesis", None)

    # 如果消息没有 message_id，基于消息内容生成稳定的伪唯一 ID
    if not msg.get("message_id"):
        # stable_payload: 用于生成哈希的拼接字符串，包含索引、发送者、接收者、消息类型、内容
        stable_payload = "|".join([
            str(index if index is not None else ""),
            str(msg.get("sender")),
            str(msg.get("receiver")),
            str(msg.get("msg_type")),
            str(msg.get("content")),
        ])
        # digest: SHA1 哈希的前 12 位，作为短唯一标识
        digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:12]
        msg["message_id"] = f"msg-{digest}"

    # correlation_id: 协作链路关联 ID，默认为自身的 message_id
    msg.setdefault("correlation_id", msg.get("message_id"))
    # related_to: 关联的上游消息 ID，默认为 None
    msg.setdefault("related_to", None)
    # hypothesis_id: 关联的假设 ID，默认为 related_to 的值
    msg.setdefault("hypothesis_id", msg.get("related_to"))
    return msg


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """批量规范化消息列表。"""
    # 遍历消息列表，每条消息调用 normalize_message 并传入 index（当前索引）
    # index 用于在无 message_id 时生成稳定的伪唯一 ID
    return [normalize_message(msg, index=idx) for idx, msg in enumerate(messages or [])]
