"""协议调度与自动响应工具。"""

from __future__ import annotations

from typing import Any, Optional

from evidence import normalize_evidence_items

from .messages import make_evidence_response
from .normalize import normalize_message, normalize_messages


def pending_requests_for(agent_name: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """返回某个 Agent 尚未响应的证据请求。"""
    # normalized: 规范化后的消息列表，确保每条消息都有默认字段
    normalized = normalize_messages(messages)
    # requests: 筛选出接收者为指定 agent_name、消息类型为 evidence_request 且状态为 open 的请求列表
    requests = [
        msg for msg in normalized
        if msg.get("receiver") == agent_name
        and msg.get("msg_type") == "evidence_request"
        and msg.get("status", "open") == "open"
    ]
    # 返回那些尚未被响应的 evidence_request，即 has_response_for 为 False 的请求
    return [request for request in requests if not has_response_for(request, normalized)]


def has_response_for(request_message: dict[str, Any], messages: Optional[list[dict[str, Any]]] = None) -> bool:
    """判断某个请求是否已有 evidence_response。"""
    # 如果消息列表为空，直接返回 False，表示没有响应
    if messages is None:
        return False
    # request: 规范化请求消息，确保包含默认字段
    request = normalize_message(request_message)
    # request_id: 请求消息的唯一标识，用于匹配对应的响应
    request_id = request["message_id"]
    # 遍历所有消息，查找是否存在 msg_type 为 evidence_response 且 related_to 等于 request_id 的消息
    for msg in normalize_messages(messages):
        if msg.get("msg_type") == "evidence_response" and msg.get("related_to") == request_id:
            return True
    return False


def auto_response_from_agent_result(
    *,
    agent_name: str,
    agent_result: dict[str, Any],
    request_message: dict[str, Any],
    supports_override: bool | None = None,
) -> dict[str, Any]:
    """当目标 Agent 已有诊断结果时，自动把结果转成 evidence_response。"""
    # request: 规范化请求消息，确保包含默认字段
    request = normalize_message(request_message)
    # supports: 是否支持该请求，如果外部传入了 supports_override 则使用，否则通过 _supports_request 判断
    supports = supports_override if supports_override is not None else _supports_request(agent_result, request)
    # evidence: 从 agent_result 中提取并规范化的证据列表
    evidence = _extract_evidence(agent_result, source_agent=agent_name, supports_hypothesis=supports)
    # diagnosis: Agent 的诊断结论文本
    diagnosis = agent_result.get("diagnosis", "")
    # confidence: Agent 诊断的置信度，默认为 0.0
    confidence = float(agent_result.get("confidence") or 0.0)
    # content: 响应的文本内容，包含诊断结论和证据数量
    content = (
        f"{agent_name} 已完成诊断，结论：{diagnosis or '无明确诊断'}；"
        f"证据数量：{len(evidence)}"
    )
    # 构造并返回 evidence_response 消息
    return make_evidence_response(
        sender=agent_name,
        receiver=request.get("sender", "broadcast"),
        request_message=request,
        evidence=evidence,
        supports_hypothesis=supports,
        content=content,
        confidence=confidence,
    )


def _extract_evidence(
    agent_result: dict[str, Any],
    *,
    source_agent: str,
    supports_hypothesis: bool | None,
) -> list[dict[str, Any]]:
    # confidence: Agent 诊断的置信度，默认为 0.0
    confidence = float(agent_result.get("confidence") or 0.0)
    # evidence: 收集到的证据列表
    evidence = []
    # explicit_evidence: Agent 结果中显式提供的证据列表
    explicit_evidence = agent_result.get("evidence") or []
    # 将显式证据或可能原因列表规范化后加入证据列表
    evidence.extend(normalize_evidence_items(
        explicit_evidence or agent_result.get("possible_causes") or [],
        source_agent=source_agent,
        supports_hypothesis=supports_hypothesis,
        confidence=confidence,
    ))
    # 将工具执行结果规范化后加入证据列表
    evidence.extend(normalize_evidence_items(
        agent_result.get("tool_results") or [],
        source_agent=source_agent,
        supports_hypothesis=supports_hypothesis,
        confidence=confidence,
    ))
    return evidence


def _supports_request(agent_result: dict[str, Any], request: dict[str, Any]) -> bool:
    # confidence: Agent 诊断的置信度，默认为 0.0
    confidence = float(agent_result.get("confidence") or 0.0)
    # diagnosis: Agent 的诊断结论文本，转为小写用于匹配失败关键词
    diagnosis = str(agent_result.get("diagnosis") or "").lower()
    # 如果诊断结论包含失败关键词，则认为不支持该请求
    if "诊断失败" in diagnosis or "无法解析" in diagnosis:
        return False
    # request_fault: 请求中指定的故障类型
    request_fault = request.get("fault_type")
    # result_fault: Agent 结果中识别出的故障类型
    result_fault = agent_result.get("fault_type")
    # 如果双方都有故障类型，则故障类型匹配或置信度足够高时认为支持
    if request_fault and result_fault:
        return request_fault == result_fault or confidence >= 0.65
    # 默认情况下，置信度达到 0.5 即认为支持
    return confidence >= 0.5
