"""协议上下文构造与格式化。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evidence import format_evidence_brief

from .normalize import normalize_messages
from .scoring import build_hypothesis_scores, choose_winning_hypothesis, score_for_hypothesis


def build_protocol_context(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """构造给 Agent/Aggregate 使用的协议上下文。"""
    # normalized: 规范化后的消息列表，确保每条消息都有默认字段（如 sender、receiver、msg_type 等）
    normalized = normalize_messages(messages)
    # hypotheses: 所有 msg_type == "hypothesis" 的消息列表，即各 Agent 发布的故障假设
    hypotheses = [m for m in normalized if m.get("msg_type") == "hypothesis"]
    # requests: 所有 msg_type == "evidence_request" 的消息列表，即向其他 Agent 请求补充证据的消息
    requests = [m for m in normalized if m.get("msg_type") == "evidence_request"]
    # responses: 所有 msg_type == "evidence_response" 的消息列表，即对其他 Agent 证据请求的响应
    responses = [m for m in normalized if m.get("msg_type") == "evidence_response"]
    # supports: 所有 msg_type == "support" 的消息列表，即支持某个假设的消息
    supports = [m for m in normalized if m.get("msg_type") == "support"]
    # challenges: 所有 msg_type == "challenge" 的消息列表，即质疑/反驳某个假设的消息
    challenges = [m for m in normalized if m.get("msg_type") == "challenge"]

    # response_count: 按 hypothesis_id 统计每个假设收到的证据响应数量
    # key = hypothesis_id, value = 该假设下的 evidence_response 数量
    response_count = defaultdict(int)
    for msg in responses:
        # hypothesis_id: 响应对应的假设 ID，优先取 hypothesis_id 字段，其次取 related_to 字段
        hypothesis_id = msg.get("hypothesis_id") or msg.get("related_to")
        if hypothesis_id:
            response_count[hypothesis_id] += 1

    # hypothesis_scores: 每个假设的可解释评分列表，包含 support_score、tool_evidence_score、confidence_score、conflict_score、final_score 等
    hypothesis_scores = build_hypothesis_scores(
        hypotheses=hypotheses,
        responses=responses,
        supports=supports,
        challenges=challenges,
        response_count=response_count,
    )
    # winning_hypothesis_id: 评分最高的假设的 message_id，作为协议推荐的最终假设
    winning_hypothesis_id = choose_winning_hypothesis(hypothesis_scores)
    # winning_score: 获胜假设的完整评分字典，用于提取 supporting_evidence_count、response_count 等统计值
    winning_score = score_for_hypothesis(hypothesis_scores, winning_hypothesis_id)
    # conflicts: 存在冲突的假设列表，每个元素包含 hypothesis_id、challenge_count、unsupported_response_count、conflict_score、reason
    # 只保留 conflicting_message_count > 0 的假设，即有至少一条反驳/不支持消息
    conflicts = [
        {
            "hypothesis_id": score["hypothesis_id"],
            "challenge_count": score["challenge_count"],
            "unsupported_response_count": score["unsupported_response_count"],
            "conflict_score": score["conflict_score"],
            "reason": score["reason"],
        }
        for score in hypothesis_scores
        if score["conflicting_message_count"] > 0
    ]

    # summary: 协议摘要字典，包含假设数量、获胜假设、支持证据数、冲突列表等统计信息
    summary = {
        "winning_hypothesis_id": winning_hypothesis_id,
        "supporting_evidence_count": winning_score.get("supporting_evidence_count", 0) if winning_score else 0,
        "response_count": winning_score.get("response_count", 0) if winning_score else 0,
        "conflicts": conflicts,
        "hypothesis_scores": hypothesis_scores,
        "hypothesis_count": len(hypotheses),
        "evidence_request_count": len(requests),
        "evidence_response_count": len(responses),
    }
    return {
        "hypotheses": hypotheses,
        "evidence_requests": requests,
        "evidence_responses": responses,
        "supports": supports,
        "challenges": challenges,
        "protocol_summary": summary,
        "text": format_protocol_context({
            "hypotheses": hypotheses,
            "evidence_requests": requests,
            "evidence_responses": responses,
            "supports": supports,
            "challenges": challenges,
            "protocol_summary": summary,
        }),
    }


def format_protocol_context(context: dict[str, Any]) -> str:
    """把协议上下文格式化为 prompt 可读文本。"""
    # lines: 最终拼接成返回文本的字符串列表，逐行收集各部分内容
    lines = []
    # summary: 协议摘要字典，包含 winning_hypothesis_id、hypothesis_scores、conflicts 等统计信息
    summary = context.get("protocol_summary") or {}
    lines.append(f"协议摘要: {summary}")

    # 遍历所有 hypothesis 消息，格式化为 "- 假设 {id} [sender] fault=... confidence=...: ..." 的行
    for msg in context.get("hypotheses", []):
        lines.append(
            f"- 假设 {msg.get('message_id')} [{msg.get('sender')}] "
            f"fault={msg.get('fault_type')} confidence={msg.get('confidence')}: "
            f"{msg.get('hypothesis') or msg.get('content')}"
        )
    # 遍历所有 evidence_request 消息，格式化为 "- 证据请求 {id} sender→receiver: 需要 ... 建议工具 ..." 的行
    for msg in context.get("evidence_requests", []):
        lines.append(
            f"- 证据请求 {msg.get('message_id')} {msg.get('sender')}→{msg.get('receiver')}: "
            f"需要 {msg.get('required_evidence')}，建议工具 {msg.get('suggested_tools')}"
        )
    # 遍历所有 evidence_response 消息，格式化为 "- 证据响应 {id} sender→receiver: supports=... evidence=..." 的行
    for msg in context.get("evidence_responses", []):
        lines.append(
            f"- 证据响应 {msg.get('message_id')} {msg.get('sender')}→{msg.get('receiver')}: "
            f"supports={msg.get('supports_hypothesis')} evidence={format_evidence_brief(msg.get('evidence'))}"
        )
    # 遍历所有 challenge 消息，格式化为 "- 反驳 sender: content evidence=..." 的行
    for msg in context.get("challenges", []):
        lines.append(f"- 反驳 {msg.get('sender')}: {msg.get('content')} evidence={format_evidence_brief(msg.get('evidence'))}")
    # 遍历所有 support 消息，格式化为 "- 支持 sender: content evidence=..." 的行
    for msg in context.get("supports", []):
        lines.append(f"- 支持 {msg.get('sender')}: {msg.get('content')} evidence={format_evidence_brief(msg.get('evidence'))}")

    # 如果 summary 中有 hypothesis_scores，追加评分详情
    if summary.get("hypothesis_scores"):
        lines.append("- 假设评分:")
        # score: 单个假设的评分字典，包含 final_score、support_score、tool_evidence_score、conflict_score、reason 等字段
        for score in summary["hypothesis_scores"]:
            lines.append(
                f"  - {score['hypothesis_id']} final={score['final_score']} "
                f"support={score['support_score']} tool={score['tool_evidence_score']} "
                f"conflict={score['conflict_score']}: {score['reason']}"
            )

    # 如果 lines 为空则返回默认提示文本，否则用换行符拼接所有行
    return "\n".join(lines) if lines else "无结构化协作消息。"
