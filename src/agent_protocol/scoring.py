"""假设裁决与可解释评分。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from evidence import count_tool_observations, format_evidence_brief, normalize_evidence_items


def build_hypothesis_scores(
    *,
    hypotheses: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    supports: list[dict[str, Any]],
    challenges: list[dict[str, Any]],
    response_count: dict[str, int],
) -> list[dict[str, Any]]:
    """为每个假设生成可解释评分。"""
    # supportive_messages: 按 hypothesis_id 分组的支持消息字典
    # key = hypothesis_id, value = 支持该假设的 evidence_response / support 消息列表
    supportive_messages = defaultdict(list)
    # conflicting_messages: 按 hypothesis_id 分组的冲突消息字典
    # key = hypothesis_id, value = 反对/质疑该假设的 evidence_response / challenge 消息列表
    conflicting_messages = defaultdict(list)
    # challenge_count: 按 hypothesis_id 统计的 challenge 消息数量
    # key = hypothesis_id, value = 该假设收到的 challenge 消息数
    challenge_count = defaultdict(int)
    # unsupported_response_count: 按 hypothesis_id 统计的不支持响应数量
    # key = hypothesis_id, value = 该假设收到的 supports_hypothesis=False 的 evidence_response 数量
    unsupported_response_count = defaultdict(int)

    # 遍历所有 evidence_response 消息，按 supports_hypothesis 分类到 supportive_messages 或 conflicting_messages
    for msg in responses:
        # hypothesis_id: 响应对应的假设 ID，优先取 hypothesis_id 字段，其次取 related_to 字段
        hypothesis_id = msg.get("hypothesis_id") or msg.get("related_to")
        if not hypothesis_id:
            continue
        if msg.get("supports_hypothesis") is True:
            supportive_messages[hypothesis_id].append(msg)
        elif msg.get("supports_hypothesis") is False:
            conflicting_messages[hypothesis_id].append(msg)
            unsupported_response_count[hypothesis_id] += 1

    # 遍历所有 support 消息，归入 supportive_messages
    for msg in supports:
        hypothesis_id = msg.get("hypothesis_id") or msg.get("related_to")
        if hypothesis_id:
            supportive_messages[hypothesis_id].append(msg)

    # 遍历所有 challenge 消息，归入 conflicting_messages 并计数
    for msg in challenges:
        hypothesis_id = msg.get("hypothesis_id") or msg.get("related_to")
        if hypothesis_id:
            conflicting_messages[hypothesis_id].append(msg)
            challenge_count[hypothesis_id] += 1

    # scores: 每个假设的评分结果列表，每个元素是一个包含多维度评分的字典
    scores = []
    for msg in hypotheses:
        # msg_id: 当前假设消息的 message_id，作为评分的唯一标识
        msg_id = msg.get("message_id")
        if not msg_id:
            continue
        # support_msgs: 支持当前假设的所有消息列表（evidence_response + support）
        support_msgs = supportive_messages[msg_id]
        # conflict_msgs: 反对当前假设的所有消息列表（evidence_response + challenge）
        conflict_msgs = conflicting_messages[msg_id]
        # own_evidence: 假设消息自身携带的结构化证据列表
        own_evidence = normalize_evidence_items(
            msg.get("evidence") or [],
            source_agent=msg.get("sender"),
            supports_hypothesis=True,
            confidence=float(msg.get("confidence") or 0.0),
        )
        # supporting_evidence: 所有支持证据的合并列表（假设自身证据 + 支持消息中的证据）
        supporting_evidence = own_evidence + [
            item
            for support_msg in support_msgs
            for item in normalize_evidence_items(
                support_msg.get("evidence") or [],
                source_agent=support_msg.get("sender"),
                supports_hypothesis=True,
                confidence=float(support_msg.get("confidence") or 0.0),
            )
        ]
        # conflicting_evidence: 所有冲突证据的合并列表（来自反对消息）
        conflicting_evidence = [
            item
            for conflict_msg in conflict_msgs
            for item in normalize_evidence_items(
                conflict_msg.get("evidence") or [],
                source_agent=conflict_msg.get("sender"),
                supports_hypothesis=False,
                confidence=float(conflict_msg.get("confidence") or 0.0),
            )
        ]

        # supporting_evidence_count: 支持证据的总条数
        supporting_evidence_count = len(supporting_evidence)
        # tool_evidence_count: 支持证据中有明确工具观测的数量（用于衡量证据质量）
        tool_evidence_count = count_tool_observations(supporting_evidence)
        # conflict_tool_count: 冲突证据中有明确工具观测的数量
        conflict_tool_count = count_tool_observations(conflicting_evidence)
        # support_score: 支持度得分 = 支持消息数 * 2.0 + min(支持证据数, 6) * 0.5
        # 消息权重高于证据，但证据数有上限（最多计 6 条）
        support_score = len(support_msgs) * 2.0 + min(supporting_evidence_count, 6) * 0.5
        # tool_evidence_score: 工具观测得分 = 工具观测数 * 1.5
        # 工具观测比纯文本证据更可靠，因此权重较高
        tool_evidence_score = tool_evidence_count * 1.5
        # confidence_score: 置信度得分 = 假设自身置信度 + 所有支持消息的置信度之和
        confidence_score = (
            float(msg.get("confidence") or 0.0)
            + sum(float(item.get("confidence") or 0.0) for item in support_msgs)
        )
        # conflict_score: 冲突得分 = 冲突消息数 * 2.0 + 冲突工具观测数 * 1.2 + 不支持响应数 * 0.5
        # 冲突得分越高表示假设越不可信，最终会减去该分数
        conflict_score = (
            len(conflict_msgs) * 2.0
            + conflict_tool_count * 1.2
            + unsupported_response_count[msg_id] * 0.5
        )
        # final_score: 最终得分 = 支持度得分 + 工具观测得分 + 置信度得分 - 冲突得分
        final_score = support_score + tool_evidence_score + confidence_score - conflict_score
        scores.append({
            "hypothesis_id": msg_id,
            "sender": msg.get("sender"),
            "fault_type": msg.get("fault_type"),
            "hypothesis": msg.get("hypothesis") or msg.get("content"),
            "support_score": round(support_score, 3),
            "tool_evidence_score": round(tool_evidence_score, 3),
            "confidence_score": round(confidence_score, 3),
            "conflict_score": round(conflict_score, 3),
            "final_score": round(final_score, 3),
            "supporting_evidence_count": supporting_evidence_count,
            "tool_evidence_count": tool_evidence_count,
            "supporting_message_count": len(support_msgs),
            "conflicting_message_count": len(conflict_msgs),
            "challenge_count": challenge_count[msg_id],
            "unsupported_response_count": unsupported_response_count[msg_id],
            "response_count": int(response_count.get(msg_id, 0)),
            "top_evidence": format_evidence_brief(supporting_evidence),
            "reason": (
                f"{supporting_evidence_count} 条支持证据，"
                f"{tool_evidence_count} 条工具观测，"
                f"{len(conflict_msgs)} 条反驳/不支持消息，"
                f"综合置信度 {round(confidence_score, 3)}"
            ),
        })
    return scores


def choose_winning_hypothesis(hypothesis_scores: list[dict[str, Any]]) -> Optional[str]:
    """从评分表中选择最终假设。"""
    # 如果评分表为空，返回 None，表示没有假设可推荐
    if not hypothesis_scores:
        return None
    # 使用 max 函数按 final_score 排序，选出得分最高的假设
    # key 函数提取每个评分项的 final_score 字段，默认值为 0.0（防止字段缺失）
    # 返回获胜假设的 hypothesis_id
    return max(hypothesis_scores, key=lambda item: item.get("final_score", 0.0)).get("hypothesis_id")


def score_for_hypothesis(
    hypothesis_scores: list[dict[str, Any]],
    hypothesis_id: Optional[str],
) -> dict[str, Any] | None:
    """按 ID 取某个假设的评分。"""
    # 如果 hypothesis_id 为空，直接返回 None，避免无意义的遍历
    if not hypothesis_id:
        return None
    # 使用 next + 生成器表达式查找匹配 hypothesis_id 的评分字典
    # 如果找不到匹配项，返回默认值 None
    return next((score for score in hypothesis_scores if score.get("hypothesis_id") == hypothesis_id), None)
