"""组装类工具：把干净字段拼成最终证据模型。"""

from __future__ import annotations

from typing import Any, Iterable

from .cleaning import ensure_list
from .conversion import coerce_evidence_dict, stable_evidence_id
from .models import EvidenceItem
from .validation import has_tool_observation, normalize_confidence, normalize_status


def normalize_evidence_items(
    items: Any,
    *,
    source_agent: str | None = None,
    supports_hypothesis: bool | None = None,
    confidence: float | None = None,
    default_status: str = "unknown",
) -> list[dict[str, Any]]:
    """把字符串、工具结果或部分结构化对象统一成 EvidenceItem 字典。"""
    # normalized: 最终返回的规范化证据字典列表
    normalized = []
    # 遍历输入 items，确保它是列表；enumerate 同时获取索引 index 和元素 item
    for index, item in enumerate(ensure_list(items)):
        # data: 把 item 强制转换成字典格式，如果 item 是字符串或其他类型会提取成字典
        data = coerce_evidence_dict(item, default_status=default_status)
        # 如果外部传入了 source_agent 且 data 中没有 source_agent，则补充默认值
        if source_agent and not data.get("source_agent"):
            data["source_agent"] = source_agent
        # 如果外部传入了 supports_hypothesis 且 data 中该字段为 None，则补充默认值
        if supports_hypothesis is not None and data.get("supports_hypothesis") is None:
            data["supports_hypothesis"] = supports_hypothesis
        # 如果外部传入了 confidence 且 data 中没有 confidence，则补充默认值
        if confidence is not None and not data.get("confidence"):
            data["confidence"] = confidence
        # 规范化 status 字段，确保它是合法的字符串值
        data["status"] = normalize_status(data.get("status"), default_status)
        # 规范化 confidence 字段，确保它在 0.0-1.0 范围内
        data["confidence"] = normalize_confidence(data.get("confidence"))
        # 如果 data 没有 evidence_id，基于内容和索引生成稳定的唯一 ID
        if not data.get("evidence_id"):
            data["evidence_id"] = stable_evidence_id(data, index)
        # 用 EvidenceItem 模型验证 data 的合法性，然后转成字典加入结果列表
        normalized.append(EvidenceItem.model_validate(data).model_dump())
    return normalized


def count_tool_observations(evidence_items: Any) -> int:
    """统计有明确工具来源或明确观测状态的证据数量。"""
    # 先规范化所有证据项，然后用生成器表达式统计满足 has_tool_observation 条件的数量
    # has_tool_observation 判断证据是否有 tool_name 或明确的观测状态
    return sum(1 for item in normalize_evidence_items(evidence_items) if has_tool_observation(item))


def format_evidence_brief(evidence_items: Any, limit: int = 3) -> list[str]:
    """生成给 prompt 和协议摘要使用的短证据文本。"""
    # lines: 最终返回的短证据文本列表
    lines = []
    # 先规范化所有证据，然后只取前 limit 条（默认 3 条），避免 prompt 过长
    for item in normalize_evidence_items(evidence_items)[:limit]:
        # prefix: 证据前缀，优先取 tool_name，其次取 source_agent，最后兜底为 "evidence"
        prefix = item.get("tool_name") or item.get("source_agent") or "evidence"
        # target: 如果有 target 字段，拼接成 " target=xxx" 的字符串；否则为空字符串
        target = f" target={item.get('target')}" if item.get("target") else ""
        # 拼接成 "prefix target=xxx status=yyy: observed" 格式的单行文本
        lines.append(f"{prefix}{target} status={item.get('status')}: {item.get('observed')}")
    return lines


def evidence_from_tool_results(
    tool_results: Iterable[dict[str, Any]],
    *,
    source_agent: str,
    supports_hypothesis: bool | None,
    confidence: float,
) -> list[dict[str, Any]]:
    """把 ReAct 工具观测批量转换成结构化证据。"""
    # evidence: 最终收集到的结构化证据列表
    evidence = []
    # 遍历所有工具结果，逐个调用 normalize_evidence_items 进行规范化
    for item in tool_results or []:
        evidence.extend(normalize_evidence_items(
            item,
            source_agent=source_agent,
            supports_hypothesis=supports_hypothesis,
            confidence=confidence,
        ))
    return evidence
