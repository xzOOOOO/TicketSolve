"""证据请求覆盖判定。"""

from __future__ import annotations

import re
from typing import Any

from evidence import normalize_evidence_items
from .normalize import normalize_message


def agent_result_covers_request(agent_result: dict[str, Any], request_message: dict[str, Any]) -> bool:
    """判断已有 Agent 结果是否覆盖本次 evidence_request。

    覆盖条件：
    - 请求指定了 suggested_tools：已有证据中至少命中一个建议工具；
    - 请求指定了 required_evidence：每个证据需求都能在观测文本中找到明显匹配；
    - 两者都没指定：只要已有明确证据即可复用。
    """
    # request: 规范化后的 evidence_request 消息字典，确保包含 suggested_tools、required_evidence 等字段
    request = normalize_message(request_message)
    # evidence_items: 从 Agent 诊断结果中提取的结构化证据列表
    evidence_items = evidence_items_from_agent_result(agent_result)
    # 如果 Agent 结果中没有任何证据，直接判定为未覆盖
    if not evidence_items:
        return False

    # suggested_tools: 请求中建议的工具名集合，全部转为小写以实现不区分大小写的匹配
    suggested_tools = {str(tool).lower() for tool in request.get("suggested_tools") or []}
    if suggested_tools:
        # observed_tools: Agent 结果中实际使用的工具名集合，同样转为小写
        observed_tools = {
            str(item.get("tool_name")).lower()
            for item in evidence_items
            if item.get("tool_name")
        }
        # 如果实际工具与建议工具有交集，说明覆盖了工具层面的要求
        if observed_tools & suggested_tools:
            return True

    # required: 请求中要求的证据项名称列表，过滤掉空字符串
    required = [str(item) for item in request.get("required_evidence") or [] if str(item).strip()]
    if required:
        # evidence_texts: 把每条证据转换成可搜索的文本，用于后续匹配
        evidence_texts = [_evidence_search_text(item) for item in evidence_items]
        # 只有当所有 required_evidence 都能在 evidence_texts 中找到匹配时，才判定为覆盖
        return all(_requirement_is_covered(requirement, evidence_texts) for requirement in required)

    # 如果请求既没有指定 suggested_tools 也没有指定 required_evidence，只要有证据就算覆盖
    return True


def evidence_items_from_agent_result(agent_result: dict[str, Any]) -> list[dict[str, Any]]:
    """从 Agent 结果里抽取可用于覆盖判定的结构化证据。"""
    # confidence: Agent 诊断结果中的置信度，用于给提取的证据赋予默认置信度
    confidence = float(agent_result.get("confidence") or 0.0)
    # evidence: 最终提取的结构化证据列表，会从多个来源收集
    evidence = []
    # explicit_evidence: Agent 结果中显式的 evidence 字段
    explicit_evidence = agent_result.get("evidence") or []
    # 把显式证据（或 possible_causes 作为后备）归一化成 EvidenceItem 并加入列表
    evidence.extend(normalize_evidence_items(
        explicit_evidence or agent_result.get("possible_causes") or [],
        source_agent=agent_result.get("agent_name") or "unknown",
        confidence=confidence,
    ))
    # 把工具调用结果（tool_results）也归一化成 EvidenceItem 并加入列表
    evidence.extend(normalize_evidence_items(
        agent_result.get("tool_results") or [],
        source_agent=agent_result.get("agent_name") or "unknown",
        confidence=confidence,
    ))
    return evidence


def _requirement_is_covered(requirement: str, evidence_texts: list[str]) -> bool:
    # normalized_requirement: 规范化后的证据需求文本，转为小写并去除多余空白
    normalized_requirement = _normalize_text(requirement)
    # 如果规范化后为空字符串，认为该需求无需匹配，直接返回 True
    if not normalized_requirement:
        return True
    # requirement_tokens: 把需求文本拆分成关键词列表，用于模糊匹配
    requirement_tokens = _tokens(normalized_requirement)
    # 遍历所有证据文本，检查是否有证据能覆盖该需求
    for evidence_text in evidence_texts:
        # 如果需求文本完整出现在某条证据中，直接判定为覆盖
        if normalized_requirement in evidence_text:
            return True
        # evidence_tokens: 当前证据文本的关键词集合
        evidence_tokens = set(_tokens(evidence_text))
        # 如果需求没有关键词，跳过模糊匹配
        if not requirement_tokens:
            continue
        # overlap: 需求关键词在证据中出现的次数（支持完整子串匹配和关键词匹配）
        overlap = sum(1 for token in requirement_tokens if token in evidence_tokens or token in evidence_text)
        # threshold: 匹配阈值，取需求关键词数和 2 的较小值
        # 即：需求有 1 个关键词时需命中 1 个，有 2 个及以上时需命中至少 2 个
        threshold = min(len(requirement_tokens), 2)
        # 如果命中数达到阈值，判定为覆盖
        if overlap >= threshold:
            return True
    # 所有证据都不匹配，判定为未覆盖
    return False


def _evidence_search_text(item: dict[str, Any]) -> str:
    # fields: 从证据字典中提取的用于搜索匹配的字段列表
    # 包含工具名、观测目标、状态、实际观测、预期结果、原始输出引用
    fields = [
        item.get("tool_name"),
        item.get("target"),
        item.get("status"),
        item.get("observed"),
        item.get("expected"),
        item.get("raw_output_ref"),
    ]
    # 把所有非空字段拼接成字符串，再规范化（转小写、去多余空白）
    return _normalize_text(" ".join(str(field) for field in fields if field))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9_\u4e00-\u9fff]+", value) if token]
