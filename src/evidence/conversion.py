"""转换类工具：从原始工具返回中抽取标准证据字段。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .cleaning import clean_text, summarize_value
from .models import EvidenceItem
from .validation import normalize_status


NESTED_OBSERVATION_KEYS = ("http_probe", "tcp_probe", "container", "cache_endpoint_probe")
TARGET_KEYS = ("target", "url", "container", "host", "domain")


def coerce_evidence_dict(item: Any, *, default_status: str) -> dict[str, Any]:
    """把单条输入转换成证据字典。"""
    # 如果 item 已经是 EvidenceItem 对象，直接转成字典返回
    if isinstance(item, EvidenceItem):
        return item.model_dump()

    # 如果 item 是字典，尝试从嵌套结构中提取各字段
    if isinstance(item, dict):
        # data: 输入字典的副本，避免修改原始字典
        data = dict(item)
        # payload: 从 data 中提取的真正载荷（可能是嵌套 JSON 或 text 字段）
        payload = extract_payload(data)
        # tool_name: 优先取 data["tool"] 作为工具名，如果不存在则留空
        data.setdefault("tool_name", data.get("tool"))
        # status: 先从 payload 提取状态，再从 data 提取，最后使用 default_status 兜底
        data.setdefault("status", extract_status(payload) or extract_status(data) or default_status)
        # target: 先从 payload 提取目标，再从 data 提取
        data.setdefault("target", extract_target(payload) or extract_target(data))
        # observed: 先从 payload 提取人可读观测，如果没有则用 summarize_value 生成摘要
        data.setdefault("observed", extract_observed(payload) or summarize_value(data))
        # 如果 data 中有 result 但没有 raw_output_ref，为 result 生成短引用 ID
        if data.get("result") is not None and not data.get("raw_output_ref"):
            data["raw_output_ref"] = raw_output_ref(data.get("result"))
        # 再次规范化 status，确保它是合法值
        data["status"] = normalize_status(data.get("status"), default_status)
        # 清理 observed 文本，去除多余空白，如果为空则使用默认提示
        data["observed"] = clean_text(data.get("observed"), fallback="未提供具体观测")
        return data

    # 如果 item 既不是 EvidenceItem 也不是字典（如字符串），构造最简证据字典
    return {
        "status": default_status,
        "observed": clean_text(item, fallback="未提供具体观测"),
    }


def extract_payload(data: dict[str, Any]) -> Any:
    """从 result/text/JSON 字符串等嵌套结构中取出真正载荷。"""
    # result: 优先取 data["result"] 字段，如果不存在则把整个 data 当作结果
    result = data.get("result", data)
    # 如果 result 是列表且非空，取第一个元素尝试解析其中的 text 字段
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and first.get("text"):
            # 先尝试把 text 解析成 JSON，失败则返回原始 text 字符串
            return parse_json_text(first.get("text")) or first.get("text")
    # 如果 result 是字典且有 text 字段，同样尝试解析 JSON
    if isinstance(result, dict) and result.get("text"):
        return parse_json_text(result.get("text")) or result.get("text")
    # 如果 result 是纯字符串，尝试解析 JSON，失败则返回原字符串
    if isinstance(result, str):
        return parse_json_text(result) or result
    # 其他情况直接返回 result 本身
    return result


def extract_status(payload: Any) -> str | None:
    """从载荷中提取状态字段。"""
    # 如果 payload 是字典，先尝试直接取 status 字段
    if isinstance(payload, dict):
        status = payload.get("status")
        if status:
            return normalize_status(status)
        # 如果顶层没有 status，遍历 NESTED_OBSERVATION_KEYS 查找嵌套状态
        for key in NESTED_OBSERVATION_KEYS:
            child = payload.get(key)
            if isinstance(child, dict) and child.get("status"):
                return normalize_status(child["status"])
    # payload 不是字典或没找到 status，返回 None
    return None


def extract_target(payload: Any) -> str | None:
    """从载荷中提取被观测目标。"""
    # 如果 payload 是字典，先遍历 TARGET_KEYS 查找直接的目标字段
    if isinstance(payload, dict):
        for key in TARGET_KEYS:
            if payload.get(key):
                return str(payload[key])
        # 如果顶层没找到，遍历 NESTED_OBSERVATION_KEYS 递归查找嵌套目标
        for key in NESTED_OBSERVATION_KEYS:
            child = payload.get(key)
            if isinstance(child, dict):
                nested = extract_target(child)
                if nested:
                    return nested
    # payload 不是字典或没找到目标，返回 None
    return None


def extract_observed(payload: Any) -> str | None:
    """从载荷中提取人可读观测描述。"""
    # 如果 payload 是字典，先尝试取 evidence 字段
    if isinstance(payload, dict):
        evidence = payload.get("evidence")
        if evidence:
            # 把 evidence 列表拼接成 "item1; item2; item3" 的字符串
            return "; ".join(str(item) for item in evidence)
        # 如果 payload 中有 status 字段，用 summarize_value 生成摘要作为 observed
        if extract_status(payload):
            return summarize_value(payload)
    # payload 不是字典或无法提取观测，返回 None
    return None


def parse_json_text(value: str) -> Any:
    """安全解析 JSON 字符串，失败时返回 None。"""
    try:
        # 尝试把 value 解析成 Python 对象（dict/list 等）
        return json.loads(value)
    except Exception:
        # 解析失败（如不是合法 JSON），返回 None，不抛异常
        return None


def raw_output_ref(value: Any) -> str:
    """给原始工具输出生成短引用，避免消息里塞大段文本。"""
    # payload: 把原始输出 summarize 成不超过 2000 字符的文本摘要
    payload = summarize_value(value, limit=2000)
    # digest: 对摘要文本计算 SHA1 哈希，取前 12 位作为短唯一标识
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"raw-{digest}"


def stable_evidence_id(data: dict[str, Any], index: int) -> str:
    """基于证据关键字段生成稳定 ID。"""
    # payload: 把证据的关键字段和索引拼接成字符串，用于生成哈希
    # 包含：索引、来源 Agent、工具名、目标、状态、观测结果、原始输出引用
    payload = "|".join([
        str(index),
        str(data.get("source_agent", "")),
        str(data.get("tool_name", "")),
        str(data.get("target", "")),
        str(data.get("status", "")),
        str(data.get("observed", "")),
        str(data.get("raw_output_ref", "")),
    ])
    # digest: 对拼接字符串计算 SHA1 哈希，取前 12 位作为短唯一标识
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"ev-{digest}"
