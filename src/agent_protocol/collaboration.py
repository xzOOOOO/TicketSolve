"""诊断输出中的协作请求解析。"""

from __future__ import annotations

from typing import Any

from .constants import VALID_AGENTS


def collaboration_requests_from_result(result_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """从诊断输出中提取结构化协作请求。"""
    requests = []
    for item in result_dict.get("collaboration_requests") or []:
        if not isinstance(item, dict):
            continue
        target = item.get("target_agent")
        if target in VALID_AGENTS:
            requests.append({
                "target_agent": target,
                "required_evidence": list(item.get("required_evidence") or []),
                "reason": item.get("reason") or "需要补充跨域诊断证据",
                "suggested_tools": list(item.get("suggested_tools") or []),
            })
    return requests
