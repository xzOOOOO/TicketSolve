"""校验类工具：检查和规范证据字段。"""

from __future__ import annotations

from typing import Any


EMPTY_STATUS = {None, "", "unknown"}
STATUS_ALIASES = {
    "success": "ok",
    "healthy": "ok",
    "running": "ok",
    "error": "failed",
    "fail": "failed",
    "failure": "failed",
    "unhealthy": "failed",
}


def normalize_status(value: Any, default: str = "unknown") -> str:
    """把不同工具返回的状态词规范成稳定的小写状态。"""
    if value is None:
        return default
    status = str(value).strip().lower()
    if not status:
        return default
    return STATUS_ALIASES.get(status, status)


def normalize_confidence(value: Any, default: float = 0.0) -> float:
    """把置信度规范到 0.0 到 1.0。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def has_tool_observation(item: dict[str, Any]) -> bool:
    """判断一条证据是否包含明确工具观测。"""
    has_tool = bool(item.get("tool_name"))
    has_status = item.get("status") not in EMPTY_STATUS
    return has_tool or has_status
