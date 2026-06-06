"""清洗类工具：把脏输入变成可处理的干净文本或列表。"""

from __future__ import annotations

import json
from typing import Any

from .models import EvidenceItem


def ensure_list(items: Any) -> list[Any]:
    """把 None、单对象、字符串或可迭代对象统一成列表。"""
    # 如果 items 是 None，返回空列表
    if items is None:
        return []
    # 如果 items 是 EvidenceItem 对象，包装成单元素列表
    if isinstance(items, EvidenceItem):
        return [items]
    # 如果 items 是字典，包装成单元素列表
    if isinstance(items, dict):
        return [items]
    # 如果 items 是字符串或字节串，包装成单元素列表
    if isinstance(items, (str, bytes)):
        return [items]
    # 其他情况（如 list、tuple、set 等可迭代对象），直接转成列表
    return list(items)


def clean_text(value: Any, *, fallback: str = "") -> str:
    """把任意值转成去掉多余空白的字符串。"""
    # 如果 value 是 None，直接返回 fallback 默认值
    if value is None:
        return fallback
    # text: 把 value 转成字符串；如果是 bytes 则先按 UTF-8 解码，解码失败时用替换字符
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
    # 用 split() 把连续空白（空格、换行、制表符等）拆成列表，再用单个空格 join，实现去多余空白
    text = " ".join(text.split())
    # 如果清理后为空字符串，返回 fallback；否则返回清理后的文本
    return text or fallback


def summarize_value(value: Any, limit: int = 500) -> str:
    """把任意值转成短文本，避免大段原始输出污染证据字段。"""
    # 如果 value 已经是字符串，直接使用
    if isinstance(value, str):
        text = value
    else:
        # 否则用 json.dumps 把对象序列化成 JSON 字符串，ensure_ascii=False 保证中文可读
        # default=str 处理不可序列化的对象（如 datetime）
        text = json.dumps(value, ensure_ascii=False, default=str)
    # 截取前 limit 个字符，然后调用 clean_text 清理空白，如果为空则返回 "未提供具体观测"
    return clean_text(text[:limit], fallback="未提供具体观测")
