"""结构化证据包。

只对外暴露少量入口：
- EvidenceItem: 标准证据模型
- normalize_evidence_items: 把任意原始输入组装成标准证据
- count_tool_observations / format_evidence_brief: 给聚合打分和 prompt 摘要使用
"""

from .assembly import (
    count_tool_observations,
    evidence_from_tool_results,
    format_evidence_brief,
    normalize_evidence_items,
)
from .models import EvidenceItem


__all__ = [
    "EvidenceItem",
    "normalize_evidence_items",
    "count_tool_observations",
    "format_evidence_brief",
    "evidence_from_tool_results",
]
