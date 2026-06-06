"""证据模型定义。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """可被聚合层打分的单条标准证据。"""

    # evidence_id: 证据唯一标识符，默认使用 uuid4 生成，前缀为 "ev-"
    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid4().hex}", description="证据唯一ID")
    # source_agent: 产生这条证据的 Agent 名称，如 "db_agent"，默认值为 "unknown"
    source_agent: str = Field("unknown", description="产生证据的 Agent")
    # tool_name: 产生证据的工具名称，如 "ping"、"curl"；非工具证据可为 None
    tool_name: str | None = Field(None, description="产生证据的工具名；非工具证据可为空")
    # target: 被观测的目标对象，如容器名、URL、端口号、数据库实例名
    target: str | None = Field(None, description="被观测对象，如容器、URL、端口、数据库实例")
    # status: 观测状态，如 "ok"（正常）、"failed"（失败）、"degraded"（降级）、"unreachable"（不可达），默认值为 "unknown"
    status: str = Field("unknown", description="观测状态，如 ok/failed/degraded/unreachable")
    # observed: 实际观测到的结果文本，默认值为 "未提供具体观测"
    observed: str = Field("未提供具体观测", description="实际观测到的结果")
    # expected: 期望的结果文本，用于与 observed 对比；无法明确时可为 None
    expected: str | None = Field(None, description="期望结果；无法明确时为空")
    # supports_hypothesis: 该证据是否支持关联的故障假设，True=支持，False=反对，None=未明确
    supports_hypothesis: bool | None = Field(None, description="该证据是否支持关联假设")
    # confidence: 证据的置信度，范围 0.0-1.0，默认值为 0.0
    confidence: float = Field(0.0, description="证据置信度，范围 0-1")
    # raw_output_ref: 原始工具输出的引用标识，避免把大段输出塞进协议消息，可为 None
    raw_output_ref: str | None = Field(None, description="原始工具输出引用，避免把大段输出塞进协议消息")
    # timestamp: 证据生成时间，默认值为当前 UTC 时间的 ISO 格式字符串
    timestamp: str = Field(default_factory=lambda: utc_now_iso(), description="证据生成时间")


def utc_now_iso() -> str:
    """返回带时区的 UTC ISO 时间戳。"""
    return datetime.now(timezone.utc).isoformat()
