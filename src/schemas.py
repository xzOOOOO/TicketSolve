from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class TicketCreateRequest(BaseModel):
    ticket_id: str = Field(..., description="工单ID")
    symptom: str = Field(..., description="故障现象描述")

class ApprovalRequest(BaseModel):
    approved: bool = Field(..., description="是否批准")
    comments: Optional[str] = Field(None, description="审批意见")

class TicketResponse(BaseModel):
    id: str
    ticket_id: str
    symptom: str
    diagnosis_type: Optional[str]
    urgency: Optional[str]
    status: str
    diagnosis_result: Optional[Dict[str, Any]]
    fix_plan: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    approval_status: Optional[str]
    approver_comments: Optional[str]
    messages: Optional[List[str]]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class APIResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[Any] = None


# ============================================================
# Structured Output 模型（用于 LLM with_structured_output）
# ============================================================
# 以下模型同时服务于两个目的：
# 1. 作为 API 请求/响应的数据模型
# 2. 作为 LLM 结构化输出的 schema（通过 Pydantic 生成 JSON Schema）
#
# 使用 with_structured_output 后，LLM 会通过 function calling 机制
# 直接返回符合 schema 的结构化数据，无需手动解析 JSON 字符串。
# ============================================================


class SupervisorDecisionOutput(BaseModel):
    """Supervisor 调度决策输出

    用于 SupervisorAgent 分析故障现象后，决定派发哪些诊断 Agent。
    对应原 parse_json_content 解析的 {"diagnosis_type", "urgency", "dispatch", "reasoning"} 结构。
    """
    diagnosis_type: str = Field(description="诊断类型，必须是以下之一: app(应用问题)/db(数据库问题)/net(网络问题)/other(其他问题或无法判断)。如果无法明确判断类型，必须填 other，不要填 unknown 或其他值。")
    urgency: str = Field(description="紧急程度: low/medium/high/critical")
    dispatch: List[str] = Field(description="需要派发的Agent列表，如 ['db_agent', 'net_agent', 'app_agent']")
    reasoning: str = Field(description="派发理由")


class DiagnosisOutput(BaseModel):
    """诊断 Agent 输出

    用于 DBAgent/NetAgent/AppAgent 返回诊断结论。
    三个诊断 Agent 共用此模型，因为它们输出结构完全一致。
    对应原 parse_json_content 解析的 {"diagnosis", "possible_causes", "confidence", "need_collaboration"} 结构。
    """
    diagnosis: str = Field(description="具体诊断结论")
    possible_causes: List[str] = Field(description="可能的原因列表")
    confidence: float = Field(description="诊断置信度，范围 0-1")
    need_collaboration: List[str] = Field(description="需要协作的Agent名称列表，如不需要协作则为空列表")


class FixStepOutput(BaseModel):
    """修复步骤输出

    FixPlanOutput 的嵌套子模型，描述单个修复步骤的详细信息。
    比 state.py 中的 FixStep 多了 expected_output/on_failure/rollback_command 三个必填字段，
    因为 LLM 生成方案时这些字段都有具体值。
    """
    step_id: int = Field(description="步骤编号")
    action: str = Field(description="具体动作描述")
    command: str = Field(description="可直接执行的完整命令")
    risk_level: str = Field(default="low", description="风险等级: low/medium/high")
    expected_output: str = Field(description="预期输出")
    on_failure: str = Field(description="失败时的处理方式")
    rollback_command: str = Field(description="回滚命令")


class VerificationOutput(BaseModel):
    """验证方法输出

    FixPlanOutput 的嵌套子模型，描述修复后的验证方式。
    """
    commands: List[str] = Field(description="验证命令列表")
    expected_result: str = Field(description="预期验证结果")


class FixPlanOutput(BaseModel):
    """修复方案输出

    用于 FixAgent 生成完整的修复方案。
    对应原 parse_json_content 解析的复杂嵌套 JSON 结构，
    包含 steps（FixStepOutput 列表）和 verification（VerificationOutput）。
    """
    plan_id: str = Field(description="方案ID，如 PLAN-001")
    description: str = Field(description="方案简述")
    risk_level: str = Field(description="风险等级: low/medium/high")
    prerequisites: List[str] = Field(description="前置条件列表")
    steps: List[FixStepOutput] = Field(description="修复步骤列表")
    verification: VerificationOutput = Field(description="验证方法")
    estimated_time: str = Field(description="预计执行时间")


class AggregateOutput(BaseModel):
    """聚合诊断输出

    用于 aggregate 节点综合多个 Agent 的诊断结果。
    对应原 parse_json_content 解析的 {"diagnosis", "possible_causes", "confidence", "contributing_agents", "reasoning"} 结构。
    """
    diagnosis: str = Field(description="最终诊断结论")
    possible_causes: List[str] = Field(description="可能的原因列表")
    confidence: float = Field(description="诊断置信度，范围 0-1")
    contributing_agents: List[str] = Field(description="贡献诊断的Agent列表")
    reasoning: str = Field(description="聚合推理过程")
