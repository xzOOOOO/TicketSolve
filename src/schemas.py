# field_validator：在赋值前自动校验/转换字段，这里用于把 LLM 偶尔输出的字符串证据归一化成 EvidenceItem
from pydantic import BaseModel, Field, field_validator
# EvidenceItem 是结构化证据对象；normalize_evidence_items 负责把字符串/字典混合格式统一成对象列表
from evidence import EvidenceItem, normalize_evidence_items
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
    诊断输出同时包含结论、证据和结构化协作请求。

    相比旧版本的变化：
    - 新增 fault_type：标准化故障类型，让下游 FixAgent 能直接映射到 Action DSL（如 RECOVER_FAULT）
    - 新增 hypothesis：一句话可验证假设，是证据协作协议的起点
    - evidence 从 List[str] 升级为 List[EvidenceItem]，每个证据都带结构化字段
    - 新增 collaboration_requests：Agent 主动请求其他 Agent 协助验证假设
    """
    diagnosis: str = Field(description="具体诊断结论")
    possible_causes: List[str] = Field(description="可能的原因列表")
    confidence: float = Field(description="诊断置信度，范围 0-1")
    # fault_type：标准化故障类型枚举值，用于后续匹配修复动作。
    # 例如 DB_CONN_FAIL → 用 RECOVER_FAULT 动作恢复数据库连接。
    # 如果 LLM 无法判断，允许为空，不要编造假值。
    fault_type: Optional[str] = Field(None, description="结构化故障类型，如 DB_CONN_FAIL/APP_PROCESS_DOWN/NGINX_BAD_ROUTE；无法判断则为空")
    # hypothesis：核心设计。一句话假设驱动后续所有协作。
    # 好假设的标准：可验证（能用工具证实或证伪）、具体（指出哪个组件出什么问题）。
    hypothesis: Optional[str] = Field(None, description="当前 Agent 的可验证故障假设")
    # evidence：结构化证据列表。每个 EvidenceItem 包含 source_agent（谁发现的）、
    # tool_name（用什么工具）、target（检查对象）、status（结果状态）、
    # observed（实际看到什么）、expected（预期应该看到什么）、
    # supports_hypothesis（是否支持假设）、confidence（证据可信度）。
    # 这种结构让 Aggregate 节点可以按字段做加权计算，而不是纯文本匹配。
    evidence: List[EvidenceItem] = Field(default_factory=list, description="支持该诊断的结构化证据列表")
    # collaboration_requests：Agent 诊断后如果发现证据不足，可以主动向其他 Agent "下单" 请求补充证据。
    # 每项是一个字典，必须包含：
    #   - target_agent：找谁帮忙（仅限 db_agent/net_agent/app_agent）
    #   - required_evidence：需要对方提供什么证据（字符串列表）
    #   - reason：为什么需要这个证据（给 LLM 看的上下文）
    #   - suggested_tools：建议对方用什么工具查（降低对方决策成本）
    collaboration_requests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="结构化协作请求列表，每项包含 target_agent/required_evidence/reason/suggested_tools",
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, value):
        """把模型偶尔输出的短文本证据归一成结构化证据对象。

        LLM with_structured_output 并不总是严格按 schema 输出 evidence：
        - 有时会输出字符串列表（如 ["连接被拒绝"]）
        - 有时会输出字典列表但字段不全
        normalize_evidence_items 会统一补全缺失字段（如 source_agent、confidence），
        保证下游消费代码拿到的是规范的 EvidenceItem 对象。
        """
        return normalize_evidence_items(value or [])


class FixStepOutput(BaseModel):
    """修复步骤输出

    FixPlanOutput 的嵌套子模型，描述单个修复步骤的详细信息。
    比 state.py 中的 FixStep 多了 expected_output/on_failure/rollback_command 三个必填字段，
    因为 LLM 生成方案时这些字段都有具体值。

    结构化动作 DSL 字段说明（安全执行的核心）：
    - action_type + target: LLM 声明"想做什么 + 作用在哪"，由 action_dsl.py 编译成安全命令
    - command: 兼容旧模式的自由文本命令，存在结构化动作时仅用于展示/日志
    - rollback_action_type + rollback_target: 结构化回滚动作，同样由 action_dsl.py 编译
    - rollback_command: 兼容旧模式的回滚命令

    执行优先级：action_type + target > command（自由文本）
    这是安全设计：LLM 无法通过 command 字段注入任意 shell 命令。
    """
    step_id: int = Field(description="步骤编号，必须是纯数字如 1/2/3")
    action: str = Field(description="具体动作描述（人可读）")
    action_type: Optional[str] = Field(None, description="结构化动作类型，如 RECOVER_FAULT/START_CONTAINER/HTTP_PROBE 等；优先于 command 执行")
    target: Optional[str] = Field(None, description="结构化动作目标，如 APP_PROCESS_DOWN/srebench-app 等；与 action_type 配合由本地编译器生成安全命令")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化动作可选参数（预留扩展）")
    command: Optional[str] = Field(None, description="兼容旧方案的完整命令字符串；当存在 action_type + target 时，此字段仅用于展示和日志，不会被执行")
    risk_level: str = Field(default="low", description="风险等级: low/medium/high")
    expected_output: str = Field(description="预期输出（用于验证步骤是否成功）")
    on_failure: str = Field(description="失败时的处理方式描述")
    rollback_action_type: Optional[str] = Field(None, description="结构化回滚动作类型，与 action_type 对称；用于执行失败时自动回滚")
    rollback_target: Optional[str] = Field(None, description="结构化回滚动作目标，与 target 对称")
    rollback_parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化回滚动作可选参数（预留扩展）")
    rollback_command: Optional[str] = Field(None, description="兼容旧方案的回滚命令字符串；存在结构化回滚动作时仅用于展示")


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

    新增 protocol_summary：把 Agent 间协作协议的结论也纳入聚合结果。
    这样 FixAgent 不仅知道 "诊断是什么"，还知道 "这个诊断在协议中战胜了哪些竞争假设"。
    """
    diagnosis: str = Field(description="最终诊断结论")
    possible_causes: List[str] = Field(description="可能的原因列表")
    confidence: float = Field(description="诊断置信度，范围 0-1")
    contributing_agents: List[str] = Field(description="贡献诊断的Agent列表")
    reasoning: str = Field(description="聚合推理过程")
    # protocol_summary：Agent 协作协议的统计摘要。
    # 由 agent_protocol.build_protocol_context() 生成，包含：
    #   - winning_hypothesis_id：得分最高的假设 ID
    #   - hypothesis_scores：每个假设的详细得分（support_score/tool_evidence_score/confidence_score/conflict_score/final_score）
    #   - conflicts：Agent 之间的冲突记录
    # Aggregate 节点把这个摘要传给 LLM，让它在聚合推理时参考协议层面的共识/分歧。
    protocol_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent 协作协议摘要，包含 winning_hypothesis_id/hypothesis_scores/conflicts 等",
    )


# ============================================================
# 安全护栏 & 闭环执行器 模型
# ============================================================


class GuardrailViolation(BaseModel):
    """单条护栏违规记录"""
    rule_id: str = Field(description="违规规则ID，如 DANGEROUS_CMD_001")
    severity: str = Field(description="严重程度: critical/warning")
    step_id: Optional[int] = Field(None, description="违规步骤编号")
    message: str = Field(description="违规描述（人可读）")
    detail: str = Field("", description="违规详情（具体匹配到的内容）")


class GuardrailResult(BaseModel):
    """护栏检查结果"""
    passed: bool = Field(description="是否通过检查")
    violations: List[GuardrailViolation] = Field(default_factory=list, description="违规列表")
    checked_at: str = Field("", description="检查时间戳")


class CommandExecutionResult(BaseModel):
    """单步命令执行结果（Mock 或真实）"""
    step_id: int = Field(description="步骤编号")
    command: str = Field(description="执行的命令")
    exit_code: int = Field(description="退出码: 0=成功, 非0=失败")
    stdout: str = Field("", description="标准输出")
    stderr: str = Field("", description="标准错误")
    success: bool = Field(description="是否成功")
    execution_time_ms: int = Field(0, description="执行耗时(毫秒)")


class ErrorAnalysisOutput(BaseModel):
    """LLM 错误分析输出

    执行失败时，LLM 分析错误信息后决定下一步动作。
    这是闭环执行器的核心决策点——由真实错误驱动，不是 LLM 凭空决定。
    """
    action: str = Field(description="决策动作: retry(重试当前步骤) / adjust(调整命令后重试) / rollback(执行回滚) / skip(跳过继续)")
    adjusted_command: Optional[str] = Field(None, description="调整后的命令（action=adjust 时必填）")
    reasoning: str = Field(description="决策理由")
    estimated_fix_probability: float = Field(0.0, description="预估修复成功概率 0-1")
