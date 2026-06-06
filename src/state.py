"""
工单状态模型 - 定义整个工作流的状态结构
"""
# 引入 field_validator：用于在 Pydantic 模型赋值前自动校验/转换字段值
# 这里主要把 LLM 输出的字符串证据统一转成结构化 EvidenceItem 对象
from pydantic import BaseModel, Field, field_validator
# EvidenceItem 是结构化证据对象；normalize_evidence_items 是归一化工具函数
from evidence import EvidenceItem, normalize_evidence_items
# uuid4 用于生成全局唯一消息 ID，保证每条 agent_message 都有独立标识
from uuid import uuid4

from typing import Optional, List, Dict, Any, Annotated
from enum import Enum
import operator


class DiagnosisType(str, Enum):
    """诊断类型枚举"""
    APP = "app"      # 应用问题（进程、内存、CPU等）
    DB = "db"        # 数据库问题（连接、慢查询、死锁等）
    NET = "net"      # 网络问题（连通性、延迟、路由等）
    OTHER = "other"  # 其他问题（如配置错误、权限问题等）

class Urgency(str, Enum):
    """紧急程度枚举"""
    LOW = "low"           # 低：可延后处理
    MEDIUM = "medium"     # 中：24小时内处理
    HIGH = "high"         # 高：需尽快处理
    CRITICAL = "critical" # 紧急：立即处理

class ApprovalStatus(str, Enum):
    """审批状态枚举"""
    PENDING = "pending"     # 待审批
    APPROVED = "approved"   # 已批准
    REJECTED = "rejected"   # 已拒绝


class FixStep(BaseModel):
    """
    修复步骤（工作流状态中的步骤模型）。

    与 schemas.py 中的 FixStepOutput 对应，但字段更宽松（部分字段可选），
    因为工作流执行过程中步骤可能被部分填充。

    安全执行设计：
    - action_type + target 组成结构化动作 DSL，由 action_dsl.py 编译成安全命令
    - command 为兼容旧模式的自由文本命令，存在结构化动作时不参与执行
    - 回滚同理：rollback_action_type + rollback_target > rollback_command
    """
    step_id: int = Field(..., description="步骤编号")
    action: str = Field(..., description="修复动作描述（人可读）")
    action_type: Optional[str] = Field(None, description="结构化动作类型，如 RECOVER_FAULT/START_CONTAINER 等；优先于 command 执行")
    target: Optional[str] = Field(None, description="结构化动作目标，如 APP_PROCESS_DOWN/srebench-app 等；与 action_type 配合由本地编译器生成安全命令")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化动作可选参数（预留扩展）")
    command: Optional[str] = Field(None, description="兼容旧方案的执行命令字符串；存在 action_type + target 时仅用于展示")
    risk_level: str = Field("low", description="风险等级: low/medium/high")
    expected_output: Optional[str] = Field(None, description="预期输出（用于验证步骤是否成功）")
    on_failure: Optional[str] = Field(None, description="失败时的处理方式描述")
    rollback_action_type: Optional[str] = Field(None, description="结构化回滚动作类型，与 action_type 对称")
    rollback_target: Optional[str] = Field(None, description="结构化回滚动作目标，与 target 对称")
    rollback_parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化回滚动作可选参数（预留扩展）")
    rollback_command: Optional[str] = Field(None, description="兼容旧方案的回滚命令字符串；存在结构化回滚动作时仅用于展示")


class FixPlan(BaseModel):
    """修复方案"""
    plan_id: str = Field(..., description="方案ID")
    description: str = Field(..., description="方案描述")
    risk_level: str = Field("low", description="风险等级: low/medium/high")
    prerequisites: List[str] = Field(default_factory=list, description="前置条件")
    steps: List[FixStep] = Field(default_factory=list, description="修复步骤列表")
    verification: Dict[str, Any] = Field(default_factory=dict, description="验证方法")
    estimated_time: str = Field("", description="预计执行时间")


class AgentMessage(BaseModel):
    """Agent 间通信消息

    v1 证据协作协议消息。通过 CommunicationBus 创建的新消息会自动带 message_id。

    设计思路：
    - 每条消息都有唯一 message_id，像快递单号一样可追溯
    - correlation_id 把同一批协作消息串成一条链，比如 "验证 Postgres 宕机" 这个假设下的所有请求/响应共用同一个 correlation_id
    - related_to 指向具体的上游消息，比如 evidence_response 要指明回应的是哪条 evidence_request
    - hypothesis + fault_type 让消息自带结构化语义，不再只是纯文本
    """
    # message_id：默认用 uuid4 生成，保证全局唯一。
    # 格式 msg-{hex} 是为了肉眼一眼认出这是消息 ID。
    message_id: str = Field(default_factory=lambda: f"msg-{uuid4().hex}", description="消息唯一ID，用于证据请求/响应关联")
    # correlation_id：同一假设下的所有消息共用此 ID。
    # 例如 db_agent 发布 hypothesis 时 correlation_id = 自己的 message_id，
    # 后续所有 evidence_request/evidence_response 都继承这个 ID，方便按链路查询。
    correlation_id: Optional[str] = Field(None, description="同一协作链路的关联ID，通常取 hypothesis 的 message_id")
    # related_to：点对点关联。evidence_response 必须填它回应的 evidence_request 的 message_id。
    related_to: Optional[str] = Field(None, description="关联的上游消息ID，如 evidence_response 指向 evidence_request")
    # hypothesis_id：如果这条消息是在验证某个假设，填那个假设的 message_id。
    hypothesis_id: Optional[str] = Field(None, description="该消息验证或支持的假设ID")
    # status：消息生命周期状态。
    # open = 刚发出还没人理；responded = 有人回复了；closed = 协作结束。
    status: str = Field("open", description="消息状态: open/responded/closed")
    sender: str = Field(..., description="发送者 Agent 名称")
    receiver: str = Field("broadcast", description="接收者，broadcast 表示广播")
    content: str = Field(..., description="消息内容")
    # msg_type 扩展为协议类型：hypothesis（假设）、evidence_request（求证据）、
    # evidence_response（给证据）、challenge（质疑）、support（支持）、diagnosis（诊断结论）、info（普通信息）
    msg_type: str = Field("info", description="消息类型: hypothesis/evidence_request/evidence_response/challenge/support/diagnosis/info")
    confidence: float = Field(0.0, description="置信度 0-1")
    # evidence 从 List[str] 升级为 List[EvidenceItem]，每个证据都是结构化对象，
    # 包含 source_agent/tool_name/target/status/observed/expected 等字段，方便机器处理。
    evidence: List[EvidenceItem] = Field(default_factory=list, description="结构化支撑证据")
    # hypothesis：一句话可验证假设，例如 "Postgres 容器停止导致数据库连接失败"
    hypothesis: Optional[str] = Field(None, description="结构化故障假设")
    # fault_type：标准化故障类型，用于后续按类型匹配修复动作（如 RECOVER_FAULT 对应 APP_PROCESS_DOWN）
    fault_type: Optional[str] = Field(None, description="假设对应的故障类型，如 APP_PROCESS_DOWN/DB_CONN_FAIL")
    # required_evidence：evidence_request 专用，列出需要对方提供的证据项名称。
    # 例如 ["nginx route status", "app direct health"]
    required_evidence: List[str] = Field(default_factory=list, description="请求对方提供的证据项")
    # suggested_tools：evidence_request 专用，建议对方调用哪些工具来收集证据。
    # 降低对方 Agent 的决策负担，直接告诉它 "你用 check_network_http_route 查一下"
    suggested_tools: List[str] = Field(default_factory=list, description="建议对方调用的工具名")
    # supports_hypothesis：evidence_response 专用，明确回答 "我找到的证据是否支持你的假设"。
    # True = 支持；False = 反对；None = 无法判断。
    supports_hypothesis: Optional[bool] = Field(None, description="证据是否支持 related_to 指向的假设")

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, value):
        """允许上游传入字符串或工具结果，但状态内统一保存结构化证据。

        为什么需要这个 validator：
        - LLM with_structured_output 偶尔会把 evidence 输出成字符串列表
        - 下游代码（如 Aggregate、DynamicCheck）只认 EvidenceItem 对象
        - 在这里做一次归一化，避免每个消费点都写兼容逻辑
        """
        return normalize_evidence_items(value or [])


class SystemState(BaseModel):
    """LangGraph工作流状态模型"""

    # ========== 输入信息 ==========
    ticket_id: str = Field(..., description="工单ID")
    symptom: str = Field(..., description="故障现象描述")

    # ========== Agent Memory / Case Library ==========
    # 案例库相关字段：检索历史相似案例，供 Supervisor 和 FixAgent 复用经验
    similar_cases: List[Dict[str, Any]] = Field(default_factory=list, description="相似历史案例")
    case_context: str = Field("无相似历史案例。", description="给 Supervisor/FixAgent 使用的案例上下文")
    case_memory: Optional[Dict[str, Any]] = Field(None, description="案例库检索元数据")

    # ========== Supervisor 决策 ==========
    diagnosis_type: Optional[DiagnosisType] = Field(None, description="诊断类型: app/db/net/other")
    urgency: Optional[Urgency] = Field(None, description="紧急程度: low/medium/high/critical")
    supervisor_decision: Optional[Dict[str, Any]] = Field(None, description="Supervisor派发决策")
    dispatched_agents: List[str] = Field(default_factory=list, description="被派发的Agent列表")

    # ========== Agent诊断结果 ==========
    db_agent_result: Optional[Dict[str, Any]] = Field(None, description="数据库Agent诊断结果")
    net_agent_result: Optional[Dict[str, Any]] = Field(None, description="网络Agent诊断结果")
    app_agent_result: Optional[Dict[str, Any]] = Field(None, description="应用Agent诊断结果")

    # ========== 聚合诊断 ==========
    aggregated_diagnosis: Optional[Dict[str, Any]] = Field(None, description="综合诊断结果")

    # ========== 动态调度 ==========
    dispatch_round: int = Field(0, description="当前调度轮次（防止无限循环）")
    max_dispatch_rounds: int = Field(3, description="最大动态调度轮次")
    # force_dispatched_agents：DynamicCheck 发现某 Agent 的旧结果不足以回答 evidence_request 时，
    # 会把该 Agent 名字放进来，Dispatch 节点下一轮会强制重新执行它（跳过已有结果的缓存）。
    force_dispatched_agents: List[str] = Field(default_factory=list, description="需要忽略缓存结果、强制重跑的 Agent")
    # redispatched_request_ids：记录哪些 evidence_request 已经触发过重跑。
    # 用 Annotated[..., operator.add] 是因为 LangGraph 状态合并时列表默认是替换，
    # operator.add 让它变成追加，这样多次 DynamicCheck 都能累积记录。
    # 目的是防止同一个请求因为证据覆盖不足被无限重派发。
    redispatched_request_ids: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="已经触发过定向重派发的 evidence_request ID，用于防止无限重跑"
    )

    # ========== Agent 间通信 ==========
    agent_messages: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list, description="Agent间通信消息")

    # ========== 修复方案 ==========
    fix_plan: Optional[FixPlan] = Field(None, description="Fix Agent生成的修复方案")

    # ========== 人工审批 ==========
    approval_status: ApprovalStatus = Field(ApprovalStatus.PENDING, description="审批状态")
    approver_comments: Optional[str] = Field(None, description="审批意见")

    # ========== 安全护栏 ==========
    guardrail_result: Optional[Dict[str, Any]] = Field(None, description="安全护栏检查结果")

    # ========== 闭环执行器 ==========
    execution_result: Optional[Dict[str, Any]] = Field(None, description="执行结果")
    execution_trace: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="闭环执行器轨迹：每一步的执行结果、LLM决策、重试/回滚记录"
    )

    # ========== 恢复验证 ==========
    verification_result: Optional[Dict[str, Any]] = Field(None, description="恢复验证结果")

    # ========== 执行失败后重规划 ==========
    replanner_result: Optional[Dict[str, Any]] = Field(
        None, description="Replanner/Critic 的决策结果，包含 decision、failure_type、reason 等"
    )
    replanner_round: int = Field(
        0, description="Replanner 已介入的轮次，每调用一次 Replanner 节点就 +1"
    )
    max_replanner_rounds: int = Field(
        2, description="Replanner 允许的最大介入轮次，超过则强制 escalate（升级人工处理）"
    )

    # ========== 审计日志（用于可追溯性） ==========
    audit_logs: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="Agent 操作审计日志，用于追溯工单处理流程"
    )

    # ========== Standard Trace Events ==========
    # 标准化 Trace 事件列表，LangGraph 节点返回的 trace_events 会通过 operator.add 自动累加到这里
    # 用于评测、前端可视化、单步分析等场景
    trace_events: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="标准 Trace 事件，用于评测和单步流程分析"
    )

    # ========== 辅助字段 ==========
    messages: Annotated[List[str], operator.add] = Field(default_factory=list, description="处理过程中的消息记录")
