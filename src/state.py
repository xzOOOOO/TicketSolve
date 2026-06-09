"""
工单状态模型 - 定义整个工作流的状态结构

本模块是 LangGraph 工作流的数据基础，所有节点共享同一个 SystemState 对象。
设计原则：
1. 状态扁平化：所有信息都存在顶层字段，方便节点间传递
2. 类型安全：用 Pydantic 模型做运行时校验，防止 LLM 输出格式错误
3. 可追溯：每个字段都有 description，生成 JSON Schema 供 LLM 理解
4. 防循环：dispatch_round / max_dispatch_rounds 控制最大调度轮次
"""

# field_validator：Pydantic 提供的字段校验装饰器，在赋值前自动调用
# 这里用于把 LLM 输出的字符串证据统一转成结构化 EvidenceItem 对象
from pydantic import BaseModel, Field, field_validator

# EvidenceItem：结构化证据对象，包含 source_agent/tool_name/target/status/observed/expected 等字段
# normalize_evidence_items：归一化工具函数，把字符串列表或字典列表统一转成 EvidenceItem 列表
from evidence import EvidenceItem, normalize_evidence_items

# uuid4：生成全局唯一标识符（128位随机数），保证每条 agent_message 都有独立标识
from uuid import uuid4

# Optional：可选类型，如 Optional[str] 表示 str 或 None
# List：列表类型
# Dict：字典类型
# Any：任意类型
# Annotated：带元数据的类型，这里配合 operator.add 使用
from typing import Optional, List, Dict, Any, Annotated

# Enum：枚举基类，用于定义有限取值的字符串枚举
from enum import Enum

# operator：Python 内置运算符模块
# operator.add 用于 LangGraph 状态合并：让列表字段在节点间合并时是"追加"而非"替换"
import operator


# ───────────────────────────────────────────────
# 枚举类型定义
# ───────────────────────────────────────────────

class DiagnosisType(str, Enum):
    """
    诊断类型枚举

    让 LLM 的输出标准化，避免它随意编造分类名称。
    同时也给前端展示用，知道这是什么类型的问题。
    """
    APP = "app"      # 应用问题（进程崩溃、内存泄漏、CPU 过高等）
    DB = "db"        # 数据库问题（连接失败、慢查询、死锁、主从延迟等）
    NET = "net"      # 网络问题（连通性、延迟、路由、DNS 等）
    OTHER = "other"  # 其他问题（配置错误、权限问题、未知类型等）


class Urgency(str, Enum):
    """
    紧急程度枚举

    Supervisor Agent 根据症状严重程度判断，用于决定响应速度。
    """
    LOW = "low"           # 低：可延后处理，如非核心功能异常
    MEDIUM = "medium"     # 中：24 小时内处理，如性能下降但服务可用
    HIGH = "high"         # 高：需尽快处理，如核心功能受影响
    CRITICAL = "critical" # 紧急：立即处理，如服务完全不可用


class ApprovalStatus(str, Enum):
    """
    审批状态枚举

    跟踪人工审批的完整生命周期。
    """
    PENDING = "pending"     # 待审批：修复方案已生成，等待人工确认
    APPROVED = "approved"   # 已批准：可以进入执行阶段
    REJECTED = "rejected"   # 已拒绝：方案被驳回，需要重新诊断或升级


# ───────────────────────────────────────────────
# 修复步骤与方案模型
# ───────────────────────────────────────────────

class FixStep(BaseModel):
    """
    修复步骤（工作流状态中的步骤模型）

    与 schemas.py 中的 FixStepOutput 对应，但字段更宽松（部分字段可选），
    因为工作流执行过程中步骤可能被部分填充（如先生成 action，再补充 command）。

    安全执行设计：
    - action_type + target 组成结构化动作 DSL，由 action_dsl.py 编译成安全命令
    - command 为兼容旧模式的自由文本命令，存在结构化动作时不参与执行
    - 回滚同理：rollback_action_type + rollback_target 的优先级高于 rollback_command

    为什么这样设计：
    LLM 喜欢输出自由文本命令（如 "docker start xxx"），但直接执行有注入风险。
    所以我们让 LLM 只选"动作类型"和"目标"，具体命令由本地白名单编译生成。
    """

    # step_id：步骤编号，从 1 开始递增，用于标识执行顺序和日志关联
    step_id: int = Field(..., description="步骤编号")

    # action：人可读的修复动作描述，如"启动 PostgreSQL 容器"
    # 这个字段只用于展示，不参与实际执行
    action: str = Field(..., description="修复动作描述（人可读）")

    # action_type：结构化动作类型，如 RECOVER_FAULT / START_CONTAINER / RESTART_CONTAINER 等
    # 优先于 command 执行，是安全 DSL 的核心字段
    action_type: Optional[str] = Field(None, description="结构化动作类型，如 RECOVER_FAULT/START_CONTAINER 等；优先于 command 执行")

    # target：结构化动作目标，如 APP_PROCESS_DOWN / srebench-app 等
    # 与 action_type 配合，由本地编译器生成安全命令
    target: Optional[str] = Field(None, description="结构化动作目标，如 APP_PROCESS_DOWN/srebench-app 等；与 action_type 配合由本地编译器生成安全命令")

    # parameters：结构化动作的可选参数（预留扩展字段）
    # 当前版本未使用，为未来复杂动作（如带超时参数的重启）预留
    parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化动作可选参数（预留扩展）")

    # command：兼容旧方案的执行命令字符串
    # 存在 action_type + target 时仅用于展示，实际执行用编译后的命令
    command: Optional[str] = Field(None, description="兼容旧方案的执行命令字符串；存在 action_type + target 时仅用于展示")

    # risk_level：风险等级，用于 Guardrail 检查和人审参考
    # low = 安全操作（如查看日志）；medium = 可能影响服务（如重启）；high = 高风险（如删数据）
    risk_level: str = Field("low", description="风险等级: low/medium/high")

    # expected_output：预期输出，用于验证步骤是否成功
    # 如 "container srebench-postgres is running"
    expected_output: Optional[str] = Field(None, description="预期输出（用于验证步骤是否成功）")

    # on_failure：失败时的处理方式描述，给人看的，不自动执行
    # 如 "检查 Docker 服务状态，尝试手动启动"
    on_failure: Optional[str] = Field(None, description="失败时的处理方式描述")

    # rollback_action_type：结构化回滚动作类型，与 action_type 对称
    # 当正向执行失败时，用回滚动作撤销已执行的操作
    rollback_action_type: Optional[str] = Field(None, description="结构化回滚动作类型，与 action_type 对称")

    # rollback_target：结构化回滚动作目标，与 target 对称
    rollback_target: Optional[str] = Field(None, description="结构化回滚动作目标，与 target 对称")

    # rollback_parameters：结构化回滚动作的可选参数（预留扩展）
    rollback_parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化回滚动作可选参数（预留扩展）")

    # rollback_command：兼容旧方案的回滚命令字符串
    # 存在结构化回滚动作时仅用于展示
    rollback_command: Optional[str] = Field(None, description="兼容旧方案的回滚命令字符串；存在结构化回滚动作时仅用于展示")


class FixPlan(BaseModel):
    """
    修复方案

    一个完整的修复方案包含多个步骤（FixStep），以及验证方法和元数据。
    FixAgent 生成 FixPlan，然后经 RepairPlanner 规范化，再经 Guardrail 安全检查。
    """

    # plan_id：方案唯一标识，通常用 uuid 或递增序号
    plan_id: str = Field(..., description="方案ID")

    # description：方案的整体描述，给人看的摘要
    description: str = Field(..., description="方案描述")

    # risk_level：整体风险等级，取所有步骤中最高的风险等级
    risk_level: str = Field("low", description="风险等级: low/medium/high")

    # prerequisites：前置条件列表，执行前必须满足的条件
    # 如 ["确保 Docker 服务运行中", "备份数据库"]
    prerequisites: List[str] = Field(default_factory=list, description="前置条件")

    # steps：修复步骤列表，按顺序执行
    steps: List[FixStep] = Field(default_factory=list, description="修复步骤列表")

    # verification：验证方法，描述如何确认修复成功
    # 如 {"method": "http_probe", "url": "http://localhost:18080/health"}
    verification: Dict[str, Any] = Field(default_factory=dict, description="验证方法")

    # estimated_time：预计执行时间，给人看的参考
    estimated_time: str = Field("", description="预计执行时间")


# ───────────────────────────────────────────────
# Agent 间通信消息模型
# ───────────────────────────────────────────────

class AgentMessage(BaseModel):
    """
    Agent 间通信消息

    v1 证据协作协议消息。通过 CommunicationBus 创建的新消息会自动带 message_id。

    设计思路：
    - 每条消息都有唯一 message_id，像快递单号一样可追溯
    - correlation_id 把同一批协作消息串成一条链，比如 "验证 Postgres 宕机" 这个假设下的所有请求/响应共用同一个 correlation_id
    - related_to 指向具体的上游消息，比如 evidence_response 要指明回应的是哪条 evidence_request
    - hypothesis + fault_type 让消息自带结构化语义，不再只是纯文本

    为什么需要这个协议：
    多 Agent 协作时，Agent A 可能向 Agent B 请求证据（如"请检查数据库连接"）。
    如果没有标准化消息格式，Agent B 可能看不懂请求，或者回应无法被自动处理。
    """

    # message_id：消息唯一标识，默认用 uuid4 生成
    # 格式 msg-{hex} 是为了肉眼一眼认出这是消息 ID
    message_id: str = Field(
        default_factory=lambda: f"msg-{uuid4().hex}",
        description="消息唯一ID，用于证据请求/响应关联"
    )

    # correlation_id：同一假设下的所有消息共用此 ID
    # 例如 db_agent 发布 hypothesis 时 correlation_id = 自己的 message_id，
    # 后续所有 evidence_request/evidence_response 都继承这个 ID，方便按链路查询
    correlation_id: Optional[str] = Field(
        None,
        description="同一协作链路的关联ID，通常取 hypothesis 的 message_id"
    )

    # related_to：点对点关联
    # evidence_response 必须填它回应的 evidence_request 的 message_id
    related_to: Optional[str] = Field(
        None,
        description="关联的上游消息ID，如 evidence_response 指向 evidence_request"
    )

    # hypothesis_id：如果这条消息是在验证某个假设，填那个假设的 message_id
    hypothesis_id: Optional[str] = Field(
        None,
        description="该消息验证或支持的假设ID"
    )

    # status：消息生命周期状态
    # open = 刚发出还没人理；responded = 有人回复了；closed = 协作结束
    status: str = Field("open", description="消息状态: open/responded/closed")

    # sender：发送者 Agent 名称，如 "db_agent" / "net_agent" / "supervisor"
    sender: str = Field(..., description="发送者 Agent 名称")

    # receiver：接收者 Agent 名称，"broadcast" 表示广播给所有 Agent
    receiver: str = Field("broadcast", description="接收者，broadcast 表示广播")

    # content：消息内容，人可读的文本描述
    content: str = Field(..., description="消息内容")

    # msg_type：消息类型，扩展为协议类型
    # hypothesis（假设）、evidence_request（求证据）、evidence_response（给证据）、
    # challenge（质疑）、support（支持）、diagnosis（诊断结论）、info（普通信息）
    msg_type: str = Field(
        "info",
        description="消息类型: hypothesis/evidence_request/evidence_response/challenge/support/diagnosis/info"
    )

    # confidence：置信度 0-1，表示发送者对该消息内容的确定程度
    confidence: float = Field(0.0, description="置信度 0-1")

    # evidence：结构化支撑证据列表
    # 每个证据都是 EvidenceItem 对象，包含 source_agent/tool_name/target/status/observed/expected 等字段
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="结构化支撑证据"
    )

    # hypothesis：一句话可验证假设
    # 例如 "Postgres 容器停止导致数据库连接失败"
    hypothesis: Optional[str] = Field(None, description="结构化故障假设")

    # fault_type：标准化故障类型
    # 用于后续按类型匹配修复动作（如 RECOVER_FAULT 对应 APP_PROCESS_DOWN）
    fault_type: Optional[str] = Field(
        None,
        description="假设对应的故障类型，如 APP_PROCESS_DOWN/DB_CONN_FAIL"
    )

    # required_evidence：evidence_request 专用
    # 列出需要对方提供的证据项名称，如 ["nginx route status", "app direct health"]
    required_evidence: List[str] = Field(
        default_factory=list,
        description="请求对方提供的证据项"
    )

    # suggested_tools：evidence_request 专用
    # 建议对方调用哪些工具来收集证据，降低对方 Agent 的决策负担
    # 直接告诉它 "你用 check_network_http_route 查一下"
    suggested_tools: List[str] = Field(
        default_factory=list,
        description="建议对方调用的工具名"
    )

    # supports_hypothesis：evidence_response 专用
    # 明确回答 "我找到的证据是否支持你的假设"
    # True = 支持；False = 反对；None = 无法判断
    supports_hypothesis: Optional[bool] = Field(
        None,
        description="证据是否支持 related_to 指向的假设"
    )

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, value):
        """
        允许上游传入字符串或工具结果，但状态内统一保存结构化证据。

        为什么需要这个 validator：
        - LLM with_structured_output 偶尔会把 evidence 输出成字符串列表
        - 下游代码（如 Aggregate、DynamicCheck）只认 EvidenceItem 对象
        - 在这里做一次归一化，避免每个消费点都写兼容逻辑

        参数：
            value: 上游传入的 evidence 值，可能是字符串列表、字典列表或 EvidenceItem 列表

        返回：
            标准化的 EvidenceItem 列表
        """
        return normalize_evidence_items(value or [])


# ───────────────────────────────────────────────
# 核心状态模型：SystemState
# ───────────────────────────────────────────────

class SystemState(BaseModel):
    """
    LangGraph 工作流状态模型

    这是整个 Multi-Agent 系统的"全局状态"，所有节点共享这个对象。
    LangGraph 会自动处理状态合并：节点返回的字典会合并到当前状态中。

    关键设计：
    1. 用 Annotated[..., operator.add] 的列表字段会在节点间自动累加
       如 agent_messages、audit_logs、trace_events 等
    2. 普通字段会被节点返回的新值覆盖
    3. 所有字段都有默认值，方便创建初始状态
    """

    # ═══════════════════════════════════════════════
    # 输入信息（创建工单时填充）
    # ═══════════════════════════════════════════════

    # ticket_id：工单唯一标识，通常由前端或 API 层生成
    ticket_id: str = Field(..., description="工单ID")

    # symptom：故障现象描述，由用户或监控系统提供
    # 这是整个工作流的起点，所有 Agent 都基于这个描述进行分析
    symptom: str = Field(..., description="故障现象描述")

    # ═══════════════════════════════════════════════
    # Agent Memory / Case Library（案例库相关）
    # ═══════════════════════════════════════════════

    # similar_cases：检索到的相似历史案例列表
    # 每个案例是字典，包含 case_id、symptom、diagnosis、fix_plan 等字段
    similar_cases: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="相似历史案例"
    )

    # case_context：给 Supervisor/FixAgent 使用的案例上下文文本
    # 由 format_case_context() 函数从 similar_cases 生成，是压缩后的人可读摘要
    case_context: str = Field(
        "无相似历史案例。",
        description="给 Supervisor/FixAgent 使用的案例上下文"
    )

    # case_memory：案例库检索元数据
    # 如 library_path、similar_case_count、retrieved_at 等，用于审计和调试
    case_memory: Optional[Dict[str, Any]] = Field(
        None,
        description="案例库检索元数据"
    )

    # ═══════════════════════════════════════════════
    # Supervisor 决策（由 SupervisorAgent 填充）
    # ═══════════════════════════════════════════════

    # diagnosis_type：诊断类型，由 Supervisor 根据症状判断
    diagnosis_type: Optional[DiagnosisType] = Field(
        None,
        description="诊断类型: app/db/net/other"
    )

    # urgency：紧急程度，由 Supervisor 根据症状严重程度判断
    urgency: Optional[Urgency] = Field(
        None,
        description="紧急程度: low/medium/high/critical"
    )

    # supervisor_decision：Supervisor 的完整决策字典
    # 包含 dispatched_agents、reasoning、confidence 等字段
    supervisor_decision: Optional[Dict[str, Any]] = Field(
        None,
        description="Supervisor派发决策"
    )

    # dispatched_agents：被派发的 Agent 列表
    # 如 ["db_agent", "net_agent"]，Dispatch 节点根据这个列表并行执行
    dispatched_agents: List[str] = Field(
        default_factory=list,
        description="被派发的Agent列表"
    )

    # ═══════════════════════════════════════════════
    # Agent 诊断结果（由各个 Agent 填充）
    # ═══════════════════════════════════════════════

    # db_agent_result：数据库 Agent 的诊断结果
    # 包含 diagnosis、evidence、confidence、recommended_actions 等字段
    db_agent_result: Optional[Dict[str, Any]] = Field(
        None,
        description="数据库Agent诊断结果"
    )

    # net_agent_result：网络 Agent 的诊断结果
    net_agent_result: Optional[Dict[str, Any]] = Field(
        None,
        description="网络Agent诊断结果"
    )

    # app_agent_result：应用 Agent 的诊断结果
    app_agent_result: Optional[Dict[str, Any]] = Field(
        None,
        description="应用Agent诊断结果"
    )

    # ═══════════════════════════════════════════════
    # 聚合诊断（由 Aggregate 节点填充）
    # ═══════════════════════════════════════════════

    # aggregated_diagnosis：综合诊断结果
    # 合并多个 Agent 的诊断，解决冲突，形成统一结论
    aggregated_diagnosis: Optional[Dict[str, Any]] = Field(
        None,
        description="综合诊断结果"
    )

    # ═══════════════════════════════════════════════
    # 动态调度（防止无限循环）
    # ═══════════════════════════════════════════════

    # dispatch_round：当前调度轮次，从 0 开始
    # 每经过一次 Dispatch 节点就 +1，用于控制最大循环次数
    dispatch_round: int = Field(
        0,
        description="当前调度轮次（防止无限循环）"
    )

    # max_dispatch_rounds：最大动态调度轮次
    # 超过这个轮次后，DynamicCheck 不再追加派发，只生成自动响应
    max_dispatch_rounds: int = Field(
        3,
        description="最大动态调度轮次"
    )

    # force_dispatched_agents：需要忽略缓存结果、强制重跑的 Agent 列表
    # DynamicCheck 发现某 Agent 的旧结果不足以回答 evidence_request 时，
    # 会把该 Agent 名字放进来，Dispatch 节点下一轮会强制重新执行它
    force_dispatched_agents: List[str] = Field(
        default_factory=list,
        description="需要忽略缓存结果、强制重跑的 Agent"
    )

    # redispatched_request_ids：已经触发过定向重派发的 evidence_request ID 列表
    # 用 Annotated[..., operator.add] 是因为 LangGraph 状态合并时列表默认是替换，
    # operator.add 让它变成追加，这样多次 DynamicCheck 都能累积记录
    # 目的是防止同一个请求因为证据覆盖不足被无限重派发
    redispatched_request_ids: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="已经触发过定向重派发的 evidence_request ID，用于防止无限重跑"
    )

    # ═══════════════════════════════════════════════
    # Agent 间通信（通过 CommunicationBus）
    # ═══════════════════════════════════════════════

    # agent_messages：Agent 间通信消息列表
    # 用 Annotated[..., operator.add] 实现自动累加，每条新消息都会追加到列表末尾
    agent_messages: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="Agent间通信消息"
    )

    # ═══════════════════════════════════════════════
    # 修复方案（由 FixAgent 填充）
    # ═══════════════════════════════════════════════

    # fix_plan：Fix Agent 生成的修复方案
    # 包含 plan_id、description、steps、verification 等字段
    fix_plan: Optional[FixPlan] = Field(
        None,
        description="Fix Agent生成的修复方案"
    )

    # ═══════════════════════════════════════════════
    # 人工审批（由 Human Approval 节点填充）
    # ═══════════════════════════════════════════════

    # approval_status：当前审批状态
    approval_status: ApprovalStatus = Field(
        ApprovalStatus.PENDING,
        description="审批状态"
    )

    # approver_comments：审批人填写的意见或备注
    approver_comments: Optional[str] = Field(
        None,
        description="审批意见"
    )

    # ═══════════════════════════════════════════════
    # 安全护栏（由 Guardrail 节点填充）
    # ═══════════════════════════════════════════════

    # guardrail_result：安全护栏检查结果
    # 包含 passed（是否通过）、violations（违规项列表）等字段
    guardrail_result: Optional[Dict[str, Any]] = Field(
        None,
        description="安全护栏检查结果"
    )

    # ═══════════════════════════════════════════════
    # 闭环执行器（由 Executor 节点填充）
    # ═══════════════════════════════════════════════

    # execution_result：执行结果摘要
    # 包含 overall_status、completed_steps、failed_steps 等字段
    execution_result: Optional[Dict[str, Any]] = Field(
        None,
        description="执行结果"
    )

    # execution_trace：闭环执行器轨迹
    # 每一步的执行结果、LLM 决策、重试/回滚记录
    # 用 Annotated[..., operator.add] 实现跨轮次累加
    execution_trace: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="闭环执行器轨迹：每一步的执行结果、LLM决策、重试/回滚记录"
    )

    # ═══════════════════════════════════════════════
    # 恢复验证（由 Verify 节点填充）
    # ═══════════════════════════════════════════════

    # verification_result：恢复验证结果
    # 包含 verified（是否通过）、verification_probe（探测详情）、recovered_at（恢复时间）等字段
    verification_result: Optional[Dict[str, Any]] = Field(
        None,
        description="恢复验证结果"
    )

    # ═══════════════════════════════════════════════
    # 执行失败后重规划（由 Replanner 节点填充）
    # ═══════════════════════════════════════════════

    # replanner_result：Replanner/Critic 的决策结果
    # 包含 decision（verify/retry/re-diagnose/escalate）、failure_type、reason 等字段
    replanner_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Replanner/Critic 的决策结果，包含 decision、failure_type、reason 等"
    )

    # replanner_round：Replanner 已介入的轮次
    # 每调用一次 Replanner 节点就 +1，用于控制最大重规划次数
    replanner_round: int = Field(
        0,
        description="Replanner 已介入的轮次，每调用一次 Replanner 节点就 +1"
    )

    # max_replanner_rounds：Replanner 允许的最大介入轮次
    # 超过则强制 escalate（升级人工处理），防止无限重试
    max_replanner_rounds: int = Field(
        2,
        description="Replanner 允许的最大介入轮次，超过则强制 escalate（升级人工处理）"
    )

    # ═══════════════════════════════════════════════
    # 审计日志（用于可追溯性）
    # ═══════════════════════════════════════════════

    # audit_logs：Agent 操作审计日志列表
    # 每个日志包含 ticket_id、agent_name、action_type、action_detail、input_context、output_result 等字段
    # 用 Annotated[..., operator.add] 实现跨节点累加
    audit_logs: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="Agent 操作审计日志，用于追溯工单处理流程"
    )

    # ═══════════════════════════════════════════════
    # Standard Trace Events（标准化轨迹事件）
    # ═══════════════════════════════════════════════

    # trace_events：标准化 Trace 事件列表
    # LangGraph 节点返回的 trace_events 会通过 operator.add 自动累加到这里
    # 用于评测、前端可视化、单步分析等场景
    # 每个事件包含 event_type、ticket_id、agent_name、status、input_data、output_data、metadata 等字段
    trace_events: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="标准 Trace 事件，用于评测和单步流程分析"
    )

    # ═══════════════════════════════════════════════
    # 辅助字段
    # ═══════════════════════════════════════════════

    # messages：处理过程中的消息记录
    # 人可读的文本日志，用于前端展示或调试
    # 用 Annotated[..., operator.add] 实现跨节点累加
    messages: Annotated[List[str], operator.add] = Field(
        default_factory=list,
        description="处理过程中的消息记录"
    )
