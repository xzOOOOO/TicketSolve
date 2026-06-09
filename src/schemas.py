# ============================================================
# schemas.py：Pydantic 数据模型定义文件
#
# 作用：
#   1. 定义 API 请求/响应的数据结构（FastAPI 自动校验）
#   2. 定义 LLM 结构化输出的 schema（with_structured_output 用）
#
# 核心概念：
#   - Pydantic BaseModel：Python 的数据类，带自动类型校验和 JSON 序列化
#   - Field(...)：给字段加描述、默认值、校验规则
#   - field_validator：在赋值前自动校验/转换字段值
#
# 为什么需要结构化输出？
#   LLM 原本输出的是自由文本，需要正则/JSON 解析，容易出错。
#   with_structured_output 让 LLM 通过 function calling 直接返回
#   符合 schema 的字典，Pydantic 自动转成对象，100% 可解析。
# ============================================================

# ConfigDict：Pydantic v2 的模型配置类型，用于替代旧版 class Config
# field_validator：Pydantic 提供的装饰器，在字段赋值前自动执行校验/转换逻辑
# 这里用于把 LLM 偶尔输出的字符串证据归一化成 EvidenceItem 对象
from pydantic import BaseModel, ConfigDict, Field, field_validator
# EvidenceItem：结构化证据对象，定义在 evidence.py
# normalize_evidence_items：把字符串/字典混合格式统一成 EvidenceItem 对象列表
from evidence import EvidenceItem, normalize_evidence_items
# typing 模块：Python 类型注解工具
# Optional[X] = X | None，表示"可以是 X 或 None"
# Dict/Any/List 用于标注字典、任意类型、列表
from typing import Optional, Dict, Any, List
# datetime：Python 日期时间类型，用于 created_at/updated_at 字段
from datetime import datetime


# ═══════════════════════════════════════════════════════════
# 一、API 请求/响应模型（FastAPI 用）
# ═══════════════════════════════════════════════════════════


class TicketCreateRequest(BaseModel):
    """创建工单请求体

    用户调用 POST /api/tickets 时传入的 JSON 数据。
    FastAPI 会自动用这个模型校验请求体：
    - ticket_id 缺失 → 返回 422 错误
    - symptom 不是字符串 → 返回 422 错误
    """
    # ticket_id：业务工单号，由调用方生成，必须唯一
    # ... 表示必填（没有默认值）
    ticket_id: str = Field(..., description="工单ID")
    # symptom：故障现象描述，用户输入的原始文本
    symptom: str = Field(..., description="故障现象描述")


class ApprovalRequest(BaseModel):
    """审批请求体

    管理员调用审批接口时传入的 JSON 数据。
    approved=True 表示批准执行修复方案，approved=False 表示驳回。
    """
    # approved：是否批准，布尔值，必填
    approved: bool = Field(..., description="是否批准")
    # comments：审批意见/备注，可选（None 表示没写意见）
    comments: Optional[str] = Field(None, description="审批意见")


class TicketResponse(BaseModel):
    """工单详情响应体

    查询工单详情时返回的完整数据结构。
    包含工单从创建到关闭的全生命周期字段。

    与数据库模型 Ticket（database.py）对应，但这里是 Pydantic 模型，
    用于 API 序列化（把 Python 对象转成 JSON 返回给客户端）。
    """
    # id：数据库主键，UUID 字符串格式
    id: str
    # ticket_id：业务工单号
    ticket_id: str
    # symptom：故障现象描述
    symptom: str
    # diagnosis_type：诊断类型（app/db/net/other），可能为空（还没诊断完）
    diagnosis_type: Optional[str]
    # urgency：紧急程度（low/medium/high/critical）
    urgency: Optional[str]
    # status：工单状态（pending/diagnosing/fixing/executing/approved/rejected/completed/failed）
    status: str
    # diagnosis_result：诊断结果字典，包含各 Agent 的诊断结论
    diagnosis_result: Optional[Dict[str, Any]]
    # fix_plan：修复方案字典，FixAgent 生成
    fix_plan: Optional[Dict[str, Any]]
    # execution_result：执行结果字典，闭环执行器生成
    execution_result: Optional[Dict[str, Any]]
    # approval_status：审批状态（pending/approved/rejected）
    approval_status: Optional[str]
    # approver_comments：审批人填写的意见
    approver_comments: Optional[str]
    # messages：工作流运行过程中产生的消息列表（用于前端展示进度）
    messages: Optional[List[str]]
    # created_at：创建时间
    created_at: Optional[datetime]
    # updated_at：最后更新时间
    updated_at: Optional[datetime]

    # model_config：Pydantic v2 模型配置，允许从 SQLAlchemy ORM 对象直接构造响应模型
    model_config = ConfigDict(from_attributes=True)


class APIResponse(BaseModel):
    """通用 API 响应包装器

    所有接口统一返回的结构：
    {
        "code": 200,
        "message": "success",
        "data": { ... 实际数据 ... }
    }

    好处：前端处理逻辑统一，不用判断每种接口返回不同结构。
    """
    # code：业务状态码，200 表示成功，其他值表示各种错误
    code: int = 200
    # message：人可读的状态描述
    message: str = "success"
    # data：实际业务数据，类型不限（Any），可以是字典/列表/字符串等
    data: Optional[Any] = None


# ============================================================
# 二、Structured Output 模型（用于 LLM with_structured_output）
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

    为什么需要这个模型？
    - 让 LLM 输出"可编程"的结构，而不是自由文本
    - dispatch 列表直接决定工作流走哪些分支
    """
    # diagnosis_type：故障分类结果，决定后续调用哪些诊断 Agent
    # 必须是四个固定值之一，LLM 被 prompt 约束只能输出这些值
    diagnosis_type: str = Field(description="诊断类型，必须是以下之一: app(应用问题)/db(数据库问题)/net(网络问题)/other(其他问题或无法判断)。如果无法明确判断类型，必须填 other，不要填 unknown 或其他值。")
    # urgency：紧急程度，影响人工审批的优先级提示
    urgency: str = Field(description="紧急程度: low/medium/high/critical")
    # dispatch：需要执行的 Agent 名称列表
    # 如 ["db_agent", "net_agent"] 表示同时诊断数据库和网络
    dispatch: List[str] = Field(description="需要派发的Agent列表，如 ['db_agent', 'net_agent', 'app_agent']")
    # reasoning：LLM 的推理过程，用于日志和调试
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
    # diagnosis：人可读的诊断结论，如 "PostgreSQL 主库连接超时"
    diagnosis: str = Field(description="具体诊断结论")
    # possible_causes：可能的原因列表，用于 FixAgent 生成修复方案时参考
    possible_causes: List[str] = Field(description="可能的原因列表")
    # confidence：诊断置信度，0-1 之间，用于 Aggregate 节点加权汇总
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

        参数：
            value：原始值，可能是 None、字符串列表、字典列表或 EvidenceItem 列表

        返回：
            规范化后的 EvidenceItem 列表
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
    # step_id：步骤编号，从 1 开始递增，用于执行顺序和日志标识
    step_id: int = Field(description="步骤编号，必须是纯数字如 1/2/3")
    # action：人可读的动作描述，如 "重启 PostgreSQL 服务"
    action: str = Field(description="具体动作描述（人可读）")
    # action_type：结构化动作类型，如 RECOVER_FAULT/START_CONTAINER/HTTP_PROBE
    # 与 target 配合，由 action_dsl.py 编译成实际命令
    action_type: Optional[str] = Field(None, description="结构化动作类型，如 RECOVER_FAULT/START_CONTAINER/HTTP_PROBE 等；优先于 command 执行")
    # target：结构化动作目标，如 APP_PROCESS_DOWN、srebench-app
    # 与 action_type 配合确定"对谁做什么"
    target: Optional[str] = Field(None, description="结构化动作目标，如 APP_PROCESS_DOWN/srebench-app 等；与 action_type 配合由本地编译器生成安全命令")
    # parameters：结构化动作的额外参数，预留扩展用
    # 如 HTTP_PROBE 可能需要 url、timeout 等参数
    parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化动作可选参数（预留扩展）")
    # command：兼容旧方案的完整命令字符串
    # 当存在 action_type + target 时，此字段仅用于展示和日志，不会被执行
    command: Optional[str] = Field(None, description="兼容旧方案的完整命令字符串；当存在 action_type + target 时，此字段仅用于展示和日志，不会被执行")
    # risk_level：风险等级，影响人工审批的提示和自动执行的策略
    risk_level: str = Field(default="low", description="风险等级: low/medium/high")
    # expected_output：预期输出，执行后用来验证步骤是否成功
    # 如 "active (running)" 表示服务已启动
    expected_output: str = Field(description="预期输出（用于验证步骤是否成功）")
    # on_failure：失败时的处理策略描述，给执行器参考
    on_failure: str = Field(description="失败时的处理方式描述")
    # rollback_action_type：结构化回滚动作类型，与 action_type 对称
    # 执行失败时，执行器用这对字段生成回滚命令
    rollback_action_type: Optional[str] = Field(None, description="结构化回滚动作类型，与 action_type 对称；用于执行失败时自动回滚")
    # rollback_target：结构化回滚动作目标，与 target 对称
    rollback_target: Optional[str] = Field(None, description="结构化回滚动作目标，与 target 对称")
    # rollback_parameters：回滚动作的额外参数
    rollback_parameters: Dict[str, Any] = Field(default_factory=dict, description="结构化回滚动作可选参数（预留扩展）")
    # rollback_command：兼容旧方案的回滚命令字符串
    # 存在结构化回滚动作时仅用于展示
    rollback_command: Optional[str] = Field(None, description="兼容旧方案的回滚命令字符串；存在结构化回滚动作时仅用于展示")


class VerificationOutput(BaseModel):
    """验证方法输出

    FixPlanOutput 的嵌套子模型，描述修复后的验证方式。
    修复执行完成后，执行器会运行这些命令确认问题已解决。
    """
    # commands：验证命令列表，如 ["curl -s http://localhost/health", "pg_isready"]
    commands: List[str] = Field(description="验证命令列表")
    # expected_result：预期验证结果，如 "HTTP 200" 或 " accepting connections"
    expected_result: str = Field(description="预期验证结果")


class FixPlanOutput(BaseModel):
    """修复方案输出

    用于 FixAgent 生成完整的修复方案。
    对应原 parse_json_content 解析的复杂嵌套 JSON 结构，
    包含 steps（FixStepOutput 列表）和 verification（VerificationOutput）。
    """
    # plan_id：方案编号，如 "PLAN-001"，用于日志和追踪
    plan_id: str = Field(description="方案ID，如 PLAN-001")
    # description：方案一句话简述，用于审批页面展示
    description: str = Field(description="方案简述")
    # risk_level：整体风险等级，取 steps 中最高的风险等级
    risk_level: str = Field(description="风险等级: low/medium/high")
    # prerequisites：前置条件列表，执行前必须满足的条件
    # 如 ["确保有数据库备份", "通知运维团队"]
    prerequisites: List[str] = Field(description="前置条件列表")
    # steps：修复步骤列表，按顺序执行
    steps: List[FixStepOutput] = Field(description="修复步骤列表")
    # verification：修复完成后的验证方法
    verification: VerificationOutput = Field(description="验证方法")
    # estimated_time：预计执行时间，如 "5分钟"，用于审批参考
    estimated_time: str = Field(description="预计执行时间")


class AggregateOutput(BaseModel):
    """聚合诊断输出

    用于 aggregate 节点综合多个 Agent 的诊断结果。
    对应原 parse_json_content 解析的 {"diagnosis", "possible_causes", "confidence", "contributing_agents", "reasoning"} 结构。

    新增 protocol_summary：把 Agent 间协作协议的结论也纳入聚合结果。
    这样 FixAgent 不仅知道 "诊断是什么"，还知道 "这个诊断在协议中战胜了哪些竞争假设"。
    """
    # diagnosis：最终诊断结论，综合所有 Agent 意见后的结果
    diagnosis: str = Field(description="最终诊断结论")
    # possible_causes：汇总所有 Agent 提出的可能原因，去重排序
    possible_causes: List[str] = Field(description="可能的原因列表")
    # confidence：聚合后的整体置信度，通常取加权平均或最高值
    confidence: float = Field(description="诊断置信度，范围 0-1")
    # contributing_agents：参与诊断并贡献了有效结论的 Agent 列表
    contributing_agents: List[str] = Field(description="贡献诊断的Agent列表")
    # reasoning：聚合推理过程，说明为什么得出这个结论
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
# 三、安全护栏 & 闭环执行器 模型
# ============================================================
# 这些模型用于 guardrail.py（安全校验）和 executor_v2.py（闭环执行）
# ============================================================


class GuardrailViolation(BaseModel):
    """单条护栏违规记录

    guardrail.py 检查修复方案时，每发现一条违规就创建一个此对象。
    多条违规汇总成 GuardrailResult。
    """
    # rule_id：违规规则标识，如 "DANGEROUS_CMD_001"
    # 用于快速定位触发了哪条规则
    rule_id: str = Field(description="违规规则ID，如 DANGEROUS_CMD_001")
    # severity：严重程度
    # critical = 不允许执行，必须修改方案
    # warning = 提示风险，但可以继续
    severity: str = Field(description="严重程度: critical/warning")
    # step_id：违规发生的步骤编号，None 表示全局违规（不针对特定步骤）
    step_id: Optional[int] = Field(None, description="违规步骤编号")
    # message：人可读的违规描述，用于展示给审批人
    message: str = Field(description="违规描述（人可读）")
    # detail：违规详情，如匹配到的具体命令内容
    detail: str = Field("", description="违规详情（具体匹配到的内容）")


class GuardrailResult(BaseModel):
    """护栏检查结果

    guardrail.py 的返回类型，表示一次完整检查的结果。
    passed=True 表示方案安全，可以进入审批/执行环节。
    """
    # passed：是否通过所有检查
    # 只要有任意一条 critical 级别违规，passed 就为 False
    passed: bool = Field(description="是否通过检查")
    # violations：发现的违规列表，空列表表示完全通过
    violations: List[GuardrailViolation] = Field(default_factory=list, description="违规列表")
    # checked_at：检查时间戳（ISO 8601 格式），用于审计
    checked_at: str = Field("", description="检查时间戳")


class CommandExecutionResult(BaseModel):
    """单步命令执行结果（Mock 或真实）

    executor_v2.py 中 CommandRunner.run() 的返回类型。
    无论是 Mock 执行还是真实子进程执行，都返回此统一结构。
    """
    # step_id：步骤编号，对应 FixStepOutput.step_id
    step_id: int = Field(description="步骤编号")
    # command：实际执行的命令字符串
    command: str = Field(description="执行的命令")
    # exit_code：进程退出码，0 表示成功，非 0 表示失败
    # 具体非零值含义取决于命令（如 curl 返回 HTTP 状态码）
    exit_code: int = Field(description="退出码: 0=成功, 非0=失败")
    # stdout：标准输出内容，通常包含命令的成功结果
    stdout: str = Field("", description="标准输出")
    # stderr：标准错误内容，通常包含错误信息
    stderr: str = Field("", description="标准错误")
    # success：是否成功（exit_code == 0）
    # 这是一个便捷字段，避免调用方自己判断 exit_code
    success: bool = Field(description="是否成功")
    # execution_time_ms：执行耗时（毫秒），用于性能分析
    execution_time_ms: int = Field(0, description="执行耗时(毫秒)")


class ErrorAnalysisOutput(BaseModel):
    """LLM 错误分析输出

    执行失败时，LLM 分析错误信息后决定下一步动作。
    这是闭环执行器的核心决策点——由真实错误驱动，不是 LLM 凭空决定。

    决策流程：
    1. 执行器运行命令失败
    2. 把 stdout/stderr/exit_code 传给 LLM
    3. LLM 返回 ErrorAnalysisOutput
    4. 执行器根据 action 字段决定：重试/调整/回滚/跳过
    """
    # action：决策动作
    # retry = 原封不动重试当前步骤（可能是临时故障）
    # adjust = 用 adjusted_command 修改后的命令重试
    # rollback = 执行回滚，终止修复
    # skip = 跳过当前步骤，继续执行下一步
    action: str = Field(description="决策动作: retry(重试当前步骤) / adjust(调整命令后重试) / rollback(执行回滚) / skip(跳过继续)")
    # adjusted_command：调整后的命令，action=adjust 时必填
    # 例如把 "systemctl start postgres" 改成 "systemctl restart postgres"
    adjusted_command: Optional[str] = Field(None, description="调整后的命令（action=adjust 时必填）")
    # reasoning：决策理由，用于日志和人工复核
    reasoning: str = Field(description="决策理由")
    # estimated_fix_probability：预估修复成功概率，0-1 之间
    # 低于某个阈值时可能直接触发回滚
    estimated_fix_probability: float = Field(0.0, description="预估修复成功概率 0-1")
