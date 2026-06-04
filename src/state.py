"""
工单状态模型 - 定义整个工作流的状态结构
"""
from typing import Optional, List, Dict, Any,Annotated
from pydantic import BaseModel, Field
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
    """Agent 间通信消息"""
    sender: str = Field(..., description="发送者 Agent 名称")
    receiver: str = Field("broadcast", description="接收者，broadcast 表示广播")
    content: str = Field(..., description="消息内容")
    msg_type: str = Field("info", description="消息类型: diagnosis/question/request_help/disagreement")
    confidence: float = Field(0.0, description="置信度 0-1")
    evidence: List[str] = Field(default_factory=list, description="支撑证据")


class SystemState(BaseModel):
    """LangGraph工作流状态模型"""

    # ========== 输入信息 ==========
    ticket_id: str = Field(..., description="工单ID")
    symptom: str = Field(..., description="故障现象描述")

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

    # ========== 审计日志（用于可追溯性） ==========
    audit_logs: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="Agent 操作审计日志，用于追溯工单处理流程"
    )

    # ========== 辅助字段 ==========
    messages: Annotated[List[str], operator.add] = Field(default_factory=list, description="处理过程中的消息记录")
