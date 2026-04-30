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
    """修复步骤

    Structured Output 改造说明：
    - 新增 expected_output、on_failure、rollback_command 三个 Optional 字段
    - 原因：FixPlanOutput（LLM 结构化输出模型）中的步骤包含这些字段
    - 设计为 Optional 以保持向后兼容：已有数据不会报错，新数据可以完整存储
    """
    step_id: int = Field(..., description="步骤编号")
    action: str = Field(..., description="修复动作描述")
    command: Optional[str] = Field(None, description="执行的命令")
    risk_level: str = Field("low", description="风险等级: low/medium/high")
    expected_output: Optional[str] = Field(None, description="预期输出")
    on_failure: Optional[str] = Field(None, description="失败时的处理方式")
    rollback_command: Optional[str] = Field(None, description="回滚命令")


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
    """
    LangGraph工作流状态模型

    Multi-Agent 改造:
    - supervisor_decision: Supervisor 的派发决策（替代原 router 单路由）
    - agent_messages: Agent 间通信消息（支持协作诊断）
    - dispatched_agents: 当前被派发的 Agent 列表
    """

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

    # ========== 执行结果 ==========
    execution_result: Optional[Dict[str, Any]] = Field(None, description="执行结果")

    # ========== 审计日志（用于可追溯性） ==========
    audit_logs: Annotated[List[Dict[str, Any]], operator.add] = Field(
        default_factory=list,
        description="Agent 操作审计日志，用于追溯工单处理流程"
    )

    # ========== 辅助字段 ==========
    messages: Annotated[List[str], operator.add] = Field(default_factory=list, description="处理过程中的消息记录")
