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
    """修复步骤"""
    step_id: int = Field(..., description="步骤编号")
    action: str = Field(..., description="修复动作描述")
    command: Optional[str] = Field(None, description="执行的命令")
    risk_level: str = Field("low", description="风险等级: low/medium/high")


class FixPlan(BaseModel):
    """修复方案"""
    plan_id: str = Field(..., description="方案ID")
    description: str = Field(..., description="方案描述")
    steps: List[FixStep] = Field(default_factory=list, description="修复步骤列表")
    estimated_time: str = Field("", description="预计执行时间")


class SystemState(BaseModel):
    """
    LangGraph工作流状态模型

    这个类定义了工单处理全流程的状态，包括：
    - 输入：工单症状
    - 路由决策：诊断类型
    - Agent诊断结果：DB/Net/App三个Agent的输出
    - 聚合诊断：综合分析结果
    - 修复方案：Fix Agent生成的方案
    - 人工审批：审批状态和意见
    - 执行结果：最终执行反馈
    """

    # ========== 输入信息 ==========
    ticket_id: str = Field(..., description="工单ID")
    symptom: str = Field(..., description="故障现象描述")

    # ========== 路由决策 ==========
    diagnosis_type: Optional[DiagnosisType] = Field(None, description="诊断类型: app/db/net/other")
    urgency: Optional[Urgency] = Field(None, description="紧急程度: low/medium/high/critical")


    # ========== Agent诊断结果 ==========
    db_agent_result: Optional[Dict[str, Any]] = Field(None, description="数据库Agent诊断结果")
    net_agent_result: Optional[Dict[str, Any]] = Field(None, description="网络Agent诊断结果")
    app_agent_result: Optional[Dict[str, Any]] = Field(None, description="应用Agent诊断结果")

    # ========== 聚合诊断 ==========
    aggregated_diagnosis: Optional[Dict[str, Any]] = Field(None, description="综合诊断结果")

    # ========== 修复方案 ==========
    fix_plan: Optional[FixPlan] = Field(None, description="Fix Agent生成的修复方案")

    # ========== 人工审批 ==========
    approval_status: ApprovalStatus = Field(ApprovalStatus.PENDING, description="审批状态")
    approver_comments: Optional[str] = Field(None, description="审批意见")

    # ========== 执行结果 ==========
    execution_result: Optional[Dict[str, Any]] = Field(None, description="执行结果")

    # ========== 辅助字段 ==========
    messages: Annotated[List[str], operator.add] = Field(default_factory=list, description="处理过程中的消息记录")
