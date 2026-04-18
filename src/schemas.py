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