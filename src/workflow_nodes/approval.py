"""
人工审批与其他处理节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

async def _save_pending_approval_snapshot(state: SystemState, plan_id: str | None) -> None:
    """保存进入人工审批前的待审批快照。

    参数说明：
    - state: 当前工作流状态，包含诊断、修复方案、审计日志和 Trace
    - plan_id: 当前修复方案 ID，用于审批审计日志展示

    返回值说明：
    - 无

    异常说明：
    - 数据库保存失败时向上抛出异常，避免接口返回一个前端查不到的待审批工单
    """
    async with AsyncSessionLocal() as db:
        # pending_audit_log：前端 Agent 流程里展示的人工审批请求节点
        pending_audit_log = {
            "ticket_id": state.ticket_id,
            "agent_name": "human_approval",
            "action_type": "approval_requested",
            "action_detail": {
                "plan_id": plan_id,
                "approval_status": ApprovalStatus.PENDING,
            },
            "input_context": {
                "ticket_id": state.ticket_id,
                "fix_plan": state.fix_plan,
            },
            "output_result": {
                "status": "pending",
                "message": "等待人工审批",
            },
            "dispatch_round": state.dispatch_round,
        }
        # pending_state：保存到数据库的待审批状态快照
        pending_state = {
            **state.__dict__,
            "approval_status": ApprovalStatus.PENDING,
            "approver_comments": None,
            "audit_logs": list(state.audit_logs) + [pending_audit_log],
            "messages": list(state.messages) + ["人工审批: 等待审批"],
        }

        logger.info(f"审批节点: 保存待审批快照 {state.ticket_id}")
        await save_ticket(db, pending_state)


def create_human_approval_node():
    """
    创建人工审批节点工厂函数。

    使用 LangGraph 的 interrupt 功能暂停工作流，等待外部人工审批输入。
    这是工作流中的"断点"，支持：
    - 审批通过 → 进入执行节点
    - 审批拒绝 → 直接保存工单（不执行）
    - 工作流中断后恢复 → 从断点继续

    为什么需要这个节点：
    修复方案可能涉及高风险操作（如重启服务、修改配置），需要人工确认。
    interrupt 机制让工作流可以安全暂停，等待外部系统（如 Web UI）传入审批结果。

    返回：
        异步节点函数 human_approval_node(state) -> dict
    """
    async def human_approval_node(state: SystemState) -> dict:
        try:
            logger.info(f"审批节点: 请求审批工单 {state.ticket_id}")
            # plan_id：从 fix_plan（可能是 dict 或 Pydantic 对象）中提取方案 ID
            plan_id = (
                state.fix_plan.get("plan_id")
                if isinstance(state.fix_plan, dict)
                else getattr(state.fix_plan, "plan_id", None)
            )

            # 先保存待审批快照到数据库，这样前端可以查询到待审批工单
            await _save_pending_approval_snapshot(state, plan_id)

            # LangGraph interrupt：暂停工作流，等待外部人工审批输入
            # 工作流会在这里暂停，状态会被 checkpointer 保存
            # 外部系统调用 resume 并传入 approval 字典后，工作流从断点继续
            approval = interrupt({
                "type": "approval_required",        # 中断类型：需要审批
                "ticket_id": state.ticket_id,       # 工单 ID
                "fix_plan": state.fix_plan,         # 修复方案详情
                "message": f"请审批修复方案: {plan_id}"
            })

            if approval.get("approved", False):
                # 审批通过，生成 success 状态的标准化 Trace 事件
                logger.info(f"审批节点: 工单 {state.ticket_id} 已审批通过, 备注: {approval.get('comments', '')}")
                # audit_log：记录人工审批通过动作，供前端 Agent 流程展示
                audit_log = {
                    "ticket_id": state.ticket_id,
                    "agent_name": "human_approval",
                    "action_type": "approval_approved",
                    "action_detail": {
                        "plan_id": plan_id,
                        "comments": approval.get("comments", ""),
                    },
                    "input_context": {
                        "approval_payload": approval,
                    },
                    "output_result": {
                        "approval_status": ApprovalStatus.APPROVED,
                    },
                    "dispatch_round": state.dispatch_round,
                }
                return {
                    "approval_status": ApprovalStatus.APPROVED,
                    "approver_comments": approval.get("comments", ""),
                    "messages": [f"人工审批: 已批准 - {approval.get('comments', '')}"],
                    "audit_logs": [audit_log],
                    "trace_events": [make_trace_event(
                        "approval_received",
                        ticket_id=state.ticket_id,
                        agent_name="human_approval",
                        input_data={"plan_id": plan_id},
                        output_data={"approved": True, "comments": approval.get("comments", "")},
                        metadata={"dispatch_round": state.dispatch_round},
                    )],
                }
            else:
                # 审批拒绝，生成 failure 状态的标准化 Trace 事件
                logger.info(f"审批节点: 工单 {state.ticket_id} 已拒绝, 备注: {approval.get('comments', '')}")
                # audit_log：记录人工审批拒绝动作，供前端 Agent 流程展示
                audit_log = {
                    "ticket_id": state.ticket_id,
                    "agent_name": "human_approval",
                    "action_type": "approval_rejected",
                    "action_detail": {
                        "plan_id": plan_id,
                        "comments": approval.get("comments", ""),
                    },
                    "input_context": {
                        "approval_payload": approval,
                    },
                    "output_result": {
                        "approval_status": ApprovalStatus.REJECTED,
                    },
                    "dispatch_round": state.dispatch_round,
                }
                return {
                    "approval_status": ApprovalStatus.REJECTED,
                    "approver_comments": approval.get("comments", ""),
                    "messages": [f"人工审批: 已拒绝 - {approval.get('comments', '')}"],
                    "audit_logs": [audit_log],
                    "trace_events": [make_trace_event(
                        "approval_received",
                        ticket_id=state.ticket_id,
                        agent_name="human_approval",
                        status="failure",
                        input_data={"plan_id": plan_id},
                        output_data={"approved": False, "comments": approval.get("comments", "")},
                        metadata={"dispatch_round": state.dispatch_round},
                    )],
                }
        except GraphInterrupt:
            # LangGraph 中断异常需要原样抛出，不能吞掉
            raise
        except Exception as e:
            # 审批节点异常时也要生成标准化 Trace 事件，保证失败路径可追溯
            logger.exception(f"审批节点执行失败: {e}")
            return {
                "approval_status": ApprovalStatus.REJECTED,
                "approver_comments": f"审批异常: {str(e)}",
                "messages": [f"人工审批: 异常 - {str(e)}"],
                "trace_events": [make_trace_event(
                    "approval_received",
                    ticket_id=state.ticket_id,
                    agent_name="human_approval",
                    status="failure",
                    error=str(e),
                    metadata={"dispatch_round": state.dispatch_round},
                )],
            }
    return human_approval_node


def create_other_handler_node():
    """
    创建其他类型工单处理节点工厂函数。

    当 Supervisor 判断工单不属于技术故障（如咨询、投诉、需求等）时，
    工单会路由到这个节点，直接记录并归档，不经过诊断-修复流程。

    返回：
        异步节点函数 other_handler_node(state) -> dict
    """
    async def other_handler_node(state: SystemState) -> dict:
        async with AsyncSessionLocal() as db:
            try:
                logger.info(f"Other Handler: 工单 {state.ticket_id} 被分类为other类型，记录并归档")

                # result：本节点生成的结果字典
                result = {
                    "messages": [
                        f"Other Handler: 工单 {state.ticket_id} 被分类为other类型",
                        f"Other Handler: 症状: {state.symptom}",
                        f"Other Handler: 紧急程度: {state.urgency}",
                        f"Other Handler: 已记录并归档，无需进一步处理"
                    ]
                }

                # merged_state：合并当前状态和本节点结果，用于保存到数据库
                merged_state = {**state.__dict__, **result}
                merged_state["messages"] = state.messages + result["messages"]

                # 保存工单到数据库
                ticket = await save_ticket(db, merged_state)
                result["messages"].append(f"归档: 工单 {ticket.ticket_id} 已保存")

                logger.info(f"Other Handler: 工单 {ticket.ticket_id} 已保存")

                return result
            except Exception as e:
                logger.exception(f"Other Handler节点执行失败: {e}")
                return {
                    "messages": [f"Other Handler: 保存工单失败 - {str(e)}"]
                }
            finally:
                # 确保数据库会话被关闭，避免连接泄漏
                await db.close()
                logger.debug("Other Handler: 数据库会话已关闭")
    return other_handler_node
