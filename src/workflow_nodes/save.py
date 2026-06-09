"""
统一归档节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def create_save_node():
    """
    创建统一归档节点工厂函数。

    工作流的最后一步，负责：
    1. 将当前工单状态保存到数据库
    2. 如果验证通过，将案例沉淀到案例库（供未来检索复用）

    案例沉淀条件：
    - verification_result.verified == True（服务已恢复）
    - 有完整的诊断和修复方案

    返回：
        异步节点函数 save_node(state) -> dict
    """

    async def save_node(state: SystemState) -> dict:
        async with AsyncSessionLocal() as db:
            try:
                logger.info(f"[Save] 开始归档工单: ticket_id={state.ticket_id}")
                # saved_case：沉淀到案例库的案例数据（如果验证通过）
                saved_case = None
                # case_audit_log：案例沉淀的审计日志
                case_audit_log = None
                try:
                    # 尝试将当前工单沉淀为案例（案例库会自动判断是否满足沉淀条件）
                    saved_case = upsert_case_from_state(state)
                    if saved_case:
                        logger.info(
                            f"[Save] 已沉淀案例: case_id={saved_case.get('case_id')}"
                        )
                        case_audit_log = {
                            "ticket_id": state.ticket_id,
                            "agent_name": "case_memory",
                            "action_type": "case_upsert",
                            "action_detail": {
                                "case_id": saved_case.get("case_id"),
                                "library_path": str(DEFAULT_CASE_LIBRARY_PATH),
                            },
                            "input_context": {
                                "verified": (state.verification_result or {}).get("verified"),
                            },
                            "output_result": saved_case,
                            "dispatch_round": state.dispatch_round,
                        }
                except Exception as exc:
                    # 案例沉淀失败不影响工单保存，记录警告即可
                    logger.warning(f"[Save] 案例沉淀失败，不影响工单保存: {exc}")

                # state_dict：将当前状态转为字典，用于保存到数据库
                state_dict = {**state.__dict__}
                if case_audit_log:
                    state_dict["audit_logs"] = list(state.audit_logs) + [case_audit_log]

                # 保存工单到数据库
                ticket = await save_ticket(db, state_dict)
                logger.info(f"[Save] 工单 {ticket.ticket_id} 已保存")
                messages = [f"归档: 工单 {ticket.ticket_id} 已保存"]
                result = {"messages": messages}
                if saved_case:
                    messages.append(f"CaseMemory: 已沉淀案例 {saved_case.get('case_id')}")
                    result["case_memory"] = {
                        **(state.case_memory or {}),
                        "last_saved_case_id": saved_case.get("case_id"),
                        "library_path": str(DEFAULT_CASE_LIBRARY_PATH),
                    }
                if case_audit_log:
                    result["audit_logs"] = [case_audit_log]
                return result
            except Exception as exc:
                logger.exception(f"[Save] 归档失败: {exc}")
                return {
                    "messages": [f"归档: 保存工单失败 - {str(exc)}"],
                }
            finally:
                # 确保数据库会话被关闭
                await db.close()
                logger.debug("[Save] 数据库会话已关闭")

    return save_node
