"""
重规划节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def create_replanner_node():
    """
    创建执行失败后的 Replanner/Critic 节点工厂函数。

    Replanner 读取 Executor 的 stdout/stderr/trace，对失败进行分类并决策：
    - command_not_allowed: 命令或 Action DSL 不被允许
    - environment_not_ready: 环境或靶场临时未就绪
    - diagnosis_mismatch: 诊断结论或目标不匹配
    - tooling_gap: 缺少必要的诊断工具或上下文
    - permission_or_privilege: 权限不足或涉及高风险操作

    最终输出 decision: retry / re-diagnose / rollback / escalate / verify。

    为什么需要这个节点：
    执行器可能失败（如命令不存在、环境未就绪），Replanner 分析失败原因，
    决定是重试、重新诊断、回滚、上报还是直接验证。

    返回：
        异步节点函数 replanner_node(state) -> dict
    """

    async def replanner_node(state: SystemState) -> dict:
        # decision：调用纯规则决策引擎，获取下一步该走哪条路
        # make_replanner_decision 是确定性规则引擎，不依赖 LLM
        decision = make_replanner_decision(
            execution_result=state.execution_result,
            execution_trace=state.execution_trace,
            replanner_round=state.replanner_round,
            max_replanner_rounds=state.max_replanner_rounds,
        )

        # action：决策动作（retry/re-diagnose/rollback/escalate/verify）
        action = decision.get("decision", "escalate")
        # failure_type：失败分类（command_not_allowed/environment_not_ready/...）
        failure_type = decision.get("failure_type", "unknown")
        logger.info(
            f"[Replanner] decision={action}, failure_type={failure_type}, "
            f"round={decision.get('replanner_round')}/{state.max_replanner_rounds}"
        )

        # 把 Replanner 的决策也写回 execution_result，方便后续节点查看
        execution_result = {
            **(state.execution_result or {}),
            "replanner_decision": decision,
        }

        # updates：构造需要更新到工作流状态的字典
        updates = {
            "replanner_result": decision,
            "replanner_round": decision.get("replanner_round", state.replanner_round),
            "execution_result": execution_result,
            "messages": [
                f"Replanner: {action} - {failure_type} - {decision.get('reason')}"
            ],
            "trace_events": [make_trace_event(
                "diagnosis_generated",
                ticket_id=state.ticket_id,
                agent_name="replanner",
                # verify 表示成功路径，其他都是失败路径
                status="success" if action == "verify" else "failure",
                input_data={
                    "execution_result": state.execution_result,
                    "trace_count": len(state.execution_trace),
                },
                output_data=decision,
                metadata={
                    "decision": action,
                    "failure_type": failure_type,
                    "round": decision.get("replanner_round"),
                    "max_rounds": state.max_replanner_rounds,
                    "dispatch_round": state.dispatch_round,
                },
            )],
            "audit_logs": [{
                "ticket_id": state.ticket_id,
                "agent_name": "replanner",
                "action_type": "replan",
                "action_detail": {
                    "decision": action,
                    "failure_type": failure_type,
                    "reason": decision.get("reason"),
                    "round": decision.get("replanner_round"),
                    "max_rounds": state.max_replanner_rounds,
                },
                "input_context": {
                    "execution_result": state.execution_result,
                    "trace_count": len(state.execution_trace),
                },
                "output_result": decision,
                "dispatch_round": state.dispatch_round,
            }],
        }

        # 如果决策是重新诊断，需要把工作流状态回退到诊断之前，清空上一轮的结果
        # 这样 Supervisor 会重新分析症状，可能派发不同的 Agent
        if action == "re-diagnose":
            updates.update({
                "dispatched_agents": ["db_agent", "net_agent", "app_agent"],
                "dispatch_round": 0,
                "db_agent_result": None,
                "net_agent_result": None,
                "app_agent_result": None,
                "aggregated_diagnosis": None,
                "fix_plan": None,
                "guardrail_result": None,
                "approval_status": ApprovalStatus.PENDING,
                "verification_result": None,
            })

        return updates

    return replanner_node
