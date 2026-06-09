"""
安全护栏节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def create_guardrail_node():
    """
    创建安全护栏节点工厂函数。

    在 Fix Agent 生成修复方案后、人工审批前，执行确定性安全检查。
    这是"代码规则硬检查"，不是 LLM 评分，保证 100% 可复现。

    检查规则：
    1. 危险命令黑名单（DROP TABLE、rm -rf 等）
    2. 回滚完整性（高风险步骤必须有回滚命令）
    3. 步骤顺序合理性（先停服务再改配置）
    4. 命令注入检测

    输出是确定性的 pass/fail + 具体违规项。
    只要有 critical 级别违规，方案就被拦截，不会进入人工审批。

    返回：
        异步节点函数 guardrail_node(state) -> dict
    """
    async def guardrail_node(state: SystemState) -> dict:
        fix_plan = state.fix_plan

        # 无修复方案时直接跳过护栏检查，生成 skipped 状态的标准化 Trace 事件
        if not fix_plan:
            logger.warning("[Guardrail] 无修复方案，跳过检查")
            return {
                "guardrail_result": {"passed": True, "violations": [], "checked_at": ""},
                "trace_events": [make_trace_event(
                    "policy_checked",                   # 事件类型：策略检查
                    ticket_id=state.ticket_id,          # 工单 ID
                    agent_name="guardrail",             # 产生事件的节点
                    status="skipped",                   # 状态：跳过（无方案可检查）
                    output_data={"passed": True, "violations": [], "checked_at": ""},
                    metadata={"dispatch_round": state.dispatch_round},
                )],
                "messages": ["Guardrail: 无修复方案，跳过检查"],
            }

        # plan_dict：fix_plan 可能是 FixPlan 对象或 dict，统一转成字典处理
        plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump()

        logger.info(f"[Guardrail] 开始检查修复方案: plan_id={plan_dict.get('plan_id')}")

        # 执行确定性护栏检查（非 LLM 评分，而是代码规则硬检查）
        result = run_guardrail(plan_dict)

        # result_dict：将 Pydantic 结果转为字典，便于序列化和返回
        result_dict = result.model_dump()

        if result.passed:
            # 护栏检查通过，生成 success 状态的标准化 Trace 事件
            logger.info(f"[Guardrail] 检查通过，方案可进入审批")
            return {
                "guardrail_result": result_dict,
                "trace_events": [make_trace_event(
                    "policy_checked",
                    ticket_id=state.ticket_id,
                    agent_name="guardrail",
                    status="success",
                    input_data={"plan_id": plan_dict.get("plan_id")},
                    output_data=result_dict,
                    metadata={
                        "violation_count": len(result.violations),
                        "dispatch_round": state.dispatch_round,
                    },
                )],
                "messages": [
                    f"Guardrail: 检查通过 ({len(result.violations)} 条 warning)"
                ],
            }
        else:
            # 护栏检查未通过，生成 failure 状态的标准化 Trace 事件
            # violations：所有违规项列表
            violations = result_dict.get("violations", [])
            # critical_violations：critical 级别违规（会导致方案被拦截）
            critical_violations = [v for v in violations if v.get("severity") == "critical"]
            # warning_violations：warning 级别违规（不会拦截，但会记录）
            warning_violations = [v for v in violations if v.get("severity") == "warning"]
            # violation_summary：critical 违规的摘要文本，用于日志和消息
            violation_summary = "; ".join(v["message"] for v in critical_violations)
            logger.warning(f"[Guardrail] 检查未通过: {violation_summary}")

            return {
                "guardrail_result": result_dict,
                "trace_events": [make_trace_event(
                    "policy_checked",
                    ticket_id=state.ticket_id,
                    agent_name="guardrail",
                    status="failure",
                    input_data={"plan_id": plan_dict.get("plan_id")},
                    output_data=result_dict,
                    metadata={
                        "critical_violation_count": len(critical_violations),
                        "warning_violation_count": len(warning_violations),
                        "dispatch_round": state.dispatch_round,
                    },
                )],
                "messages": [
                    f"Guardrail: 检查未通过 - {len(critical_violations)} 条 critical, "
                    f"{len(warning_violations)} 条 warning。"
                    f"违规项: {violation_summary}"
                ],
            }

    return guardrail_node
