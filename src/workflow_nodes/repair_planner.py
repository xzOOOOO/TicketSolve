"""
修复规划节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def create_repair_planner_node():
    """
    创建修复规划节点工厂函数。

    RepairPlanner 位于 FixAgent 和 Guardrail 之间，负责把 FixAgent 生成的
    平铺 Action DSL 规范化为可审计的修复计划：
    - action_type + target 合法时，编译出 canonical command 供审批展示
    - rollback_action_type + rollback_target 合法时，编译出 rollback_command
    - 非法 DSL 不在这里吞掉，保留给 Guardrail 拦截并给出违规明细

    为什么需要这个节点：
    FixAgent 生成的是"意图"（如"重启 nginx"），RepairPlanner 把它翻译成
    具体的、可执行的命令字符串（如"docker restart nginx"），并生成回滚命令。

    返回：
        异步节点函数 repair_planner_node(state) -> dict
    """

    async def repair_planner_node(state: SystemState) -> dict:
        fix_plan = state.fix_plan

        # 无修复方案时直接跳过，生成 skipped Trace 事件
        if not fix_plan:
            logger.warning("[RepairPlanner] 无修复方案，跳过规划")
            return {
                "messages": ["RepairPlanner: 无修复方案，跳过"],
                "trace_events": [make_trace_event(
                    "plan_generated",
                    ticket_id=state.ticket_id,
                    agent_name="repair_planner",
                    status="skipped",
                    output_data={"fix_plan": None},
                    metadata={"dispatch_round": state.dispatch_round},
                )],
            }

        # plan_dict：统一转为字典格式（FixPlan Pydantic 对象或 dict）
        plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump()
        plan_dict = {**plan_dict}  # 深拷贝，避免修改原始对象
        # raw_steps：FixAgent 生成的原始步骤列表
        raw_steps = plan_dict.get("steps", []) or []
        # planned_steps：编译后的步骤列表
        planned_steps = []
        # 统计计数器
        compiled_count = 0           # 成功编译的正向动作数
        rollback_compiled_count = 0  # 成功编译的回滚动作数
        invalid_count = 0            # 非法 DSL 数

        # 逐步骤编译 Action DSL
        for raw_step in raw_steps:
            step = raw_step if isinstance(raw_step, dict) else raw_step.model_dump()
            step = {**step}  # 深拷贝
            step_id = step.get("step_id", "?")

            # 编译正向动作（如 RESTART_SERVICE → "docker restart nginx"）
            try:
                compiled = compile_step_action(step)
                if compiled:
                    step["action_type"] = compiled.action_type
                    step["target"] = compiled.target
                    step["command"] = compiled.command
                    compiled_count += 1
            except ActionDSLValidationError as exc:
                invalid_count += 1
                logger.warning(f"[RepairPlanner] 步骤 {step_id} 动作 DSL 非法: {exc}")

            # 编译回滚动作（如 RESTART_SERVICE 的回滚也是 RESTART_SERVICE）
            try:
                rollback = compile_rollback_action(step)
                if rollback:
                    step["rollback_action_type"] = rollback.action_type
                    step["rollback_target"] = rollback.target
                    step["rollback_command"] = rollback.command
                    rollback_compiled_count += 1
            except ActionDSLValidationError as exc:
                invalid_count += 1
                logger.warning(f"[RepairPlanner] 步骤 {step_id} 回滚 DSL 非法: {exc}")

            # 确保 parameters 和 rollback_parameters 字段存在（空字典兜底）
            step.setdefault("parameters", {})
            step.setdefault("rollback_parameters", {})
            planned_steps.append(step)

        plan_dict["steps"] = planned_steps

        # audit_log：审计日志，记录规划过程统计
        audit_log = {
            "ticket_id": state.ticket_id,
            "agent_name": "repair_planner",
            "action_type": "repair_plan",
            "action_detail": {
                "plan_id": plan_dict.get("plan_id"),
                "steps_count": len(planned_steps),
                "compiled_actions": compiled_count,
                "compiled_rollbacks": rollback_compiled_count,
                "invalid_action_specs": invalid_count,
            },
            "input_context": {
                "source": "fix_agent",
                "plan_id": plan_dict.get("plan_id"),
            },
            "output_result": plan_dict,
            "dispatch_round": state.dispatch_round,
        }

        logger.info(
            f"[RepairPlanner] 规划完成: plan_id={plan_dict.get('plan_id')}, "
            f"steps={len(planned_steps)}, compiled={compiled_count}, "
            f"rollback_compiled={rollback_compiled_count}, invalid={invalid_count}"
        )

        return {
            "fix_plan": plan_dict,
            "messages": [
                f"RepairPlanner: 规划完成 - {compiled_count}/{len(planned_steps)} 个动作已编译"
            ],
            "audit_logs": [audit_log],
            "trace_events": [make_trace_event(
                "plan_generated",
                ticket_id=state.ticket_id,
                agent_name="repair_planner",
                input_data={"source": "fix_agent", "plan_id": plan_dict.get("plan_id")},
                output_data=plan_dict,
                metadata={
                    "steps_count": len(planned_steps),
                    "compiled_actions": compiled_count,
                    "compiled_rollbacks": rollback_compiled_count,
                    "invalid_action_specs": invalid_count,
                    "dispatch_round": state.dispatch_round,
                },
            )],
        }

    return repair_planner_node
