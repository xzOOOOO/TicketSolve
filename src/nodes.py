"""
工作流节点定义 - dispatch, aggregate, human_approval, guardrail, executor, other_handler

Agent 类（DBAgent/NetAgent/AppAgent/SupervisorAgent/FixAgent）在 agents/ 目录
本文件只保留非 Agent 类的工作流节点函数。
"""

import asyncio
from typing import Callable, Awaitable
from state import SystemState, ApprovalStatus
from prompts import AGGREGATE_PROMPT, ERROR_ANALYSIS_PROMPT
from schemas import AggregateOutput
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from database import AsyncSessionLocal, save_ticket
from guardrail import run_guardrail
from action_dsl import ActionDSLValidationError, compile_rollback_action, compile_step_action
from executor_v2 import ClosedLoopExecutor, MockCommandRunner, SafeDockerCommandRunner
from logger import logger
from config import settings


def create_dispatch_node(agent_runners: dict[str, Callable[[SystemState], Awaitable[dict]]]):
    """
    创建并行派发节点

    根据 Supervisor 的 dispatched_agents 列表，并行调用被派发的 Agent。
    使用 asyncio.gather 实现并行执行，各 Agent 结果合并写入 state。

    动态调度增强:
    - 跳过本轮已有结果的 Agent（避免重复执行）
    - 递增 dispatch_round 计数器

    Args:
        agent_runners: Agent 名称 → run 方法的映射
            {"db_agent": db_agent.run, "net_agent": net_agent.run, ...}
    """
    _result_fields = {
        "db_agent": "db_agent_result",
        "net_agent": "net_agent_result",
        "app_agent": "app_agent_result",
    }

    async def dispatch_node(state: SystemState) -> dict:
        dispatched = state.dispatched_agents

        if not dispatched:
            logger.info("[Dispatch] 无 Agent 被派发，跳过诊断")
            return {"messages": ["Dispatch: 无需诊断Agent，直接处理"]}

        to_run = []
        for agent_name in dispatched:
            field = _result_fields.get(agent_name)
            already_done = field and getattr(state, field, None) is not None
            if already_done:
                logger.info(f"[Dispatch] {agent_name} 已有结果，跳过本轮执行")
            else:
                to_run.append(agent_name)

        if not to_run:
            logger.info("[Dispatch] 所有被派发 Agent 均已有结果，跳过")
            return {"messages": ["Dispatch: 所有Agent已完成，无需重复执行"]}

        logger.info(f"[Dispatch] 并行派发 Agent: {to_run} (轮次 {state.dispatch_round + 1})")

        tasks = []
        agent_names = []
        for agent_name in to_run:
            runner = agent_runners.get(agent_name)
            if runner:
                tasks.append(runner(state))
                agent_names.append(agent_name)
            else:
                logger.warning(f"[Dispatch] 未找到 Agent: {agent_name}")

        if not tasks:
            logger.warning("[Dispatch] 没有可执行的 Agent")
            return {"messages": ["Dispatch: 无可用Agent执行"]}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = {"messages": [], "dispatch_round": state.dispatch_round + 1}
        for agent_name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                logger.error(f"[Dispatch] Agent {agent_name} 执行异常: {result}")
                merged["messages"].append(f"Dispatch: {agent_name} 执行异常 - {str(result)}")
                continue

            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "messages":
                        merged["messages"].extend(value)
                    elif key == "agent_messages":
                        merged.setdefault("agent_messages", []).extend(value)
                    elif key == "audit_logs":
                        merged.setdefault("audit_logs", []).extend(value)
                    else:
                        merged[key] = value

        logger.info(f"[Dispatch] 并行执行完成，{len(agent_names)} 个 Agent 返回结果")
        return merged

    return dispatch_node


def create_dynamic_check_node():
    """
    创建动态检查节点

    扫描 agent_messages 中的 request_help 消息，提取需要追加派发的 Agent。
    如果存在未执行的请求且未超过最大轮次，则更新 dispatched_agents 进入下一轮 dispatch。
    否则进入 aggregate 节点。
    """
    async def dynamic_check_node(state: SystemState) -> dict:
        if state.dispatch_round >= state.max_dispatch_rounds:
            logger.info(
                f"[DynamicCheck] 已达最大轮次 {state.max_dispatch_rounds}，进入聚合"
            )
            return {"dispatched_agents": []}

        requested = set()
        for msg in state.agent_messages:
            if msg.get("msg_type") == "request_help":
                receiver = msg.get("receiver", "")
                if receiver in {"db_agent", "net_agent", "app_agent"}:
                    requested.add(receiver)

        _result_fields = {
            "db_agent": "db_agent_result",
            "net_agent": "net_agent_result",
            "app_agent": "app_agent_result",
        }

        new_dispatch = []
        for agent_name in requested:
            field = _result_fields.get(agent_name)
            already_done = field and getattr(state, field, None) is not None
            if not already_done:
                new_dispatch.append(agent_name)

        if new_dispatch:
            logger.info(
                f"[DynamicCheck] 发现协作请求，追加派发: {new_dispatch} "
                f"(轮次 {state.dispatch_round}/{state.max_dispatch_rounds})"
            )
            return {"dispatched_agents": new_dispatch}

        logger.info("[DynamicCheck] 无协作请求，进入聚合")
        return {"dispatched_agents": []}

    return dynamic_check_node


def create_aggregate_node(llm, communication_bus=None):
    """创建聚合推理节点

    综合多个 Agent 的诊断结果，给出最终诊断结论。
    - 只有一个 Agent 返回结果 → 直接采用
    - 多个 Agent 返回结果 → LLM 聚合推理，加权判断

    Args:
        llm: LLM 实例，用于聚合推理
        communication_bus: CommunicationBus 实例（可选），用于读取 Agent 间通信消息
    """
    async def aggregate_node(state: SystemState) -> dict:
        agent_results = {}

        if state.db_agent_result:
            agent_results["db_agent"] = state.db_agent_result
        if state.net_agent_result:
            agent_results["net_agent"] = state.net_agent_result
        if state.app_agent_result:
            agent_results["app_agent"] = state.app_agent_result

        if not agent_results:
            logger.info("[Aggregate] 无 Agent 诊断结果，跳过聚合")
            return {
                "aggregated_diagnosis": None,
                "messages": ["Aggregate: 无诊断结果可聚合"],
            }

        if len(agent_results) == 1:
            agent_name = list(agent_results.keys())[0]
            single_result = agent_results[agent_name]
            logger.info(f"[Aggregate] 只有 {agent_name} 返回结果，直接采用")

            aggregated = {
                "diagnosis": single_result.get("diagnosis", "未知"),
                "possible_causes": single_result.get("possible_causes", []),
                "confidence": 0.7,
                "contributing_agents": [agent_name],
                "reasoning": f"仅 {agent_name} 返回诊断结果，直接采用",
            }
            return {
                "aggregated_diagnosis": aggregated,
                "messages": [f"Aggregate: 采用 {agent_name} 的诊断结论"],
            }

        logger.info(f"[Aggregate] 聚合 {len(agent_results)} 个 Agent 的诊断结果: {list(agent_results.keys())}")

        try:
            results_str = ""
            for name, result in agent_results.items():
                results_str += f"\n--- {name} ---\n"
                results_str += f"诊断: {result.get('diagnosis', '未知')}\n"
                results_str += f"可能原因: {result.get('possible_causes', [])}\n"

            if communication_bus and state.agent_messages:
                relevant_msgs = communication_bus.receive("aggregate", state.agent_messages)
                if relevant_msgs:
                    results_str += "\n--- Agent 间通信 ---\n"
                    for msg in relevant_msgs:
                        results_str += f"[{msg['sender']}→{msg['receiver']}] ({msg['msg_type']}, 置信度:{msg.get('confidence', 0)}) {msg['content']}\n"

            # 使用 Structured Output 进行聚合推理
            # 在函数内部创建 structured_llm（因为 aggregate 是函数式节点，无 __init__）
            structured_llm = llm.with_structured_output(AggregateOutput)
            result = await (AGGREGATE_PROMPT | structured_llm).with_retry(**settings.get_retry_config()).ainvoke({
                "symptom": state.symptom,
                "agent_results": results_str,
            })

            # 兜底处理
            if result is None:
                aggregated = {
                    "diagnosis": "聚合分析失败",
                    "possible_causes": [],
                    "confidence": 0.0,
                    "contributing_agents": list(agent_results.keys()),
                    "reasoning": "Structured Output 解析失败",
                }
            else:
                # Pydantic 对象转 dict，保持与 SystemState 的兼容性
                aggregated = result.model_dump()

            logger.info(
                f"[Aggregate] 聚合完成: diagnosis={aggregated.get('diagnosis')}, "
                f"confidence={aggregated.get('confidence')}"
            )

            # 记录审计日志：聚合推理
            audit_log = {
                "ticket_id": state.ticket_id,
                "agent_name": "aggregate",
                "action_type": "aggregate",
                "action_detail": {
                    "contributing_agents": aggregated.get("contributing_agents", []),
                    "diagnosis": aggregated.get("diagnosis"),
                    "confidence": aggregated.get("confidence"),
                    "reasoning": aggregated.get("reasoning"),
                },
                "input_context": {
                    "agent_results": results_str,
                    "symptom": state.symptom,
                },
                "output_result": aggregated,
                "dispatch_round": state.dispatch_round,
            }

            return {
                "aggregated_diagnosis": aggregated,
                "messages": [
                    f"Aggregate: 综合诊断={aggregated.get('diagnosis')}, "
                    f"置信度={aggregated.get('confidence')}"
                ],
                "audit_logs": [audit_log],
            }
        except Exception as e:
            logger.exception(f"[Aggregate] 聚合推理失败: {e}")
            return {
                "aggregated_diagnosis": {
                    "diagnosis": "聚合推理异常",
                    "possible_causes": [],
                    "confidence": 0.0,
                    "contributing_agents": list(agent_results.keys()),
                    "reasoning": f"异常: {str(e)}",
                },
                "messages": [f"Aggregate: 聚合推理失败 - {str(e)}"],
            }

    return aggregate_node


def create_human_approval_node():
    async def human_approval_node(state: SystemState) -> dict:
        try:
            logger.info(f"审批节点: 请求审批工单 {state.ticket_id}")

            approval = interrupt({
                "type": "approval_required",
                "ticket_id": state.ticket_id,
                "fix_plan": state.fix_plan,
                "message": f"请审批修复方案: {state.fix_plan.plan_id}"
            })

            if approval.get("approved", False):
                logger.info(f"审批节点: 工单 {state.ticket_id} 已审批通过, 备注: {approval.get('comments', '')}")
                return {
                    "approval_status": ApprovalStatus.APPROVED,
                    "approver_comments": approval.get("comments", ""),
                    "messages": [f"人工审批: 已批准 - {approval.get('comments', '')}"]
                }
            else:
                logger.info(f"审批节点: 工单 {state.ticket_id} 已拒绝, 备注: {approval.get('comments', '')}")
                return {
                    "approval_status": ApprovalStatus.REJECTED,
                    "approver_comments": approval.get("comments", ""),
                    "messages": [f"人工审批: 已拒绝 - {approval.get('comments', '')}"]
                }
        except GraphInterrupt:
            raise
        except Exception as e:
            logger.exception(f"审批节点执行失败: {e}")
            return {
                "approval_status": ApprovalStatus.REJECTED,
                "approver_comments": f"审批异常: {str(e)}",
                "messages": [f"人工审批: 异常 - {str(e)}"]
            }
    return human_approval_node


def create_other_handler_node():
    async def other_handler_node(state: SystemState) -> dict:
        async with AsyncSessionLocal() as db:
            try:
                logger.info(f"Other Handler: 工单 {state.ticket_id} 被分类为other类型，记录并归档")

                result = {
                    "messages": [
                        f"Other Handler: 工单 {state.ticket_id} 被分类为other类型",
                        f"Other Handler: 症状: {state.symptom}",
                        f"Other Handler: 紧急程度: {state.urgency}",
                        f"Other Handler: 已记录并归档，无需进一步处理"
                    ]
                }

                merged_state = {**state.__dict__, **result}
                merged_state["messages"] = state.messages + result["messages"]

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
                await db.close()
                logger.debug("Other Handler: 数据库会话已关闭")
    return other_handler_node


def create_repair_planner_node():
    """
    创建修复规划节点。

    RepairPlanner 位于 FixAgent 和 Guardrail 之间，负责把 FixAgent 生成的
    平铺 Action DSL 规范化为可审计的修复计划：
    - action_type + target 合法时，编译出 canonical command 供审批展示
    - rollback_action_type + rollback_target 合法时，编译出 rollback_command
    - 非法 DSL 不在这里吞掉，保留给 Guardrail 拦截并给出违规明细
    """

    async def repair_planner_node(state: SystemState) -> dict:
        fix_plan = state.fix_plan

        if not fix_plan:
            logger.warning("[RepairPlanner] 无修复方案，跳过规划")
            return {"messages": ["RepairPlanner: 无修复方案，跳过"]}

        plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump()
        plan_dict = {**plan_dict}
        raw_steps = plan_dict.get("steps", []) or []
        planned_steps = []
        compiled_count = 0
        rollback_compiled_count = 0
        invalid_count = 0

        for raw_step in raw_steps:
            step = raw_step if isinstance(raw_step, dict) else raw_step.model_dump()
            step = {**step}
            step_id = step.get("step_id", "?")

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

            step.setdefault("parameters", {})
            step.setdefault("rollback_parameters", {})
            planned_steps.append(step)

        plan_dict["steps"] = planned_steps

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
        }

    return repair_planner_node


def create_guardrail_node():
    """
    创建安全护栏节点

    在 Fix Agent 生成修复方案后、人工审批前，执行确定性安全检查。
    检查规则：
    1. 危险命令黑名单（DROP TABLE、rm -rf 等）
    2. 回滚完整性（高风险步骤必须有回滚命令）
    3. 步骤顺序合理性（先停服务再改配置）
    4. 命令注入检测

    输出是确定性的 pass/fail + 具体违规项，不是 LLM 猜的分数。
    只要有 critical 级别违规，方案就被拦截，不会进入人工审批。
    """
    async def guardrail_node(state: SystemState) -> dict:
        fix_plan = state.fix_plan

        if not fix_plan:
            logger.warning("[Guardrail] 无修复方案，跳过检查")
            return {
                "guardrail_result": {"passed": True, "violations": [], "checked_at": ""},
                "messages": ["Guardrail: 无修复方案，跳过检查"],
            }

        # fix_plan 可能是 FixPlan 对象或 dict
        plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump()

        logger.info(f"[Guardrail] 开始检查修复方案: plan_id={plan_dict.get('plan_id')}")

        # 执行确定性护栏检查
        result = run_guardrail(plan_dict)

        result_dict = result.model_dump()

        if result.passed:
            logger.info(f"[Guardrail] 检查通过，方案可进入审批")
            return {
                "guardrail_result": result_dict,
                "messages": [
                    f"Guardrail: 检查通过 ({len(result.violations)} 条 warning)"
                ],
            }
        else:
            violations = result_dict.get("violations", [])
            critical_violations = [v for v in violations if v.get("severity") == "critical"]
            warning_violations = [v for v in violations if v.get("severity") == "warning"]
            violation_summary = "; ".join(v["message"] for v in critical_violations)
            logger.warning(f"[Guardrail] 检查未通过: {violation_summary}")

            return {
                "guardrail_result": result_dict,
                "messages": [
                    f"Guardrail: 检查未通过 - {len(critical_violations)} 条 critical, "
                    f"{len(warning_violations)} 条 warning。"
                    f"违规项: {violation_summary}"
                ],
            }

    return guardrail_node


def create_executor_node(llm=None):
    """
    创建闭环执行器节点

    核心改造：不是一次性跑完所有步骤，而是每一步都观察真实结果，
    根据结果决策下一步（重试/调整/回滚）。

    执行流程：
        执行步骤1 → 观察真实输出 → [成功] → 执行步骤2
                                 → [失败] → LLM分析 → [可重试] → 重试/调整
                                                     → [不可重试] → 回滚 → 报告失败

    执行模式：
    - EXECUTOR_MODE=mock: 使用 MockCommandRunner 模拟命令执行
    - EXECUTOR_MODE=docker_lab: 使用 SafeDockerCommandRunner 执行靶场白名单命令
    """
    async def executor_node(state: SystemState) -> dict:
        async with AsyncSessionLocal() as db:
            try:
                fix_plan = state.fix_plan
                plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump() if fix_plan else {}

                if not plan_dict or not plan_dict.get("steps"):
                    logger.warning("[Executor] 无修复方案或步骤为空")
                    return {
                        "execution_result": {
                            "plan_id": plan_dict.get("plan_id"),
                            "executed_steps": [],
                            "overall_status": "skipped",
                            "summary": "无修复方案或步骤为空",
                        },
                        "messages": ["Executor: 无修复方案可执行"],
                    }

                logger.info(
                    f"[Executor] 开始闭环执行: plan_id={plan_dict.get('plan_id')}, "
                    f"共 {len(plan_dict.get('steps', []))} 个步骤"
                )

                # 创建闭环执行器。默认仍使用 mock，只有显式配置 docker_lab 才真实执行。
                executor_mode = settings.EXECUTOR_MODE.lower()
                if executor_mode == "docker_lab":
                    runner = SafeDockerCommandRunner()
                else:
                    runner = MockCommandRunner(failure_rate=0.15)

                logger.info(f"[Executor] 执行器模式: {executor_mode}")
                executor = ClosedLoopExecutor(
                    command_runner=runner,
                    llm=llm,
                    max_retries_per_step=2,
                )

                # 构建错误分析链（如果有 LLM）
                error_analyzer = None
                if llm:
                    from schemas import ErrorAnalysisOutput

                    structured_llm = llm.with_structured_output(ErrorAnalysisOutput)
                    error_analyzer = ERROR_ANALYSIS_PROMPT | structured_llm

                # 执行修复方案（闭环）
                exec_output = await executor.execute_plan(
                    fix_plan=plan_dict,
                    error_analyzer=error_analyzer,
                )

                execution_result = exec_output["execution_result"]
                execution_trace = exec_output["execution_trace"]

                # 记录审计日志
                audit_log = {
                    "ticket_id": state.ticket_id,
                    "agent_name": "executor",
                    "action_type": "execute",
                    "action_detail": {
                        "plan_id": plan_dict.get("plan_id"),
                        "overall_status": execution_result.get("overall_status"),
                        "completed_steps": execution_result.get("completed_steps"),
                        "total_steps": execution_result.get("total_steps"),
                        "trace_count": len(execution_trace),
                        "executor_mode": executor_mode,
                    },
                    "input_context": {
                        "plan_id": plan_dict.get("plan_id"),
                        "steps_count": len(plan_dict.get("steps", [])),
                    },
                    "output_result": execution_result,
                    "dispatch_round": state.dispatch_round,
                }

                result = {
                    "execution_result": execution_result,
                    "execution_trace": execution_trace,
                    "messages": [
                        f"Executor: 执行{execution_result.get('overall_status', 'unknown')} - "
                        f"{execution_result.get('completed_steps', 0)}/{execution_result.get('total_steps', 0)} 步骤完成 "
                        f"(mode={executor_mode})"
                    ],
                    "audit_logs": [audit_log],
                }

                # 保存工单
                merged_state = {**state.__dict__, **result}
                merged_state["messages"] = state.messages + result["messages"]
                ticket = await save_ticket(db, merged_state)
                result["messages"].append(f"归档: 工单 {ticket.ticket_id} 已保存")

                logger.info(f"[Executor] 执行完成，工单 {ticket.ticket_id} 已保存")

                return result
            except Exception as e:
                logger.exception(f"[Executor] 执行失败: {e}")
                return {
                    "execution_result": {
                        "plan_id": plan_dict.get("plan_id") if fix_plan else None,
                        "executed_steps": [],
                        "overall_status": "failed",
                        "summary": f"执行异常: {str(e)}",
                    },
                    "messages": [f"Executor: 执行失败 - {str(e)}"],
                }
            finally:
                await db.close()
                logger.debug("[Executor] 数据库会话已关闭")

    return executor_node
