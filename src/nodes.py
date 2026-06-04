"""
工作流节点定义 - dispatch, aggregate, human_approval, guardrail, executor, other_handler

Agent 类（DBAgent/NetAgent/AppAgent/SupervisorAgent/FixAgent）在 agents/ 目录
本文件只保留非 Agent 类的工作流节点函数。
"""

import asyncio
from typing import Callable, Awaitable
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from state import SystemState, ApprovalStatus
from prompts import AGGREGATE_PROMPT, ERROR_ANALYSIS_PROMPT
from schemas import AggregateOutput
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from database import AsyncSessionLocal, save_ticket
from guardrail import run_guardrail
from action_dsl import ActionDSLValidationError, compile_rollback_action, compile_step_action
from case_library import (
    DEFAULT_CASE_LIBRARY_PATH,
    format_case_context,
    retrieve_similar_cases,
    upsert_case_from_state,
)
from executor_v2 import ClosedLoopExecutor, MockCommandRunner, SafeDockerCommandRunner
from replanner import make_replanner_decision
# 引入标准化 Trace 事件工厂和状态转换工具
from trace_events import make_trace_event, status_from_success
from logger import logger
from config import settings


VERIFY_PROBES = [
    {"name": "health", "url": "http://localhost:18080/health"},
    {"name": "cache_ping", "url": "http://localhost:18080/cache/ping"},
    {"name": "orders_pending", "url": "http://localhost:18080/orders/pending"},
]


def create_case_memory_node(limit: int = 3):
    """
    创建案例记忆检索节点。

    新工单进入 Supervisor 前，先用症状检索历史相似案例，并把压缩后的
    case_context 写入 state，供 Supervisor 和 FixAgent 复用。
    """

    async def case_memory_node(state: SystemState) -> dict:
        cases = retrieve_similar_cases(state.symptom, limit=limit)
        case_context = format_case_context(cases)

        logger.info(
            f"[CaseMemory] 检索完成: ticket_id={state.ticket_id}, "
            f"similar_cases={len(cases)}"
        )

        audit_log = {
            "ticket_id": state.ticket_id,
            "agent_name": "case_memory",
            "action_type": "case_retrieval",
            "action_detail": {
                "case_count": len(cases),
                "case_ids": [case.get("case_id") for case in cases],
                "library_path": str(DEFAULT_CASE_LIBRARY_PATH),
            },
            "input_context": {
                "symptom": state.symptom,
            },
            "output_result": {
                "similar_cases": cases,
            },
            "dispatch_round": state.dispatch_round,
        }
        # 生成标准化 Trace 事件：案例检索结果作为 observation_received 记录
        trace_event = make_trace_event(
            "observation_received",
            ticket_id=state.ticket_id,
            agent_name="case_memory",
            input_data={"symptom": state.symptom},
            output_data={"similar_cases": cases},
            metadata={
                "case_count": len(cases),
                "case_ids": [case.get("case_id") for case in cases],
                "dispatch_round": state.dispatch_round,
            },
        )

        return {
            "similar_cases": cases,
            "case_context": case_context,
            "case_memory": {
                "library_path": str(DEFAULT_CASE_LIBRARY_PATH),
                "similar_case_count": len(cases),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
            "messages": [f"CaseMemory: 检索到 {len(cases)} 个相似历史案例"],
            "audit_logs": [audit_log],
            "trace_events": [trace_event],
        }

    return case_memory_node


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
                    # 收集各 Agent 返回的标准化 Trace 事件，通过 operator.add 自动累加到 State
                    elif key == "trace_events":
                        merged.setdefault("trace_events", []).extend(value)
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
        # 收集各 Agent 的诊断结果
        agent_results = {}
        if state.db_agent_result:
            agent_results["db_agent"] = state.db_agent_result
        if state.net_agent_result:
            agent_results["net_agent"] = state.net_agent_result
        if state.app_agent_result:
            agent_results["app_agent"] = state.app_agent_result

        # 无任何诊断结果时，生成 skipped 状态的标准化 Trace 事件
        if not agent_results:
            logger.info("[Aggregate] 无 Agent 诊断结果，跳过聚合")
            return {
                "aggregated_diagnosis": None,
                "trace_events": [make_trace_event(
                    "diagnosis_generated",
                    ticket_id=state.ticket_id,
                    agent_name="aggregate",
                    status="skipped",
                    input_data={"agent_result_count": 0},
                    output_data={"aggregated_diagnosis": None},
                    metadata={"dispatch_round": state.dispatch_round},
                )],
                "messages": ["Aggregate: 无诊断结果可聚合"],
            }

        # 只有一个 Agent 返回结果时，直接采用，无需 LLM 聚合
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
                "trace_events": [make_trace_event(
                    "diagnosis_generated",
                    ticket_id=state.ticket_id,
                    agent_name="aggregate",
                    input_data={"contributing_agents": [agent_name]},
                    output_data=aggregated,
                    metadata={"dispatch_round": state.dispatch_round},
                )],
                "messages": [f"Aggregate: 采用 {agent_name} 的诊断结论"],
            }

        # 多个 Agent 返回结果时，使用 LLM 进行聚合推理
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
                # 聚合完成，生成 diagnosis_generated 标准化 Trace 事件
                "trace_events": [make_trace_event(
                    "diagnosis_generated",
                    ticket_id=state.ticket_id,
                    agent_name="aggregate",
                    input_data={"contributing_agents": list(agent_results.keys())},
                    output_data=aggregated,
                    metadata={"dispatch_round": state.dispatch_round},
                )],
                "messages": [
                    f"Aggregate: 综合诊断={aggregated.get('diagnosis')}, "
                    f"置信度={aggregated.get('confidence')}"
                ],
                "audit_logs": [audit_log],
            }
        except Exception as e:
            # 聚合推理异常时也要生成 failure 状态的标准化 Trace 事件
            logger.exception(f"[Aggregate] 聚合推理失败: {e}")
            return {
                "aggregated_diagnosis": {
                    "diagnosis": "聚合推理异常",
                    "possible_causes": [],
                    "confidence": 0.0,
                    "contributing_agents": list(agent_results.keys()),
                    "reasoning": f"异常: {str(e)}",
                },
                "trace_events": [make_trace_event(
                    "diagnosis_generated",
                    ticket_id=state.ticket_id,
                    agent_name="aggregate",
                    status="failure",
                    input_data={"contributing_agents": list(agent_results.keys())},
                    error=str(e),
                    metadata={"dispatch_round": state.dispatch_round},
                )],
                "messages": [f"Aggregate: 聚合推理失败 - {str(e)}"],
            }

    return aggregate_node


def create_human_approval_node():
    async def human_approval_node(state: SystemState) -> dict:
        try:
            logger.info(f"审批节点: 请求审批工单 {state.ticket_id}")
            # 从 fix_plan（可能是 dict 或 Pydantic 对象）中提取 plan_id
            plan_id = (
                state.fix_plan.get("plan_id")
                if isinstance(state.fix_plan, dict)
                else getattr(state.fix_plan, "plan_id", None)
            )

            # LangGraph interrupt：暂停工作流，等待外部人工审批输入
            approval = interrupt({
                "type": "approval_required",
                "ticket_id": state.ticket_id,
                "fix_plan": state.fix_plan,
                "message": f"请审批修复方案: {plan_id}"
            })

            if approval.get("approved", False):
                # 审批通过，生成 success 状态的标准化 Trace 事件
                logger.info(f"审批节点: 工单 {state.ticket_id} 已审批通过, 备注: {approval.get('comments', '')}")
                return {
                    "approval_status": ApprovalStatus.APPROVED,
                    "approver_comments": approval.get("comments", ""),
                    "messages": [f"人工审批: 已批准 - {approval.get('comments', '')}"],
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
                return {
                    "approval_status": ApprovalStatus.REJECTED,
                    "approver_comments": approval.get("comments", ""),
                    "messages": [f"人工审批: 已拒绝 - {approval.get('comments', '')}"],
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


def _run_verification_probe(name: str, url: str, timeout: int = 5) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {
                "name": name,
                "url": url,
                "status_code": resp.status,
                "success": 200 <= resp.status < 300,
                "body": body[:1000],
                "error": "",
                "checked_at": started_at,
            }
    except HTTPError as exc:
        return {
            "name": name,
            "url": url,
            "status_code": exc.code,
            "success": False,
            "body": "",
            "error": f"HTTP {exc.code}: {exc.reason}",
            "checked_at": started_at,
        }
    except URLError as exc:
        return {
            "name": name,
            "url": url,
            "status_code": None,
            "success": False,
            "body": "",
            "error": str(exc.reason),
            "checked_at": started_at,
        }
    except Exception as exc:
        return {
            "name": name,
            "url": url,
            "status_code": None,
            "success": False,
            "body": "",
            "error": str(exc),
            "checked_at": started_at,
        }


def create_verify_node():
    """
    创建恢复验证节点。

    Verify 位于 Executor 之后、Save 之前，固定探测 SREBench Lite 的三个
    关键恢复接口，并把结果写入 verification_result，同时合并进
    execution_result，方便现有工单表持久化。
    """

    async def verify_node(state: SystemState) -> dict:
        logger.info(f"[Verify] 开始恢复验证: ticket_id={state.ticket_id}")

        # 并行探测三个恢复接口（health、cache_ping、orders_pending）
        tasks = [
            asyncio.to_thread(_run_verification_probe, probe["name"], probe["url"])
            for probe in VERIFY_PROBES
        ]
        probes = await asyncio.gather(*tasks)
        # 所有探测都成功才算验证通过
        verified = all(probe.get("success", False) for probe in probes)
        recovered_at = datetime.now(timezone.utc).isoformat() if verified else None

        verification_result = {
            "verified": verified,
            "verification_probe": probes,
            "recovered_at": recovered_at,
        }
        # 生成标准化 Trace 事件：verification_passed，状态由探测结果决定
        trace_event = make_trace_event(
            "verification_passed",
            ticket_id=state.ticket_id,
            agent_name="verify",
            status=status_from_success(verified),
            input_data={
                "execution_status": (state.execution_result or {}).get("overall_status"),
                "probe_urls": [probe["url"] for probe in VERIFY_PROBES],
            },
            output_data=verification_result,
            metadata={
                "probe_count": len(probes),
                "passed_count": sum(1 for probe in probes if probe.get("success")),
                "dispatch_round": state.dispatch_round,
            },
        )

        execution_result = {
            **(state.execution_result or {}),
            **verification_result,
        }

        audit_log = {
            "ticket_id": state.ticket_id,
            "agent_name": "verify",
            "action_type": "verify",
            "action_detail": {
                "verified": verified,
                "probe_count": len(probes),
                "recovered_at": recovered_at,
            },
            "input_context": {
                "execution_status": (state.execution_result or {}).get("overall_status"),
            },
            "output_result": verification_result,
            "dispatch_round": state.dispatch_round,
        }

        logger.info(
            f"[Verify] 恢复验证完成: verified={verified}, "
            f"passed={sum(1 for probe in probes if probe.get('success'))}/{len(probes)}"
        )

        return {
            "verification_result": verification_result,
            "execution_result": execution_result,
            "messages": [
                f"Verify: 恢复验证{'通过' if verified else '未通过'} "
                f"({sum(1 for probe in probes if probe.get('success'))}/{len(probes)})"
            ],
            "audit_logs": [audit_log],
            "trace_events": [trace_event],
        }

    return verify_node


def create_save_node():
    """创建统一归档节点。"""

    async def save_node(state: SystemState) -> dict:
        async with AsyncSessionLocal() as db:
            try:
                logger.info(f"[Save] 开始归档工单: ticket_id={state.ticket_id}")
                saved_case = None
                case_audit_log = None
                try:
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
                    logger.warning(f"[Save] 案例沉淀失败，不影响工单保存: {exc}")

                state_dict = {**state.__dict__}
                if case_audit_log:
                    state_dict["audit_logs"] = list(state.audit_logs) + [case_audit_log]

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
                await db.close()
                logger.debug("[Save] 数据库会话已关闭")

    return save_node


def create_replanner_node():
    """创建执行失败后的 Replanner/Critic 节点。

    Replanner 读取 Executor 的 stdout/stderr/trace，对失败进行分类并决策：
    - command_not_allowed: 命令或 Action DSL 不被允许
    - environment_not_ready: 环境或靶场临时未就绪
    - diagnosis_mismatch: 诊断结论或目标不匹配
    - tooling_gap: 缺少必要的诊断工具或上下文
    - permission_or_privilege: 权限不足或涉及高风险操作

    最终输出 decision: retry / re-diagnose / rollback / escalate / verify。
    """

    async def replanner_node(state: SystemState) -> dict:
        # 调用纯规则决策引擎，获取下一步该走哪条路
        decision = make_replanner_decision(
            execution_result=state.execution_result,
            execution_trace=state.execution_trace,
            replanner_round=state.replanner_round,
            max_replanner_rounds=state.max_replanner_rounds,
        )

        action = decision.get("decision", "escalate")
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

        # 构造需要更新到工作流状态的字典
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

        # 无修复方案时直接跳过护栏检查，生成 skipped 状态的标准化 Trace 事件
        if not fix_plan:
            logger.warning("[Guardrail] 无修复方案，跳过检查")
            return {
                "guardrail_result": {"passed": True, "violations": [], "checked_at": ""},
                "trace_events": [make_trace_event(
                    "policy_checked",
                    ticket_id=state.ticket_id,
                    agent_name="guardrail",
                    status="skipped",
                    output_data={"passed": True, "violations": [], "checked_at": ""},
                    metadata={"dispatch_round": state.dispatch_round},
                )],
                "messages": ["Guardrail: 无修复方案，跳过检查"],
            }

        # fix_plan 可能是 FixPlan 对象或 dict，统一转成字典处理
        plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump()

        logger.info(f"[Guardrail] 开始检查修复方案: plan_id={plan_dict.get('plan_id')}")

        # 执行确定性护栏检查（非 LLM 评分，而是代码规则硬检查）
        result = run_guardrail(plan_dict)

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
            # 护栏检查未通过，生成 failure 状态的标准化 Trace 事件，并统计 critical/warning 违规数
            violations = result_dict.get("violations", [])
            critical_violations = [v for v in violations if v.get("severity") == "critical"]
            warning_violations = [v for v in violations if v.get("severity") == "warning"]
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


def _build_action_trace_events(
    *,
    ticket_id: str,
    plan_dict: dict,
    execution_trace: list[dict],
    dispatch_round: int,
) -> list[dict]:
    """将执行器的 execution_trace 逐条拆分为标准化的 action_executed 事件。

    原来 execution_trace 是一个列表，retry/rollback/失败混在一起难以单步分析。
    现在每条执行记录都变成独立事件，trace_type 标明是 execute/retry/rollback，
    方便外部评测系统按 step_id 或 timestamp 排序后逐条分析。
    """
    events = []
    for item in execution_trace:
        # 从执行记录中提取成功标志，布尔值转标准状态字符串
        success = item.get("success")
        status = status_from_success(success if isinstance(success, bool) else None)
        # trace_type 标明动作类型：execute（正常执行）、retry（重试）、rollback（回滚）
        trace_type = item.get("trace_type", "execute")
        events.append(make_trace_event(
            "action_executed",
            ticket_id=ticket_id,
            agent_name="executor",
            status=status,
            input_data={
                "plan_id": plan_dict.get("plan_id"),
                "step_id": item.get("step_id"),
                "command": item.get("command"),
                "trace_type": trace_type,
            },
            output_data=item,
            error=item.get("stderr") if status == "failure" else None,
            metadata={
                "plan_id": plan_dict.get("plan_id"),
                "trace_type": trace_type,
                "attempt": item.get("attempt"),
                "dispatch_round": dispatch_round,
            },
            timestamp=item.get("timestamp"),
        ))
    return events


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
        fix_plan = state.fix_plan
        plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump() if fix_plan else {}

        try:
            # 无修复方案或步骤为空时，直接返回 skipped 状态的事件，避免空跑
            if not plan_dict or not plan_dict.get("steps"):
                logger.warning("[Executor] 无修复方案或步骤为空")
                execution_result = {
                    "plan_id": plan_dict.get("plan_id"),
                    "executed_steps": [],
                    "overall_status": "skipped",
                    "summary": "无修复方案或步骤为空",
                }
                return {
                    "execution_result": execution_result,
                    "messages": ["Executor: 无修复方案可执行"],
                    # 生成标准化 Trace 事件：无方案可执行，标记为 skipped
                    "trace_events": [make_trace_event(
                        "action_executed",
                        ticket_id=state.ticket_id,
                        agent_name="executor",
                        status="skipped",
                        input_data={"plan_id": plan_dict.get("plan_id")},
                        output_data=execution_result,
                        metadata={"dispatch_round": state.dispatch_round},
                    )],
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

            logger.info(f"[Executor] 执行完成: plan_id={plan_dict.get('plan_id')}")

            return {
                "execution_result": execution_result,
                "execution_trace": execution_trace,
                # 将闭环执行器的 trace 逐条拆分为标准化 action_executed 事件
                "trace_events": _build_action_trace_events(
                    ticket_id=state.ticket_id,
                    plan_dict=plan_dict,
                    execution_trace=execution_trace,
                    dispatch_round=state.dispatch_round,
                ),
                "messages": [
                    f"Executor: 执行{execution_result.get('overall_status', 'unknown')} - "
                    f"{execution_result.get('completed_steps', 0)}/{execution_result.get('total_steps', 0)} 步骤完成 "
                    f"(mode={executor_mode})"
                ],
                "audit_logs": [audit_log],
            }
        except Exception as e:
            # 执行器异常时也要生成标准化 Trace 事件，保证失败路径可追溯
            logger.exception(f"[Executor] 执行失败: {e}")
            execution_result = {
                "plan_id": plan_dict.get("plan_id") if plan_dict else None,
                "executed_steps": [],
                "overall_status": "failed",
                "summary": f"执行异常: {str(e)}",
            }
            return {
                "execution_result": execution_result,
                "messages": [f"Executor: 执行失败 - {str(e)}"],
                "trace_events": [make_trace_event(
                    "action_executed",
                    ticket_id=state.ticket_id,
                    agent_name="executor",
                    status="failure",
                    input_data={"plan_id": plan_dict.get("plan_id") if plan_dict else None},
                    output_data=execution_result,
                    error=str(e),
                    metadata={"dispatch_round": state.dispatch_round},
                )],
            }

    return executor_node
