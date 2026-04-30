"""
工作流节点定义 - dispatch, aggregate, human_approval, executor, other_handler

Agent 类（DBAgent/NetAgent/AppAgent/SupervisorAgent/FixAgent）在 agents/ 目录
本文件只保留非 Agent 类的工作流节点函数。
"""

import asyncio
from typing import Callable, Awaitable
from state import SystemState, ApprovalStatus
from prompts import AGGREGATE_PROMPT
from schemas import AggregateOutput
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from database import AsyncSessionLocal, save_ticket
from logger import logger


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

    Structured Output 改造说明：
    - 原方案：LLM 输出 JSON 字符串 -> parse_json_content 解析 -> dict
    - 新方案：llm.with_structured_output(AggregateOutput)
             直接返回 Pydantic 对象
    - 注意：aggregate 是函数式节点（非类），所以 structured_llm 在函数内部创建

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
            result = await (AGGREGATE_PROMPT | structured_llm).ainvoke({
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


def create_executor_node():
    async def executor_node(state: SystemState) -> dict:
        async with AsyncSessionLocal() as db:
            try:
                fix_plan = state.fix_plan
                steps = fix_plan.steps if fix_plan else []
                executed_steps = []

                logger.info(f"执行节点: 开始执行修复方案 {fix_plan.plan_id if fix_plan else '未知'}, 共 {len(steps)} 个步骤")

                for step in steps:
                    logger.debug(f"执行步骤 {step.step_id}: {step.action}")
                    executed_steps.append({
                        "step_id": step.step_id,
                        "action": step.action,
                        "command": step.command,
                        "status": "success",
                        "output": f"Mock执行成功: {step.command}"
                    })
                    logger.debug(f"步骤 {step.step_id} 执行完成")

                result = {
                    "execution_result": {
                        "plan_id": fix_plan.plan_id if fix_plan else None,
                        "executed_steps": executed_steps,
                        "overall_status": "success",
                        "summary": f"执行完成，共 {len(executed_steps)} 个步骤"
                    },
                    "messages": [f"执行节点: 完成修复方案执行 - {len(executed_steps)} 个步骤"]
                }

                merged_state = {**state.__dict__, **result}
                merged_state["messages"] = state.messages + result["messages"]

                ticket = await save_ticket(db, merged_state)
                result["messages"].append(f"归档: 工单 {ticket.ticket_id} 已保存")

                logger.info(f"执行节点: 修复方案执行完成，工单 {ticket.ticket_id} 已保存")

                return result
            except Exception as e:
                logger.exception(f"执行节点执行失败: {e}")
                return {
                    "execution_result": {
                        "plan_id": fix_plan.plan_id if fix_plan else None,
                        "executed_steps": [],
                        "overall_status": "failed",
                        "summary": f"执行失败: {str(e)}"
                    },
                    "messages": [f"执行节点: 执行失败 - {str(e)}"]
                }
            finally:
                await db.close()
                logger.debug("执行节点: 数据库会话已关闭")
    return executor_node
