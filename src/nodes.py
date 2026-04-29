"""
所有节点定义 - 整合 supervisor, dispatch, agents, aggregate, fix, approval, executor

Multi-Agent 改造说明:
- Agent 类（DBAgent/NetAgent/AppAgent/SupervisorAgent/FixAgent）在 agents/ 目录
- 本文件保留工作流节点函数（非 Agent 类的节点）
- 新增 dispatch_node: 根据 Supervisor 决策并行派发 Agent
- 新增 aggregate_node: 综合多个 Agent 诊断结果
- 原有节点（human_approval/executor/other_handler）保持不变

MCP集成说明:
- 工具由 workflow.py 在创建时通过 langchain-mcp-adapters 一次性获取
- Agent 节点通过参数接收工具列表，不管理 MCP 连接
- 节点内部只做: 绑定工具 → LLM决策 → 执行工具调用 → 生成诊断

技术栈:
- LangGraph: 工作流编排
- LangChain: LLM链式调用 + 工具绑定
- MCP (Model Context Protocol): 工具调用协议
- langchain-mcp-adapters: MCP工具自动适配
"""

import asyncio
import json
from typing import Callable, Awaitable
from state import SystemState, DiagnosisType, ApprovalStatus
from prompts import (
    ROUTER_PROMPT, DB_PROMPT, DB_DIAGNOSIS_PROMPT,
    NET_PROMPT, NET_DIAGNOSIS_PROMPT,
    APP_PROMPT, APP_DIAGNOSIS_PROMPT,
    FIX_PROMPT, AGGREGATE_PROMPT
)
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from database import AsyncSessionLocal, save_ticket
from utils import parse_json_content
from logger import logger


def parse_json_content(content: str) -> dict:
    """解析JSON内容，支持多种格式"""
    result = None
    try:
        result = json.loads(content)
        logger.debug("JSON解析成功")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON解析失败,尝试提取：{e}")
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON解析失败,提取失败：{e}")
    return result


async def _execute_tool_calls(response, tools):
    """执行LLM返回的工具调用，返回工具结果列表"""
    tool_results = []
    for tool_call in response.tool_calls:
        logger.debug(f"调用MCP工具: {tool_call['name']}, args={tool_call['args']}")
        for tool in tools:
            if tool.name == tool_call["name"]:
                result = await tool.ainvoke(tool_call["args"])
                tool_results.append({"tool": tool_call["name"], "result": result})
                logger.debug(f"MCP工具 {tool_call['name']} 返回成功")
                break
    return tool_results


def create_router_node(llm):
    chain = ROUTER_PROMPT | llm

    async def router_node(state: SystemState) -> dict:
        """路由节点：分析故障现象并分类"""
        try:
            logger.info(f"Router开始分析: symptom={state.symptom[:50]}...")
            response = await chain.ainvoke({"symptom": state.symptom})
            result = parse_json_content(response.content) or {"diagnosis_type": "other", "urgency": "medium"}
            
            diagnosis_type = result.get("diagnosis_type", "other")
            urgency = result.get("urgency", "medium")
            
            logger.info(f"Router分析完成: diagnosis_type={diagnosis_type}, urgency={urgency}")
            
            return {
                "diagnosis_type": diagnosis_type,
                "urgency": urgency,
                "messages": [f"Router: 诊断={diagnosis_type}, 紧急度={urgency}"]
            }
        except Exception as e:
            logger.exception(f"Router节点执行失败: {e}")
            return {
                "diagnosis_type": "other",
                "urgency": "medium",
                "messages": [f"Router: 分析失败，使用默认值"]
            }
    return router_node


def create_db_agent_node(llm, tools):
    """
    数据库诊断Agent节点

    Args:
        llm: LLM实例
        tools: MCP工具列表（由workflow注入，已过滤为db相关）
    """
    async def db_agent_node(state: SystemState) -> dict:
        try:
            logger.info(f"DB Agent开始诊断: symptom={state.symptom[:50]}...")

            response = await (DB_PROMPT | llm.bind_tools(tools)).ainvoke({"symptom": state.symptom})
            tool_results = await _execute_tool_calls(response, tools)

            diagnosis = await (DB_DIAGNOSIS_PROMPT | llm).ainvoke({
                "symptom": state.symptom,
                "tool_calls": str(response.tool_calls),
                "tool_results": str(tool_results)
            })
            result = parse_json_content(diagnosis.content) or {"diagnosis": "无法解析", "possible_causes": []}

            logger.info(f"DB Agent诊断完成: diagnosis={result.get('diagnosis')}")
            return {
                "db_agent_result": {**result, "tool_results": tool_results},
                "messages": [f"DB Agent (MCP): {result.get('diagnosis')}"]
            }
        except Exception as e:
            logger.exception(f"DB Agent节点执行失败: {e}")
            return {
                "db_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"DB Agent: 诊断失败 - {str(e)}"]
            }
    return db_agent_node


def create_net_agent_node(llm, tools):
    """
    网络诊断Agent节点

    Args:
        llm: LLM实例
        tools: MCP工具列表（由workflow注入，已过滤为net相关）
    """
    async def net_agent_node(state: SystemState) -> dict:
        try:
            logger.info(f"Net Agent开始诊断: symptom={state.symptom[:50]}...")

            response = await (NET_PROMPT | llm.bind_tools(tools)).ainvoke({"symptom": state.symptom})
            tool_results = await _execute_tool_calls(response, tools)

            diagnosis = await (NET_DIAGNOSIS_PROMPT | llm).ainvoke({
                "symptom": state.symptom,
                "tool_calls": str(response.tool_calls),
                "tool_results": str(tool_results)
            })
            result = parse_json_content(diagnosis.content) or {"diagnosis": "无法解析", "possible_causes": []}

            logger.info(f"Net Agent诊断完成: diagnosis={result.get('diagnosis')}")
            return {
                "net_agent_result": {**result, "tool_results": tool_results},
                "messages": [f"Net Agent (MCP): {result.get('diagnosis')}"]
            }
        except Exception as e:
            logger.exception(f"Net Agent节点执行失败: {e}")
            return {
                "net_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"Net Agent: 诊断失败 - {str(e)}"]
            }
    return net_agent_node


def create_app_agent_node(llm, tools):
    """
    应用诊断Agent节点

    Args:
        llm: LLM实例
        tools: MCP工具列表（由workflow注入，已过滤为app相关）
    """
    async def app_agent_node(state: SystemState) -> dict:
        try:
            logger.info(f"App Agent开始诊断: symptom={state.symptom[:50]}...")

            response = await (APP_PROMPT | llm.bind_tools(tools)).ainvoke({"symptom": state.symptom})
            tool_results = await _execute_tool_calls(response, tools)

            diagnosis = await (APP_DIAGNOSIS_PROMPT | llm).ainvoke({
                "symptom": state.symptom,
                "tool_calls": str(response.tool_calls),
                "tool_results": str(tool_results)
            })
            result = parse_json_content(diagnosis.content) or {"diagnosis": "无法解析", "possible_causes": []}

            logger.info(f"App Agent诊断完成: diagnosis={result.get('diagnosis')}")
            return {
                "app_agent_result": {**result, "tool_results": tool_results},
                "messages": [f"App Agent (MCP): {result.get('diagnosis')}"]
            }
        except Exception as e:
            logger.exception(f"App Agent节点执行失败: {e}")
            return {
                "app_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"App Agent: 诊断失败 - {str(e)}"]
            }
    return app_agent_node


def create_fix_agent_node(llm):
    chain = FIX_PROMPT | llm

    async def fix_agent_node(state: SystemState) -> dict:
        """修复方案生成Agent节点"""
        try:
            diagnosis_type = state.diagnosis_type
            if diagnosis_type == "db":
                diagnosis_result = state.db_agent_result
            elif diagnosis_type == "net":
                diagnosis_result = state.net_agent_result
            elif diagnosis_type == "app":
                diagnosis_result = state.app_agent_result
            else:
                diagnosis_result = {}
            
            logger.info(f"Fix Agent开始生成修复方案: diagnosis_type={diagnosis_type}")
            
            response = await chain.ainvoke({
                "diagnosis_type": diagnosis_type,
                "diagnosis_result": str(diagnosis_result)
            })
            result = parse_json_content(response.content) or {
                "plan_id": "PLAN-ERROR",
                "description": "无法生成方案",
                "risk_level": "unknown",
                "prerequisites": [],
                "steps": [],
                "verification": {"commands": [], "expected_result": ""},
                "estimated_time": "0"
            }
            
            logger.info(f"Fix Agent方案生成完成: plan_id={result.get('plan_id')}, risk_level={result.get('risk_level')}")
            
            return {
                "fix_plan": result,
                "messages": [f"Fix Agent: 生成修复方案 {result.get('plan_id')} - 风险等级: {result.get('risk_level')}"]
            }
        except Exception as e:
            logger.exception(f"Fix Agent节点执行失败: {e}")
            return {
                "fix_plan": {
                    "plan_id": "PLAN-ERROR",
                    "description": "方案生成失败",
                    "risk_level": "unknown",
                    "prerequisites": [],
                    "steps": [],
                    "verification": {"commands": [], "expected_result": ""},
                    "estimated_time": "0"
                },
                "messages": [f"Fix Agent: 方案生成失败 - {str(e)}"]
            }
    return fix_agent_node


def create_human_approval_node():
    async def human_approval_node(state: SystemState) -> dict:
        """人工审批节点"""
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
        """处理other类型工单：记录日志并保存工单"""
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
        """执行节点：执行修复方案并保存工单"""
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


# ============================================================
# Multi-Agent 节点: 并行派发 + 聚合推理
# ============================================================

def create_dispatch_node(agent_runners: dict[str, Callable[[SystemState], Awaitable[dict]]]):
    """
    创建并行派发节点

    根据 Supervisor 的 dispatched_agents 列表，并行调用被派发的 Agent。
    使用 asyncio.gather 实现并行执行，各 Agent 结果合并写入 state。

    Args:
        agent_runners: Agent 名称 → run 方法的映射
            {"db_agent": db_agent.run, "net_agent": net_agent.run, ...}
    """
    async def dispatch_node(state: SystemState) -> dict:
        dispatched = state.dispatched_agents

        if not dispatched:
            logger.info("[Dispatch] 无 Agent 被派发，跳过诊断")
            return {"messages": ["Dispatch: 无需诊断Agent，直接处理"]}

        logger.info(f"[Dispatch] 并行派发 Agent: {dispatched}")

        tasks = []
        agent_names = []
        for agent_name in dispatched:
            runner = agent_runners.get(agent_name)
            if runner:
                tasks.append(runner(state))
                agent_names.append(agent_name)
            else:
                logger.warning(f"[Dispatch] 未找到 Agent: {agent_name}")

        if not tasks:
            logger.warning("[Dispatch] 没有可执行的 Agent")
            return {"messages": ["Dispatch: 无可用Agent执行"]}

        # 并行执行所有被派发的 Agent
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并各 Agent 的返回结果
        merged = {"messages": []}
        for agent_name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                logger.error(f"[Dispatch] Agent {agent_name} 执行异常: {result}")
                merged["messages"].append(f"Dispatch: {agent_name} 执行异常 - {str(result)}")
                continue

            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "messages":
                        merged["messages"].extend(value)
                    else:
                        merged[key] = value

        logger.info(f"[Dispatch] 并行执行完成，{len(agent_names)} 个 Agent 返回结果")
        return merged

    return dispatch_node


def create_aggregate_node(llm):
    """
    创建聚合推理节点

    综合多个 Agent 的诊断结果，给出最终诊断结论。
    - 只有一个 Agent 返回结果 → 直接采用
    - 多个 Agent 返回结果 → LLM 聚合推理，加权判断

    Args:
        llm: LLM 实例，用于聚合推理
    """
    async def aggregate_node(state: SystemState) -> dict:
        # 收集所有已执行的 Agent 诊断结果
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

        # 单 Agent 结果直接采用，无需 LLM 聚合
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

        # 多 Agent 结果需要 LLM 聚合推理
        logger.info(f"[Aggregate] 聚合 {len(agent_results)} 个 Agent 的诊断结果: {list(agent_results.keys())}")

        try:
            results_str = ""
            for name, result in agent_results.items():
                results_str += f"\n--- {name} ---\n"
                results_str += f"诊断: {result.get('diagnosis', '未知')}\n"
                results_str += f"可能原因: {result.get('possible_causes', [])}\n"

            response = await (AGGREGATE_PROMPT | llm).ainvoke({
                "symptom": state.symptom,
                "agent_results": results_str,
            })
            aggregated = parse_json_content(response.content) or {
                "diagnosis": "聚合分析失败",
                "possible_causes": [],
                "confidence": 0.0,
                "contributing_agents": list(agent_results.keys()),
                "reasoning": "LLM 聚合推理失败",
            }

            logger.info(
                f"[Aggregate] 聚合完成: diagnosis={aggregated.get('diagnosis')}, "
                f"confidence={aggregated.get('confidence')}"
            )

            return {
                "aggregated_diagnosis": aggregated,
                "messages": [
                    f"Aggregate: 综合诊断={aggregated.get('diagnosis')}, "
                    f"置信度={aggregated.get('confidence')}"
                ],
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
