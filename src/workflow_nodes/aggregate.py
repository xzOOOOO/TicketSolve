"""
聚合诊断节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def create_aggregate_node(llm, communication_bus=None):
    """
    创建聚合推理节点工厂函数。

    综合多个 Agent 的诊断结果，给出最终诊断结论。
    - 只有一个 Agent 返回结果 → 直接采用，无需 LLM
    - 多个 Agent 返回结果 → 使用 LLM 聚合推理，加权判断

    v1 协议增强：
    - 引入 protocol_context：把 Agent 间协作协议的假设、证据、冲突也纳入聚合输入
    - LLM 不仅看诊断结论，还看 "哪个假设在协议中胜出"、"有哪些支持/反对证据"
    - 聚合结果包含 protocol_summary，供 FixAgent 参考

    参数：
        llm: LLM 实例，用于聚合推理（多 Agent 场景才需要）
        communication_bus: CommunicationBus 实例（可选），用于读取 Agent 间通信消息

    返回：
        异步节点函数 aggregate_node(state) -> dict
    """
    async def aggregate_node(state: SystemState) -> dict:
        # protocol_context：构建协议上下文，从所有 Agent 消息中提取假设、证据请求、响应、冲突等
        # protocol_context["text"]：人可读的文本摘要，会拼接到 LLM 输入中
        # protocol_context["protocol_summary"]：结构化的统计字典，会写入聚合结果
        protocol_context = build_protocol_context(state.agent_messages)
        protocol_summary = protocol_context.get("protocol_summary", {})

        # agent_results：收集各 Agent 的诊断结果，key 是 Agent 名称，value 是诊断结果字典
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
                    "diagnosis_generated",              # 事件类型：诊断生成
                    ticket_id=state.ticket_id,          # 工单 ID
                    agent_name="aggregate",             # 产生事件的节点
                    status="skipped",                   # 状态：跳过（无结果可聚合）
                    input_data={"agent_result_count": 0},
                    output_data={"aggregated_diagnosis": None},
                    metadata={"dispatch_round": state.dispatch_round},
                )],
                "messages": ["Aggregate: 无诊断结果可聚合"],
            }

        # 只有一个 Agent 返回结果时，直接采用，无需 LLM 聚合（节省 token 和延迟）
        if len(agent_results) == 1:
            agent_name = list(agent_results.keys())[0]
            single_result = agent_results[agent_name]
            logger.info(f"[Aggregate] 只有 {agent_name} 返回结果，直接采用")

            # aggregated：聚合诊断结果字典
            # confidence 固定为 0.7（单 Agent 的置信度不如多 Agent 聚合高）
            aggregated = {
                "diagnosis": single_result.get("diagnosis", "未知"),
                "possible_causes": single_result.get("possible_causes", []),
                "confidence": 0.7,
                "contributing_agents": [agent_name],
                "reasoning": f"仅 {agent_name} 返回诊断结果，直接采用",
                "protocol_summary": protocol_summary,
            }
            return {
                "aggregated_diagnosis": aggregated,
                "trace_events": [make_trace_event(
                    "diagnosis_generated",
                    ticket_id=state.ticket_id,
                    agent_name="aggregate",
                    input_data={"contributing_agents": [agent_name]},
                    output_data=aggregated,
                    metadata={
                        "dispatch_round": state.dispatch_round,
                        # 单 Agent 场景也把 protocol_summary 带上，保持输出格式一致
                        "protocol_summary": protocol_summary,
                    },
                )],
                "messages": [f"Aggregate: 采用 {agent_name} 的诊断结论"],
            }

        # 多个 Agent 返回结果时，使用 LLM 进行聚合推理
        logger.info(f"[Aggregate] 聚合 {len(agent_results)} 个 Agent 的诊断结果: {list(agent_results.keys())}")

        try:
            # results_str：把各 Agent 的诊断结果拼接成文本，作为 LLM 的输入
            results_str = ""
            for name, result in agent_results.items():
                results_str += f"\n--- {name} ---\n"
                results_str += f"诊断: {result.get('diagnosis', '未知')}\n"
                results_str += f"可能原因: {result.get('possible_causes', [])}\n"
                results_str += f"故障类型: {result.get('fault_type')}\n"
                results_str += f"假设: {result.get('hypothesis')}\n"
                results_str += f"证据: {result.get('evidence', [])}\n"

            # 如果传入了 communication_bus，读取发给 aggregate 的消息（广播消息）
            if communication_bus and state.agent_messages:
                relevant_msgs = communication_bus.receive("aggregate", state.agent_messages)
                if relevant_msgs:
                    results_str += "\n--- Agent 间通信 ---\n"
                    for msg in relevant_msgs:
                        results_str += f"[{msg['sender']}→{msg['receiver']}] ({msg['msg_type']}, 置信度:{msg.get('confidence', 0)}) {msg['content']}\n"
            # 把协议上下文也拼接到输入中，让 LLM 知道各 Agent 的假设谁支持、谁反对
            if state.agent_messages:
                results_str += "\n--- 结构化协作协议上下文 ---\n"
                results_str += protocol_context.get("text", "无结构化协作消息。")

            # 使用 Structured Output 进行聚合推理
            # structured_llm：包装后的 LLM，输出会被强制解析为 AggregateOutput Pydantic 模型
            # 在函数内部创建（因为 aggregate 是函数式节点，无 __init__ 初始化时机）
            structured_llm = llm.with_structured_output(AggregateOutput)
            # AGGREGATE_PROMPT | structured_llm：LangChain 管道语法，先应用 Prompt 模板，再调用 LLM
            # .with_retry()：自动重试（应对 LLM 偶尔的输出格式错误）
            # .ainvoke()：异步调用
            result = await (AGGREGATE_PROMPT | structured_llm).with_retry(**settings.get_retry_config()).ainvoke({
                "symptom": state.symptom,
                "agent_results": results_str,
            })

            # 兜底处理：如果 LLM 返回 None（解析失败），生成默认的失败诊断
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
            # 兜底：如果 LLM 没有输出 protocol_summary，把协议上下文中的摘要补进去
            # 这样 FixAgent 始终能拿到协议层面的统计信息
            if not aggregated.get("protocol_summary"):
                aggregated["protocol_summary"] = protocol_summary

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
                    # 把协议摘要也记入审计日志，方便事后追溯 "为什么选了这个诊断"
                    "protocol_summary": aggregated.get("protocol_summary"),
                },
                "input_context": {
                    "agent_results": results_str,
                    "symptom": state.symptom,
                    # 把完整的协议上下文也记入输入，审计时能还原当时的协作全貌
                    "protocol_context": protocol_context,
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
                    metadata={
                        "dispatch_round": state.dispatch_round,
                        "protocol_summary": aggregated.get("protocol_summary"),
                    },
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
                    "protocol_summary": protocol_summary,
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

