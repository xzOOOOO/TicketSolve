from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from agents.base import BaseAgent
from agents.communication import CommunicationBus
from state import SystemState
from prompts import APP_PROMPT, APP_DIAGNOSIS_PROMPT
from schemas import DiagnosisOutput
# make_trace_event：标准化 Trace 事件工厂，所有 Agent 都用同一套事件格式，方便外部评测系统解析
from trace_events import make_trace_event
# build_protocol_context：从消息列表中提取协议上下文（含假设、证据、冲突等）
# collaboration_requests_from_result：从诊断结果中提取结构化协作请求
from agent_protocol import build_protocol_context, collaboration_requests_from_result
# normalize_evidence_items：把 LLM 输出的各种格式证据统一成 EvidenceItem 对象
from evidence import normalize_evidence_items
from logger import logger
from config import settings

# 允许协作的目标 Agent 白名单，防止 LLM 编造不存在的 Agent 名字
_VALID_AGENTS = {"db_agent", "net_agent", "app_agent"}


class AppAgent(BaseAgent):
    """应用诊断专家 Agent

    职责：使用应用相关工具诊断故障，返回结构化诊断结论。

    v1 协议增强：
    - 诊断前接收其他 Agent 的协作消息（peer_messages）
    - 诊断后发布结构化假设（hypothesis）和证据请求（evidence_request）
    - 通过 CommunicationBus 与 db_agent/net_agent/app_agent 协作
    """
    name = "app_agent"
    role = "应用诊断专家"

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool],
                 communication_bus: CommunicationBus = None):
        super().__init__(llm, tools)
        # communication_bus：通信总线实例，用于发送/接收协议消息
        # 为 None 时 Agent 退化为单机模式，不发送协作消息
        self.bus = communication_bus
        # with_structured_output：让 LLM 直接返回符合 DiagnosisOutput schema 的对象
        self._structured_llm = self.llm.with_structured_output(DiagnosisOutput)
        # 诊断链 = PromptTemplate → StructuredLLM → 自动重试
        self._diagnosis_chain = (APP_DIAGNOSIS_PROMPT | self._structured_llm).with_retry(**settings.get_retry_config())

    async def run(self, state: SystemState) -> dict:
        """执行应用诊断

        流程：
        1. 接收其他 Agent 的协作消息（如果有）
        2. ReAct 循环：调用工具收集信息（check_app_process 等）
        3. 使用 Structured Output 调用 LLM 生成诊断结论
        4. 广播诊断结果，向需要协作的 Agent 发送求助消息
        """
        # 初始化 Trace 事件列表，首先记录 agent_started 事件
        trace_events = [make_trace_event(
            "agent_started",
            ticket_id=state.ticket_id,
            agent_name=self.name,
            input_data={"symptom": state.symptom},
            metadata={"dispatch_round": state.dispatch_round},
        )]
        try:
            logger.info(f"[{self.name}] 开始诊断: symptom={state.symptom[:50]}...")

            # 从通信总线接收其他 Agent 发来的消息
            peer_messages = ""
            if self.bus and state.agent_messages:
                incoming = self.bus.receive(self.name, state.agent_messages)
                if incoming:
                    logger.info(f"[{self.name}] 收到 {len(incoming)} 条消息")
                    protocol_context = build_protocol_context(incoming)
                    peer_messages = protocol_context["text"]

            # ReAct 循环：Think -> Act -> Observe，最多 3 轮
            tool_results, tool_calls_info = await self.react_loop(
                APP_PROMPT, state.symptom, max_iterations=3
            )

            # 使用 Structured Output 生成诊断结论
            result = await self._diagnosis_chain.ainvoke({
                "symptom": state.symptom,
                "tool_calls": str(tool_calls_info),
                "tool_results": str(tool_results),
                "peer_messages": peer_messages or "无",
            })

            # 兜底处理
            if result is None:
                result = DiagnosisOutput(
                    diagnosis="无法解析",
                    possible_causes=[],
                    confidence=0.0,
                    collaboration_requests=[],
                )

            # 转为 dict 以保持与 state 的兼容性
            result_dict = result.model_dump()
            result_dict["evidence"] = normalize_evidence_items(
                result_dict.get("evidence") or result_dict.get("possible_causes", []),
                source_agent=self.name,
                supports_hypothesis=True,
                confidence=float(result_dict.get("confidence") or 0.0),
            )

            logger.info(f"[{self.name}] 诊断完成: diagnosis={result_dict.get('diagnosis')}")

            # 记录审计日志：工具调用 + 诊断结论
            audit_logs = []
            # 记录工具调用标准化 Trace 事件（如果有调用工具）
            if tool_calls_info:
                trace_events.append(make_trace_event(
                    "tool_called",
                    ticket_id=state.ticket_id,
                    agent_name=self.name,
                    input_data={"symptom": state.symptom, "peer_messages": peer_messages or "无"},
                    output_data={"tool_calls": tool_calls_info},
                    metadata={
                        "tool_count": len(tool_calls_info),
                        "dispatch_round": state.dispatch_round,
                    },
                ))
            # 记录工具观察结果标准化 Trace 事件（如果有返回结果）
            if tool_results:
                trace_events.append(make_trace_event(
                    "observation_received",
                    ticket_id=state.ticket_id,
                    agent_name=self.name,
                    input_data={"tool_call_count": len(tool_calls_info)},
                    output_data={"tool_results": tool_results},
                    metadata={
                        "observation_count": len(tool_results),
                        "dispatch_round": state.dispatch_round,
                    },
                ))
            # 记录诊断结论生成的标准化 Trace 事件
            trace_events.append(make_trace_event(
                "diagnosis_generated",
                ticket_id=state.ticket_id,
                agent_name=self.name,
                input_data={
                    "symptom": state.symptom,
                    "tool_calls": str(tool_calls_info),
                    "tool_results": str(tool_results),
                    "peer_messages": peer_messages or "无",
                },
                output_data=result_dict,
                metadata={
                    "confidence": result_dict.get("confidence"),
                    "dispatch_round": state.dispatch_round,
                },
            ))

            # 1. 记录工具调用
            if tool_calls_info:
                audit_logs.append({
                    "ticket_id": state.ticket_id,
                    "agent_name": self.name,
                    "action_type": "tool_call",
                    "action_detail": {
                        "tools_called": [t.get("name", t.get("tool", "unknown")) for t in tool_calls_info],
                        "tool_results_summary": [
                            {"tool": t.get("name", t.get("tool", "unknown")), "status": "success" if "error" not in str(t).lower() else "error"}
                            for t in tool_results
                        ],
                    },
                    "input_context": {"symptom": state.symptom, "peer_messages": peer_messages or "无"},
                    "output_result": {"tool_results": tool_results},
                    "dispatch_round": state.dispatch_round,
                })

            # 2. 记录诊断结论
            audit_logs.append({
                "ticket_id": state.ticket_id,
                "agent_name": self.name,
                "action_type": "diagnosis",
                "action_detail": {
                    "diagnosis": result_dict.get("diagnosis"),
                    "possible_causes": result_dict.get("possible_causes", []),
                    "confidence": result_dict.get("confidence"),
                },
                "input_context": {
                    "symptom": state.symptom,
                    "tool_calls": str(tool_calls_info),
                    "tool_results": str(tool_results),
                    "peer_messages": peer_messages or "无",
                },
                "output_result": result_dict,
                "dispatch_round": state.dispatch_round,
            })

            # 3. 记录协作请求（如果有）
            collaboration_requests = collaboration_requests_from_result(result_dict)
            for request in collaboration_requests:
                target = request["target_agent"]
                if target in _VALID_AGENTS and target != self.name:
                    audit_logs.append({
                        "ticket_id": state.ticket_id,
                        "agent_name": self.name,
                        "action_type": "collaborate",
                        "action_detail": {
                            "target_agent": target,
                            "reason": f"应用诊断发现可能涉及{target}领域的问题",
                            "content": f"应用诊断发现可能涉及{target}领域的问题: {result_dict.get('diagnosis')}，请协助确认",
                        },
                        "input_context": {"diagnosis": result_dict.get("diagnosis")},
                        "output_result": {"request_sent": True},
                        "dispatch_round": state.dispatch_round,
                    })

            # 构造状态更新
            update = {
                "app_agent_result": {**result_dict, "tool_results": tool_results},
                "messages": [f"App Agent (MCP): {result_dict.get('diagnosis')}"],
                "audit_logs": audit_logs,
                "trace_events": trace_events,
            }

            # ========== 通过通信总线发送协议消息 ==========
            if self.bus:
                agent_messages = []

                # 第 1 步：发布结构化故障假设（hypothesis）
                # 这是证据协作协议的起点。其他 Agent 看到这条消息后，
                # 可以决定是否要请求补充证据、支持或质疑这个假设。
                hypothesis_messages = self.bus.publish_hypothesis(
                    sender=self.name,
                    content=f"应用诊断假设: {result_dict.get('hypothesis') or result_dict.get('diagnosis')}",
                    hypothesis=result_dict.get("hypothesis") or result_dict.get("diagnosis", ""),
                    fault_type=result_dict.get("fault_type"),
                    confidence=result_dict.get("confidence", 0.0),
                    evidence=result_dict.get("evidence") or result_dict.get("possible_causes", []),
                )
                agent_messages.extend(hypothesis_messages)
                # 保存 hypothesis_message，后续 evidence_request 需要引用它作为关联源头
                hypothesis_message = hypothesis_messages[0]

                # 第 2 步：广播诊断结论（diagnosis）
                # 这条消息是对外公开的诊断结果，所有 Agent 都能看到。
                agent_messages.extend(self.bus.broadcast(
                    sender=self.name,
                    content=f"诊断结论: {result_dict.get('diagnosis')}，可能原因: {result_dict.get('possible_causes', [])}",
                    msg_type="diagnosis",
                    confidence=result_dict.get("confidence", 0.0),
                    evidence=result_dict.get("evidence") or result_dict.get("possible_causes", []),
                    hypothesis=result_dict.get("hypothesis"),
                    fault_type=result_dict.get("fault_type"),
                ))

                # 第 3 步：向需要协作的 Agent 发送结构化证据请求（evidence_request）
                # collaboration_requests 来自 LLM 的诊断输出，表示 "我还需要谁帮我验证什么"
                for request in collaboration_requests:
                    target = request["target_agent"]
                    # 白名单校验：只允许向真实存在的 Agent 发请求，防止 LLM 编造 Agent 名字
                    if target in _VALID_AGENTS and target != self.name:
                        request_messages = self.bus.request_evidence(
                            sender=self.name,
                            receiver=target,
                            # hypothesis_message 作为关联源头，evidence_request 会继承它的 correlation_id
                            hypothesis_message=hypothesis_message,
                            required_evidence=request.get("required_evidence", []),
                            reason=request.get("reason") or f"应用诊断需要 {target} 补充证据",
                            suggested_tools=request.get("suggested_tools", []),
                            confidence=result_dict.get("confidence", 0.0),
                        )
                        agent_messages.extend(request_messages)
                        # 每条 evidence_request 都记录一个 handoff_requested Trace 事件
                        for msg in request_messages:
                            trace_events.append(make_trace_event(
                                "handoff_requested",
                                ticket_id=state.ticket_id,
                                agent_name=self.name,
                                input_data={"diagnosis": result_dict.get("diagnosis")},
                                output_data=msg,
                                metadata={
                                    "dispatch_round": state.dispatch_round,
                                    "message_id": msg.get("message_id"),
                                    "correlation_id": msg.get("correlation_id"),
                                    "msg_type": msg.get("msg_type"),
                                },
                            ))

                # 把所有协议消息合并到状态更新中
                if agent_messages:
                    update["agent_messages"] = agent_messages

            return update
        except Exception as e:
            # Agent 异常时也要记录 failure 状态的 Trace 事件，保证失败路径可追溯
            logger.exception(f"[{self.name}] 执行失败: {e}")
            trace_events.append(make_trace_event(
                "diagnosis_generated",
                ticket_id=state.ticket_id,
                agent_name=self.name,
                status="failure",
                input_data={"symptom": state.symptom},
                error=str(e),
                metadata={"dispatch_round": state.dispatch_round},
            ))
            return {
                "app_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"App Agent: 诊断失败 - {str(e)}"],
                # 异常路径也要返回 trace_events，避免前面已记录的事件丢失
                "trace_events": trace_events,
            }
