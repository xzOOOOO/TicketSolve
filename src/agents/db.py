from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from agents.base import BaseAgent
from agents.communication import CommunicationBus
from state import SystemState
from prompts import DB_PROMPT, DB_DIAGNOSIS_PROMPT
from schemas import DiagnosisOutput
# 引入标准化 Trace 事件工厂，用于生成统一的追踪事件
from trace_events import make_trace_event
from logger import logger
from config import settings

_VALID_AGENTS = {"db_agent", "net_agent", "app_agent"}


class DBAgent(BaseAgent):
    """数据库诊断专家 Agent

    职责：使用数据库相关工具诊断故障，返回结构化诊断结论。
    """
    name = "db_agent"
    role = "数据库诊断专家"

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool],
                 communication_bus: CommunicationBus = None):
        super().__init__(llm, tools)
        self.bus = communication_bus
        # 创建结构化 LLM，输出格式由 DiagnosisOutput 的 Pydantic schema 约束
        self._structured_llm = self.llm.with_structured_output(DiagnosisOutput)
        self._diagnosis_chain = (DB_DIAGNOSIS_PROMPT | self._structured_llm).with_retry(**settings.get_retry_config())

    async def run(self, state: SystemState) -> dict:
        """执行数据库诊断

        流程：
        1. 接收其他 Agent 的协作消息（如果有）
        2. ReAct 循环：调用工具收集信息（check_db_connection 等）
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
                    for msg in incoming:
                        peer_messages += f"[{msg['sender']}→{msg.get('receiver','')}] ({msg['msg_type']}) {msg['content']}\n"

            # ReAct 循环：Think -> Act -> Observe，最多 3 轮
            tool_results, tool_calls_info = await self.react_loop(
                DB_PROMPT, state.symptom, max_iterations=3
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
                    need_collaboration=[],
                )

            # 转为 dict 以保持与 state 的兼容性
            result_dict = result.model_dump()

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
            # 记录向其他 Agent 请求协作的标准化 Trace 事件
            for target in result_dict.get("need_collaboration", []):
                if target in _VALID_AGENTS and target != self.name:
                    trace_events.append(make_trace_event(
                        "handoff_requested",
                        ticket_id=state.ticket_id,
                        agent_name=self.name,
                        input_data={"diagnosis": result_dict.get("diagnosis")},
                        output_data={"target_agent": target, "request_sent": True},
                        metadata={"dispatch_round": state.dispatch_round},
                    ))
                    audit_logs.append({
                        "ticket_id": state.ticket_id,
                        "agent_name": self.name,
                        "action_type": "collaborate",
                        "action_detail": {
                            "target_agent": target,
                            "reason": f"数据库诊断发现可能涉及{target}领域的问题",
                            "content": f"数据库诊断发现可能涉及{target}领域的问题: {result_dict.get('diagnosis')}，请协助确认",
                        },
                        "input_context": {"diagnosis": result_dict.get("diagnosis")},
                        "output_result": {"request_sent": True},
                        "dispatch_round": state.dispatch_round,
                    })

            # 构造状态更新
            update = {
                "db_agent_result": {**result_dict, "tool_results": tool_results},
                "messages": [f"DB Agent (MCP): {result_dict.get('diagnosis')}"],
                "audit_logs": audit_logs,
                "trace_events": trace_events,
            }

            # 通过通信总线发送消息
            if self.bus:
                agent_messages = []

                # 广播诊断结论给所有 Agent
                agent_messages.extend(self.bus.broadcast(
                    sender=self.name,
                    content=f"诊断结论: {result_dict.get('diagnosis')}，可能原因: {result_dict.get('possible_causes', [])}",
                    msg_type="diagnosis",
                    confidence=result_dict.get("confidence", 0.0),
                    evidence=result_dict.get("possible_causes", []),
                ))

                # 向需要协作的 Agent 发送求助消息
                for target in result_dict.get("need_collaboration", []):
                    if target in _VALID_AGENTS and target != self.name:
                        agent_messages.extend(self.bus.send(
                            sender=self.name,
                            receiver=target,
                            content=f"数据库诊断发现可能涉及{target}领域的问题: {result_dict.get('diagnosis')}，请协助确认",
                            msg_type="request_help",
                            confidence=result_dict.get("confidence", 0.0),
                            evidence=result_dict.get("possible_causes", []),
                        ))

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
                "db_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"DB Agent: 诊断失败 - {str(e)}"],
                # 异常路径也要返回 trace_events，避免前面已记录的事件丢失
                "trace_events": trace_events,
            }
