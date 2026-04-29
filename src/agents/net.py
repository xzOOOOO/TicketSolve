"""
NetAgent - 网络诊断专家
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from agents.base import BaseAgent
from agents.communication import CommunicationBus
from state import SystemState
from prompts import NET_PROMPT, NET_DIAGNOSIS_PROMPT
from utils import parse_json_content
from logger import logger

_VALID_AGENTS = {"db_agent", "net_agent", "app_agent"}


class NetAgent(BaseAgent):
    name = "net_agent"
    role = "网络诊断专家"

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool],
                 communication_bus: CommunicationBus = None):
        super().__init__(llm, tools)
        self.bus = communication_bus

    async def run(self, state: SystemState) -> dict:
        try:
            logger.info(f"[{self.name}] 开始诊断: symptom={state.symptom[:50]}...")

            peer_messages = ""
            if self.bus and state.agent_messages:
                incoming = self.bus.receive(self.name, state.agent_messages)
                if incoming:
                    logger.info(f"[{self.name}] 收到 {len(incoming)} 条消息")
                    for msg in incoming:
                        peer_messages += f"[{msg['sender']}→{msg.get('receiver','')}] ({msg['msg_type']}) {msg['content']}\n"

            tool_results, tool_calls_info = await self.react_loop(
                NET_PROMPT, state.symptom, max_iterations=3
            )

            diagnosis = await (NET_DIAGNOSIS_PROMPT | self.llm).ainvoke({
                "symptom": state.symptom,
                "tool_calls": str(tool_calls_info),
                "tool_results": str(tool_results),
                "peer_messages": peer_messages or "无",
            })
            result = parse_json_content(diagnosis.content) or {
                "diagnosis": "无法解析",
                "possible_causes": [],
                "confidence": 0.0,
                "need_collaboration": [],
            }

            logger.info(f"[{self.name}] 诊断完成: diagnosis={result.get('diagnosis')}")

            update = {
                "net_agent_result": {**result, "tool_results": tool_results},
                "messages": [f"Net Agent (MCP): {result.get('diagnosis')}"],
            }

            if self.bus:
                agent_messages = []

                agent_messages.extend(self.bus.broadcast(
                    sender=self.name,
                    content=f"诊断结论: {result.get('diagnosis')}，可能原因: {result.get('possible_causes', [])}",
                    msg_type="diagnosis",
                    confidence=result.get("confidence", 0.0),
                    evidence=result.get("possible_causes", []),
                ))

                for target in result.get("need_collaboration", []):
                    if target in _VALID_AGENTS and target != self.name:
                        agent_messages.extend(self.bus.send(
                            sender=self.name,
                            receiver=target,
                            content=f"网络诊断发现可能涉及{target}领域的问题: {result.get('diagnosis')}，请协助确认",
                            msg_type="request_help",
                            confidence=result.get("confidence", 0.0),
                            evidence=result.get("possible_causes", []),
                        ))

                if agent_messages:
                    update["agent_messages"] = agent_messages

            return update
        except Exception as e:
            logger.exception(f"[{self.name}] 执行失败: {e}")
            return {
                "net_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"Net Agent: 诊断失败 - {str(e)}"],
            }
