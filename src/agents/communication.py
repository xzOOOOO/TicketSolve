"""
CommunicationBus - Agent 间通信总线

设计原则:
- 基于 state.agent_messages 字段，不引入外部依赖
- Agent 通过 send/broadcast 写入消息
- Agent 通过 receive 读取发给自己的消息
- 消息追加式写入（operator.add），不覆盖
"""

from typing import Optional
from state import AgentMessage
from logger import logger


class CommunicationBus:
    """
    Agent 间通信总线

    使用方式:
    1. Agent 诊断过程中发现需要其他 Agent 协助 → bus.send()
    2. Agent 诊断完成后广播结论 → bus.broadcast()
    3. Agent 开始诊断前检查是否有其他 Agent 的消息 → bus.receive()
    """

    def __init__(self):
        self._pending: list[dict] = []

    def send(self, sender: str, receiver: str, content: str,
             msg_type: str = "info", confidence: float = 0.0,
             evidence: Optional[list[str]] = None) -> dict:
        """
        发送消息给指定 Agent

        Args:
            sender: 发送者名称
            receiver: 接收者名称
            content: 消息内容
            msg_type: 消息类型 (diagnosis/question/request_help/disagreement)
            confidence: 置信度 0-1
            evidence: 支撑证据

        Returns:
            可写入 state.agent_messages 的字典
        """
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            confidence=confidence,
            evidence=evidence or [],
        )
        msg_dict = msg.model_dump()
        self._pending.append(msg_dict)
        logger.debug(f"[Bus] {sender} → {receiver}: [{msg_type}] {content[:50]}...")
        return [msg_dict]

    def broadcast(self, sender: str, content: str,
                  msg_type: str = "diagnosis", confidence: float = 0.0,
                  evidence: Optional[list[str]] = None) -> list[dict]:
        """
        广播消息给所有 Agent

        Returns:
            可写入 state.agent_messages 的列表
        """
        msg = AgentMessage(
            sender=sender,
            receiver="broadcast",
            content=content,
            msg_type=msg_type,
            confidence=confidence,
            evidence=evidence or [],
        )
        msg_dict = msg.model_dump()
        self._pending.append(msg_dict)
        logger.debug(f"[Bus] {sender} → broadcast: [{msg_type}] {content[:50]}...")
        return [msg_dict]

    def receive(self, agent_name: str, state_messages: list[dict]) -> list[dict]:
        """
        获取发给指定 Agent 的消息

        Args:
            agent_name: 接收者名称
            state_messages: state.agent_messages 中的已有消息

        Returns:
            发给该 Agent 的消息列表（包括广播消息）
        """
        received = []
        for msg in state_messages:
            if msg.get("receiver") == agent_name or msg.get("receiver") == "broadcast":
                if msg.get("sender") != agent_name:
                    received.append(msg)
        return received

    def flush(self) -> list[dict]:
        """
        取出所有待发送消息并清空缓冲区

        Returns:
            所有待发送消息列表，用于写入 state.agent_messages
        """
        pending = self._pending.copy()
        self._pending.clear()
        return pending
