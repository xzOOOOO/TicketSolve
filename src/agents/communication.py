"""
CommunicationBus - Agent 间通信总线

设计原则:
- 无状态：不维护内部缓冲区，所有消息通过 state.agent_messages 流转
- Agent 通过 send/broadcast 生成消息字典，返回值写入 state
- Agent 通过 receive 从 state.agent_messages 中过滤属于自己的消息
- 消息追加式写入（operator.add），不覆盖
"""

from typing import Optional
from state import AgentMessage
from logger import logger


class CommunicationBus:
    """
    Agent 间通信总线（无状态）

    使用方式:
    1. Agent 诊断过程中发现需要其他 Agent 协助 → bus.send()，返回值写入 state.agent_messages
    2. Agent 诊断完成后广播结论 → bus.broadcast()，返回值写入 state.agent_messages
    3. Agent 开始诊断前检查是否有其他 Agent 的消息 → bus.receive()
    """

    def send(self, sender: str, receiver: str, content: str,
             msg_type: str = "info", confidence: float = 0.0,
             evidence: Optional[list[str]] = None) -> list[dict]:
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
            可写入 state.agent_messages 的消息列表
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
        logger.debug(f"[Bus] {sender} → {receiver}: [{msg_type}] {content[:50]}...")
        return [msg_dict]

    def broadcast(self, sender: str, content: str,
                  msg_type: str = "diagnosis", confidence: float = 0.0,
                  evidence: Optional[list[str]] = None) -> list[dict]:
        """
        广播消息给所有 Agent

        Returns:
            可写入 state.agent_messages 的消息列表
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
