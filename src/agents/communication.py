"""
CommunicationBus - Agent 间通信总线

设计原则:
- 无状态：不维护内部缓冲区，所有消息通过 state.agent_messages 流转
- Agent 通过 send/broadcast 生成消息字典，返回值写入 state
- Agent 通过 receive 从 state.agent_messages 中过滤属于自己的消息
- 消息追加式写入（operator.add），不覆盖

v1 协议扩展：
- 新增 publish_hypothesis：发布结构化故障假设，作为证据协作的起点
- 新增 request_evidence：向特定 Agent 请求补充证据
- 新增 respond_evidence：响应证据请求（通常在 DynamicCheck 中自动生成）
- 这些方法的返回值都是 list[dict]，可以直接合并到 state.agent_messages
"""

from typing import Any, Optional
# normalize_evidence_items：把字符串/字典混合格式统一成 EvidenceItem 对象列表
from evidence import normalize_evidence_items
from state import AgentMessage
# agent_protocol 模块实现了证据协作协议的核心逻辑：
# - make_hypothesis：构造 hypothesis 消息
# - make_evidence_request：构造 evidence_request 消息
# - make_evidence_response：构造 evidence_response 消息
# - normalize_messages：把 state 中存储的字典消息归一化成标准格式
# - pending_requests_for：找出某个 Agent 尚未响应的证据请求
# - has_response_for：检查某条请求是否已有响应
# - agent_result_covers_request：判断 Agent 的诊断结果是否覆盖了请求要求的证据
# - auto_response_from_agent_result：用已有诊断结果自动生成 evidence_response
# - build_protocol_context：从所有消息中构建协议上下文（含统计摘要）
from agent_protocol import (
    make_evidence_request,
    make_evidence_response,
    make_hypothesis,
    normalize_messages,
)
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
             evidence: Optional[list[Any]] = None, **protocol_fields) -> list[dict]:
        """
        发送消息给指定 Agent

        Args:
            sender: 发送者名称
            receiver: 接收者名称
            content: 消息内容
            msg_type: 消息类型 (hypothesis/evidence_request/evidence_response/challenge/support/diagnosis/info)
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
            evidence=normalize_evidence_items(
                evidence or [],
                source_agent=sender,
                supports_hypothesis=protocol_fields.get("supports_hypothesis"),
                confidence=confidence,
            ),
            **protocol_fields,
        )
        msg_dict = msg.model_dump()
        logger.debug(f"[Bus] {sender} → {receiver}: [{msg_type}] {content[:50]}...")
        return [msg_dict]

    def broadcast(self, sender: str, content: str,
                  msg_type: str = "diagnosis", confidence: float = 0.0,
                  evidence: Optional[list[Any]] = None, **protocol_fields) -> list[dict]:
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
            evidence=normalize_evidence_items(
                evidence or [],
                source_agent=sender,
                supports_hypothesis=protocol_fields.get("supports_hypothesis"),
                confidence=confidence,
            ),
            **protocol_fields,
        )
        msg_dict = msg.model_dump()
        logger.debug(f"[Bus] {sender} → broadcast: [{msg_type}] {content[:50]}...")
        return [msg_dict]

    def publish_hypothesis(
        self,
        *,
        sender: str,
        content: str,
        hypothesis: str,
        fault_type: Optional[str] = None,
        confidence: float = 0.0,
        evidence: Optional[list[Any]] = None,
    ) -> list[dict]:
        """发布结构化故障假设。

        这是证据协作协议的起点。Agent 诊断后提出一个可验证假设，
        其他 Agent 可以基于此假设发起证据请求或提出质疑。

        参数说明：
        - sender：发布假设的 Agent 名称
        - content：人可读的假设描述
        - hypothesis：结构化假设文本（一句话）
        - fault_type：标准化故障类型，用于匹配修复动作
        - confidence：对假设的置信度
        - evidence：支持假设的初始证据列表

        返回：包含一条 hypothesis 消息的列表，可直接追加到 state.agent_messages。
        """
        # 调用 agent_protocol.make_hypothesis 构造标准格式的 hypothesis 消息
        # 它会自动填充 message_id、correlation_id、msg_type 等协议字段
        msg = make_hypothesis(
            sender=sender,
            content=content,
            hypothesis=hypothesis,
            fault_type=fault_type,
            confidence=confidence,
            evidence=evidence or [],
        )
        logger.debug(f"[Bus] {sender} → broadcast: [hypothesis] {hypothesis[:50]}...")
        return [msg]

    def request_evidence(
        self,
        *,
        sender: str,
        receiver: str,
        hypothesis_message: dict,
        required_evidence: list[str],
        reason: str,
        suggested_tools: Optional[list[str]] = None,
        confidence: float = 0.0,
    ) -> list[dict]:
        """请求其他 Agent 对某个假设补充证据。

        使用场景：
        - db_agent 假设 "Postgres 宕机"，但需要 net_agent 确认 "网络是否连通"
        - db_agent 调用 bus.request_evidence(receiver="net_agent", ...)

        参数说明：
        - sender：请求方 Agent 名称
        - receiver：被请求方 Agent 名称
        - hypothesis_message：关联的 hypothesis 消息字典（用于继承 correlation_id 和 hypothesis_id）
        - required_evidence：需要对方提供的证据项名称列表，如 ["ping status", "dns resolution"]
        - reason：为什么需要这些证据（给被请求方 LLM 看的上下文）
        - suggested_tools：建议对方使用的工具名列表，降低对方决策成本
        - confidence：请求方当前置信度

        返回：包含一条 evidence_request 消息的列表。
        """
        # make_evidence_request 会自动：
        # 1. 从 hypothesis_message 继承 correlation_id 和 hypothesis_id
        # 2. 设置 related_to 指向 hypothesis_message 的 message_id
        # 3. 生成新的 message_id
        msg = make_evidence_request(
            sender=sender,
            receiver=receiver,
            hypothesis_message=hypothesis_message,
            required_evidence=required_evidence,
            reason=reason,
            suggested_tools=suggested_tools or [],
            confidence=confidence,
        )
        logger.debug(f"[Bus] {sender} → {receiver}: [evidence_request] {reason[:50]}...")
        return [msg]

    def respond_evidence(
        self,
        *,
        sender: str,
        receiver: str,
        request_message: dict,
        evidence: list[Any],
        supports_hypothesis: bool,
        content: Optional[str] = None,
        confidence: float = 0.0,
    ) -> list[dict]:
        """响应证据请求。

        使用场景：
        - 通常在 DynamicCheck 节点中自动生成，而不是由 Agent 手动调用
        - 当目标 Agent 已有诊断结果且能覆盖请求时，auto_response_from_agent_result 会调用此方法

        参数说明：
        - sender：响应方 Agent 名称
        - receiver：请求方 Agent 名称
        - request_message：对应的 evidence_request 消息字典
        - evidence：响应方提供的证据列表
        - supports_hypothesis：明确回答 "我找到的证据是否支持你的假设"
        - content：人可读的响应说明
        - confidence：响应方对证据的置信度

        返回：包含一条 evidence_response 消息的列表。
        """
        # make_evidence_response 会自动：
        # 1. 从 request_message 继承 correlation_id 和 hypothesis_id
        # 2. 设置 related_to 指向 request_message 的 message_id
        # 3. 把 supports_hypothesis 写入消息字段
        msg = make_evidence_response(
            sender=sender,
            receiver=receiver,
            request_message=request_message,
            evidence=evidence,
            supports_hypothesis=supports_hypothesis,
            content=content,
            confidence=confidence,
        )
        logger.debug(f"[Bus] {sender} → {receiver}: [evidence_response] {msg.get('message_id')}")
        return [msg]

    def receive(
        self,
        agent_name: str,
        state_messages: list[dict],
        msg_types: Optional[set[str] | list[str]] = None,
        *,
        include_broadcast: bool = True,
        exclude_self: bool = True,
    ) -> list[dict]:
        """
        获取发给指定 Agent 的消息

        Args:
            agent_name: 接收者名称
            state_messages: state.agent_messages 中的已有消息

        Returns:
            发给该 Agent 的消息列表（包括广播消息）
        """
        allowed_types = set(msg_types) if msg_types else None
        received = []
        for msg in normalize_messages(state_messages):
            if allowed_types and msg.get("msg_type") not in allowed_types:
                continue
            is_receiver = msg.get("receiver") == agent_name
            is_broadcast = include_broadcast and msg.get("receiver") == "broadcast"
            if is_receiver or is_broadcast:
                if exclude_self and msg.get("sender") == agent_name:
                    continue
                received.append(msg)
        return received
