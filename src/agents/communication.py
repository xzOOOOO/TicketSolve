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

为什么需要 CommunicationBus：
多 Agent 协作时，Agent A 可能需要 Agent B 的协助（如"请检查网络连通性"）。
CommunicationBus 提供了标准化的消息格式和接口，避免各 Agent 自行构造消息字典，
确保消息格式统一、可追溯、可审计。
"""

# Any：任意类型；Optional：可选类型（可为 None）
from typing import Any, Optional
# normalize_evidence_items：把字符串/字典混合格式统一成 EvidenceItem 对象列表
from evidence import normalize_evidence_items
# AgentMessage：Pydantic 消息模型，定义了 Agent 间通信的标准消息格式
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
# logger：项目统一日志记录器
from logger import logger


class CommunicationBus:
    """
    Agent 间通信总线（无状态设计）

    使用方式:
    1. Agent 诊断过程中发现需要其他 Agent 协助 → bus.send()，返回值写入 state.agent_messages
    2. Agent 诊断完成后广播结论 → bus.broadcast()，返回值写入 state.agent_messages
    3. Agent 开始诊断前检查是否有其他 Agent 的消息 → bus.receive()

    设计要点：
    - 无状态：不维护内部缓冲区，所有消息通过 state.agent_messages 流转
    - 消息追加式写入（operator.add），不覆盖已有消息
    - 所有方法返回 list[dict]，可直接合并到 state.agent_messages
    """

    def send(self, sender: str, receiver: str, content: str,
             msg_type: str = "info", confidence: float = 0.0,
             evidence: Optional[list[Any]] = None, **protocol_fields) -> list[dict]:
        """
        发送消息给指定 Agent

        参数：
            sender: 发送者名称（如 "db_agent"）
            receiver: 接收者名称（如 "net_agent"）
            content: 消息内容（人可读的文本）
            msg_type: 消息类型
                - hypothesis：故障假设
                - evidence_request：证据请求
                - evidence_response：证据响应
                - challenge：质疑
                - support：支持
                - diagnosis：诊断结论
                - info：普通信息
            confidence: 置信度 0-1
            evidence: 支撑证据列表
            **protocol_fields: 额外的协议字段（如 supports_hypothesis、hypothesis_id 等）

        返回：
            可写入 state.agent_messages 的消息列表（单元素列表）
        """
        # AgentMessage：Pydantic 模型，自动校验字段类型和必填项
        msg = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            msg_type=msg_type,
            confidence=confidence,
            # normalize_evidence_items：统一证据格式，确保每条证据都有 source_agent、confidence 等字段
            evidence=normalize_evidence_items(
                evidence or [],
                source_agent=sender,
                supports_hypothesis=protocol_fields.get("supports_hypothesis"),
                confidence=confidence,
            ),
            **protocol_fields,
        )
        # model_dump()：Pydantic v2 方法，将模型转为字典（兼容 JSON 序列化）
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

        参数：
            agent_name: 接收者名称（如 "db_agent"）
            state_messages: state.agent_messages 中的已有消息列表
            msg_types: 过滤的消息类型集合/列表（如 {"evidence_request", "hypothesis"}）
            include_broadcast: 是否包含广播消息（receiver="broadcast"）
            exclude_self: 是否排除自己发给自己的消息

        返回：
            发给该 Agent 的消息列表（包括广播消息，如果 include_broadcast=True）
        """
        # allowed_types：将 msg_types 转为集合，提高查找效率
        allowed_types = set(msg_types) if msg_types else None
        received = []
        # normalize_messages：归一化消息格式，处理旧消息可能缺失的字段
        for msg in normalize_messages(state_messages):
            # 按消息类型过滤
            if allowed_types and msg.get("msg_type") not in allowed_types:
                continue
            # is_receiver：消息直接发给该 Agent
            is_receiver = msg.get("receiver") == agent_name
            # is_broadcast：消息是广播，且允许接收广播
            is_broadcast = include_broadcast and msg.get("receiver") == "broadcast"
            # 匹配接收者或广播
            if is_receiver or is_broadcast:
                # exclude_self：排除自己发给自己的消息（避免自循环）
                if exclude_self and msg.get("sender") == agent_name:
                    continue
                received.append(msg)
        return received
