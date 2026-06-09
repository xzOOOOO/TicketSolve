"""
诊断类 Agent 公共模板。

本模块收敛 DBAgent、NetAgent、AppAgent 的共同执行流程：
1. 接收协作消息
2. ReAct 调用诊断工具
3. 结构化生成诊断结论
4. 记录审计日志和 Trace
5. 发布协作协议消息
"""

# BaseChatModel：LangChain 聊天模型基类，用于结构化诊断生成
from langchain_core.language_models import BaseChatModel
# ChatPromptTemplate：LangChain Prompt 模板类型，用于工具调用阶段和诊断阶段
from langchain_core.prompts import ChatPromptTemplate
# BaseTool：LangChain 工具基类，用于注入 MCP 诊断工具
from langchain_core.tools import BaseTool

# agent_protocol：证据协作协议工具函数
from agent_protocol import build_protocol_context, collaboration_requests_from_result
# BaseAgent：所有 Agent 的基础类，提供 ReAct 工具调用循环
from agents.base import BaseAgent
# CommunicationBus：Agent 间通信总线
from agents.communication import CommunicationBus
# settings：运行时配置，提供 LLM 重试策略
from config import settings
# normalize_evidence_items：统一 LLM 输出证据格式
from evidence import normalize_evidence_items
# logger：项目统一日志记录器
from logger import logger
# DiagnosisOutput：诊断阶段结构化输出模型
from schemas import DiagnosisOutput
# SystemState：工作流全局状态类型
from state import SystemState
# make_trace_event：标准化 Trace 事件工厂
from trace_events import make_trace_event


# VALID_COLLABORATION_AGENTS：允许参与协作的 Agent 名称白名单
VALID_COLLABORATION_AGENTS = {"db_agent", "net_agent", "app_agent"}


class DiagnosticAgent(BaseAgent):
    """
    诊断类 Agent 模板。

    用途：
        为数据库、网络、应用三个领域 Agent 提供统一的 run() 实现。

    子类属性：
        tool_prompt: 工具调用阶段 Prompt
        diagnosis_prompt: 结构化诊断阶段 Prompt
        result_field: 写回 SystemState 的结果字段名
        domain_label: 中文领域名称，用于审计和协作消息
        message_label: 用户可见消息前缀
    """

    # tool_prompt：ReAct 工具调用阶段使用的 Prompt，由子类覆盖
    tool_prompt: ChatPromptTemplate | None = None
    # diagnosis_prompt：结构化诊断阶段使用的 Prompt，由子类覆盖
    diagnosis_prompt: ChatPromptTemplate | None = None
    # result_field：诊断结果写入 SystemState 的字段名，由子类覆盖
    result_field: str = ""
    # domain_label：中文领域名称，用于协作消息和审计日志
    domain_label: str = ""
    # message_label：状态消息前缀，用于前端展示
    message_label: str = ""
    # max_react_iterations：ReAct 最大轮数，防止工具调用无限循环
    max_react_iterations: int = 3

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        communication_bus: CommunicationBus | None = None,
    ):
        """
        初始化诊断 Agent。

        参数：
            llm: LangChain 聊天模型实例
            tools: 当前领域可用的 MCP 诊断工具
            communication_bus: Agent 间通信总线，为 None 时不发送协作消息

        返回：
            None
        """
        super().__init__(llm, tools)
        # bus：通信总线实例，用于接收和发送协作协议消息
        self.bus = communication_bus
        # structured_llm：强制 LLM 输出 DiagnosisOutput，避免自由文本解析
        self._structured_llm = self.llm.with_structured_output(DiagnosisOutput)
        # diagnosis_chain：诊断 Prompt 与结构化 LLM 组合，并应用统一重试策略
        self._diagnosis_chain = (
            self._require_diagnosis_prompt() | self._structured_llm
        ).with_retry(**settings.get_retry_config())

    async def run(self, state: SystemState) -> dict:
        """
        执行诊断流程。

        参数：
            state: 当前工作流全局状态

        返回：
            状态更新字典，包含诊断结果、消息、审计日志、Trace 事件和协作消息

        异常说明：
            捕获内部异常并转为失败状态更新，避免单个 Agent 中断工作流。
        """
        # trace_events：当前 Agent 产生的标准化 Trace 事件列表
        trace_events = [self._make_agent_started_event(state)]
        try:
            logger.info(f"[{self.name}] 开始诊断: symptom={state.symptom[:50]}...")

            # peer_messages：来自其他 Agent 的协作上下文文本
            peer_messages = self._receive_peer_messages(state)
            # tool_results/tool_calls_info：ReAct 阶段收集到的工具结果和调用信息
            tool_results, tool_calls_info = await self.react_loop(
                self._require_tool_prompt(),
                state.symptom,
                max_iterations=self.max_react_iterations,
            )
            # result_dict：结构化诊断结果字典，写入状态和审计日志
            result_dict = await self._build_diagnosis_result(
                state=state,
                tool_calls_info=tool_calls_info,
                tool_results=tool_results,
                peer_messages=peer_messages,
            )

            logger.info(f"[{self.name}] 诊断完成: diagnosis={result_dict.get('diagnosis')}")

            # audit_logs：本次诊断产生的审计日志列表
            audit_logs = self._build_audit_logs(
                state=state,
                result_dict=result_dict,
                tool_calls_info=tool_calls_info,
                tool_results=tool_results,
                peer_messages=peer_messages,
            )
            # trace_events：追加工具调用、观测结果和诊断结论 Trace
            trace_events.extend(self._build_diagnosis_trace_events(
                state=state,
                result_dict=result_dict,
                tool_calls_info=tool_calls_info,
                tool_results=tool_results,
                peer_messages=peer_messages,
            ))

            # update：写回 LangGraph 状态的增量字典
            update = {
                self.result_field: {**result_dict, "tool_results": tool_results},
                "messages": [f"{self.message_label}: {result_dict.get('diagnosis')}"],
                "audit_logs": audit_logs,
                "trace_events": trace_events,
            }

            # collaboration_requests：结构化协作请求，用于生成 Agent 间协议消息
            collaboration_requests = collaboration_requests_from_result(result_dict)
            # agent_messages：通过通信总线生成的协议消息
            agent_messages = self._build_agent_messages(
                state=state,
                result_dict=result_dict,
                collaboration_requests=collaboration_requests,
                trace_events=trace_events,
            )
            # 如果存在协作消息，则追加到状态更新中
            if agent_messages:
                update["agent_messages"] = agent_messages

            return update
        except Exception as exc:
            # Agent 异常时记录失败 Trace，保证失败路径仍可追溯
            logger.exception(f"[{self.name}] 执行失败: {exc}")
            trace_events.append(make_trace_event(
                "diagnosis_generated",
                ticket_id=state.ticket_id,
                agent_name=self.name,
                status="failure",
                input_data={"symptom": state.symptom},
                error=str(exc),
                metadata={"dispatch_round": state.dispatch_round},
            ))
            return {
                self.result_field: {"diagnosis": "诊断失败", "possible_causes": [str(exc)]},
                "messages": [f"{self.message_label}: 诊断失败 - {str(exc)}"],
                "trace_events": trace_events,
            }

    def _require_tool_prompt(self) -> ChatPromptTemplate:
        """
        获取工具调用 Prompt。

        参数：
            无

        返回：
            ChatPromptTemplate 工具调用 Prompt

        异常说明：
            当子类未配置 tool_prompt 时抛出 ValueError。
        """
        # 如果子类没有配置工具 Prompt，则说明 Agent 配置不完整
        if self.tool_prompt is None:
            raise ValueError(f"{self.name} 未配置 tool_prompt")
        return self.tool_prompt

    def _require_diagnosis_prompt(self) -> ChatPromptTemplate:
        """
        获取结构化诊断 Prompt。

        参数：
            无

        返回：
            ChatPromptTemplate 结构化诊断 Prompt

        异常说明：
            当子类未配置 diagnosis_prompt 时抛出 ValueError。
        """
        # 如果子类没有配置诊断 Prompt，则说明 Agent 配置不完整
        if self.diagnosis_prompt is None:
            raise ValueError(f"{self.name} 未配置 diagnosis_prompt")
        return self.diagnosis_prompt

    def _make_agent_started_event(self, state: SystemState) -> dict:
        """
        创建 Agent 启动 Trace 事件。

        参数：
            state: 当前工作流状态

        返回：
            标准化 Trace 事件字典
        """
        return make_trace_event(
            "agent_started",
            ticket_id=state.ticket_id,
            agent_name=self.name,
            input_data={"symptom": state.symptom},
            metadata={"dispatch_round": state.dispatch_round},
        )

    def _receive_peer_messages(self, state: SystemState) -> str:
        """
        接收其他 Agent 发来的协作消息。

        参数：
            state: 当前工作流状态

        返回：
            协作上下文文本；没有消息时返回空字符串
        """
        # 没有通信总线或状态里没有消息时，直接返回空上下文
        if not self.bus or not state.agent_messages:
            return ""

        # incoming：当前 Agent 可见的协作消息列表
        incoming = self.bus.receive(self.name, state.agent_messages)
        # 没有可见消息时，直接返回空上下文
        if not incoming:
            return ""

        logger.info(f"[{self.name}] 收到 {len(incoming)} 条消息")
        # protocol_context：协议上下文，包含可读文本和结构化摘要
        protocol_context = build_protocol_context(incoming)
        return protocol_context["text"]

    async def _build_diagnosis_result(
        self,
        state: SystemState,
        tool_calls_info: list[dict],
        tool_results: list[dict],
        peer_messages: str,
    ) -> dict:
        """
        调用结构化 LLM 生成诊断结果。

        参数：
            state: 当前工作流状态
            tool_calls_info: ReAct 工具调用信息
            tool_results: ReAct 工具返回结果
            peer_messages: 其他 Agent 的协作上下文

        返回：
            标准化诊断结果字典
        """
        # result：LLM 结构化输出对象，可能因异常兼容而为空
        result = await self._diagnosis_chain.ainvoke({
            "symptom": state.symptom,
            "tool_calls": str(tool_calls_info),
            "tool_results": str(tool_results),
            "peer_messages": peer_messages or "无",
        })
        # 如果 LLM 返回空结果，则使用固定兜底诊断对象
        if result is None:
            result = DiagnosisOutput(
                diagnosis="无法解析",
                possible_causes=[],
                confidence=0.0,
                collaboration_requests=[],
            )

        # result_dict：Pydantic 模型转字典，便于写入状态和持久化
        result_dict = result.model_dump()
        # evidence：统一证据结构，保证协议层和前端展示格式一致
        result_dict["evidence"] = normalize_evidence_items(
            result_dict.get("evidence") or result_dict.get("possible_causes", []),
            source_agent=self.name,
            supports_hypothesis=True,
            confidence=float(result_dict.get("confidence") or 0.0),
        )
        return result_dict

    def _build_diagnosis_trace_events(
        self,
        state: SystemState,
        result_dict: dict,
        tool_calls_info: list[dict],
        tool_results: list[dict],
        peer_messages: str,
    ) -> list[dict]:
        """
        构造诊断阶段 Trace 事件。

        参数：
            state: 当前工作流状态
            result_dict: 结构化诊断结果
            tool_calls_info: 工具调用信息
            tool_results: 工具返回结果
            peer_messages: 协作上下文文本

        返回：
            Trace 事件列表
        """
        # trace_events：当前方法构造的 Trace 事件列表
        trace_events = []
        # 如果有工具调用，则记录 tool_called 事件
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
        # 如果有工具返回结果，则记录 observation_received 事件
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
        return trace_events

    def _build_audit_logs(
        self,
        state: SystemState,
        result_dict: dict,
        tool_calls_info: list[dict],
        tool_results: list[dict],
        peer_messages: str,
    ) -> list[dict]:
        """
        构造诊断审计日志。

        参数：
            state: 当前工作流状态
            result_dict: 结构化诊断结果
            tool_calls_info: 工具调用信息
            tool_results: 工具返回结果
            peer_messages: 协作上下文文本

        返回：
            审计日志字典列表
        """
        # audit_logs：本次诊断产生的审计日志列表
        audit_logs = []
        # 如果有工具调用，则记录工具调用审计日志
        if tool_calls_info:
            audit_logs.append({
                "ticket_id": state.ticket_id,
                "agent_name": self.name,
                "action_type": "tool_call",
                "action_detail": {
                    "tools_called": [
                        tool_call.get("name", tool_call.get("tool", "unknown"))
                        for tool_call in tool_calls_info
                    ],
                    "tool_results_summary": [
                        {
                            "tool": tool_result.get("name", tool_result.get("tool", "unknown")),
                            "status": "success" if "error" not in str(tool_result).lower() else "error",
                        }
                        for tool_result in tool_results
                    ],
                },
                "input_context": {"symptom": state.symptom, "peer_messages": peer_messages or "无"},
                "output_result": {"tool_results": tool_results},
                "dispatch_round": state.dispatch_round,
            })

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

        # collaboration_requests：LLM 诊断结果里的结构化协作请求
        collaboration_requests = collaboration_requests_from_result(result_dict)
        # 遍历协作请求，把有效请求写入审计日志
        for request in collaboration_requests:
            # target：协作请求目标 Agent 名称
            target = request["target_agent"]
            # 只记录白名单内且不是自己的协作请求
            if target in VALID_COLLABORATION_AGENTS and target != self.name:
                audit_logs.append({
                    "ticket_id": state.ticket_id,
                    "agent_name": self.name,
                    "action_type": "collaborate",
                    "action_detail": {
                        "target_agent": target,
                        "reason": f"{self.domain_label}诊断发现可能涉及{target}领域的问题",
                        "content": (
                            f"{self.domain_label}诊断发现可能涉及{target}领域的问题: "
                            f"{result_dict.get('diagnosis')}，请协助确认"
                        ),
                    },
                    "input_context": {"diagnosis": result_dict.get("diagnosis")},
                    "output_result": {"request_sent": True},
                    "dispatch_round": state.dispatch_round,
                })
        return audit_logs

    def _build_agent_messages(
        self,
        state: SystemState,
        result_dict: dict,
        collaboration_requests: list[dict],
        trace_events: list[dict],
    ) -> list[dict]:
        """
        构造 Agent 间协作协议消息。

        参数：
            state: 当前工作流状态
            result_dict: 结构化诊断结果
            collaboration_requests: 结构化协作请求列表
            trace_events: 当前 Agent 的 Trace 事件列表，会原地追加 handoff 事件

        返回：
            可追加到 state.agent_messages 的协议消息列表
        """
        # 如果没有通信总线，则不进入协作模式
        if not self.bus:
            return []

        # agent_messages：本次诊断发布的协议消息列表
        agent_messages = []
        # hypothesis_messages：结构化故障假设消息，是后续证据请求的关联源头
        hypothesis_messages = self.bus.publish_hypothesis(
            sender=self.name,
            content=f"{self.domain_label}诊断假设: {result_dict.get('hypothesis') or result_dict.get('diagnosis')}",
            hypothesis=result_dict.get("hypothesis") or result_dict.get("diagnosis", ""),
            fault_type=result_dict.get("fault_type"),
            confidence=result_dict.get("confidence", 0.0),
            evidence=result_dict.get("evidence") or result_dict.get("possible_causes", []),
        )
        agent_messages.extend(hypothesis_messages)
        # hypothesis_message：后续 evidence_request 需要引用的假设消息
        hypothesis_message = hypothesis_messages[0]

        agent_messages.extend(self.bus.broadcast(
            sender=self.name,
            content=f"诊断结论: {result_dict.get('diagnosis')}，可能原因: {result_dict.get('possible_causes', [])}",
            msg_type="diagnosis",
            confidence=result_dict.get("confidence", 0.0),
            evidence=result_dict.get("evidence") or result_dict.get("possible_causes", []),
            hypothesis=result_dict.get("hypothesis"),
            fault_type=result_dict.get("fault_type"),
        ))

        # 遍历协作请求，向目标 Agent 发送证据请求
        for request in collaboration_requests:
            # target：协作请求目标 Agent 名称
            target = request["target_agent"]
            # 只允许向白名单内的其他 Agent 发证据请求
            if target in VALID_COLLABORATION_AGENTS and target != self.name:
                # request_messages：通信总线生成的一组 evidence_request 消息
                request_messages = self.bus.request_evidence(
                    sender=self.name,
                    receiver=target,
                    hypothesis_message=hypothesis_message,
                    required_evidence=request.get("required_evidence", []),
                    reason=request.get("reason") or f"{self.domain_label}诊断需要 {target} 补充证据",
                    suggested_tools=request.get("suggested_tools", []),
                    confidence=result_dict.get("confidence", 0.0),
                )
                agent_messages.extend(request_messages)
                # 每条 evidence_request 都记录一个 handoff_requested Trace 事件
                for message in request_messages:
                    trace_events.append(make_trace_event(
                        "handoff_requested",
                        ticket_id=state.ticket_id,
                        agent_name=self.name,
                        input_data={"diagnosis": result_dict.get("diagnosis")},
                        output_data=message,
                        metadata={
                            "dispatch_round": state.dispatch_round,
                            "message_id": message.get("message_id"),
                            "correlation_id": message.get("correlation_id"),
                            "msg_type": message.get("msg_type"),
                        },
                    ))

        return agent_messages
