# BaseChatModel：LangChain 聊天模型基类
from langchain_core.language_models import BaseChatModel
# BaseAgent：Agent 抽象基类
from agents.base import BaseAgent
# SystemState：工作流全局状态
from state import SystemState
# SUPERVISOR_PROMPT：Supervisor 的 LLM Prompt 模板
from prompts import SUPERVISOR_PROMPT
# SupervisorDecisionOutput：Supervisor 决策的结构化输出模型
from schemas import SupervisorDecisionOutput
# make_trace_event：标准化 Trace 事件工厂
from trace_events import make_trace_event
# logger：项目统一日志记录器
from logger import logger
# settings：项目配置对象
from config import settings


class SupervisorAgent(BaseAgent):
    """智能调度主管 Agent

    职责：分析故障现象，决定派发哪些诊断 Agent 去调查。
    """
    name = "supervisor"
    role = "智能调度主管"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, tools=None)
        # 在初始化时创建结构化 LLM，避免每次调用都重复创建
        # with_structured_output 会将 Pydantic 模型转换为 JSON Schema
        # 通过 function calling 机制约束 LLM 的输出格式
        self._structured_llm = self.llm.with_structured_output(SupervisorDecisionOutput)
        self._chain = (SUPERVISOR_PROMPT | self._structured_llm).with_retry(**settings.get_retry_config())

    async def run(self, state: SystemState) -> dict:
        """执行 Supervisor 调度决策

        流程：
        1. 调用 LLM 分析故障现象（使用 Structured Output）
        2. 获取 diagnosis_type、urgency、dispatch、reasoning
        3. 如果 dispatch 为空，根据 diagnosis_type 推断默认派发
        4. 过滤无效的 Agent 名称
        5. 返回状态更新字典
        """
        # 初始化 Trace 事件列表，首先记录 agent_started 事件
        trace_events = [make_trace_event(
            "agent_started",
            ticket_id=state.ticket_id,
            agent_name=self.name,
            input_data={
                "symptom": state.symptom,
                "case_context": state.case_context,
            },
            metadata={
                "dispatch_round": state.dispatch_round,
                "similar_case_ids": [case.get("case_id") for case in state.similar_cases],
            },
        )]
        try:
            logger.info(f"[{self.name}] 开始分析: symptom={state.symptom[:50]}...")

            # 使用 Structured Output 调用 LLM，直接返回 SupervisorDecisionOutput 对象
            # 无需再手动解析 JSON 字符串
            result = await self._chain.ainvoke(
                {
                    "symptom": state.symptom,
                    "case_context": state.case_context or "无相似历史案例。",
                }
            )

            # 兜底：极少数情况下 with_structured_output 可能返回 None
            if result is None:
                result = SupervisorDecisionOutput(
                    diagnosis_type="other",
                    urgency="medium",
                    dispatch=[],
                    reasoning="Structured Output 解析失败，使用默认值",
                )

            # Pydantic 对象转 dict，保持与 SystemState 的兼容性
            result_dict = result.model_dump()
            diagnosis_type = result.diagnosis_type
            urgency = result.urgency
            dispatch = result.dispatch
            reasoning = result.reasoning

            # 过滤非法的 diagnosis_type，防止 LLM 输出 unknown 等无效值导致 SystemState 校验失败
            valid_types = {"app", "db", "net", "other"}
            if diagnosis_type not in valid_types:
                logger.warning(
                    f"[{self.name}] LLM 返回非法 diagnosis_type='{diagnosis_type}'，强制修正为 'other'"
                )
                diagnosis_type = "other"
                result_dict["diagnosis_type"] = "other"

            # 如果 LLM 没有给出 dispatch 列表，根据诊断类型推断默认值
            if not dispatch:
                if diagnosis_type == "db":
                    dispatch = ["db_agent"]
                elif diagnosis_type == "net":
                    dispatch = ["net_agent"]
                elif diagnosis_type == "app":
                    dispatch = ["app_agent"]
                else:
                    dispatch = []

            # 过滤无效的 Agent 名称，防止 LLM  hallucination
            valid_agents = {"db_agent", "net_agent", "app_agent"}
            dispatch = [a for a in dispatch if a in valid_agents]

            logger.info(
                f"[{self.name}] 决策完成: type={diagnosis_type}, "
                f"urgency={urgency}, dispatch={dispatch}, "
                f"reasoning={reasoning[:80]}"
            )

            # 记录审计日志：Supervisor 的调度决策
            audit_log = {
                "ticket_id": state.ticket_id,
                "agent_name": self.name,
                "action_type": "dispatch",
                "action_detail": {
                    "diagnosis_type": diagnosis_type,
                    "urgency": urgency,
                    "dispatched_agents": dispatch,
                    "reasoning": reasoning,
                },
                "input_context": {
                    "symptom": state.symptom,
                    "case_context": state.case_context,
                    "similar_case_ids": [case.get("case_id") for case in state.similar_cases],
                },
                "output_result": result_dict,
                "dispatch_round": state.dispatch_round,
            }
            # 生成诊断完成和交接请求的标准化 Trace 事件
            trace_events.extend([
                make_trace_event(
                    "diagnosis_generated",
                    ticket_id=state.ticket_id,
                    agent_name=self.name,
                    input_data={"symptom": state.symptom},
                    output_data=result_dict,
                    metadata={
                        "diagnosis_type": diagnosis_type,
                        "urgency": urgency,
                        "dispatch_round": state.dispatch_round,
                    },
                ),
                make_trace_event(
                    "handoff_requested",
                    ticket_id=state.ticket_id,
                    agent_name=self.name,
                    status="success" if dispatch else "skipped",
                    input_data={"diagnosis_type": diagnosis_type},
                    output_data={"dispatched_agents": dispatch, "reasoning": reasoning},
                    metadata={
                        "handoff_count": len(dispatch),
                        "dispatch_round": state.dispatch_round,
                    },
                ),
            ])

            return {
                "diagnosis_type": diagnosis_type,
                "urgency": urgency,
                "supervisor_decision": result_dict,
                "dispatched_agents": dispatch,
                "messages": [
                    f"Supervisor: 诊断={diagnosis_type}, 紧急度={urgency}, "
                    f"派发={dispatch}"
                ],
                "audit_logs": [audit_log],
                "trace_events": trace_events,
            }
        except Exception as e:
            # Supervisor 异常时也要记录 failure 状态的 Trace 事件，保证失败路径可追溯
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
                "diagnosis_type": "other",
                "urgency": "medium",
                "supervisor_decision": {},
                "dispatched_agents": [],
                "messages": ["Supervisor: 分析失败，使用默认值"],
                # 异常路径也要返回 trace_events，避免前面已记录的事件丢失
                "trace_events": trace_events,
            }
