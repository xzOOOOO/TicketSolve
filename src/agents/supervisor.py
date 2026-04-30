from langchain_core.language_models import BaseChatModel
from agents.base import BaseAgent
from state import SystemState
from prompts import SUPERVISOR_PROMPT
from schemas import SupervisorDecisionOutput
from logger import logger


class SupervisorAgent(BaseAgent):
    """智能调度主管 Agent

    职责：分析故障现象，决定派发哪些诊断 Agent 去调查。

    Structured Output 改造说明：
    - 原方案：LLM 输出 JSON 字符串 -> parse_json_content 手动解析 -> dict
    - 新方案：self.llm.with_structured_output(SupervisorDecisionOutput)
             LLM 通过 function calling 直接返回 Pydantic 对象
    - 优势：类型安全、零解析失败、Prompt 更简洁
    """
    name = "supervisor"
    role = "智能调度主管"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, tools=None)
        # 在初始化时创建结构化 LLM，避免每次调用都重复创建
        # with_structured_output 会将 Pydantic 模型转换为 JSON Schema
        # 通过 function calling 机制约束 LLM 的输出格式
        self._structured_llm = self.llm.with_structured_output(SupervisorDecisionOutput)

    async def run(self, state: SystemState) -> dict:
        """执行 Supervisor 调度决策

        流程：
        1. 调用 LLM 分析故障现象（使用 Structured Output）
        2. 获取 diagnosis_type、urgency、dispatch、reasoning
        3. 如果 dispatch 为空，根据 diagnosis_type 推断默认派发
        4. 过滤无效的 Agent 名称
        5. 返回状态更新字典
        """
        try:
            logger.info(f"[{self.name}] 开始分析: symptom={state.symptom[:50]}...")

            # 使用 Structured Output 调用 LLM，直接返回 SupervisorDecisionOutput 对象
            # 无需再手动解析 JSON 字符串
            result = await (SUPERVISOR_PROMPT | self._structured_llm).ainvoke(
                {"symptom": state.symptom}
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
                "input_context": {"symptom": state.symptom},
                "output_result": result_dict,
                "dispatch_round": state.dispatch_round,
            }

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
            }
        except Exception as e:
            logger.exception(f"[{self.name}] 执行失败: {e}")
            return {
                "diagnosis_type": "other",
                "urgency": "medium",
                "supervisor_decision": {},
                "dispatched_agents": [],
                "messages": ["Supervisor: 分析失败，使用默认值"],
            }
