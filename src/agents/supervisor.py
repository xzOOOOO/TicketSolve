"""
SupervisorAgent - 智能调度主管

替代原 Router 的单路由决策，支持:
- 分析故障现象，决定派发哪些 Agent
- 可以并行派发多个 Agent
- 输出 dispatch 列表供并行派发节点使用
"""

from langchain_core.language_models import BaseChatModel
from agents.base import BaseAgent
from state import SystemState
from prompts import SUPERVISOR_PROMPT
from utils import parse_json_content
from logger import logger


class SupervisorAgent(BaseAgent):
    name = "supervisor"
    role = "智能调度主管"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, tools=None)

    async def run(self, state: SystemState) -> dict:
        try:
            logger.info(f"[{self.name}] 开始分析: symptom={state.symptom[:50]}...")

            response = await (SUPERVISOR_PROMPT | self.llm).ainvoke(
                {"symptom": state.symptom}
            )
            result = parse_json_content(response.content) or {
                "diagnosis_type": "other",
                "urgency": "medium",
                "dispatch": [],
                "reasoning": "解析失败，使用默认值",
            }

            diagnosis_type = result.get("diagnosis_type", "other")
            urgency = result.get("urgency", "medium")
            dispatch = result.get("dispatch", [])
            reasoning = result.get("reasoning", "")

            if not dispatch:
                if diagnosis_type == "db":
                    dispatch = ["db_agent"]
                elif diagnosis_type == "net":
                    dispatch = ["net_agent"]
                elif diagnosis_type == "app":
                    dispatch = ["app_agent"]
                else:
                    dispatch = []

            valid_agents = {"db_agent", "net_agent", "app_agent"}
            dispatch = [a for a in dispatch if a in valid_agents]

            logger.info(
                f"[{self.name}] 决策完成: type={diagnosis_type}, "
                f"urgency={urgency}, dispatch={dispatch}, "
                f"reasoning={reasoning[:80]}"
            )

            return {
                "diagnosis_type": diagnosis_type,
                "urgency": urgency,
                "supervisor_decision": result,
                "dispatched_agents": dispatch,
                "messages": [
                    f"Supervisor: 诊断={diagnosis_type}, 紧急度={urgency}, "
                    f"派发={dispatch}"
                ],
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
