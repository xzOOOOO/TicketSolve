"""
RouterAgent - 工单路由分类 Agent

改造说明:
- 使用 utils.parse_json_content 消除重复代码
- 无需 ReAct 循环（不调用工具）
- 返回格式与原节点完全一致
"""

from langchain_core.language_models import BaseChatModel
from agents.base import BaseAgent
from state import SystemState
from prompts import ROUTER_PROMPT
from utils import parse_json_content
from logger import logger


class RouterAgent(BaseAgent):
    name = "router"
    role = "工单路由分类Agent"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, tools=None)

    async def run(self, state: SystemState) -> dict:
        try:
            logger.info(f"[{self.name}] 开始分析: symptom={state.symptom[:50]}...")
            response = await (ROUTER_PROMPT | self.llm).ainvoke({"symptom": state.symptom})
            result = parse_json_content(response.content) or {
                "diagnosis_type": "other",
                "urgency": "medium",
            }

            diagnosis_type = result.get("diagnosis_type", "other")
            urgency = result.get("urgency", "medium")

            logger.info(
                f"[{self.name}] 分析完成: diagnosis_type={diagnosis_type}, urgency={urgency}"
            )

            return {
                "diagnosis_type": diagnosis_type,
                "urgency": urgency,
                "messages": [f"Router: 诊断={diagnosis_type}, 紧急度={urgency}"],
            }
        except Exception as e:
            logger.exception(f"[{self.name}] 执行失败: {e}")
            return {
                "diagnosis_type": "other",
                "urgency": "medium",
                "messages": ["Router: 分析失败，使用默认值"],
            }
