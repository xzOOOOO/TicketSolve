"""
FixAgent - 修复方案生成专家

改造说明:
- 使用 utils.parse_json_content 消除重复代码
- 无需 ReAct 循环（不调用工具）
- 返回格式与原节点完全一致
"""

from langchain_core.language_models import BaseChatModel
from agents.base import BaseAgent
from state import SystemState
from prompts import FIX_PROMPT
from utils import parse_json_content
from logger import logger

_DEFAULT_FIX_PLAN = {
    "plan_id": "PLAN-ERROR",
    "description": "无法生成方案",
    "risk_level": "unknown",
    "prerequisites": [],
    "steps": [],
    "verification": {"commands": [], "expected_result": ""},
    "estimated_time": "0",
}


class FixAgent(BaseAgent):
    name = "fix_agent"
    role = "修复方案生成专家"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, tools=None)

    async def run(self, state: SystemState) -> dict:
        try:
            diagnosis_type = state.diagnosis_type
            if diagnosis_type == "db":
                diagnosis_result = state.db_agent_result
            elif diagnosis_type == "net":
                diagnosis_result = state.net_agent_result
            elif diagnosis_type == "app":
                diagnosis_result = state.app_agent_result
            else:
                diagnosis_result = {}

            logger.info(f"[{self.name}] 开始生成修复方案: diagnosis_type={diagnosis_type}")

            response = await (FIX_PROMPT | self.llm).ainvoke({
                "diagnosis_type": diagnosis_type,
                "diagnosis_result": str(diagnosis_result),
            })
            result = parse_json_content(response.content) or _DEFAULT_FIX_PLAN.copy()

            logger.info(
                f"[{self.name}] 方案生成完成: plan_id={result.get('plan_id')}, "
                f"risk_level={result.get('risk_level')}"
            )

            return {
                "fix_plan": result,
                "messages": [
                    f"Fix Agent: 生成修复方案 {result.get('plan_id')} - "
                    f"风险等级: {result.get('risk_level')}"
                ],
            }
        except Exception as e:
            logger.exception(f"[{self.name}] 执行失败: {e}")
            return {
                "fix_plan": _DEFAULT_FIX_PLAN.copy(),
                "messages": [f"Fix Agent: 方案生成失败 - {str(e)}"],
            }
