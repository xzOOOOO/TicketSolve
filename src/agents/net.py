"""
NetAgent - 网络诊断专家

改造说明:
- 使用 BaseAgent.react_loop() 实现多轮工具调用
- 使用 utils.parse_json_content / execute_tool_calls 消除重复代码
- 返回格式与原节点完全一致
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from agents.base import BaseAgent
from state import SystemState
from prompts import NET_PROMPT, NET_DIAGNOSIS_PROMPT
from utils import parse_json_content
from logger import logger


class NetAgent(BaseAgent):
    name = "net_agent"
    role = "网络诊断专家"

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        super().__init__(llm, tools)

    async def run(self, state: SystemState) -> dict:
        try:
            logger.info(f"[{self.name}] 开始诊断: symptom={state.symptom[:50]}...")

            tool_results, tool_calls_info = await self.react_loop(
                NET_PROMPT, state.symptom, max_iterations=3
            )

            diagnosis = await (NET_DIAGNOSIS_PROMPT | self.llm).ainvoke({
                "symptom": state.symptom,
                "tool_calls": str(tool_calls_info),
                "tool_results": str(tool_results),
            })
            result = parse_json_content(diagnosis.content) or {
                "diagnosis": "无法解析",
                "possible_causes": [],
            }

            logger.info(f"[{self.name}] 诊断完成: diagnosis={result.get('diagnosis')}")
            return {
                "net_agent_result": {**result, "tool_results": tool_results},
                "messages": [f"Net Agent (MCP): {result.get('diagnosis')}"],
            }
        except Exception as e:
            logger.exception(f"[{self.name}] 执行失败: {e}")
            return {
                "net_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"Net Agent: 诊断失败 - {str(e)}"],
            }
