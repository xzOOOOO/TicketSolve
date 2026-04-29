"""
BaseAgent - 所有 Agent 的抽象基类

设计原则:
- 每个 Agent 拥有独立身份（name, role）
- Agent 通过 run() 方法被 LangGraph 节点调用
- 返回值格式与原 nodes.py 完全一致，确保兼容
- 子类只需实现 run() 方法
- 诊断类 Agent 可使用 react_loop() 实现多轮工具调用
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from utils import execute_tool_calls
from logger import logger


class BaseAgent(ABC):
    """
    Agent 抽象基类

    Attributes:
        name: Agent 唯一标识，用于日志和通信
        role: Agent 角色描述，用于 prompt 和身份识别
        llm: 独立的 LLM 实例
        tools: Agent 可用的工具列表
    """

    name: str = "base_agent"
    role: str = "基础Agent"

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[list[BaseTool]] = None,
    ):
        self.llm = llm
        self.tools = tools or []
        logger.debug(f"Agent [{self.name}] 初始化完成, 工具数: {len(self.tools)}")

    @abstractmethod
    async def run(self, state: Any) -> dict:
        """
        执行 Agent 逻辑

        Args:
            state: LangGraph 工作流状态 (SystemState)

        Returns:
            状态更新字典，与原 nodes.py 返回格式一致
        """
        ...

    async def react_loop(
        self,
        prompt_template: ChatPromptTemplate,
        symptom: str,
        max_iterations: int = 3,
    ) -> tuple[list[dict], list[dict]]:
        """
        ReAct 循环: Think → Act → Observe → 重复直到信息充足

        流程:
        1. LLM 根据症状决定调用哪些工具 (Think)
        2. 执行工具调用 (Act)
        3. 将工具结果反馈给 LLM (Observe)
        4. LLM 判断是否需要更多信息:
           - 需要更多 → 继续调用工具，回到步骤 2
           - 信息充足 → 停止调用工具，退出循环
        5. 达到 max_iterations 强制退出

        Args:
            prompt_template: 初始 prompt 模板（含 system + human）
            symptom: 故障现象描述
            max_iterations: 最大迭代轮数，默认 3

        Returns:
            (all_tool_results, all_tool_calls_info)
            - all_tool_results: 所有工具调用结果 [{"tool": name, "result": ...}, ...]
            - all_tool_calls_info: 所有工具调用信息 [{"name": ..., "args": ..., "id": ...}, ...]
        """
        all_tool_results = []
        all_tool_calls_info = []

        if not self.tools:
            logger.warning(f"[{self.name}] 无可用工具，跳过 ReAct 循环")
            return all_tool_results, all_tool_calls_info

        messages = prompt_template.format_messages(symptom=symptom)
        bound_llm = self.llm.bind_tools(self.tools)

        for iteration in range(max_iterations):
            logger.info(
                f"[{self.name}] ReAct 第 {iteration + 1}/{max_iterations} 轮"
            )

            response = await bound_llm.ainvoke(messages)

            if not response.tool_calls:
                logger.info(f"[{self.name}] LLM 未请求工具调用，信息收集完成")
                break

            all_tool_calls_info.extend(response.tool_calls)
            tool_results = await execute_tool_calls(response, self.tools, self.name)
            all_tool_results.extend(tool_results)

            messages.append(response)
            for tool_call in response.tool_calls:
                matching = next(
                    (tr for tr in tool_results if tr["tool"] == tool_call["name"]),
                    None,
                )
                if matching:
                    messages.append(
                        ToolMessage(
                            content=str(matching["result"]),
                            tool_call_id=tool_call["id"],
                        )
                    )

            logger.info(
                f"[{self.name}] 本轮调用 {len(response.tool_calls)} 个工具，"
                f"累计 {len(all_tool_results)} 个结果"
            )

            if iteration < max_iterations - 1:
                from langchain_core.messages import HumanMessage

                messages.append(
                    HumanMessage(
                        content="请根据以上工具返回结果继续分析。"
                        "如果还需要更多信息请调用工具，如果信息已足够请直接回复分析结论，不要再调用工具。"
                    )
                )
        else:
            logger.warning(
                f"[{self.name}] 达到最大迭代次数 {max_iterations}，强制结束 ReAct 循环"
            )

        return all_tool_results, all_tool_calls_info


