"""
BaseAgent - 所有 Agent 的抽象基类

设计原则:
- 每个 Agent 拥有独立身份（name, role）
- Agent 通过 run() 方法被 LangGraph 节点调用
- 返回值格式与原 nodes.py 完全一致，确保兼容
- 子类只需实现 run() 方法
- 诊断类 Agent 可使用 react_loop() 实现多轮工具调用

ReAct 循环说明:
ReAct = Reasoning + Acting，是一种让 LLM 交替进行"思考"和"行动"的范式。
在本项目中，诊断类 Agent（DBAgent/NetAgent/AppAgent）使用 ReAct 循环：
1. LLM 根据症状决定调用哪些诊断工具（Think）
2. 系统执行工具调用（Act）
3. 将工具结果反馈给 LLM（Observe）
4. LLM 判断是否需要更多信息，循环或退出
"""

# abc：Python 抽象基类模块，用于定义抽象类和抽象方法
from abc import ABC, abstractmethod
# Any：任意类型；Optional：可选类型（可为 None）
from typing import Any, Optional
# BaseChatModel：LangChain 聊天模型基类，所有 LLM（如 GPT、Claude）都继承自它
from langchain_core.language_models import BaseChatModel
# BaseTool：LangChain 工具基类，所有诊断工具都继承自它
from langchain_core.tools import BaseTool
# ToolMessage：LangChain 工具消息类型，用于将工具结果反馈给 LLM
from langchain_core.messages import ToolMessage
# ChatPromptTemplate：LangChain 聊天 Prompt 模板，用于构造 system + human 消息
from langchain_core.prompts import ChatPromptTemplate
# execute_tool_calls：项目工具函数，批量执行 LLM 请求的工具调用
from utils import execute_tool_calls
# logger：项目统一日志记录器
from logger import logger
# settings：项目配置对象，包含重试配置等
from config import settings


class BaseAgent(ABC):
    """
    Agent 抽象基类

    所有 Agent（SupervisorAgent、DBAgent、NetAgent、AppAgent、FixAgent）都继承自此类。
    子类必须实现 run() 方法，定义自己的业务逻辑。

    类属性：
        name: Agent 唯一标识，用于日志和通信
        role: Agent 角色描述，用于 prompt 和身份识别

    实例属性：
        llm: 独立的 LLM 实例
        tools: Agent 可用的工具列表（诊断类 Agent 使用）
    """

    # name：类属性，子类必须覆盖，如 "db_agent"
    name: str = "base_agent"
    # role：类属性，子类必须覆盖，如 "数据库诊断专家"
    role: str = "基础Agent"

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[list[BaseTool]] = None,
    ):
        # llm：LangChain 聊天模型实例，用于推理和生成
        self.llm = llm
        # tools：该 Agent 可调用的工具列表，如 [check_db_connection, check_db_slow_queries]
        # 默认为空列表（SupervisorAgent 和 FixAgent 不需要工具）
        self.tools = tools or []
        logger.debug(f"Agent [{self.name}] 初始化完成, 工具数: {len(self.tools)}")

    @abstractmethod
    async def run(self, state: Any) -> dict:
        """
        执行 Agent 逻辑（子类必须实现）

        参数：
            state: LangGraph 工作流状态 (SystemState)

        返回：
            状态更新字典，与原 nodes.py 返回格式一致
            典型格式：{"db_agent_result": {...}, "messages": [...], "audit_logs": [...]}
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

        ReAct（Reasoning + Acting）是诊断类 Agent 的核心工作模式：
        1. LLM 根据症状决定调用哪些诊断工具 (Think)
        2. 系统执行工具调用 (Act)
        3. 将工具结果反馈给 LLM (Observe)
        4. LLM 判断是否需要更多信息:
           - 需要更多 → 继续调用工具，回到步骤 2
           - 信息充足 → 停止调用工具，退出循环
        5. 达到 max_iterations 强制退出（防无限循环）

        参数：
            prompt_template: 初始 prompt 模板（含 system + human 消息）
            symptom: 故障现象描述
            max_iterations: 最大迭代轮数，默认 3

        返回：
            (all_tool_results, all_tool_calls_info) 元组
            - all_tool_results: 所有工具调用结果 [{"tool": name, "result": ...}, ...]
            - all_tool_calls_info: 所有工具调用信息 [{"name": ..., "args": ..., "id": ...}, ...]
        """
        # all_tool_results：收集所有工具调用结果
        all_tool_results = []
        # all_tool_calls_info：收集所有工具调用信息（用于审计和 Trace）
        all_tool_calls_info = []

        # 无可用工具时直接返回，避免空跑
        if not self.tools:
            logger.warning(f"[{self.name}] 无可用工具，跳过 ReAct 循环")
            return all_tool_results, all_tool_calls_info

        # messages：LLM 对话历史，初始为 Prompt 模板格式化后的消息
        messages = prompt_template.format_messages(symptom=symptom)
        # bound_llm：绑定工具的 LLM 实例，LLM 输出会包含 tool_calls 字段
        # .with_retry()：自动重试（应对 LLM 偶尔的输出格式错误）
        bound_llm = self.llm.bind_tools(self.tools).with_retry(**settings.get_retry_config())

        # ReAct 循环：最多 max_iterations 轮
        for iteration in range(max_iterations):
            logger.info(
                f"[{self.name}] ReAct 第 {iteration + 1}/{max_iterations} 轮"
            )

            # 调用 LLM，获取响应（可能包含 tool_calls）
            response = await bound_llm.ainvoke(messages)

            # LLM 没有请求工具调用 → 信息收集完成，退出循环
            if not response.tool_calls:
                logger.info(f"[{self.name}] LLM 未请求工具调用，信息收集完成")
                break

            # 收集工具调用信息
            all_tool_calls_info.extend(response.tool_calls)
            # 执行工具调用（批量执行，可能并行）
            tool_results = await execute_tool_calls(response, self.tools, self.name)
            all_tool_results.extend(tool_results)

            # 将 LLM 响应和工具结果追加到对话历史
            messages.append(response)
            for tool_call in response.tool_calls:
                # matching：找到与当前 tool_call 对应的结果
                matching = next(
                    (tr for tr in tool_results if tr["tool"] == tool_call["name"]),
                    None,
                )
                if matching:
                    # ToolMessage：将工具结果反馈给 LLM，必须包含 tool_call_id
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

            # 如果不是最后一轮，追加提示消息让 LLM 继续分析或退出
            if iteration < max_iterations - 1:
                from langchain_core.messages import HumanMessage

                messages.append(
                    HumanMessage(
                        content="请根据以上工具返回结果继续分析。"
                        "如果还需要更多信息请调用工具，如果信息已足够请直接回复分析结论，不要再调用工具。"
                    )
                )
        else:
            # for-else 语法：循环正常结束（未 break）时执行
            # 表示达到了最大迭代次数，强制退出
            logger.warning(
                f"[{self.name}] 达到最大迭代次数 {max_iterations}，强制结束 ReAct 循环"
            )

        return all_tool_results, all_tool_calls_info


