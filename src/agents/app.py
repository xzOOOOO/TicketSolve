"""
应用诊断 Agent。

本模块只保留应用领域的差异化配置，通用诊断流程由 DiagnosticAgent 提供。
"""

# BaseChatModel：LangChain 聊天模型基类，用于诊断推理
from langchain_core.language_models import BaseChatModel
# BaseTool：LangChain 工具基类，用于注入应用诊断工具
from langchain_core.tools import BaseTool

# DiagnosticAgent：诊断类 Agent 公共模板
from agents.diagnostic import DiagnosticAgent
# CommunicationBus：Agent 间通信总线
from agents.communication import CommunicationBus
# APP_DIAGNOSIS_PROMPT：应用诊断结论生成 Prompt
# APP_PROMPT：应用工具调用阶段 Prompt
from prompts import APP_DIAGNOSIS_PROMPT, APP_PROMPT


class AppAgent(DiagnosticAgent):
    """
    应用诊断专家 Agent。

    用途：
        使用应用相关 MCP 工具诊断进程、健康检查、缓存依赖等故障。
    """

    # name：Agent 唯一标识，用于调度、通信和日志
    name = "app_agent"
    # role：Agent 角色描述，用于身份识别
    role = "应用诊断专家"
    # tool_prompt：ReAct 工具调用阶段 Prompt
    tool_prompt = APP_PROMPT
    # diagnosis_prompt：结构化诊断阶段 Prompt
    diagnosis_prompt = APP_DIAGNOSIS_PROMPT
    # result_field：诊断结果写回 SystemState 的字段名
    result_field = "app_agent_result"
    # domain_label：中文领域名称，用于协作消息和审计日志
    domain_label = "应用"
    # message_label：用户可见消息前缀
    message_label = "App Agent (MCP)"

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        communication_bus: CommunicationBus | None = None,
    ):
        """
        初始化应用诊断 Agent。

        参数：
            llm: LangChain 聊天模型实例
            tools: 应用诊断工具列表
            communication_bus: Agent 间通信总线，为 None 时不发送协作消息

        返回：
            None
        """
        super().__init__(llm, tools, communication_bus)
