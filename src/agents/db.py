"""
数据库诊断 Agent。

本模块只保留数据库领域的差异化配置，通用诊断流程由 DiagnosticAgent 提供。
"""

# BaseChatModel：LangChain 聊天模型基类，用于诊断推理
from langchain_core.language_models import BaseChatModel
# BaseTool：LangChain 工具基类，用于注入数据库诊断工具
from langchain_core.tools import BaseTool

# DiagnosticAgent：诊断类 Agent 公共模板
from agents.diagnostic import DiagnosticAgent
# CommunicationBus：Agent 间通信总线
from agents.communication import CommunicationBus
# DB_DIAGNOSIS_PROMPT：数据库诊断结论生成 Prompt
# DB_PROMPT：数据库工具调用阶段 Prompt
from prompts import DB_DIAGNOSIS_PROMPT, DB_PROMPT


class DBAgent(DiagnosticAgent):
    """
    数据库诊断专家 Agent。

    用途：
        使用数据库相关 MCP 工具诊断数据库连接、慢查询、锁等待等故障。
    """

    # name：Agent 唯一标识，用于调度、通信和日志
    name = "db_agent"
    # role：Agent 角色描述，用于身份识别
    role = "数据库诊断专家"
    # tool_prompt：ReAct 工具调用阶段 Prompt
    tool_prompt = DB_PROMPT
    # diagnosis_prompt：结构化诊断阶段 Prompt
    diagnosis_prompt = DB_DIAGNOSIS_PROMPT
    # result_field：诊断结果写回 SystemState 的字段名
    result_field = "db_agent_result"
    # domain_label：中文领域名称，用于协作消息和审计日志
    domain_label = "数据库"
    # message_label：用户可见消息前缀
    message_label = "DB Agent (MCP)"

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        communication_bus: CommunicationBus | None = None,
    ):
        """
        初始化数据库诊断 Agent。

        参数：
            llm: LangChain 聊天模型实例
            tools: 数据库诊断工具列表
            communication_bus: Agent 间通信总线，为 None 时不发送协作消息

        返回：
            None
        """
        super().__init__(llm, tools, communication_bus)
