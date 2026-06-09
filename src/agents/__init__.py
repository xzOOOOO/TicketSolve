# BaseAgent：所有 Agent 的抽象基类，定义通用接口
from agents.base import BaseAgent
# DiagnosticAgent：诊断类 Agent 公共模板，收敛 DB/Net/App 的重复流程
from agents.diagnostic import DiagnosticAgent
# SupervisorAgent：调度中心 Agent，决定派发哪些专业 Agent
from agents.supervisor import SupervisorAgent
# DBAgent：数据库诊断 Agent
from agents.db import DBAgent
# NetAgent：网络诊断 Agent
from agents.net import NetAgent
# AppAgent：应用诊断 Agent
from agents.app import AppAgent
# FixAgent：修复方案生成 Agent
from agents.fix import FixAgent
# CommunicationBus：Agent 间通信总线，用于传递 evidence_request/evidence_response
from agents.communication import CommunicationBus

# __all__：控制 from agents import * 时导出的符号列表
__all__ = [
    "BaseAgent",
    "DiagnosticAgent",
    "SupervisorAgent",
    "DBAgent",
    "NetAgent",
    "AppAgent",
    "FixAgent",
    "CommunicationBus",
]
