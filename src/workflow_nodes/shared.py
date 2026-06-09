"""
工作流节点共享依赖。

本模块集中保存拆分后各节点模块需要复用的导入和恢复验证配置。
"""

# asyncio：Python 异步编程库，用于并行执行多个 Agent
import asyncio
# Callable：可调用类型；Awaitable：可等待类型（异步函数返回值）
from typing import Callable, Awaitable
# datetime：日期时间库；timezone：时区支持
from datetime import datetime, timezone
# HTTPError/URLError：HTTP 请求异常类，用于验证节点的错误处理
from urllib.error import HTTPError, URLError
# Request/urlopen：用于发送 HTTP 请求，验证恢复状态
from urllib.request import Request, urlopen

# SystemState：工作流全局状态；ApprovalStatus：审批状态枚举
from state import SystemState, ApprovalStatus
# AGGREGATE_PROMPT：聚合诊断的 LLM Prompt 模板
# ERROR_ANALYSIS_PROMPT：执行错误分析的 LLM Prompt 模板
from prompts import AGGREGATE_PROMPT, ERROR_ANALYSIS_PROMPT
# AggregateOutput：聚合诊断的结构化输出模型
from schemas import AggregateOutput
# interrupt：LangGraph 提供的暂停工作流功能，用于人工审批节点
from langgraph.types import interrupt
# GraphInterrupt：LangGraph 中断异常，需要原样抛出不能吞掉
from langgraph.errors import GraphInterrupt
# AsyncSessionLocal：异步数据库会话；save_ticket：保存工单到数据库
from database import AsyncSessionLocal, save_ticket
# run_guardrail：执行确定性安全护栏检查
from guardrail import run_guardrail
# ActionDSLValidationError：Action DSL 校验异常
# compile_rollback_action/compile_step_action：编译回滚/正向动作
from action_dsl import ActionDSLValidationError, compile_rollback_action, compile_step_action

# agent_protocol 模块：证据协作协议的核心工具函数
# - agent_result_covers_request：判断 Agent 的诊断结果是否覆盖了证据请求要求的证据项
# - auto_response_from_agent_result：用已有诊断结果自动生成 evidence_response 消息
# - build_protocol_context：从所有消息中构建协议上下文（含统计摘要）
# - has_response_for：检查某条 evidence_request 是否已有对应的 evidence_response
# - normalize_messages：把 state 中存储的字典消息归一化成标准格式
# - pending_requests_for：找出某个 Agent 尚未响应的证据请求列表
from agent_protocol import (
    agent_result_covers_request,
    auto_response_from_agent_result,
    build_protocol_context,
    has_response_for,
    normalize_messages,
    pending_requests_for,
)

# case_library 模块：案例库相关函数
# - DEFAULT_CASE_LIBRARY_PATH：默认案例库文件路径
# - format_case_context：将案例列表格式化为文本上下文
# - retrieve_similar_cases：根据症状检索相似历史案例
# - upsert_case_from_state：从当前状态沉淀案例到案例库
from case_library import (
    DEFAULT_CASE_LIBRARY_PATH,
    format_case_context,
    retrieve_similar_cases,
    upsert_case_from_state,
)

# executor_v2 模块：闭环执行器
# - ClosedLoopExecutor：核心执行器类，支持 Observe-Decide-Act 循环
# - MockCommandRunner：模拟命令执行（用于测试）
# - SafeDockerCommandRunner：安全的 Docker 命令执行（用于靶场）
from executor_v2 import ClosedLoopExecutor, MockCommandRunner, SafeDockerCommandRunner
# make_replanner_decision：纯规则决策引擎，根据执行结果决定下一步动作
from replanner import make_replanner_decision

# trace_events 模块：标准化 Trace 事件
# - make_trace_event：创建标准化 Trace 事件字典
# - status_from_success：将布尔成功标志转换为标准状态字符串
from trace_events import make_trace_event, status_from_success
# logger：项目统一日志记录器
from logger import logger
# settings：项目配置对象，包含 EXECUTOR_MODE 等运行时参数
from config import settings


# VERIFY_PROBES：恢复验证时探测的 HTTP 端点列表
# 所有探测都成功才认为服务已恢复
VERIFY_PROBES = [
    # health：Nginx 入口健康检查，验证整体服务可用性
    {"name": "health", "url": "http://localhost:18080/health"},
    # cache_ping：Redis 缓存接口，验证缓存服务连通性
    {"name": "cache_ping", "url": "http://localhost:18080/cache/ping"},
    # orders_pending：数据库查询接口，验证数据库连通性和查询能力
    {"name": "orders_pending", "url": "http://localhost:18080/orders/pending"},
]
