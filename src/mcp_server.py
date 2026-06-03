"""
MCP Server - 工单系统诊断工具服务
基于 FastMCP SDK 实现，提供标准化的工具调用接口

技术栈:
- FastMCP: 高性能 MCP Server 框架
- asyncio: 异步IO处理
- json: 结构化数据返回

运行方式:
1. stdio 模式: python mcp_server.py (用于本地LangGraph集成)
2. sse 模式: 待扩展
"""

import asyncio
import json
from typing import Any
from mcp.server.fastmcp import FastMCP
from lab_diagnostics import (
    check_app_health_result,
    check_app_port_result,
    check_app_process_result,
    check_app_redis_connection_result,
    check_db_connection_result,
    check_db_deadlock_result,
    check_db_slow_query_result,
    check_network_dns_result,
    check_network_http_route_result,
    check_network_ping_result,
)

# 初始化 MCP Server
mcp = FastMCP("diagnosis-server")


# ============================================================
# 数据库诊断工具集
# ============================================================

@mcp.tool()
def check_db_connection() -> str:
    """
    检查数据库连接状态

    Returns:
        JSON字符串，包含连接状态、错误信息和可能的问题分析
    """
    result = check_db_connection_result()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def check_db_slow_query() -> str:
    """
    检查数据库慢查询

    Returns:
        JSON字符串，包含慢查询列表和可能的问题分析
    """
    result = check_db_slow_query_result()
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def check_db_deadlock() -> str:
    """
    检查数据库死锁情况

    Returns:
        JSON字符串，包含死锁检测结果
    """
    result = check_db_deadlock_result()
    return json.dumps(result, ensure_ascii=False)


# ============================================================
# 网络诊断工具集
# ============================================================

@mcp.tool()
def check_network_ping(host: str) -> str:
    """
    检查网络连通性

    Args:
        host: 目标主机地址 (IP或域名)

    Returns:
        JSON字符串，包含ping测试结果
    """
    result = check_network_ping_result(host)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def check_network_dns(domain: str) -> str:
    """
    检查DNS解析状态

    Args:
        domain: 待解析的域名

    Returns:
        JSON字符串，包含DNS解析结果
    """
    result = check_network_dns_result(domain)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def check_network_http_route(url: str = "http://localhost:18080/health") -> str:
    """
    检查 Nginx 到应用的 HTTP 路由状态

    Args:
        url: 入口 URL，默认检查 SREBench Lite 的 nginx health 路由

    Returns:
        JSON字符串，包含 nginx 入口与直连 app 的对比结果
    """
    result = check_network_http_route_result(url)
    return json.dumps(result, ensure_ascii=False)


# ============================================================
# 应用诊断工具集
# ============================================================

@mcp.tool()
def check_app_process(process_name: str) -> str:
    """
    检查应用进程状态

    Args:
        process_name: 进程名称

    Returns:
        JSON字符串，包含进程运行状态、资源占用等
    """
    result = check_app_process_result(process_name)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def check_app_port(port: int) -> str:
    """
    检查应用端口状态

    Args:
        port: 端口号

    Returns:
        JSON字符串，包含端口监听状态和连接数
    """
    result = check_app_port_result(port)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def check_app_health(url: str = "http://localhost:18081/health") -> str:
    """
    检查应用健康接口

    Args:
        url: 应用健康检查 URL，默认直连 SREBench Lite app

    Returns:
        JSON字符串，包含应用容器状态和 HTTP 健康检查结果
    """
    result = check_app_health_result(url)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def check_app_redis_connection() -> str:
    """
    检查应用依赖的 Redis 连接

    Returns:
        JSON字符串，包含 Redis 容器、端口和业务 cache endpoint 检查结果
    """
    result = check_app_redis_connection_result()
    return json.dumps(result, ensure_ascii=False)


# ============================================================
# 系统信息工具集 
# ============================================================

@mcp.tool()
def get_system_info() -> str:
    """
    获取MCP Server系统信息

    Returns:
        JSON字符串，包含Server版本、可用工具列表等元信息
    """
    result = {
        "server_name": "diagnosis-server",
        "version": "1.0.0",
        "protocol_version": "2026-4-27",
        "tools": [
            "check_db_connection",
            "check_db_slow_query",
            "check_db_deadlock",
            "check_network_ping",
            "check_network_dns",
            "check_network_http_route",
            "check_app_process",
            "check_app_port",
            "check_app_health",
            "check_app_redis_connection"
        ],
        "tech_stack": {
            "framework": "FastMCP",
            "language": "Python 3.12+",
            "transport": ["stdio", "sse"]
        }
    }
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # 默认以 stdio 模式运行，用于本地 LangGraph 集成
    mcp.run(transport='stdio')
