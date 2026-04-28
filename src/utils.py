"""
公共工具函数

从各 Agent 中提取的共享逻辑:
- parse_json_content: JSON 解析（兼容 markdown 代码块等格式）
- execute_tool_calls: 执行 LLM 返回的工具调用
"""

import json
from typing import Optional

from logger import logger


def parse_json_content(content: str) -> Optional[dict]:
    """
    解析 JSON 内容，兼容多种格式:
    - 标准 JSON
    - ```json ... ``` 代码块
    - ``` ... ``` 代码块
    - 混合文本中提取 JSON
    """
    result = None
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
        except Exception:
            pass
    return result


async def execute_tool_calls(response, tools, agent_name: str = "agent") -> list:
    """
    执行 LLM 返回的工具调用

    Args:
        response: LLM 响应对象（含 tool_calls）
        tools: 可用工具列表
        agent_name: Agent 名称（用于日志）

    Returns:
        工具结果列表: [{"tool": name, "result": result}, ...]
    """
    tool_results = []
    for tool_call in response.tool_calls:
        logger.debug(f"[{agent_name}] 调用工具: {tool_call['name']}, args={tool_call['args']}")
        for tool in tools:
            if tool.name == tool_call["name"]:
                result = await tool.ainvoke(tool_call["args"])
                tool_results.append({"tool": tool_call["name"], "result": result})
                logger.debug(f"[{agent_name}] 工具 {tool_call['name']} 返回成功")
                break
    return tool_results
