from logger import logger


async def execute_tool_calls(response, tools, agent_name: str = "agent") -> list:
    """执行 LLM 返回的工具调用

    在 ReAct 循环中，LLM 决定调用哪些工具后，由本函数实际执行。
    遍历 response.tool_calls，匹配对应的工具并异步调用。

    Args:
        response: LLM 响应对象（含 tool_calls 列表）
        tools: 可用工具列表
        agent_name: Agent 名称（用于日志标识）

    Returns:
        工具结果列表，格式: [{"tool": 工具名, "result": 结果}, ...]
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
