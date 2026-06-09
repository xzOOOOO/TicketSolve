"""
并行派发节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def create_dispatch_node(agent_runners: dict[str, Callable[[SystemState], Awaitable[dict]]]):
    """
    创建并行派发节点工厂函数。

    根据 Supervisor 的 dispatched_agents 列表，并行调用被派发的 Agent。
    使用 asyncio.gather 实现并行执行，各 Agent 结果合并写入 state。

    动态调度增强:
    - 跳过本轮已有结果的 Agent（避免重复执行）
    - 支持 force_dispatched_agents 强制重跑（证据覆盖不足时）
    - 递增 dispatch_round 计数器（防循环）

    参数：
        agent_runners: Agent 名称 → run 方法的映射
            格式：{"db_agent": db_agent.run, "net_agent": net_agent.run, ...}
            每个 run 方法接收 SystemState，返回 dict

    返回：
        异步节点函数 dispatch_node(state) -> dict
    """
    # _result_fields：Agent 名称 → SystemState 结果字段的映射
    # 用于快速检查某 Agent 是否已有诊断结果（避免重复执行）
    _result_fields = {
        "db_agent": "db_agent_result",    # db_agent 的结果存在 state.db_agent_result
        "net_agent": "net_agent_result",  # net_agent 的结果存在 state.net_agent_result
        "app_agent": "app_agent_result",  # app_agent 的结果存在 state.app_agent_result
    }

    async def dispatch_node(state: SystemState) -> dict:
        # dispatched：Supervisor 决策的本次待派发 Agent 列表
        dispatched = state.dispatched_agents
        # force_dispatched：被 DynamicCheck 标记为需要强制重跑的 Agent 集合
        # 这些 Agent 即使有缓存结果也会被重新执行
        force_dispatched = set(state.force_dispatched_agents or [])

        # 如果没有 Agent 被派发，直接返回（如 other 类型工单）
        if not dispatched:
            logger.info("[Dispatch] 无 Agent 被派发，跳过诊断")
            return {
                "force_dispatched_agents": [],  # 清空强制重跑列表
                "messages": ["Dispatch: 无需诊断Agent，直接处理"],
            }

        # to_run：本轮实际需要执行的 Agent 列表
        # 过滤掉已有结果且不在 force_dispatched 中的 Agent
        to_run = []
        for agent_name in dispatched:
            field = _result_fields.get(agent_name)
            # already_done：检查该 Agent 是否已有诊断结果
            already_done = field and getattr(state, field, None) is not None
            if already_done and agent_name not in force_dispatched:
                # 已有结果且不需要强制重跑 → 跳过
                logger.info(f"[Dispatch] {agent_name} 已有结果，跳过本轮执行")
            else:
                if already_done and agent_name in force_dispatched:
                    # 已有结果但被强制重跑 → 记录日志
                    logger.info(f"[Dispatch] {agent_name} 被证据请求强制重跑")
                to_run.append(agent_name)

        # 如果所有 Agent 都跳过，直接返回
        if not to_run:
            logger.info("[Dispatch] 所有被派发 Agent 均已有结果，跳过")
            return {
                "force_dispatched_agents": [],
                "messages": ["Dispatch: 所有Agent已完成，无需重复执行"],
            }

        logger.info(f"[Dispatch] 并行派发 Agent: {to_run} (轮次 {state.dispatch_round + 1})")

        # tasks：异步任务列表，每个任务是一个 Agent 的 run 方法调用
        # agent_names：与 tasks 对应的 Agent 名称列表，用于结果匹配
        tasks = []
        agent_names = []
        for agent_name in to_run:
            runner = agent_runners.get(agent_name)
            if runner:
                # runner(state)：调用 Agent 的 run 方法，传入当前状态
                tasks.append(runner(state))
                agent_names.append(agent_name)
            else:
                logger.warning(f"[Dispatch] 未找到 Agent: {agent_name}")

        # 没有可执行的任务，返回警告
        if not tasks:
            logger.warning("[Dispatch] 没有可执行的 Agent")
            return {"messages": ["Dispatch: 无可用Agent执行"]}

        # asyncio.gather：并行执行所有任务，return_exceptions=True 表示异常不中断其他任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # merged：合并所有 Agent 返回结果的字典
        # messages：收集所有文本消息
        # dispatch_round：递增调度轮次（防循环）
        # force_dispatched_agents：清空本轮强制重跑标记
        merged = {
            "messages": [],
            "dispatch_round": state.dispatch_round + 1,
            "force_dispatched_agents": [],
        }
        # 逐个处理 Agent 返回结果
        for agent_name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                # Agent 执行异常：记录错误日志，不中断其他 Agent
                logger.error(f"[Dispatch] Agent {agent_name} 执行异常: {result}")
                merged["messages"].append(f"Dispatch: {agent_name} 执行异常 - {str(result)}")
                continue

            if isinstance(result, dict):
                # Agent 正常返回字典：按 key 合并到 merged
                for key, value in result.items():
                    if key == "messages":
                        # messages：直接扩展列表
                        merged["messages"].extend(value)
                    elif key == "agent_messages":
                        # agent_messages：Agent 间通信消息（通过 operator.add 累加）
                        merged.setdefault("agent_messages", []).extend(value)
                    elif key == "audit_logs":
                        # audit_logs：审计日志（通过 operator.add 累加）
                        merged.setdefault("audit_logs", []).extend(value)
                    elif key == "trace_events":
                        # trace_events：标准化 Trace 事件（通过 operator.add 累加）
                        merged.setdefault("trace_events", []).extend(value)
                    else:
                        # 其他字段（如 db_agent_result）：直接覆盖
                        merged[key] = value

        logger.info(f"[Dispatch] 并行执行完成，{len(agent_names)} 个 Agent 返回结果")
        return merged

    return dispatch_node
