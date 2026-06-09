"""
动态检查节点。
"""

# agent_protocol：动态检查节点只依赖证据协作协议的轻量工具函数
from agent_protocol import (
    agent_result_covers_request,
    auto_response_from_agent_result,
    has_response_for,
    normalize_messages,
    pending_requests_for,
)
# logger：项目统一日志记录器
from logger import logger
# SystemState：工作流全局状态类型
from state import SystemState
# make_trace_event：标准化 Trace 事件工厂
from trace_events import make_trace_event

def create_dynamic_check_node():
    """
    创建动态检查节点工厂函数（证据协作协议的核心调度器）。

    扫描 agent_messages 中尚未响应的 evidence_request，根据目标 Agent 的状态做三种处理：

    1. 目标 Agent 尚未执行（无诊断结果）：
       → 追加派发该 Agent，让它去执行诊断

    2. 目标 Agent 已有结果，且证据覆盖了请求要求的证据项：
       → 自动生成 evidence_response，避免重复执行 Agent

    3. 目标 Agent 已有结果，但证据覆盖不足：
       → 定向重跑该 Agent（加入 force_dispatched_agents），让它重新诊断
       → 用 redispatched_request_ids 防止同一个请求被无限重跑

    防循环设计：
    - dispatch_round < max_dispatch_rounds 时才允许追加派发/重跑
    - 已达最大轮次时，只生成 auto_responses，不再追加派发

    为什么需要这个节点：
    多 Agent 协作时，Agent A 可能向 Agent B 请求证据（如"请检查数据库连接"）。
    DynamicCheck 负责协调这些请求，决定是重新执行 Agent B，还是用已有结果自动响应。

    返回：
        异步节点函数 dynamic_check_node(state) -> dict
    """
    async def dynamic_check_node(state: SystemState) -> dict:
        # _result_fields：Agent 名称 → SystemState 结果字段的映射
        # 用于快速查找某 Agent 是否已有诊断结果
        _result_fields = {
            "db_agent": "db_agent_result",
            "net_agent": "net_agent_result",
            "app_agent": "app_agent_result",
        }

        # normalized_messages：归一化所有消息，确保字段格式统一
        # 处理旧消息可能缺失的字段，统一转成标准字典格式
        normalized_messages = normalize_messages(state.agent_messages)
        # auto_responses：自动生成的 evidence_response 消息列表
        # 当 Agent 已有结果且证据覆盖请求时，自动生成响应消息
        auto_responses = []
        # requested：需要被派发的 Agent 集合（包括首次派发和重跑）
        requested = set()
        # force_requested：需要强制重跑的 Agent 集合（证据覆盖不足时）
        # 这些 Agent 会被加入 force_dispatched_agents，Dispatch 节点会强制重新执行
        force_requested = set()
        # already_redispatched：已经触发过重跑的请求 ID 集合
        # 用于防止同一个请求因为证据覆盖不足被无限重跑
        already_redispatched = set(state.redispatched_request_ids or [])
        # new_redispatched_request_ids：本轮新触发重跑的请求 ID 列表
        # 会返回给 state，通过 operator.add 累加到 redispatched_request_ids
        new_redispatched_request_ids = []
        # coverage_trace_events：记录证据覆盖判定和定向重派发的 Trace 事件
        # 方便 demo/前端/评测系统展示调度决策过程
        coverage_trace_events = []
        # can_redispatch：是否允许继续调度（未达最大轮次）
        # 如果已达最大轮次，只生成 auto_responses，不再追加派发
        can_redispatch = state.dispatch_round < state.max_dispatch_rounds

        # 遍历每个 Agent，检查它是否有待响应的证据请求
        for agent_name, result_field in _result_fields.items():
            # pending_requests_for：找出发给该 Agent 且尚未有 response 的 evidence_request
            for request in pending_requests_for(agent_name, normalized_messages):
                # 如果这条请求已经被 auto_responses 响应过了，跳过
                if has_response_for(request, normalized_messages + auto_responses):
                    continue

                # 获取该 Agent 当前的诊断结果
                agent_result = getattr(state, result_field, None)
                if agent_result:
                    # Agent 已有结果：判断证据是否覆盖了请求要求的证据项
                    request_id = request.get("message_id")
                    covers_request = agent_result_covers_request(agent_result, request)

                    if not covers_request and request_id not in already_redispatched and can_redispatch:
                        # 场景 3：证据覆盖不足，且没重跑过，且还能调度 → 定向重跑
                        requested.add(agent_name)
                        force_requested.add(agent_name)
                        new_redispatched_request_ids.append(request_id)
                        logger.info(
                            f"[DynamicCheck] 旧结果未覆盖证据请求，定向重派发: "
                            f"request={request_id} responder={agent_name}"
                        )
                        # 生成 handoff_requested Trace 事件：记录"证据覆盖不足→定向重跑"的决策点
                        # 这个事件对外展示为什么系统要强制某个 Agent 重新执行，方便 demo/评测复盘
                        coverage_trace_events.append(make_trace_event(
                            "handoff_requested",               # trace 事件类型：任务交接/重派发请求
                            ticket_id=state.ticket_id,         # 当前工单 ID，用于关联整条工作流轨迹
                            agent_name="dynamic_check",        # 产生该事件的节点名称
                            status="pending",                  # 状态为 pending，表示重跑任务已发出但尚未完成
                            input_data={
                                "request_message": request,    # 原始证据请求消息体，包含 required_evidence/suggested_tools
                                "cached_agent_result_exists": True,  # 标记该 Agent 已有缓存结果（不是首次执行）
                            },
                            output_data={
                                "target_agent": agent_name,    # 被强制重跑的目标 Agent
                                "forced_redispatch": True,     # 标记这是一次"强制重跑"（非普通调度）
                            },
                            metadata={
                                "dispatch_round": state.dispatch_round,        # 当前调度轮次，用于追踪循环深度
                                "message_id": request_id,                      # 证据请求的唯一消息 ID
                                "correlation_id": request.get("correlation_id"),  # 关联 ID，串联请求-响应链路
                                "msg_type": request.get("msg_type"),           # 消息类型，这里是 evidence_request
                                "target_agent": agent_name,                    # 目标 Agent（冗余字段，方便过滤）
                                "coverage": False,                             # 核心字段：证据未覆盖请求要求
                                "forced_redispatch": True,                     # 与 output_data 一致，方便元数据查询
                                "required_evidence": request.get("required_evidence", []),  # 请求方要求的证据项列表
                                "suggested_tools": request.get("suggested_tools", []),      # 请求方建议使用的工具列表
                            },
                        ))
                        continue

                    # 场景 2：证据已覆盖（或无法重跑了）→ 自动生成 evidence_response
                    response = auto_response_from_agent_result(
                        agent_name=agent_name,
                        agent_result=agent_result,
                        request_message=request,
                        # supports_override：如果证据覆盖不足，强制设为 False（表示不支持假设）
                        supports_override=False if not covers_request else None,
                    )
                    auto_responses.append(response)
                    logger.info(
                        f"[DynamicCheck] 自动生成证据响应: request={request.get('message_id')} "
                        f"responder={agent_name} coverage={covers_request}"
                    )
                    # 生成 observation_received Trace 事件：记录"证据已覆盖→自动生成响应"的决策点
                    # status 根据 covers_request 动态决定：覆盖充分为 success，覆盖不足为 failure
                    coverage_trace_events.append(make_trace_event(
                        "observation_received",            # trace 事件类型：观测到 Agent 输出/响应
                        ticket_id=state.ticket_id,         # 当前工单 ID，用于关联整条工作流轨迹
                        agent_name="dynamic_check",        # 产生该事件的节点名称
                        status="success" if covers_request else "failure",  # 覆盖成功标记 success，否则 failure
                        input_data={
                            "request_message": request,    # 原始证据请求消息体
                            "cached_agent_result_exists": True,  # 标记该 Agent 已有缓存结果
                        },
                        output_data=response,              # 自动生成的 evidence_response 消息体
                        metadata={
                            "dispatch_round": state.dispatch_round,        # 当前调度轮次
                            "message_id": request.get("message_id"),       # 证据请求的唯一消息 ID
                            "correlation_id": request.get("correlation_id"),  # 关联 ID，串联请求-响应链路
                            "msg_type": "evidence_response",               # 消息类型：自动生成的证据响应
                            "target_agent": agent_name,                    # 响应该请求的目标 Agent
                            "coverage": covers_request,                    # 核心字段：证据是否覆盖了请求要求
                            "forced_redispatch": False,                    # 标记这不是重跑，而是自动响应
                            "auto_response": True,                         # 标记该响应是 DynamicCheck 自动生成（非 Agent 实时执行）
                            "required_evidence": request.get("required_evidence", []),  # 请求方要求的证据项列表
                            "suggested_tools": request.get("suggested_tools", []),      # 请求方建议使用的工具列表
                        },
                    ))
                else:
                    # 场景 1：Agent 还没执行过 → 追加派发
                    requested.add(agent_name)

        # 已达最大调度轮次：不再追加派发，只返回自动生成的响应
        if state.dispatch_round >= state.max_dispatch_rounds:
            logger.info(
                f"[DynamicCheck] 已达最大轮次 {state.max_dispatch_rounds}，进入聚合"
            )
            result = {
                "dispatched_agents": [],
                "messages": ["DynamicCheck: 达到最大轮次，停止追加派发"],
            }
            if auto_responses:
                result["agent_messages"] = auto_responses
                result["messages"].append(f"DynamicCheck: 自动补充 {len(auto_responses)} 条证据响应")
            # 把本轮收集到的 coverage 判定事件一并返回，通过 operator.add 累加到 state.trace_events
            if coverage_trace_events:
                result["trace_events"] = coverage_trace_events
            return result

        # 从 requested 中筛选出真正需要派发的 Agent
        # 规则：如果 Agent 已有结果且不在 force_requested 中，则不需要再跑
        new_dispatch = []
        for agent_name in requested:
            field = _result_fields.get(agent_name)
            already_done = field and getattr(state, field, None) is not None
            if not already_done or agent_name in force_requested:
                new_dispatch.append(agent_name)

        if new_dispatch:
            logger.info(
                f"[DynamicCheck] 发现协作请求，追加派发: {new_dispatch} "
                f"(轮次 {state.dispatch_round}/{state.max_dispatch_rounds})"
            )
            result = {
                "dispatched_agents": new_dispatch,
                # force_dispatched_agents：Dispatch 节点看到这些 Agent 会强制重跑（忽略缓存）
                "force_dispatched_agents": sorted(force_requested & set(new_dispatch)),
                "messages": [f"DynamicCheck: 追加派发 {new_dispatch}"],
            }
            if new_redispatched_request_ids:
                # redispatched_request_ids 用 operator.add 追加到 state，防止下次再重跑
                result["redispatched_request_ids"] = new_redispatched_request_ids
                result["messages"].append(
                    f"DynamicCheck: {len(new_redispatched_request_ids)} 个请求因证据覆盖不足触发定向重跑"
                )
            if auto_responses:
                result["agent_messages"] = auto_responses
                result["messages"].append(f"DynamicCheck: 自动补充 {len(auto_responses)} 条证据响应")
            # 把本轮收集到的 coverage 判定事件一并返回，通过 operator.add 累加到 state.trace_events
            if coverage_trace_events:
                result["trace_events"] = coverage_trace_events
            return result

        # 没有任何协作请求需要处理，直接进入聚合
        logger.info("[DynamicCheck] 无协作请求，进入聚合")
        result = {"dispatched_agents": [], "messages": ["DynamicCheck: 无需追加派发"]}
        if auto_responses:
            result["agent_messages"] = auto_responses
            result["messages"].append(f"DynamicCheck: 自动补充 {len(auto_responses)} 条证据响应")
        # 把本轮收集到的 coverage 判定事件一并返回，通过 operator.add 累加到 state.trace_events
        if coverage_trace_events:
            result["trace_events"] = coverage_trace_events
        return result

    return dynamic_check_node
