"""
案例记忆节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def create_case_memory_node(limit: int = 3):
    """
    创建案例记忆检索节点工厂函数。

    新工单进入 Supervisor 前，先用症状检索历史相似案例，并把压缩后的
    case_context 写入 state，供 Supervisor 和 FixAgent 复用经验。

    为什么需要案例库：
    - LLM 可能没见过某些特定故障模式，历史案例能提供参考
    - 相似案例的修复方案可以直接复用或改编，提高修复效率
    - 案例上下文帮助 Supervisor 更准确地判断该派发哪些 Agent

    参数：
        limit: 最多检索几个相似案例，默认 3 个

    返回：
        异步节点函数 case_memory_node(state) -> dict
    """

    async def case_memory_node(state: SystemState) -> dict:
        # cases：检索到的相似历史案例列表，每个案例包含 case_id、symptom、diagnosis、fix_plan 等
        cases = retrieve_similar_cases(state.symptom, limit=limit)
        # case_context：将案例列表压缩为人可读的文本摘要，供 LLM 使用
        case_context = format_case_context(cases)

        logger.info(
            f"[CaseMemory] 检索完成: ticket_id={state.ticket_id}, "
            f"similar_cases={len(cases)}"
        )

        # audit_log：审计日志记录，用于追溯案例检索操作
        audit_log = {
            "ticket_id": state.ticket_id,           # 关联工单 ID
            "agent_name": "case_memory",            # 操作者标识
            "action_type": "case_retrieval",        # 动作类型：案例检索
            "action_detail": {                      # 动作详情
                "case_count": len(cases),           # 检索到的案例数量
                "case_ids": [case.get("case_id") for case in cases],  # 案例 ID 列表
                "library_path": str(DEFAULT_CASE_LIBRARY_PATH),         # 案例库路径
            },
            "input_context": {                      # 输入上下文
                "symptom": state.symptom,           # 当前工单症状
            },
            "output_result": {                      # 输出结果
                "similar_cases": cases,             # 检索到的完整案例数据
            },
            "dispatch_round": state.dispatch_round, # 当前调度轮次
        }
        # 生成标准化 Trace 事件：案例检索结果作为 observation_received 记录
        trace_event = make_trace_event(
            "observation_received",                 # 事件类型：观测到输入/输出
            ticket_id=state.ticket_id,              # 工单 ID
            agent_name="case_memory",               # 产生事件的节点
            input_data={"symptom": state.symptom},  # 输入：症状描述
            output_data={"similar_cases": cases},   # 输出：相似案例
            metadata={                              # 元数据
                "case_count": len(cases),           # 案例数量
                "case_ids": [case.get("case_id") for case in cases],
                "dispatch_round": state.dispatch_round,
            },
        )

        # 返回字典：LangGraph 会自动合并到 SystemState 中
        return {
            "similar_cases": cases,                 # 相似案例列表（覆盖 state.similar_cases）
            "case_context": case_context,           # 案例上下文文本（覆盖 state.case_context）
            "case_memory": {                        # 案例库元数据（覆盖 state.case_memory）
                "library_path": str(DEFAULT_CASE_LIBRARY_PATH),
                "similar_case_count": len(cases),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),  # ISO 格式 UTC 时间
            },
            "messages": [f"CaseMemory: 检索到 {len(cases)} 个相似历史案例"],
            "audit_logs": [audit_log],              # 审计日志（通过 operator.add 累加）
            "trace_events": [trace_event],          # Trace 事件（通过 operator.add 累加）
        }

    return case_memory_node
