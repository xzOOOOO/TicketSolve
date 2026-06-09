"""
闭环执行器节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def _build_action_trace_events(
    *,
    ticket_id: str,
    plan_dict: dict,
    execution_trace: list[dict],
    dispatch_round: int,
) -> list[dict]:
    """
    将执行器的 execution_trace 逐条拆分为标准化的 action_executed 事件。

    原来 execution_trace 是一个列表，retry/rollback/失败混在一起难以单步分析。
    现在每条执行记录都变成独立事件，trace_type 标明是 execute/retry/rollback，
    方便外部评测系统按 step_id 或 timestamp 排序后逐条分析。

    参数：
        ticket_id: 工单 ID
        plan_dict: 修复方案字典
        execution_trace: 执行器输出的执行记录列表
        dispatch_round: 当前调度轮次

    返回：
        标准化 Trace 事件列表
    """
    events = []
    for item in execution_trace:
        # success：从执行记录中提取成功标志（布尔值或 None）
        success = item.get("success")
        # status：布尔值转标准状态字符串（success/failure/skipped）
        status = status_from_success(success if isinstance(success, bool) else None)
        # trace_type：标明动作类型：execute（正常执行）、retry（重试）、rollback（回滚）
        trace_type = item.get("trace_type", "execute")
        events.append(make_trace_event(
            "action_executed",                  # 事件类型：动作执行
            ticket_id=ticket_id,                # 工单 ID
            agent_name="executor",              # 产生事件的节点
            status=status,                      # 执行状态
            input_data={                        # 输入数据
                "plan_id": plan_dict.get("plan_id"),
                "step_id": item.get("step_id"),
                "command": item.get("command"),
                "trace_type": trace_type,
            },
            output_data=item,                   # 输出数据：完整执行记录
            # error：只有失败时才记录 stderr，否则为 None
            error=item.get("stderr") if status == "failure" else None,
            metadata={                          # 元数据
                "plan_id": plan_dict.get("plan_id"),
                "trace_type": trace_type,
                "attempt": item.get("attempt"),  # 重试次数
                "dispatch_round": dispatch_round,
            },
            timestamp=item.get("timestamp"),    # 执行时间戳
        ))
    return events


def create_executor_node(llm=None):
    """
    创建闭环执行器节点工厂函数。

    核心改造：不是一次性跑完所有步骤，而是每一步都观察真实结果，
    根据结果决策下一步（重试/调整/回滚）。

    执行流程（Observe-Decide-Act 循环）：
        执行步骤1 → 观察真实输出 → [成功] → 执行步骤2
                                 → [失败] → LLM分析 → [可重试] → 重试/调整
                                                     → [不可重试] → 回滚 → 报告失败

    执行模式：
    - EXECUTOR_MODE=mock: 使用 MockCommandRunner 模拟命令执行（默认，用于测试）
    - EXECUTOR_MODE=docker_lab: 使用 SafeDockerCommandRunner 执行靶场白名单命令

    参数：
        llm: LLM 实例（可选），用于执行失败时的错误分析

    返回：
        异步节点函数 executor_node(state) -> dict
    """
    async def executor_node(state: SystemState) -> dict:
        fix_plan = state.fix_plan
        # plan_dict：统一转为字典格式（FixPlan Pydantic 对象或 dict）
        plan_dict = fix_plan if isinstance(fix_plan, dict) else fix_plan.model_dump() if fix_plan else {}

        try:
            # 无修复方案或步骤为空时，直接返回 skipped 状态的事件，避免空跑
            if not plan_dict or not plan_dict.get("steps"):
                logger.warning("[Executor] 无修复方案或步骤为空")
                execution_result = {
                    "plan_id": plan_dict.get("plan_id"),
                    "executed_steps": [],
                    "overall_status": "skipped",
                    "summary": "无修复方案或步骤为空",
                }
                return {
                    "execution_result": execution_result,
                    "messages": ["Executor: 无修复方案可执行"],
                    # 生成标准化 Trace 事件：无方案可执行，标记为 skipped
                    "trace_events": [make_trace_event(
                        "action_executed",
                        ticket_id=state.ticket_id,
                        agent_name="executor",
                        status="skipped",
                        input_data={"plan_id": plan_dict.get("plan_id")},
                        output_data=execution_result,
                        metadata={"dispatch_round": state.dispatch_round},
                    )],
                }

            logger.info(
                f"[Executor] 开始闭环执行: plan_id={plan_dict.get('plan_id')}, "
                f"共 {len(plan_dict.get('steps', []))} 个步骤"
            )

            # 创建闭环执行器。默认仍使用 mock，只有显式配置 docker_lab 才真实执行。
            executor_mode = settings.EXECUTOR_MODE.lower()
            if executor_mode == "docker_lab":
                # docker_lab 模式：使用 SafeDockerCommandRunner，执行真实的 Docker 命令
                runner = SafeDockerCommandRunner()
            else:
                # mock 模式：使用 MockCommandRunner，模拟命令执行（15% 失败率）
                runner = MockCommandRunner(failure_rate=0.15)

            logger.info(f"[Executor] 执行器模式: {executor_mode}")
            # executor：闭环执行器实例，支持 Observe-Decide-Act 循环
            executor = ClosedLoopExecutor(
                command_runner=runner,       # 命令执行器
                llm=llm,                     # LLM（可选），用于错误分析
                max_retries_per_step=2,      # 每步最多重试 2 次
            )

            # 构建错误分析链（如果有 LLM）
            # error_analyzer：LangChain 管道，用于分析执行失败原因
            error_analyzer = None
            if llm:
                from schemas import ErrorAnalysisOutput

                # structured_llm：包装后的 LLM，输出会被强制解析为 ErrorAnalysisOutput
                structured_llm = llm.with_structured_output(ErrorAnalysisOutput)
                # ERROR_ANALYSIS_PROMPT | structured_llm：先应用 Prompt 模板，再调用 LLM
                error_analyzer = ERROR_ANALYSIS_PROMPT | structured_llm

            # 执行修复方案（闭环 Observe-Decide-Act 循环）
            exec_output = await executor.execute_plan(
                fix_plan=plan_dict,
                error_analyzer=error_analyzer,
            )

            # execution_result：执行结果摘要（overall_status、completed_steps 等）
            execution_result = exec_output["execution_result"]
            # execution_trace：执行轨迹，记录每步的执行详情（stdout、stderr、success 等）
            execution_trace = exec_output["execution_trace"]

            # audit_log：审计日志，记录执行过程统计
            audit_log = {
                "ticket_id": state.ticket_id,
                "agent_name": "executor",
                "action_type": "execute",
                "action_detail": {
                    "plan_id": plan_dict.get("plan_id"),
                    "overall_status": execution_result.get("overall_status"),
                    "completed_steps": execution_result.get("completed_steps"),
                    "total_steps": execution_result.get("total_steps"),
                    "trace_count": len(execution_trace),
                    "executor_mode": executor_mode,
                },
                "input_context": {
                    "plan_id": plan_dict.get("plan_id"),
                    "steps_count": len(plan_dict.get("steps", [])),
                },
                "output_result": execution_result,
                "dispatch_round": state.dispatch_round,
            }

            logger.info(f"[Executor] 执行完成: plan_id={plan_dict.get('plan_id')}")

            return {
                "execution_result": execution_result,
                "execution_trace": execution_trace,
                # 将闭环执行器的 trace 逐条拆分为标准化 action_executed 事件
                "trace_events": _build_action_trace_events(
                    ticket_id=state.ticket_id,
                    plan_dict=plan_dict,
                    execution_trace=execution_trace,
                    dispatch_round=state.dispatch_round,
                ),
                "messages": [
                    f"Executor: 执行{execution_result.get('overall_status', 'unknown')} - "
                    f"{execution_result.get('completed_steps', 0)}/{execution_result.get('total_steps', 0)} 步骤完成 "
                    f"(mode={executor_mode})"
                ],
                "audit_logs": [audit_log],
            }
        except Exception as e:
            # 执行器异常时也要生成标准化 Trace 事件，保证失败路径可追溯
            logger.exception(f"[Executor] 执行失败: {e}")
            execution_result = {
                "plan_id": plan_dict.get("plan_id") if plan_dict else None,
                "executed_steps": [],
                "overall_status": "failed",
                "summary": f"执行异常: {str(e)}",
            }
            return {
                "execution_result": execution_result,
                "messages": [f"Executor: 执行失败 - {str(e)}"],
                "trace_events": [make_trace_event(
                    "action_executed",
                    ticket_id=state.ticket_id,
                    agent_name="executor",
                    status="failure",
                    input_data={"plan_id": plan_dict.get("plan_id") if plan_dict else None},
                    output_data=execution_result,
                    error=str(e),
                    metadata={"dispatch_round": state.dispatch_round},
                )],
            }

    return executor_node
