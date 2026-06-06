from langchain_core.language_models import BaseChatModel
from agents.base import BaseAgent
from state import SystemState
from prompts import FIX_PROMPT
from schemas import FixPlanOutput
# make_trace_event：标准化 Trace 事件工厂，所有 Agent 都用同一套事件格式
from trace_events import make_trace_event
# build_protocol_context：从所有 Agent 消息中构建协议上下文（含假设、证据、冲突统计）
# FixAgent 使用它来了解 Agent 间协作的结论，辅助生成更精准的修复方案
from agent_protocol import build_protocol_context
from logger import logger
from config import settings

# 修复方案生成失败时的默认兜底数据
_DEFAULT_FIX_PLAN = {
    "plan_id": "PLAN-ERROR",
    "description": "无法生成方案",
    "risk_level": "unknown",
    "prerequisites": [],
    "steps": [],
    "verification": {"commands": [], "expected_result": ""},
    "estimated_time": "0",
}


class FixAgent(BaseAgent):
    """修复方案生成专家 Agent

    职责：根据诊断结果生成可执行的修复方案。
    """
    name = "fix_agent"
    role = "修复方案生成专家"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, tools=None)
        # FixPlanOutput 包含嵌套的 FixStepOutput 和 VerificationOutput
        # with_structured_output 会自动处理嵌套 schema
        self._structured_llm = self.llm.with_structured_output(FixPlanOutput)
        self._chain = (FIX_PROMPT | self._structured_llm).with_retry(**settings.get_retry_config())

    async def run(self, state: SystemState) -> dict:
        """执行修复方案生成

        流程：
        1. 根据 diagnosis_type 选择对应的诊断结果（优先使用聚合结果）
        2. 使用 Structured Output 调用 LLM 生成修复方案
        3. 返回包含 fix_plan 的状态更新字典
        """
        # 初始化 Trace 事件列表，首先记录 agent_started 事件
        trace_events = [make_trace_event(
            "agent_started",
            ticket_id=state.ticket_id,
            agent_name=self.name,
            input_data={"diagnosis_type": state.diagnosis_type},
            metadata={
                "dispatch_round": state.dispatch_round,
                "similar_case_ids": [case.get("case_id") for case in state.similar_cases],
            },
        )]
        try:
            diagnosis_type = state.diagnosis_type

            # 优先使用聚合诊断结果，否则使用单个 Agent 的诊断结果
            if state.aggregated_diagnosis:
                diagnosis_result = state.aggregated_diagnosis
                logger.info(f"[{self.name}] 使用聚合诊断结果")
            elif diagnosis_type == "db":
                diagnosis_result = state.db_agent_result
            elif diagnosis_type == "net":
                diagnosis_result = state.net_agent_result
            elif diagnosis_type == "app":
                diagnosis_result = state.app_agent_result
            else:
                diagnosis_result = {}

            # 构建协议上下文：从所有 Agent 消息中提取假设、证据、冲突等信息
            # 这样 FixAgent 不仅知道 "诊断结论是什么"，还知道：
            # - 哪个假设在协议中胜出（winning_hypothesis_id）
            # - 各假设的得分情况（hypothesis_scores）
            # - Agent 之间有没有冲突（conflicts）
            # 这些信息帮助 FixAgent 生成更精准的修复方案
            protocol_context = build_protocol_context(state.agent_messages)
            if isinstance(diagnosis_result, dict):
                # 把协议上下文和摘要注入诊断结果，一起传给 LLM
                # protocol_context["text"] 是人可读的文本摘要
                # protocol_context["protocol_summary"] 是结构化的统计字典
                diagnosis_result = {
                    **diagnosis_result,
                    "protocol_context": protocol_context.get("text"),
                    "protocol_summary": protocol_context.get("protocol_summary"),
                }

            logger.info(f"[{self.name}] 开始生成修复方案: diagnosis_type={diagnosis_type}")

            # 使用 Structured Output 生成修复方案
            # LLM 直接返回 FixPlanOutput 对象，包含嵌套的 steps 和 verification
            result = await self._chain.ainvoke({
                "diagnosis_type": diagnosis_type,
                "diagnosis_result": str(diagnosis_result),
                "case_context": state.case_context or "无相似历史案例。",
            })

            # 兜底处理
            if result is None:
                result_dict = _DEFAULT_FIX_PLAN.copy()
            else:
                result_dict = result.model_dump()

            logger.info(
                f"[{self.name}] 方案生成完成: plan_id={result_dict.get('plan_id')}, "
                f"risk_level={result_dict.get('risk_level')}"
            )

            # 记录审计日志：修复方案生成
            audit_log = {
                "ticket_id": state.ticket_id,
                "agent_name": self.name,
                "action_type": "fix_plan",
                "action_detail": {
                    "plan_id": result_dict.get("plan_id"),
                    "description": result_dict.get("description"),
                    "risk_level": result_dict.get("risk_level"),
                    "steps_count": len(result_dict.get("steps", [])),
                    "prerequisites": result_dict.get("prerequisites", []),
                    "estimated_time": result_dict.get("estimated_time"),
                },
                "input_context": {
                    "diagnosis_type": diagnosis_type,
                    "diagnosis_result": str(diagnosis_result),
                    "case_context": state.case_context,
                    "similar_case_ids": [case.get("case_id") for case in state.similar_cases],
                },
                "output_result": result_dict,
                "dispatch_round": state.dispatch_round,
            }
            # 记录修复方案生成的标准化 Trace 事件，附带 plan_id、risk_level、steps_count 等元数据
            trace_events.append(make_trace_event(
                "plan_generated",
                ticket_id=state.ticket_id,
                agent_name=self.name,
                input_data={
                    "diagnosis_type": diagnosis_type,
                    "diagnosis_result": str(diagnosis_result),
                    "case_context": state.case_context,
                },
                output_data=result_dict,
                metadata={
                    "plan_id": result_dict.get("plan_id"),
                    "risk_level": result_dict.get("risk_level"),
                    "steps_count": len(result_dict.get("steps", [])),
                    "dispatch_round": state.dispatch_round,
                },
            ))

            return {
                "fix_plan": result_dict,
                "messages": [
                    f"Fix Agent: 生成修复方案 {result_dict.get('plan_id')} - "
                    f"风险等级: {result_dict.get('risk_level')}"
                ],
                "audit_logs": [audit_log],
                "trace_events": trace_events,
            }
        except Exception as e:
            # FixAgent 异常时也要记录 failure 状态的 Trace 事件，保证失败路径可追溯
            logger.exception(f"[{self.name}] 执行失败: {e}")
            trace_events.append(make_trace_event(
                "plan_generated",
                ticket_id=state.ticket_id,
                agent_name=self.name,
                status="failure",
                output_data=_DEFAULT_FIX_PLAN.copy(),
                error=str(e),
                metadata={"dispatch_round": state.dispatch_round},
            ))
            return {
                "fix_plan": _DEFAULT_FIX_PLAN.copy(),
                "messages": [f"Fix Agent: 方案生成失败 - {str(e)}"],
                # 异常路径也要返回 trace_events，避免前面已记录的事件丢失
                "trace_events": trace_events,
            }
