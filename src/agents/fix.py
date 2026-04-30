from langchain_core.language_models import BaseChatModel
from agents.base import BaseAgent
from state import SystemState
from prompts import FIX_PROMPT
from schemas import FixPlanOutput
from logger import logger

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

    Structured Output 改造说明：
    - 原方案：LLM 输出 JSON 字符串 -> parse_json_content 解析 -> dict
    - 新方案：self.llm.with_structured_output(FixPlanOutput)
             直接返回嵌套的 Pydantic 对象（含 FixStepOutput 列表和 VerificationOutput）
    - 注意：修复方案结构最复杂，手动解析最容易出错，Structured Output 收益最大
    """
    name = "fix_agent"
    role = "修复方案生成专家"

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm, tools=None)
        # FixPlanOutput 包含嵌套的 FixStepOutput 和 VerificationOutput
        # with_structured_output 会自动处理嵌套 schema
        self._structured_llm = self.llm.with_structured_output(FixPlanOutput)

    async def run(self, state: SystemState) -> dict:
        """执行修复方案生成

        流程：
        1. 根据 diagnosis_type 选择对应的诊断结果（优先使用聚合结果）
        2. 使用 Structured Output 调用 LLM 生成修复方案
        3. 返回包含 fix_plan 的状态更新字典
        """
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

            logger.info(f"[{self.name}] 开始生成修复方案: diagnosis_type={diagnosis_type}")

            # 使用 Structured Output 生成修复方案
            # LLM 直接返回 FixPlanOutput 对象，包含嵌套的 steps 和 verification
            result = await (FIX_PROMPT | self._structured_llm).ainvoke({
                "diagnosis_type": diagnosis_type,
                "diagnosis_result": str(diagnosis_result),
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
                },
                "output_result": result_dict,
                "dispatch_round": state.dispatch_round,
            }

            return {
                "fix_plan": result_dict,
                "messages": [
                    f"Fix Agent: 生成修复方案 {result_dict.get('plan_id')} - "
                    f"风险等级: {result_dict.get('risk_level')}"
                ],
                "audit_logs": [audit_log],
            }
        except Exception as e:
            logger.exception(f"[{self.name}] 执行失败: {e}")
            return {
                "fix_plan": _DEFAULT_FIX_PLAN.copy(),
                "messages": [f"Fix Agent: 方案生成失败 - {str(e)}"],
            }
