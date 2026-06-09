# ============================================================
# replanner.py：重规划决策模块（Critic / 评判器）
#
# 作用：
#   当修复方案执行失败时，根据执行结果和轨迹做出决策：
#   重试？重新诊断？回滚？还是升级人工处理？
#
# 核心设计：
#   - 纯规则实现，不调用 LLM，确保可审计、可单元测试
#   - 基于 exit_code 和 stdout/stderr 关键词进行失败分类
#   - 有重试预算限制（max_replanner_rounds），防死循环
#
# 为什么不用 LLM 做决策？
#   1. 规则足够覆盖常见失败类型，LLM 反而可能产生幻觉
#   2. 规则执行快（毫秒级），LLM 调用慢（秒级）
#   3. 规则可解释，能明确告诉用户"为什么决定回滚"
# ============================================================

# 模块文档字符串：说明模块用途和设计意图
"""执行失败后的 Critic（评判器），用于 Executor 之后的重规划。

Critic 读取执行器的真实 stdout/stderr/trace，从以下选项中做出选择：
retry（重试）、re-diagnose（重新诊断）、rollback（回滚）、escalate（升级人工处理）。
设计上采用纯规则实现，确保可审计、可单元测试，无需再调用一次 LLM。
"""

# from __future__ import annotations：启用 PEP 563，支持 dict[str, Any] 等类型注解
from __future__ import annotations

# datetime/timezone：生成决策时间戳
from datetime import datetime, timezone
# typing.Any：任意类型，用于兼容各种输入
from typing import Any


# ═══════════════════════════════════════════════════════════
# 一、决策类型常量
# ═══════════════════════════════════════════════════════════

# RETRY：重试当前修复方案
# 使用场景：环境临时不可用（如网络超时、Docker 未就绪），可能下次就成功了
RETRY = "retry"

# RE_DIAGNOSE：重新诊断
# 使用场景：修复方案本身有问题（如命令不允许、诊断不匹配），需要回到诊断链路重新生成方案
RE_DIAGNOSE = "re-diagnose"

# ROLLBACK：执行回滚
# 使用场景：已经尝试过回滚且成功，或者权限/高风险失败且没有回滚证据
ROLLBACK = "rollback"

# ESCALATE：升级人工处理
# 使用场景：重试预算耗尽，或存在权限/高风险失败且无法自动处理
ESCALATE = "escalate"

# VERIFY：验证通过
# 使用场景：执行成功，进入验证环节
VERIFY = "verify"


# ═══════════════════════════════════════════════════════════
# 二、核心决策函数
# ═══════════════════════════════════════════════════════════

def make_replanner_decision(
    execution_result: dict[str, Any] | None,
    execution_trace: list[dict[str, Any]],
    replanner_round: int,
    max_replanner_rounds: int,
) -> dict[str, Any]:
    """根据执行结果和轨迹，做出重规划决策。

    决策流程（按优先级排序）：
    1. 执行成功 → VERIFY（进入验证）
    2. 已有成功回滚记录 → ROLLBACK（停止修复）
    3. 重试预算耗尽 → ESCALATE（升级人工）
    4. 方案/诊断问题 → RE_DIAGNOSE（重新诊断）
    5. 环境临时问题 → RETRY（重试）
    6. 权限/高风险失败 → ESCALATE 或 ROLLBACK
    7. 兜底 → RETRY（重试一次再升级）

    参数：
        execution_result: Executor 的最终结果字典，通常包含 overall_status 等字段
        execution_trace: 执行轨迹列表，每一步的执行记录（含 stdout/stderr/exit_code）
        replanner_round: 当前已介入的轮次（从 0 开始）
        max_replanner_rounds: 允许的最大介入轮次，用于防死循环

    返回：
        决策字典，包含 decision、failure_type、reason、evidence、时间戳等
    """
    # execution_result 可能为 None，统一处理成空字典
    execution_result = execution_result or {}
    # overall_status：执行器的整体状态，"success" 表示全部步骤成功
    overall_status = execution_result.get("overall_status", "unknown")

    # 分支 1：执行成功，直接进入验证环节
    if overall_status == "success":
        return _decision(
            decision=VERIFY,
            failure_type="none",
            reason="执行器成功完成",
            replanner_round=replanner_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence={"overall_status": overall_status},
        )

    # 从轨迹中提取关键信息
    # failed_execute：最近一次失败的执行记录（用于分析失败原因）
    failed_execute = _last_failed_execute(execution_trace)
    # rollback_trace：最近一次回滚记录（判断是否已尝试过回滚）
    rollback_trace = _last_rollback_trace(execution_trace)
    # evidence：收集决策所需的证据信息
    evidence = _build_evidence(execution_result, failed_execute, rollback_trace)
    # failure_type：对失败进行分类（如 environment_not_ready、permission_or_privilege 等）
    failure_type = _classify_failure(failed_execute, execution_result)
    # next_round：下一轮次，用于记录决策历史
    next_round = replanner_round + 1

    # 分支 2：如果 trace 中已经有成功的回滚操作，停止继续修复
    # 设计意图：回滚成功意味着系统已恢复到修复前状态，继续修复可能引入新问题
    if (
        rollback_trace
        and rollback_trace.get("trace_type") == "rollback"
        and rollback_trace.get("success") is True
    ):
        return _decision(
            decision=ROLLBACK,
            failure_type="rollback_already_attempted",
            reason="执行器已尝试过回滚；停止修复并保存状态",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 分支 3：重试预算已耗尽，升级到人工处理
    # 设计意图：防止无限循环，保证系统不会卡死在某一步
    if replanner_round >= max_replanner_rounds:
        return _decision(
            decision=ESCALATE,
            failure_type=failure_type,
            reason="重规划重试预算已耗尽",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 分支 4：方案本身可能有问题，回到诊断链路重新生成
    # 这些失败类型表明修复方案或诊断结论不正确，重试也没用
    if failure_type in {"command_not_allowed", "diagnosis_mismatch", "tooling_gap"}:
        return _decision(
            decision=RE_DIAGNOSE,
            failure_type=failure_type,
            reason="修复方案或诊断可能存在问题，需要基于新的诊断重新生成",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 分支 5：环境临时不可用 → 直接重试执行
    # 设计意图：网络超时、Docker 未就绪等问题通常是暂时的，重试可能成功
    if failure_type == "environment_not_ready":
        return _decision(
            decision=RETRY,
            failure_type=failure_type,
            reason="环境看起来是临时不可用；重试执行",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 分支 6：权限/高风险失败
    if failure_type in {"permission_or_privilege", "rollback_already_attempted"}:
        # 如果已经尝试过回滚，停止并保存
        if rollback_trace:
            return _decision(
                decision=ROLLBACK,
                failure_type=failure_type,
                reason="执行器已尝试过回滚；停止修复并保存状态",
                replanner_round=next_round,
                max_replanner_rounds=max_replanner_rounds,
                evidence=evidence,
            )
        # 没有回滚证据的权限/高风险失败，升级到人工处理
        return _decision(
            decision=ESCALATE,
            failure_type=failure_type,
            reason="存在权限或高风险失败，且没有回滚证据",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 分支 7：兜底策略
    # 未分类的执行失败，先重试一次，如果还是失败，下次可能走到 ESCALATE
    return _decision(
        decision=RETRY,
        failure_type=failure_type,
        reason="未分类的执行器失败；在升级前重试一次",
        replanner_round=next_round,
        max_replanner_rounds=max_replanner_rounds,
        evidence=evidence,
    )


# ═══════════════════════════════════════════════════════════
# 三、内部辅助函数
# ═══════════════════════════════════════════════════════════

def _decision(
    *,
    decision: str,
    failure_type: str,
    reason: str,
    replanner_round: int,
    max_replanner_rounds: int,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """构造统一的决策返回字典。

    所有决策分支都调用此函数，保证返回结构一致。

    参数：
        decision: 决策类型（RETRY/RE_DIAGNOSE/ROLLBACK/ESCALATE/VERIFY）
        failure_type: 失败分类
        reason: 决策理由（人可读）
        replanner_round: 当前轮次
        max_replanner_rounds: 最大轮次
        evidence: 证据字典

    返回：
        标准决策字典
    """
    return {
        "decision": decision,  # 决策类型
        "failure_type": failure_type,  # 失败分类
        "reason": reason,  # 决策理由
        "replanner_round": replanner_round,  # 当前轮次
        "max_replanner_rounds": max_replanner_rounds,  # 最大允许轮次
        "evidence": evidence,  # 决策证据
        "decided_at": datetime.now(timezone.utc).isoformat(),  # 决策时间戳
    }


def _last_failed_execute(execution_trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从执行轨迹中倒序查找最近一次失败的 execute 记录。

    参数：
        execution_trace：执行轨迹列表

    返回：
        失败的执行记录字典，如果没有返回 None
    """
    # reversed：倒序遍历，找到最近的一次失败
    for item in reversed(execution_trace or []):
        # trace_type == "execute" 且 success == False 表示执行失败
        if item.get("trace_type") == "execute" and item.get("success") is False:
            return item
    return None


def _last_rollback_trace(execution_trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从执行轨迹中倒序查找最近一次 rollback 或 rollback_skipped 记录。

    参数：
        execution_trace：执行轨迹列表

    返回：
        回滚记录字典，如果没有返回 None
    """
    for item in reversed(execution_trace or []):
        # trace_type 为 rollback 或 rollback_skipped 都表示有过回滚尝试
        if item.get("trace_type") in {"rollback", "rollback_skipped"}:
            return item
    return None


def _build_evidence(
    execution_result: dict[str, Any],
    failed_execute: dict[str, Any] | None,
    rollback_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    """收集决策所需的证据信息。

    证据包括整体状态、失败步骤的命令、退出码、输出日志等。
    对 stdout/stderr 做截断（最多 1000 字符），防止证据过大。

    参数：
        execution_result：执行器最终结果
        failed_execute：最近一次失败的执行记录
        rollback_trace：最近一次回滚记录

    返回：
        证据字典
    """
    failed_execute = failed_execute or {}
    evidence = {
        "overall_status": execution_result.get("overall_status"),  # 整体状态
        "summary": execution_result.get("summary", ""),  # 执行摘要
        "step_id": failed_execute.get("step_id"),  # 失败步骤编号
        "command": failed_execute.get("command", ""),  # 执行的命令
        "exit_code": failed_execute.get("exit_code"),  # 退出码
        # stdout/stderr 截断到 1000 字符，防止证据过大影响日志和传输
        "stdout": (failed_execute.get("stdout") or "")[:1000],
        "stderr": (failed_execute.get("stderr") or "")[:1000],
    }
    # 如果有回滚记录，也加入证据
    if rollback_trace:
        evidence["rollback"] = {
            "trace_type": rollback_trace.get("trace_type"),
            "success": rollback_trace.get("success"),
            "stderr": (rollback_trace.get("stderr") or "")[:1000],
        }
    return evidence


def _classify_failure(
    failed_execute: dict[str, Any] | None,
    execution_result: dict[str, Any],
) -> str:
    """对失败进行分类。

    基于 exit_code 和 stdout/stderr/command 中的关键词，判断失败属于哪一类：
    - command_not_allowed: 命令或 Action DSL 不被允许（exit_code=126 或关键词匹配）
    - environment_not_ready: 环境或依赖临时不可用（exit_code=124 或关键词匹配）
    - permission_or_privilege: 权限不足或高风险操作
    - diagnosis_mismatch: 诊断目标不匹配
    - tooling_gap: 缺少工具或上下文
    - unknown: 无法识别的失败

    参数：
        failed_execute：最近一次失败的执行记录
        execution_result：执行器最终结果

    返回：
        失败类型字符串
    """
    # 如果没有失败的执行记录
    if not failed_execute:
        # 如果也没有执行过任何步骤，说明缺少工具或上下文
        if not execution_result.get("executed_steps"):
            return "tooling_gap"
        return "unknown"

    # 提取关键字段
    exit_code = failed_execute.get("exit_code")
    stdout = str(failed_execute.get("stdout") or "")
    stderr = str(failed_execute.get("stderr") or "")
    command = str(failed_execute.get("command") or "")
    # text：把 stdout、stderr、command 拼接，统一小写，用于关键词匹配
    text = f"{stdout}\n{stderr}\n{command}".lower()

    # 分类 1：命令不被允许（白名单、权限、非法 DSL）
    # exit_code=126 是 Linux 标准退出码，表示"命令不可执行（权限问题或不是可执行文件）"
    if exit_code == 126 or _contains_any(text, [
        "白名单",
        "whitelist",
        "not allowed",
        "not permitted",
        "invalid action dsl",
        "unsupported action_type",
    ]):
        return "command_not_allowed"

    # 分类 2：环境未就绪（连接拒绝、超时、Docker 问题）
    # exit_code=124 通常是 timeout 命令的退出码
    if exit_code == 124 or _contains_any(text, [
        "connection refused",
        "timed out",
        "timeout",
        "service not responding",
        "temporarily unavailable",
        "resource temporarily unavailable",
        "cannot connect to the docker daemon",
        "docker command not found",
        "no such container",
        "is not running",
    ]):
        return "environment_not_ready"

    # 分类 3：权限或特权问题
    if _contains_any(text, [
        "permission denied",
        "insufficient privileges",
        "operation not permitted",
        "access denied",
    ]):
        return "permission_or_privilege"

    # 分类 4：诊断不匹配（目标错误、404、意外诊断）
    if _contains_any(text, [
        "unknown fault",
        "wrong target",
        "http 404",
        "not found",
        "unexpected diagnosis",
    ]):
        return "diagnosis_mismatch"

    # 分类 5：缺少工具或上下文
    if _contains_any(text, [
        "missing",
        "module not found",
        "no such file or directory",
        "tool",
        "diagnostic",
    ]):
        return "tooling_gap"

    # 兜底：无法识别的失败类型
    return "unknown"


def _contains_any(text: str, needles: list[str]) -> bool:
    """判断 text 中是否包含 needles 列表中的任意一个子串。

    参数：
        text：被搜索的文本
        needles：关键词列表

    返回：
        True 如果包含任意一个关键词，否则 False
    """
    return any(needle in text for needle in needles)
