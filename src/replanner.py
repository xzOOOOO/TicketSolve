"""执行失败后的 Critic（评判器），用于 Executor 之后的重规划。

Critic 读取执行器的真实 stdout/stderr/trace，从以下选项中做出选择：
retry（重试）、re-diagnose（重新诊断）、rollback（回滚）、escalate（升级人工处理）。
设计上采用纯规则实现，确保可审计、可单元测试，无需再调用一次 LLM。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


RETRY = "retry"
RE_DIAGNOSE = "re-diagnose"
ROLLBACK = "rollback"
ESCALATE = "escalate" # 升级人工处理
VERIFY = "verify"


def make_replanner_decision(
    execution_result: dict[str, Any] | None,
    execution_trace: list[dict[str, Any]],
    replanner_round: int,
    max_replanner_rounds: int,
) -> dict[str, Any]:
    """根据执行结果和轨迹，做出重规划决策。

    参数：
        execution_result: Executor 的最终结果字典，通常包含 overall_status 等字段。
        execution_trace: 执行轨迹列表，每一步的执行记录（含 stdout/stderr/exit_code）。
        replanner_round: 当前已介入的轮次（从 0 开始）。
        max_replanner_rounds: 允许的最大介入轮次，用于防死循环。

    返回：
        决策字典，包含 decision、failure_type、reason、evidence、时间戳等。
    """
    execution_result = execution_result or {}
    overall_status = execution_result.get("overall_status", "unknown")

    # 如果执行成功，直接进入验证环节
    if overall_status == "success":
        return _decision(
            decision=VERIFY,
            failure_type="none",
            reason="执行器成功完成",
            replanner_round=replanner_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence={"overall_status": overall_status},
        )

    # 从轨迹中提取最近一次失败的执行记录和回滚记录
    failed_execute = _last_failed_execute(execution_trace) # 最近一次失败的执行记录
    rollback_trace = _last_rollback_trace(execution_trace) # 最近一次回滚记录
    evidence = _build_evidence(execution_result, failed_execute, rollback_trace) # 证据
    failure_type = _classify_failure(failed_execute, execution_result) # 失败类型
    next_round = replanner_round + 1 # 下一个重规划轮次

    # 如果 trace 中已经有成功的回滚操作，停止继续修复，保存状态
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

    # 如果重试预算已耗尽，升级到人工处理
    if replanner_round >= max_replanner_rounds:
        return _decision(
            decision=ESCALATE,
            failure_type=failure_type,
            reason="重规划重试预算已耗尽",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 命令不允许、诊断不匹配、缺工具上下文 → 方案本身可能有问题，回到诊断链路重新生成
    if failure_type in {"command_not_allowed", "diagnosis_mismatch", "tooling_gap"}:
        return _decision(
            decision=RE_DIAGNOSE,
            failure_type=failure_type,
            reason="修复方案或诊断可能存在问题，需要基于新的诊断重新生成",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 环境临时不可用（连接拒绝、超时、Docker 未就绪等）→ 直接重试执行
    if failure_type == "environment_not_ready":
        return _decision(
            decision=RETRY,
            failure_type=failure_type,
            reason="环境看起来是临时不可用；重试执行",
            replanner_round=next_round,
            max_replanner_rounds=max_replanner_rounds,
            evidence=evidence,
        )

    # 权限/高风险失败
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

    # 兜底：未分类的执行失败，先重试一次，再不行就升级
    return _decision(
        decision=RETRY,
        failure_type=failure_type,
        reason="未分类的执行器失败；在升级前重试一次",
        replanner_round=next_round,
        max_replanner_rounds=max_replanner_rounds,
        evidence=evidence,
    )


def _decision(
    *,
    decision: str,
    failure_type: str,
    reason: str,
    replanner_round: int,
    max_replanner_rounds: int,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """构造统一的决策返回字典。"""
    return {
        "decision": decision,
        "failure_type": failure_type,
        "reason": reason,
        "replanner_round": replanner_round,
        "max_replanner_rounds": max_replanner_rounds,
        "evidence": evidence,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def _last_failed_execute(execution_trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从执行轨迹中倒序查找最近一次失败的 execute 记录。"""
    for item in reversed(execution_trace or []):
        if item.get("trace_type") == "execute" and item.get("success") is False:
            return item
    return None


def _last_rollback_trace(execution_trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从执行轨迹中倒序查找最近一次 rollback 或 rollback_skipped 记录。"""
    for item in reversed(execution_trace or []):
        if item.get("trace_type") in {"rollback", "rollback_skipped"}:
            return item
    return None


def _build_evidence(
    execution_result: dict[str, Any],
    failed_execute: dict[str, Any] | None,
    rollback_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    """收集决策所需的证据信息。

    包括整体状态、失败步骤的命令、退出码、输出日志等。
    对 stdout/stderr 做截断（最多 1000 字符），防止证据过大。
    """
    failed_execute = failed_execute or {}
    evidence = {
        "overall_status": execution_result.get("overall_status"),
        "summary": execution_result.get("summary", ""),
        "step_id": failed_execute.get("step_id"),
        "command": failed_execute.get("command", ""),
        "exit_code": failed_execute.get("exit_code"),
        "stdout": (failed_execute.get("stdout") or "")[:1000],
        "stderr": (failed_execute.get("stderr") or "")[:1000],
    }
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
    - command_not_allowed: 命令或 Action DSL 不被允许
    - environment_not_ready: 环境或依赖临时不可用
    - permission_or_privilege: 权限不足或高风险操作
    - diagnosis_mismatch: 诊断目标不匹配
    - tooling_gap: 缺少工具或上下文
    - unknown: 无法识别的失败
    """
    if not failed_execute:
        # 没有失败的执行记录，但也没有执行过任何步骤 → 缺工具/上下文
        if not execution_result.get("executed_steps"):
            return "tooling_gap"
        return "unknown"

    exit_code = failed_execute.get("exit_code")
    stdout = str(failed_execute.get("stdout") or "")
    stderr = str(failed_execute.get("stderr") or "")
    command = str(failed_execute.get("command") or "")
    text = f"{stdout}\n{stderr}\n{command}".lower()

    # 命令不被允许（白名单、权限、非法 DSL）
    if exit_code == 126 or _contains_any(text, [
        "白名单",
        "whitelist",
        "not allowed",
        "not permitted",
        "invalid action dsl",
        "unsupported action_type",
    ]):
        return "command_not_allowed"

    # 环境未就绪（连接拒绝、超时、Docker 问题）
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

    # 权限或特权问题
    if _contains_any(text, [
        "permission denied",
        "insufficient privileges",
        "operation not permitted",
        "access denied",
    ]):
        return "permission_or_privilege"

    # 诊断不匹配（目标错误、404、意外诊断）
    if _contains_any(text, [
        "unknown fault",
        "wrong target",
        "http 404",
        "not found",
        "unexpected diagnosis",
    ]):
        return "diagnosis_mismatch"

    # 缺少工具或上下文
    if _contains_any(text, [
        "missing",
        "module not found",
        "no such file or directory",
        "tool",
        "diagnostic",
    ]):
        return "tooling_gap"

    return "unknown"


def _contains_any(text: str, needles: list[str]) -> bool:
    """判断 text 中是否包含 needles 列表中的任意一个子串。"""
    return any(needle in text for needle in needles)
