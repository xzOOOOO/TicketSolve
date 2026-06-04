"""Agent memory / case library.

The library stores resolved incidents and retrieves similar cases for new
tickets before Supervisor/FixAgent make decisions. It intentionally uses a
small deterministic scorer so the behavior is transparent and easy to audit.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_LIBRARY_PATH = PROJECT_ROOT / "eval" / "case_library.json"

CASE_KEYWORDS = {
    "db": ["db", "database", "postgres", "postgresql", "psql", "数据库", "连接", "查询", "慢查询"],
    "app": ["app", "application", "process", "fastapi", "应用", "进程", "健康", "不可用"],
    "redis": ["redis", "cache", "缓存"],
    "nginx": ["nginx", "route", "路由", "网关", "反向代理"],
    "slow": ["slow", "latency", "timeout", "超时", "慢", "延迟"],
    "health": ["health", "健康", "探活"],
    "orders": ["orders", "pending", "订单"],
}


def load_cases(path: Path | str = DEFAULT_CASE_LIBRARY_PATH) -> list[dict[str, Any]]:
    case_path = Path(path)
    if not case_path.exists():
        return []
    try:
        data = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [case for case in data if isinstance(case, dict)]


def save_cases(cases: list[dict[str, Any]], path: Path | str = DEFAULT_CASE_LIBRARY_PATH) -> None:
    case_path = Path(path)
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def retrieve_similar_cases(
    symptom: str,
    *,
    limit: int = 3,
    path: Path | str = DEFAULT_CASE_LIBRARY_PATH,
) -> list[dict[str, Any]]:
    query_tokens = _tokenize(symptom)
    if not query_tokens:
        return []

    ranked = []
    for case in load_cases(path):
        score, matched_terms = _score_case(case, query_tokens, symptom)
        if score <= 0:
            continue
        ranked.append({
            **case,
            "score": round(score, 3),
            "matched_terms": sorted(matched_terms),
        })

    ranked.sort(key=lambda item: (item["score"], item.get("updated_at", "")), reverse=True)
    return ranked[:limit]


def format_case_context(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "无相似历史案例。"

    blocks = []
    for index, case in enumerate(cases, start=1):
        repair = case.get("successful_repair_action", {})
        verification = case.get("verification", {})
        action_text = _format_repair_action(repair)
        commands = verification.get("commands", [])
        if isinstance(commands, list):
            verification_text = ", ".join(str(command) for command in commands[:3])
        else:
            verification_text = str(commands)

        blocks.append(
            "\n".join([
                f"案例 {index}: {case.get('title') or case.get('case_id', 'unknown')}",
                f"- 症状: {_join_list(case.get('symptoms'))}",
                f"- 工具证据: {_join_list(case.get('tool_evidence'))}",
                f"- 根因: {case.get('root_cause', '未知')}",
                f"- 成功修复动作: {action_text}",
                f"- 验证方式: {verification_text or verification.get('expected_result', '未记录')}",
                f"- 匹配: score={case.get('score')}, terms={case.get('matched_terms', [])}",
            ])
        )
    return "\n\n".join(blocks)


def build_case_from_state(state: Any) -> dict[str, Any] | None:
    verification_result = _getattr_or_key(state, "verification_result") or {}
    if not verification_result.get("verified"):
        return None

    ticket_id = _getattr_or_key(state, "ticket_id") or "unknown"
    symptom = _getattr_or_key(state, "symptom") or ""
    fix_plan = _to_dict(_getattr_or_key(state, "fix_plan") or {})
    diagnosis = _best_diagnosis(state)
    tool_evidence = _collect_tool_evidence(state)
    successful_actions = _collect_successful_actions(fix_plan)

    return {
        "case_id": f"ticket:{ticket_id}",
        "title": f"Resolved ticket {ticket_id}",
        "symptoms": [symptom],
        "tool_evidence": tool_evidence,
        "root_cause": diagnosis.get("diagnosis", "未记录"),
        "possible_causes": diagnosis.get("possible_causes", []),
        "successful_repair_action": successful_actions[0] if successful_actions else {},
        "successful_repair_actions": successful_actions,
        "verification": {
            "commands": [
                probe.get("url")
                for probe in verification_result.get("verification_probe", [])
                if probe.get("url")
            ],
            "expected_result": "所有恢复探针 HTTP 2xx",
            "probe_result": verification_result,
        },
        "source_ticket_id": ticket_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def upsert_case(case: dict[str, Any], path: Path | str = DEFAULT_CASE_LIBRARY_PATH) -> dict[str, Any]:
    cases = load_cases(path)
    case_id = case.get("case_id")
    if not case_id:
        case_id = f"case:{len(cases) + 1}"
        case["case_id"] = case_id

    replaced = False
    for index, existing in enumerate(cases):
        if existing.get("case_id") == case_id:
            merged = {**existing, **case, "updated_at": datetime.now(timezone.utc).isoformat()}
            cases[index] = merged
            case = merged
            replaced = True
            break

    if not replaced:
        cases.append(case)

    save_cases(cases, path)
    return case


def upsert_case_from_state(
    state: Any,
    path: Path | str = DEFAULT_CASE_LIBRARY_PATH,
) -> dict[str, Any] | None:
    case = build_case_from_state(state)
    if not case:
        return None
    return upsert_case(case, path)


def _score_case(case: dict[str, Any], query_tokens: set[str], raw_query: str) -> tuple[float, set[str]]:
    case_text = " ".join([
        str(case.get("title", "")),
        _join_list(case.get("symptoms")),
        _join_list(case.get("tool_evidence")),
        str(case.get("root_cause", "")),
        _join_list(case.get("possible_causes")),
        json.dumps(case.get("successful_repair_action", {}), ensure_ascii=False),
    ])
    case_tokens = _tokenize(case_text)
    matched_terms = query_tokens & case_tokens
    score = float(len(matched_terms))

    raw_query_lower = raw_query.lower()
    for symptom in case.get("symptoms", []) or []:
        symptom_text = str(symptom).lower()
        if symptom_text and (symptom_text in raw_query_lower or raw_query_lower in symptom_text):
            score += 3.0

    target = str((case.get("successful_repair_action") or {}).get("target", "")).lower()
    if target and target in raw_query_lower:
        score += 2.0

    score += min(len(matched_terms), 6) * 0.25
    return score, matched_terms


def _tokenize(text: str) -> set[str]:
    normalized = str(text).lower()
    tokens = set(re.findall(r"[a-z0-9_:/.-]+", normalized))
    for canonical, keywords in CASE_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            tokens.add(canonical)
            tokens.update(keyword.lower() for keyword in keywords if keyword.lower() in normalized)
    return {token for token in tokens if len(token) > 1}


def _format_repair_action(repair: Any) -> str:
    if isinstance(repair, dict):
        action_type = repair.get("action_type", "")
        target = repair.get("target", "")
        command = repair.get("command", "")
        return " / ".join(part for part in [action_type, target, command] if part) or "未记录"
    return str(repair or "未记录")


def _join_list(value: Any) -> str:
    if isinstance(value, list):
        return "；".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _getattr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _best_diagnosis(state: Any) -> dict[str, Any]:
    for key in ["aggregated_diagnosis", "db_agent_result", "net_agent_result", "app_agent_result"]:
        value = _getattr_or_key(state, key)
        if value:
            return _to_dict(value)
    return {}


def _collect_tool_evidence(state: Any) -> list[str]:
    evidence = []
    for key in ["db_agent_result", "net_agent_result", "app_agent_result"]:
        result = _to_dict(_getattr_or_key(state, key) or {})
        for tool_result in result.get("tool_results", []) or []:
            if isinstance(tool_result, dict):
                tool = tool_result.get("tool", "unknown_tool")
                evidence.append(f"{tool}: {str(tool_result.get('result'))[:500]}")
    return evidence[:10]


def _collect_successful_actions(fix_plan: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    for step in fix_plan.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        action = {
            "step_id": step.get("step_id"),
            "action": step.get("action"),
            "action_type": step.get("action_type"),
            "target": step.get("target"),
            "command": step.get("command"),
        }
        actions.append({k: v for k, v in action.items() if v is not None})
    return actions
