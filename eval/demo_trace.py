"""生成 NGINX_BAD_ROUTE 的可展示 Demo Trace。

这个脚本不调用 LLM，也不操作 Docker Lab。
它生成一条稳定的、便于 README/面试展示的标准 Trace，用来说明系统闭环：
诊断协作 → coverage 判定 → 定向重派发 → 证据裁决 → Action DSL 修复 → 验证。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from agent_protocol import (  # noqa: E402
    agent_result_covers_request,
    auto_response_from_agent_result,
    build_protocol_context,
    make_evidence_request,
    make_hypothesis,
)
from trace_events import make_trace_event  # noqa: E402


TICKET_ID = "DEMO-NGINX-BAD-ROUTE"
TS = "2026-06-06T10:00:00+00:00"


def build_demo_trace() -> dict[str, Any]:
    """构造一条完整、稳定的 NGINX_BAD_ROUTE 展示轨迹。"""
    app_hypothesis = make_hypothesis(
        sender="app_agent",
        content="应用直连健康，入口异常更可能来自 Nginx 路由",
        hypothesis="应用直连健康但 Nginx 入口 502，疑似 upstream 配置错误",
        fault_type="NGINX_BAD_ROUTE",
        confidence=0.78,
        evidence=[{
            "source_agent": "app_agent",
            "tool_name": "check_app_health",
            "target": "http://localhost:18081/health",
            "status": "ok",
            "observed": "direct app health returned 200",
            "expected": "direct app health should return 200",
            "supports_hypothesis": True,
            "confidence": 0.78,
        }],
    )
    app_hypothesis["message_id"] = "hyp-app-direct-ok"
    app_hypothesis["correlation_id"] = "hyp-app-direct-ok"

    evidence_request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=app_hypothesis,
        required_evidence=["nginx route status", "direct app health"],
        reason="需要确认入口 Nginx 路由是否异常，以及应用直连是否健康",
        suggested_tools=["check_network_http_route"],
        confidence=0.78,
    )
    evidence_request["message_id"] = "req-net-route-evidence"
    evidence_request["correlation_id"] = "hyp-app-direct-ok"

    stale_net_result = {
        "agent_name": "net_agent",
        "diagnosis": "主机 ping 正常，但未检查 Nginx HTTP route",
        "confidence": 0.55,
        "tool_results": [{
            "tool": "check_network_ping",
            "result": {"status": "ok", "host": "localhost", "evidence": ["localhost ping ok"]},
        }],
    }
    stale_coverage = agent_result_covers_request(stale_net_result, evidence_request)

    route_net_result = {
        "agent_name": "net_agent",
        "diagnosis": "Nginx upstream 指向错误端口，入口返回 502；应用直连健康",
        "possible_causes": ["nginx route status failed", "direct app health ok"],
        "confidence": 0.92,
        "fault_type": "NGINX_BAD_ROUTE",
        "tool_results": [{
            "tool": "check_network_http_route",
            "result": {
                "status": "failed",
                "url": "http://localhost:18080/health",
                "evidence": ["nginx route status failed: 502", "direct app health ok: 200"],
            },
        }],
    }
    route_coverage = agent_result_covers_request(route_net_result, evidence_request)
    evidence_response = auto_response_from_agent_result(
        agent_name="net_agent",
        agent_result=route_net_result,
        request_message=evidence_request,
    )
    evidence_response["message_id"] = "resp-net-route-evidence"
    for message in (app_hypothesis, evidence_request, evidence_response):
        _pin_evidence_timestamp(message)

    protocol_context = build_protocol_context([
        app_hypothesis,
        evidence_request,
        evidence_response,
    ])

    fix_plan = {
        "plan_id": "PLAN-DEMO-NGINX",
        "description": "恢复 Nginx upstream 到正确 app 端口",
        "risk_level": "low",
        "steps": [{
            "step_id": 1,
            "action": "恢复 NGINX_BAD_ROUTE 故障",
            "action_type": "RECOVER_FAULT",
            "target": "NGINX_BAD_ROUTE",
            "command": "python lab/chaos.py recover NGINX_BAD_ROUTE",
        }],
        "verification": {
            "commands": ["curl http://localhost:18080/health"],
            "expected_result": "HTTP 200",
        },
    }

    trace_events = [
        make_trace_event(
            "agent_started",
            ticket_id=TICKET_ID,
            agent_name="supervisor",
            input_data={"symptom": "The app is healthy directly, but nginx returns bad gateway."},
            metadata={"dispatch_round": 0},
            timestamp=TS,
        ),
        make_trace_event(
            "handoff_requested",
            ticket_id=TICKET_ID,
            agent_name="supervisor",
            output_data={"dispatched_agents": ["app_agent", "net_agent"]},
            metadata={"dispatch_round": 0, "reason": "入口和应用健康状态需要跨域确认"},
            timestamp=TS,
        ),
        make_trace_event(
            "tool_called",
            ticket_id=TICKET_ID,
            agent_name="app_agent",
            input_data={"tool": "check_app_health", "target": "http://localhost:18081/health"},
            output_data={"status": "ok", "http_status": 200},
            metadata={"dispatch_round": 1},
            timestamp=TS,
        ),
        make_trace_event(
            "diagnosis_generated",
            ticket_id=TICKET_ID,
            agent_name="app_agent",
            output_data={"fault_type": "NGINX_BAD_ROUTE", "hypothesis": app_hypothesis["hypothesis"]},
            metadata={"message_id": app_hypothesis["message_id"], "confidence": 0.78},
            timestamp=TS,
        ),
        make_trace_event(
            "handoff_requested",
            ticket_id=TICKET_ID,
            agent_name="app_agent",
            status="pending",
            input_data={"hypothesis_id": app_hypothesis["message_id"]},
            output_data=evidence_request,
            metadata={
                "message_id": evidence_request["message_id"],
                "correlation_id": evidence_request["correlation_id"],
                "msg_type": "evidence_request",
                "target_agent": "net_agent",
                "required_evidence": evidence_request["required_evidence"],
                "suggested_tools": evidence_request["suggested_tools"],
            },
            timestamp=TS,
        ),
        make_trace_event(
            "handoff_requested",
            ticket_id=TICKET_ID,
            agent_name="dynamic_check",
            status="pending",
            input_data={"cached_agent_result_exists": True, "coverage": stale_coverage},
            output_data={"target_agent": "net_agent", "forced_redispatch": True},
            metadata={
                "message_id": evidence_request["message_id"],
                "correlation_id": evidence_request["correlation_id"],
                "msg_type": "evidence_request",
                "target_agent": "net_agent",
                "coverage": stale_coverage,
                "forced_redispatch": True,
                "required_evidence": evidence_request["required_evidence"],
                "suggested_tools": evidence_request["suggested_tools"],
            },
            timestamp=TS,
        ),
        make_trace_event(
            "tool_called",
            ticket_id=TICKET_ID,
            agent_name="net_agent",
            input_data={"tool": "check_network_http_route", "target": "http://localhost:18080/health"},
            output_data={"nginx_status": 502, "direct_app_status": 200},
            metadata={"dispatch_round": 2, "forced_redispatch": True},
            timestamp=TS,
        ),
        make_trace_event(
            "observation_received",
            ticket_id=TICKET_ID,
            agent_name="dynamic_check",
            output_data=evidence_response,
            metadata={
                "message_id": evidence_request["message_id"],
                "correlation_id": evidence_request["correlation_id"],
                "msg_type": "evidence_response",
                "target_agent": "net_agent",
                "coverage": route_coverage,
                "forced_redispatch": False,
                "auto_response": True,
                "required_evidence": evidence_request["required_evidence"],
                "suggested_tools": evidence_request["suggested_tools"],
            },
            timestamp=TS,
        ),
        make_trace_event(
            "diagnosis_generated",
            ticket_id=TICKET_ID,
            agent_name="aggregate",
            output_data={
                "diagnosis": "Nginx upstream 指向错误端口导致入口 502",
                "fault_type": "NGINX_BAD_ROUTE",
                "protocol_summary": protocol_context["protocol_summary"],
            },
            metadata={"winning_hypothesis_id": protocol_context["protocol_summary"]["winning_hypothesis_id"]},
            timestamp=TS,
        ),
        make_trace_event(
            "plan_generated",
            ticket_id=TICKET_ID,
            agent_name="fix_agent",
            output_data=fix_plan,
            metadata={"action_dsl": True, "risk_level": "low"},
            timestamp=TS,
        ),
        make_trace_event(
            "policy_checked",
            ticket_id=TICKET_ID,
            agent_name="guardrail",
            output_data={"passed": True, "violations": []},
            metadata={"dsl_only": True, "free_form_shell": False},
            timestamp=TS,
        ),
        make_trace_event(
            "approval_received",
            ticket_id=TICKET_ID,
            agent_name="human_approval",
            output_data={"approved": True, "comments": "demo auto approval"},
            metadata={"approval_required": True},
            timestamp=TS,
        ),
        make_trace_event(
            "action_executed",
            ticket_id=TICKET_ID,
            agent_name="executor",
            input_data={"action_type": "RECOVER_FAULT", "target": "NGINX_BAD_ROUTE"},
            output_data={"exit_code": 0, "compiled_command": "python lab/chaos.py recover NGINX_BAD_ROUTE"},
            metadata={"executor_mode": "docker_lab", "compiled_from_action_dsl": True},
            timestamp=TS,
        ),
        make_trace_event(
            "verification_passed",
            ticket_id=TICKET_ID,
            agent_name="verify",
            output_data={"url": "http://localhost:18080/health", "http_status": 200, "verified": True},
            metadata={"probe_count": 1},
            timestamp=TS,
        ),
    ]

    return {
        "ticket_id": TICKET_ID,
        "case": "NGINX_BAD_ROUTE",
        "purpose": "展示多 Agent 证据协作、coverage 调度、安全执行和恢复验证闭环",
        "agent_messages": [app_hypothesis, evidence_request, evidence_response],
        "protocol_summary": protocol_context["protocol_summary"],
        "trace_events": trace_events,
        "security_boundary": {
            "llm_direct_shell": False,
            "repair_output": "Action DSL",
            "guardrail": "run_guardrail",
            "executor_mode": "docker_lab",
            "allowed_scope": ["srebench-* containers", "lab/chaos.py recover/inject/reset", "health probes"],
        },
    }


def _pin_evidence_timestamp(message: dict[str, Any]) -> None:
    """固定样例证据时间戳，避免每次生成 demo JSON 都出现无意义 diff。"""
    for item in message.get("evidence") or []:
        item["timestamp"] = TS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic NGINX_BAD_ROUTE demo trace")
    parser.add_argument("--json-out", type=Path, help="Optional output path")
    args = parser.parse_args()

    payload = build_demo_trace()
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote demo trace: {args.json_out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
