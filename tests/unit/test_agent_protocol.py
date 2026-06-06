import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agent_protocol import (
    agent_result_covers_request,
    auto_response_from_agent_result,
    build_protocol_context,
    collaboration_requests_from_result,
    has_response_for,
    make_evidence_request,
    make_hypothesis,
    pending_requests_for,
)
from state import SystemState


def _load_communication_bus():
    spec = importlib.util.spec_from_file_location(
        "communication_module",
        ROOT / "src" / "agents" / "communication.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.CommunicationBus


def _load_dynamic_check_node():
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")
    from nodes import create_dynamic_check_node

    return create_dynamic_check_node


def test_agent_protocol_generates_hypothesis_request_response():
    hypothesis = make_hypothesis(
        sender="app_agent",
        content="应用容器可能停止",
        hypothesis="应用容器停止导致服务不可用",
        fault_type="APP_PROCESS_DOWN",
        confidence=0.8,
        evidence=["app health failed"],
    )
    request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=hypothesis,
        required_evidence=["nginx route status"],
        reason="需要确认入口路由是否异常",
        suggested_tools=["check_network_http_route"],
        confidence=0.8,
    )
    response = auto_response_from_agent_result(
        agent_name="net_agent",
        agent_result={
            "diagnosis": "Nginx 路由正常",
            "possible_causes": ["direct app failed too"],
            "confidence": 0.7,
        },
        request_message=request,
    )

    assert hypothesis["msg_type"] == "hypothesis"
    assert request["msg_type"] == "evidence_request"
    assert request["correlation_id"] == hypothesis["message_id"]
    assert hypothesis["evidence"][0]["observed"] == "app health failed"
    assert hypothesis["evidence"][0]["source_agent"] == "app_agent"
    assert response["msg_type"] == "evidence_response"
    assert response["related_to"] == request["message_id"]
    assert response["evidence"][0]["source_agent"] == "net_agent"
    assert "observed" in response["evidence"][0]
    assert has_response_for(request, [hypothesis, request, response]) is True


def test_pending_requests_for_ignores_already_responded_request():
    hypothesis = make_hypothesis(
        sender="db_agent",
        content="数据库连接失败",
        hypothesis="Postgres 停止",
        fault_type="DB_CONN_FAIL",
    )
    request = make_evidence_request(
        sender="db_agent",
        receiver="app_agent",
        hypothesis_message=hypothesis,
        required_evidence=["app health"],
        reason="确认应用是否也异常",
    )
    response = auto_response_from_agent_result(
        agent_name="app_agent",
        agent_result={"diagnosis": "app degraded", "confidence": 0.6},
        request_message=request,
    )

    assert pending_requests_for("app_agent", [hypothesis, request]) == [request]
    assert pending_requests_for("app_agent", [hypothesis, request, response]) == []


def test_collaboration_requests_only_uses_structured_field():
    requests = collaboration_requests_from_result({
        "diagnosis": "需要网络确认",
        "need_collaboration": ["net_agent"],
        "collaboration_requests": [],
    })

    assert requests == []


def test_build_protocol_context_returns_summary():
    hypothesis = make_hypothesis(
        sender="net_agent",
        content="Nginx upstream 错误",
        hypothesis="Nginx upstream 指向错误端口",
        fault_type="NGINX_BAD_ROUTE",
        confidence=0.9,
    )
    request = make_evidence_request(
        sender="net_agent",
        receiver="app_agent",
        hypothesis_message=hypothesis,
        required_evidence=["direct app health"],
        reason="需要确认应用直连健康",
    )
    response = auto_response_from_agent_result(
        agent_name="app_agent",
        agent_result={
            "diagnosis": "应用直连健康",
            "possible_causes": ["direct app health ok"],
            "confidence": 0.9,
            "fault_type": "NGINX_BAD_ROUTE",
        },
        request_message=request,
    )

    context = build_protocol_context([hypothesis, request, response])

    assert context["protocol_summary"]["winning_hypothesis_id"] == hypothesis["message_id"]
    assert context["protocol_summary"]["supporting_evidence_count"] == 1
    scores = context["protocol_summary"]["hypothesis_scores"]
    assert len(scores) == 1
    assert scores[0]["hypothesis_id"] == hypothesis["message_id"]
    assert scores[0]["final_score"] > 0
    assert scores[0]["support_score"] > 0
    assert "工具观测" in scores[0]["reason"]
    assert "证据响应" in context["text"]


def test_protocol_scores_prefer_tool_supported_hypothesis():
    weak_hypothesis = make_hypothesis(
        sender="app_agent",
        content="应用可能停止",
        hypothesis="应用容器停止",
        fault_type="APP_PROCESS_DOWN",
        confidence=0.4,
    )
    strong_hypothesis = make_hypothesis(
        sender="net_agent",
        content="Nginx upstream 错误",
        hypothesis="Nginx upstream 指向错误端口",
        fault_type="NGINX_BAD_ROUTE",
        confidence=0.6,
    )
    request = make_evidence_request(
        sender="net_agent",
        receiver="app_agent",
        hypothesis_message=strong_hypothesis,
        required_evidence=["direct app health"],
        reason="需要应用直连证据",
    )
    response = auto_response_from_agent_result(
        agent_name="app_agent",
        agent_result={
            "diagnosis": "应用直连健康，入口失败",
            "confidence": 0.9,
            "fault_type": "NGINX_BAD_ROUTE",
            "tool_results": [
                {
                    "tool": "check_app_health",
                    "result": {
                        "status": "ok",
                        "url": "http://localhost:18081/health",
                        "evidence": ["direct app health ok"],
                    },
                }
            ],
        },
        request_message=request,
    )

    context = build_protocol_context([weak_hypothesis, strong_hypothesis, request, response])
    winning_id = context["protocol_summary"]["winning_hypothesis_id"]
    winning_score = next(
        score for score in context["protocol_summary"]["hypothesis_scores"]
        if score["hypothesis_id"] == winning_id
    )

    assert winning_id == strong_hypothesis["message_id"]
    assert winning_score["tool_evidence_score"] > 0
    assert winning_score["top_evidence"]


def test_agent_result_coverage_checks_required_tool_evidence():
    hypothesis = make_hypothesis(
        sender="app_agent",
        content="需要网络确认",
        hypothesis="入口路由异常",
        fault_type="NGINX_BAD_ROUTE",
    )
    request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=hypothesis,
        required_evidence=["nginx route status"],
        reason="确认 nginx route",
        suggested_tools=["check_network_http_route"],
    )
    ping_only_result = {
        "diagnosis": "ping 正常",
        "confidence": 0.6,
        "tool_results": [{"tool": "check_network_ping", "result": {"status": "ok", "host": "localhost"}}],
    }
    route_result = {
        "diagnosis": "入口路由异常",
        "confidence": 0.9,
        "tool_results": [{
            "tool": "check_network_http_route",
            "result": {
                "status": "failed",
                "url": "http://localhost:18080/health",
                "evidence": ["nginx route status failed"],
            },
        }],
    }

    assert agent_result_covers_request(ping_only_result, request) is False
    assert agent_result_covers_request(route_result, request) is True


def test_communication_bus_receive_filters_by_type():
    CommunicationBus = _load_communication_bus()
    bus = CommunicationBus()
    diagnosis = bus.broadcast(
        sender="db_agent",
        content="数据库连接失败",
        msg_type="diagnosis",
    )[0]
    hypothesis = bus.publish_hypothesis(
        sender="app_agent",
        content="应用停止",
        hypothesis="应用容器停止",
    )[0]

    received = bus.receive("net_agent", [diagnosis, hypothesis], msg_types={"hypothesis"})

    assert len(received) == 1
    assert received[0]["msg_type"] == "hypothesis"


def test_dynamic_check_dispatches_unfinished_evidence_request():
    create_dynamic_check_node = _load_dynamic_check_node()
    hypothesis = make_hypothesis(
        sender="app_agent",
        content="需要网络确认",
        hypothesis="入口路由异常",
        fault_type="NGINX_BAD_ROUTE",
    )
    request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=hypothesis,
        required_evidence=["nginx route status"],
        reason="确认 nginx route",
    )
    state = SystemState(
        ticket_id="T-1",
        symptom="入口不可用",
        agent_messages=[hypothesis, request],
    )

    result = asyncio.run(create_dynamic_check_node()(state))

    assert result["dispatched_agents"] == ["net_agent"]


def test_dynamic_check_auto_responds_when_agent_result_exists():
    create_dynamic_check_node = _load_dynamic_check_node()
    hypothesis = make_hypothesis(
        sender="app_agent",
        content="需要网络确认",
        hypothesis="入口路由异常",
        fault_type="NGINX_BAD_ROUTE",
    )
    request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=hypothesis,
        required_evidence=["nginx route status"],
        reason="确认 nginx route",
        suggested_tools=["check_network_http_route"],
    )
    state = SystemState(
        ticket_id="T-2",
        symptom="入口不可用",
        agent_messages=[hypothesis, request],
        net_agent_result={
            "diagnosis": "Nginx upstream 指向错误端口",
            "confidence": 0.9,
            "fault_type": "NGINX_BAD_ROUTE",
            "tool_results": [{
                "tool": "check_network_http_route",
                "result": {
                    "status": "failed",
                    "url": "http://localhost:18080/health",
                    "evidence": ["nginx route status failed"],
                },
            }],
        },
    )

    result = asyncio.run(create_dynamic_check_node()(state))

    assert result["dispatched_agents"] == []
    assert result["agent_messages"][0]["msg_type"] == "evidence_response"
    assert result["agent_messages"][0]["related_to"] == request["message_id"]


def test_dynamic_check_forces_redispatch_when_cached_result_lacks_requested_evidence():
    create_dynamic_check_node = _load_dynamic_check_node()
    hypothesis = make_hypothesis(
        sender="app_agent",
        content="需要网络确认",
        hypothesis="入口路由异常",
        fault_type="NGINX_BAD_ROUTE",
    )
    request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=hypothesis,
        required_evidence=["nginx route status"],
        reason="确认 nginx route",
        suggested_tools=["check_network_http_route"],
    )
    state = SystemState(
        ticket_id="T-4",
        symptom="入口不可用",
        agent_messages=[hypothesis, request],
        net_agent_result={
            "diagnosis": "ping 正常",
            "confidence": 0.6,
            "tool_results": [{"tool": "check_network_ping", "result": {"status": "ok", "host": "localhost"}}],
        },
    )

    result = asyncio.run(create_dynamic_check_node()(state))

    assert result["dispatched_agents"] == ["net_agent"]
    assert result["force_dispatched_agents"] == ["net_agent"]
    assert result["redispatched_request_ids"] == [request["message_id"]]
    assert "agent_messages" not in result


def test_dynamic_check_does_not_loop_after_forced_redispatch():
    create_dynamic_check_node = _load_dynamic_check_node()
    hypothesis = make_hypothesis(
        sender="app_agent",
        content="需要网络确认",
        hypothesis="入口路由异常",
        fault_type="NGINX_BAD_ROUTE",
    )
    request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=hypothesis,
        required_evidence=["nginx route status"],
        reason="确认 nginx route",
        suggested_tools=["check_network_http_route"],
    )
    state = SystemState(
        ticket_id="T-5",
        symptom="入口不可用",
        agent_messages=[hypothesis, request],
        redispatched_request_ids=[request["message_id"]],
        net_agent_result={
            "diagnosis": "ping 正常",
            "confidence": 0.6,
            "tool_results": [{"tool": "check_network_ping", "result": {"status": "ok", "host": "localhost"}}],
        },
    )

    result = asyncio.run(create_dynamic_check_node()(state))

    assert result["dispatched_agents"] == []
    assert result["agent_messages"][0]["msg_type"] == "evidence_response"
    assert result["agent_messages"][0]["supports_hypothesis"] is False


def test_dynamic_check_does_not_duplicate_existing_response():
    create_dynamic_check_node = _load_dynamic_check_node()
    hypothesis = make_hypothesis(
        sender="app_agent",
        content="需要网络确认",
        hypothesis="入口路由异常",
        fault_type="NGINX_BAD_ROUTE",
    )
    request = make_evidence_request(
        sender="app_agent",
        receiver="net_agent",
        hypothesis_message=hypothesis,
        required_evidence=["nginx route status"],
        reason="确认 nginx route",
    )
    response = auto_response_from_agent_result(
        agent_name="net_agent",
        agent_result={"diagnosis": "route failed", "confidence": 0.9},
        request_message=request,
    )
    state = SystemState(
        ticket_id="T-3",
        symptom="入口不可用",
        agent_messages=[hypothesis, request, response],
        net_agent_result={"diagnosis": "route failed", "confidence": 0.9},
    )

    result = asyncio.run(create_dynamic_check_node()(state))

    assert result["dispatched_agents"] == []
    assert "agent_messages" not in result
