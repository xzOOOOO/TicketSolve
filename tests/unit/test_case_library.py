import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from case_library import (
    build_case_from_state,
    format_case_context,
    retrieve_similar_cases,
    save_cases,
    upsert_case,
)


def test_retrieve_similar_cases_by_symptom(tmp_path):
    path = tmp_path / "cases.json"
    save_cases([
        {
            "case_id": "redis",
            "title": "Redis cache down",
            "symptoms": ["Redis 连接失败", "缓存接口失败"],
            "tool_evidence": ["check_app_redis_connection failed"],
            "root_cause": "Redis 容器停止",
            "successful_repair_action": {
                "action_type": "RECOVER_FAULT",
                "target": "REDIS_DOWN",
            },
            "verification": {"commands": ["curl /cache/ping"]},
        },
        {
            "case_id": "nginx",
            "title": "Nginx route bad",
            "symptoms": ["nginx 路由错误"],
            "tool_evidence": ["check_network_http_route failed"],
            "root_cause": "路由配置错误",
            "successful_repair_action": {
                "action_type": "RECOVER_FAULT",
                "target": "NGINX_BAD_ROUTE",
            },
            "verification": {"commands": ["curl /health"]},
        },
    ], path)

    matches = retrieve_similar_cases("缓存接口 /cache/ping 失败，Redis 连不上", path=path)

    assert matches[0]["case_id"] == "redis"
    assert matches[0]["score"] > 0


def test_format_case_context_includes_repair_and_verification():
    context = format_case_context([{
        "case_id": "case-1",
        "title": "App down",
        "symptoms": ["应用不可用"],
        "tool_evidence": ["check_app_process failed"],
        "root_cause": "进程停止",
        "successful_repair_action": {
            "action_type": "RECOVER_FAULT",
            "target": "APP_PROCESS_DOWN",
            "command": "python lab/chaos.py recover APP_PROCESS_DOWN",
        },
        "verification": {"commands": ["curl /health"]},
        "score": 3.0,
        "matched_terms": ["app"],
    }])

    assert "App down" in context
    assert "RECOVER_FAULT" in context
    assert "curl /health" in context


def test_build_case_from_verified_state():
    state = SimpleNamespace(
        ticket_id="T-1",
        symptom="订单查询很慢",
        aggregated_diagnosis={
            "diagnosis": "缺少 orders 索引",
            "possible_causes": ["idx_orders_status_created_at missing"],
        },
        db_agent_result={"tool_results": [{"tool": "check_db_slow_query", "result": "missing index"}]},
        net_agent_result=None,
        app_agent_result=None,
        fix_plan={
            "steps": [{
                "step_id": 1,
                "action": "重建索引",
                "action_type": "REBUILD_ORDERS_INDEX",
                "target": "DB_SLOW_QUERY",
                "command": "docker exec ... create index",
            }]
        },
        verification_result={
            "verified": True,
            "verification_probe": [{"url": "http://localhost:18080/orders/pending"}],
        },
    )

    case = build_case_from_state(state)

    assert case["case_id"] == "ticket:T-1"
    assert case["root_cause"] == "缺少 orders 索引"
    assert case["successful_repair_action"]["action_type"] == "REBUILD_ORDERS_INDEX"
    assert case["verification"]["commands"] == ["http://localhost:18080/orders/pending"]


def test_upsert_case_replaces_existing(tmp_path):
    path = tmp_path / "cases.json"

    first = upsert_case({"case_id": "same", "title": "old"}, path)
    second = upsert_case({"case_id": "same", "title": "new"}, path)
    matches = retrieve_similar_cases("new", path=path)

    assert first["case_id"] == "same"
    assert second["title"] == "new"
    assert len(matches) == 1
