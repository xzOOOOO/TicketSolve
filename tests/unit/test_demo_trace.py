import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_demo_trace_module():
    spec = importlib.util.spec_from_file_location(
        "demo_trace_module",
        ROOT / "eval" / "demo_trace.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_demo_trace_contains_coverage_and_sandbox_story():
    module = _load_demo_trace_module()
    payload = module.build_demo_trace()

    trace_events = payload["trace_events"]
    redispatch_events = [
        event for event in trace_events
        if event["event_type"] == "handoff_requested"
        and event["agent_name"] == "dynamic_check"
        and event["metadata"].get("forced_redispatch") is True
    ]
    action_events = [
        event for event in trace_events
        if event["event_type"] == "action_executed"
    ]

    assert payload["case"] == "NGINX_BAD_ROUTE"
    assert redispatch_events
    assert redispatch_events[0]["metadata"]["coverage"] is False
    assert payload["protocol_summary"]["hypothesis_scores"]
    assert action_events[0]["metadata"]["compiled_from_action_dsl"] is True
    assert payload["security_boundary"]["llm_direct_shell"] is False
    assert trace_events[-1]["event_type"] == "verification_passed"
