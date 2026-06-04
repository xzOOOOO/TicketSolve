import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace_events import TRACE_SCHEMA_VERSION, make_trace_event, status_from_success


def test_make_trace_event_uses_standard_schema():
    event = make_trace_event(
        "action_executed",
        ticket_id="T-1",
        agent_name="executor",
        status="success",
        input_data={"command": "python lab/chaos.py recover APP_PROCESS_DOWN"},
        output_data={"exit_code": 0},
        metadata={"step_id": 1},
        timestamp="2026-06-04T00:00:00+00:00",
    )

    assert event == {
        "schema_version": TRACE_SCHEMA_VERSION,
        "event_type": "action_executed",
        "ticket_id": "T-1",
        "agent_name": "executor",
        "status": "success",
        "timestamp": "2026-06-04T00:00:00+00:00",
        "input": {"command": "python lab/chaos.py recover APP_PROCESS_DOWN"},
        "output": {"exit_code": 0},
        "error": None,
        "metadata": {"step_id": 1},
    }


def test_make_trace_event_rejects_unknown_event_type():
    with pytest.raises(ValueError, match="Unsupported trace event_type"):
        make_trace_event(
            "custom_debug_event",
            ticket_id="T-1",
            agent_name="debugger",
        )


def test_status_from_success():
    assert status_from_success(True) == "success"
    assert status_from_success(False) == "failure"
    assert status_from_success(None) == "skipped"
