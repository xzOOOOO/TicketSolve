import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from action_dsl import ActionDSLValidationError, compile_action
from executor_v2 import ClosedLoopExecutor, CommandRunner
from guardrail import run_guardrail
from schemas import CommandExecutionResult, FixStepOutput


class RecordingRunner(CommandRunner):
    def __init__(self):
        self.commands = []

    async def run(self, command: str, step_id: int, timeout: int = 30) -> CommandExecutionResult:
        self.commands.append(command)
        return CommandExecutionResult(
            step_id=step_id,
            command=command,
            exit_code=0,
            stdout="ok",
            stderr="",
            success=True,
            execution_time_ms=1,
        )


def test_compile_recover_fault_action():
    compiled = compile_action({
        "action_type": "RECOVER_FAULT",
        "target": "APP_PROCESS_DOWN",
        "command": "ignored display command",
    })

    assert compiled.command == "python lab/chaos.py recover APP_PROCESS_DOWN"


def test_compile_rejects_unknown_target():
    try:
        compile_action({
            "action_type": "START_CONTAINER",
            "target": "prod-db",
        })
    except ActionDSLValidationError as exc:
        assert "prod-db" in str(exc)
    else:
        raise AssertionError("expected ActionDSLValidationError")


def test_executor_prefers_action_dsl_over_free_form_command():
    runner = RecordingRunner()
    executor = ClosedLoopExecutor(command_runner=runner, max_retries_per_step=1)

    result = asyncio.run(executor.execute_plan({
        "plan_id": "PLAN-ACTION-DSL",
        "steps": [{
            "step_id": 1,
            "action": "Recover app process",
            "action_type": "RECOVER_FAULT",
            "target": "APP_PROCESS_DOWN",
            "command": "rm -rf /",
            "risk_level": "low",
        }],
    }))

    assert runner.commands == ["python lab/chaos.py recover APP_PROCESS_DOWN"]
    assert result["execution_result"]["overall_status"] == "success"
    trace = result["execution_trace"][0]
    assert trace["compiled_from_action_dsl"] is True
    assert trace["action_type"] == "RECOVER_FAULT"
    assert trace["target"] == "APP_PROCESS_DOWN"


def test_guardrail_checks_compiled_action_not_display_command():
    result = run_guardrail({
        "steps": [{
            "step_id": 1,
            "action": "Recover app process",
            "action_type": "RECOVER_FAULT",
            "target": "APP_PROCESS_DOWN",
            "command": "rm -rf /",
            "risk_level": "low",
        }],
    })

    assert result.passed is True
    assert result.violations == []


def test_guardrail_rejects_invalid_action_dsl():
    result = run_guardrail({
        "steps": [{
            "step_id": 1,
            "action": "Start unknown container",
            "action_type": "START_CONTAINER",
            "target": "prod-db",
            "risk_level": "low",
        }],
    })

    assert result.passed is False
    assert result.violations[0].rule_id == "ACTION_DSL_001"


def test_guardrail_rejects_nested_action_dsl_shape():
    result = run_guardrail({
        "steps": [{
            "step_id": 1,
            "action": "Recover app process",
            "action_spec": {
                "action_type": "RECOVER_FAULT",
                "target": "APP_PROCESS_DOWN",
            },
            "risk_level": "low",
        }],
    })

    assert result.passed is False
    assert result.violations[0].rule_id == "ACTION_DSL_001"
    assert "action_spec" in result.violations[0].detail


def test_fix_step_schema_only_exposes_flat_action_fields():
    schema = FixStepOutput.model_json_schema()
    properties = schema["properties"]

    assert "action_type" in properties
    assert "target" in properties
    assert "rollback_action_type" in properties
    assert "rollback_target" in properties
    assert "action_spec" not in properties
    assert "rollback_action" not in properties
