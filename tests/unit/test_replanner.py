import sys
from pathlib import Path

# 将项目根目录下的 src 加入模块搜索路径，以便导入 replanner
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from replanner import make_replanner_decision


def test_replanner_routes_success_to_verify():
    """场景：执行器成功完成 → 决策应为 verify，失败类型为 none。"""
    decision = make_replanner_decision(
        execution_result={"overall_status": "success"},
        execution_trace=[],
        replanner_round=0,
        max_replanner_rounds=2,
    )

    assert decision["decision"] == "verify"
    assert decision["failure_type"] == "none"


def test_replanner_rediagnoses_disallowed_command():
    """场景：命令不在白名单中（exit_code=126）→ 决策应为 re-diagnose，失败类型为 command_not_allowed。"""
    decision = make_replanner_decision(
        execution_result={"overall_status": "failed"},
        execution_trace=[{
            "trace_type": "execute",
            "success": False,
            "exit_code": 126,
            "stderr": "命令未在 SREBench Lite 白名单中",
        }],
        replanner_round=0,
        max_replanner_rounds=2,
    )

    assert decision["decision"] == "re-diagnose"
    assert decision["failure_type"] == "command_not_allowed"


def test_replanner_retries_environment_failure_before_budget_exhaustion():
    """场景：环境临时不可用（Connection refused），且预算未耗尽 → 决策应为 retry。"""
    decision = make_replanner_decision(
        execution_result={"overall_status": "failed"},
        execution_trace=[{
            "trace_type": "execute",
            "success": False,
            "exit_code": 1,
            "stderr": "Connection refused",
        }],
        replanner_round=0,
        max_replanner_rounds=2,
    )

    assert decision["decision"] == "retry"
    assert decision["failure_type"] == "environment_not_ready"


def test_replanner_escalates_after_budget_exhaustion():
    """场景：重试预算已耗尽（replanner_round >= max_replanner_rounds）→ 决策应为 escalate。"""
    decision = make_replanner_decision(
        execution_result={"overall_status": "failed"},
        execution_trace=[{
            "trace_type": "execute",
            "success": False,
            "exit_code": 124,
            "stderr": "timeout",
        }],
        replanner_round=2,
        max_replanner_rounds=2,
    )

    assert decision["decision"] == "escalate"


def test_replanner_stops_after_executor_rollback():
    """场景：执行器已尝试过回滚且成功 → 决策应为 rollback，停止继续修复。"""
    decision = make_replanner_decision(
        execution_result={"overall_status": "failed"},
        execution_trace=[
            {
                "trace_type": "execute",
                "success": False,
                "exit_code": 2,
                "stderr": "Permission denied",
            },
            {
                "trace_type": "rollback",
                "success": True,
                "stderr": "",
            },
        ],
        replanner_round=0,
        max_replanner_rounds=2,
    )

    assert decision["decision"] == "rollback"
    assert decision["failure_type"] == "rollback_already_attempted"
