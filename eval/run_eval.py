import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = ROOT / "eval" / "cases.yaml"
CHAOS = ROOT / "lab" / "chaos.py"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, text=True, check=check)


def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method=method, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def probe(url: str, timeout: int = 5) -> tuple[bool, str]:
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return 200 <= resp.status < 300, body[:300]
    except HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except URLError as exc:
        return False, str(exc.reason)
    except Exception as exc:
        return False, str(exc)


def wait_for_lab(url: str = "http://localhost:18080/health", seconds: int = 30) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        ok, _ = probe(url)
        if ok:
            return
        time.sleep(1)
    raise RuntimeError(f"Lab did not become healthy within {seconds}s")


def wait_for_recovery(url: str, seconds: int = 30) -> tuple[bool, str]:
    deadline = time.time() + seconds
    last_body = ""
    while time.time() < deadline:
        ok, body = probe(url)
        last_body = body
        if ok:
            return True, body
        time.sleep(1)
    return False, last_body


def load_cases(selected: list[str] | None) -> list[dict]:
    data = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))
    cases = data["cases"]
    if selected:
        selected_set = set(selected)
        cases = [case for case in cases if case["id"] in selected_set]
    return cases


def summarize_flow(api_base: str, ticket_id: str) -> dict:
    try:
        flow = http_json("GET", f"{api_base}/api/tickets/{ticket_id}/agent-flow")
        data = flow.get("data", {})
        return {
            "diagnosis_type": data.get("diagnosis_type"),
            "agents": data.get("dispatched_agents", []),
            "total_steps": data.get("total_steps", 0),
        }
    except Exception as exc:
        return {"diagnosis_type": None, "agents": [], "total_steps": 0, "error": str(exc)}


def run_case(api_base: str, case: dict, approve: bool) -> dict:
    ticket_id = f"EVAL-{case['id']}-{int(time.time())}"

    run([sys.executable, str(CHAOS), "reset"])
    wait_for_lab()
    run([sys.executable, str(CHAOS), "inject", case["id"]])
    time.sleep(2)

    create_result = http_json(
        "POST",
        f"{api_base}/api/tickets",
        {"ticket_id": ticket_id, "symptom": case["ticket"]},
        timeout=180,
    )

    approve_result = None
    if approve:
        approve_result = http_json(
            "POST",
            f"{api_base}/api/tickets/{ticket_id}/approve",
            {"approved": True, "comments": "eval auto-approval"},
            timeout=180,
        )

    fixed_by_agent, recovery_body = wait_for_recovery(case["recovery_check"])
    flow = summarize_flow(api_base, ticket_id)

    # 评测记录完成后再做兜底清理，避免下一条 case 继承当前故障。
    run([sys.executable, str(CHAOS), "recover", case["id"]], check=False)

    predicted = flow.get("diagnosis_type")
    return {
        "case": case["id"],
        "expected": case["expected_domain"],
        "predicted": predicted,
        "matched": predicted == case["expected_domain"],
        "fixed_by_agent": fixed_by_agent,
        "agents": ",".join(flow.get("agents", [])),
        "flow_steps": flow.get("total_steps", 0),
        "ticket_id": ticket_id,
        "create_message": create_result.get("message"),
        "approved": approve_result is not None,
        "recovery_probe": recovery_body,
    }


def print_table(rows: list[dict]) -> None:
    columns = ["case", "expected", "predicted", "matched", "fixed_by_agent", "agents", "flow_steps"]
    widths = {
        col: max(len(col), *(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    print(" | ".join(col.ljust(widths[col]) for col in columns))
    print("-+-".join("-" * widths[col] for col in columns))
    for row in rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SREBench Lite evaluations")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--cases", nargs="*", help="Case IDs to run")
    parser.add_argument("--no-approve", action="store_true", help="Do not resume approval/execution")
    parser.add_argument("--json-out", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        raise SystemExit("No cases selected")

    rows = []
    for case in cases:
        print(f"\n=== {case['id']} ===")
        rows.append(run_case(args.api_base.rstrip("/"), case, approve=not args.no_approve))

    print("\nResults")
    print_table(rows)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote report: {args.json_out}")


if __name__ == "__main__":
    main()
