"""
恢复验证节点。
"""

# shared：集中导入工作流节点需要的公共依赖和常量
from workflow_nodes.shared import *

def _run_verification_probe(name: str, url: str, timeout: int = 5) -> dict:
    """
    运行单个恢复验证探测。

    发送 HTTP GET 请求到指定 URL，检查服务是否恢复。

    参数：
        name: 探测名称（如 "health"）
        url: 探测 URL（如 "http://localhost:18080/health"）
        timeout: 请求超时时间（秒），默认 5 秒

    返回：
        探测结果字典，包含 name、url、status_code、success、body、error、checked_at
    """
    # started_at：探测开始时间，ISO 格式 UTC 时间
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        # 构造 HTTP GET 请求
        req = Request(url, method="GET")
        # urlopen：发送请求，timeout 防止长时间阻塞
        with urlopen(req, timeout=timeout) as resp:
            # body：读取响应体，最多 1000 字符，避免过大
            body = resp.read().decode("utf-8", errors="replace")
            return {
                "name": name,                           # 探测名称
                "url": url,                             # 探测 URL
                "status_code": resp.status,             # HTTP 状态码
                "success": 200 <= resp.status < 300,    # 2xx 视为成功
                "body": body[:1000],                    # 响应体截断到 1000 字符
                "error": "",                            # 无错误
                "checked_at": started_at,               # 探测时间戳
            }
    except HTTPError as exc:
        # HTTP 错误（如 404、500）：返回错误状态码和原因
        return {
            "name": name,
            "url": url,
            "status_code": exc.code,
            "success": False,
            "body": "",
            "error": f"HTTP {exc.code}: {exc.reason}",
            "checked_at": started_at,
        }
    except URLError as exc:
        # URL 错误（如连接失败、DNS 错误）：无状态码
        return {
            "name": name,
            "url": url,
            "status_code": None,
            "success": False,
            "body": "",
            "error": str(exc.reason),
            "checked_at": started_at,
        }
    except Exception as exc:
        # 其他异常（如超时）：捕获所有异常，避免验证节点崩溃
        return {
            "name": name,
            "url": url,
            "status_code": None,
            "success": False,
            "body": "",
            "error": str(exc),
            "checked_at": started_at,
        }


def create_verify_node():
    """
    创建恢复验证节点工厂函数。

    Verify 位于 Executor 之后、Save 之前，固定探测三个关键恢复接口：
    - /health：Nginx 入口健康检查
    - /cache/ping：Redis 缓存连通性
    - /orders/pending：数据库查询能力

    所有探测都成功才认为服务已恢复。
    结果写入 verification_result，同时合并进 execution_result，方便工单表持久化。

    返回：
        异步节点函数 verify_node(state) -> dict
    """

    async def verify_node(state: SystemState) -> dict:
        logger.info(f"[Verify] 开始恢复验证: ticket_id={state.ticket_id}")

        # 并行探测三个恢复接口（health、cache_ping、orders_pending）
        # asyncio.to_thread：将同步的 HTTP 请求放到线程池中执行，避免阻塞事件循环
        tasks = [
            asyncio.to_thread(_run_verification_probe, probe["name"], probe["url"])
            for probe in VERIFY_PROBES
        ]
        probes = await asyncio.gather(*tasks)
        # verified：所有探测都成功才算验证通过（逻辑与）
        verified = all(probe.get("success", False) for probe in probes)
        # recovered_at：恢复时间戳，只有验证通过时才记录
        recovered_at = datetime.now(timezone.utc).isoformat() if verified else None

        verification_result = {
            "verified": verified,
            "verification_probe": probes,
            "recovered_at": recovered_at,
        }
        # 生成标准化 Trace 事件：verification_passed，状态由探测结果决定
        trace_event = make_trace_event(
            "verification_passed",
            ticket_id=state.ticket_id,
            agent_name="verify",
            status=status_from_success(verified),
            input_data={
                "execution_status": (state.execution_result or {}).get("overall_status"),
                "probe_urls": [probe["url"] for probe in VERIFY_PROBES],
            },
            output_data=verification_result,
            metadata={
                "probe_count": len(probes),
                "passed_count": sum(1 for probe in probes if probe.get("success")),
                "dispatch_round": state.dispatch_round,
            },
        )

        execution_result = {
            **(state.execution_result or {}),
            **verification_result,
        }

        audit_log = {
            "ticket_id": state.ticket_id,
            "agent_name": "verify",
            "action_type": "verify",
            "action_detail": {
                "verified": verified,
                "probe_count": len(probes),
                "recovered_at": recovered_at,
            },
            "input_context": {
                "execution_status": (state.execution_result or {}).get("overall_status"),
            },
            "output_result": verification_result,
            "dispatch_round": state.dispatch_round,
        }

        logger.info(
            f"[Verify] 恢复验证完成: verified={verified}, "
            f"passed={sum(1 for probe in probes if probe.get('success'))}/{len(probes)}"
        )

        return {
            "verification_result": verification_result,
            "execution_result": execution_result,
            "messages": [
                f"Verify: 恢复验证{'通过' if verified else '未通过'} "
                f"({sum(1 for probe in probes if probe.get('success'))}/{len(probes)})"
            ],
            "audit_logs": [audit_log],
            "trace_events": [trace_event],
        }

    return verify_node
