"""
SREBench Lite Docker 实验环境的诊断工具集。

本模块提供一系列诊断函数，用于检测实验环境中各组件的运行状态。
设计原则：保持 MCP 工具函数精简和确定性，底层探测只使用
标准库的进程/套接字/HTTP 探测，以及 psycopg2（项目已有依赖）。
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# 实验环境中的容器名称映射：逻辑名 → Docker容器名
LAB_CONTAINERS = {
    "app": "srebench-app",
    "postgres": "srebench-postgres",
    "redis": "srebench-redis",
    "nginx": "srebench-nginx",
}

# 数据库连接配置（连接宿主机映射的端口，不是容器内部端口）
DB_CONFIG = {
    "host": "localhost",
    "port": 15432,          # 宿主机映射端口，对应容器内的5432
    "dbname": "labdb",
    "user": "labuser",
    "password": "labpass",
    "connect_timeout": 3,   # 连接超时3秒，避免长时间阻塞
}


def _now() -> float:
    """返回当前时间戳（秒，保留3位小数），用于诊断结果的时间标记。"""
    return round(time.time(), 3)


def _docker(*args: str, timeout: int = 10) -> tuple[bool, str]:
    """执行 docker 命令并返回执行结果。

    Args:
        *args: 传递给 docker 命令的参数，如 "inspect", "容器名"
        timeout: 命令超时时间（秒），默认10秒

    Returns:
        (是否成功, 输出内容) 的元组。成功时输出为 stdout，失败时为 stderr 或错误信息。
    """
    try:
        completed = subprocess.run(
            ["docker", *args],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,    # 不自动抛异常，由返回码判断成功与否
        )
        output = (completed.stdout or completed.stderr or "").strip()
        return completed.returncode == 0, output
    except FileNotFoundError:
        return False, "docker command not found"       # 系统没装docker
    except subprocess.TimeoutExpired:
        return False, "docker command timed out"       # 命令超时
    except Exception as exc:
        return False, str(exc)                         # 其他异常


def _container_status(container_name: str) -> dict[str, Any]:
    """通过 docker inspect 获取容器的运行状态。

    Returns:
        包含容器信息的字典：exists(是否存在)、status(状态)、running(是否运行中)、
        exit_code(退出码)、error(错误信息)。
    """
    ok, output = _docker("inspect", container_name)
    if not ok:
        # 容器不存在或docker命令失败
        return {
            "container": container_name,
            "exists": False,
            "status": "unknown",
            "error": output,
        }

    try:
        data = json.loads(output)[0]            # inspect 输出是JSON数组，取第一个元素
        state = data.get("State", {})
        return {
            "container": container_name,
            "exists": True,
            "status": state.get("Status", "unknown"),   # 如 "running"、"exited"
            "running": bool(state.get("Running")),       # 是否正在运行
            "exit_code": state.get("ExitCode"),          # 退出码，0=正常退出
            "error": state.get("Error") or "",           # 容器错误信息
        }
    except Exception as exc:
        return {
            "container": container_name,
            "exists": True,
            "status": "unknown",
            "error": f"failed to parse docker inspect output: {exc}",
        }


def _http_get(url: str, timeout: int = 5) -> dict[str, Any]:
    """发送 HTTP GET 请求并记录响应状态和耗时。

    Args:
        url: 请求的目标URL
        timeout: 请求超时时间（秒），默认5秒

    Returns:
        包含请求结果的字典：status(ok/failed)、http_status(HTTP状态码)、
        elapsed_ms(耗时毫秒)、body(响应体前500字符)、error(错误信息)。
    """
    started = time.perf_counter()
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return {
                "url": url,
                "status": "ok" if 200 <= resp.status < 300 else "failed",  # 2xx视为成功
                "http_status": resp.status,
                "elapsed_ms": elapsed_ms,
                "body": body[:500],   # 只保留前500字符，避免返回数据过大
            }
    except HTTPError as exc:
        # 服务器返回了HTTP错误响应（4xx/5xx）
        return {
            "url": url,
            "status": "failed",
            "http_status": exc.code,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "error": exc.reason,
        }
    except URLError as exc:
        # 网络层错误（DNS解析失败、连接被拒等）
        return {
            "url": url,
            "status": "failed",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "error": str(exc.reason),
        }
    except Exception as exc:
        # 其他未知异常
        return {
            "url": url,
            "status": "failed",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "error": str(exc),
        }


def _tcp_probe(host: str, port: int, timeout: float = 3.0) -> dict[str, Any]:
    """TCP 连通性探测：尝试建立 TCP 连接来检测端口是否可达。

    Args:
        host: 目标主机地址
        port: 目标端口号
        timeout: 连接超时时间（秒），默认3秒

    Returns:
        包含探测结果的字典：status(reachable/unreachable)、latency_ms(延迟毫秒)。
    """
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {
                "target": f"{host}:{port}",
                "status": "reachable",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
    except Exception as exc:
        return {
            "target": f"{host}:{port}",
            "status": "unreachable",
            "latency_ms": None,
            "error": str(exc),
        }


def _db_query(sql: str) -> tuple[bool, Any]:
    """在实验数据库上执行 SQL 查询。

    Args:
        sql: 要执行的SQL语句

    Returns:
        (是否成功, 查询结果) 的元组。SELECT 语句返回行列表，
        非查询语句返回 None；失败时第二个元素为错误信息字符串。
    """
    try:
        import psycopg2

        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description:     # 有结果集的查询（如SELECT）
                    return True, cur.fetchall()
                return True, None       # 无结果集的操作（如DROP/CREATE）
    except Exception as exc:
        return False, str(exc)


def check_db_connection_result() -> dict[str, Any]:
    """检查数据库连接是否正常。

    三层检测：1) 容器是否运行  2) TCP端口是否可达  3) 能否执行简单查询。
    只有三层全部通过才判定为 ok，否则收集失败证据。
    """
    container = _container_status(LAB_CONTAINERS["postgres"])
    tcp = _tcp_probe("localhost", 15432)
    ok, query_result = _db_query("select 1;")

    # 三层检测全部通过才算ok
    status = "ok" if container.get("running") and tcp["status"] == "reachable" and ok else "failed"
    evidence = []
    if not container.get("running"):
        evidence.append(f"container {LAB_CONTAINERS['postgres']} is not running")
    if tcp["status"] != "reachable":
        evidence.append(f"tcp probe failed: {tcp.get('error')}")
    if not ok:
        evidence.append(f"database query failed: {query_result}")

    return {
        "tool": "check_db_connection",
        "status": status,
        "container": container,
        "tcp_probe": tcp,
        "query_result": query_result if ok else None,
        "evidence": evidence or ["postgres container, port and query are healthy"],
        "suggested_domain": "db_agent",    # 建议由数据库领域的agent处理
        "timestamp": _now(),
    }


def check_db_slow_query_result() -> dict[str, Any]:
    """检查是否存在慢查询问题。

    检测逻辑：1) 查询 orders 表的关键索引是否存在  2) 请求待处理订单接口看响应时间。
    如果索引缺失或接口响应超过800ms，判定为 slow_or_failed。
    """
    ok, index_rows = _db_query(
        """
        select indexname
        from pg_indexes
        where tablename = 'orders'
          and indexname = 'idx_orders_status_created_at';
        """
    )
    http = _http_get("http://localhost:18080/orders/pending")
    index_exists = bool(ok and index_rows)
    status = "ok" if index_exists and http.get("status") == "ok" else "slow_or_failed"

    evidence = []
    if not ok:
        evidence.append(f"failed to inspect indexes: {index_rows}")
    elif not index_exists:
        evidence.append("idx_orders_status_created_at is missing")  # 索引被删了
    if http.get("status") != "ok":
        evidence.append(f"pending-orders endpoint failed: {http.get('error') or http.get('http_status')}")
    if http.get("elapsed_ms", 0) > 800:
        evidence.append(f"pending-orders endpoint is slow: {http.get('elapsed_ms')}ms")  # 响应超过800ms视为慢查询

    return {
        "tool": "check_db_slow_query",
        "status": status,
        "index_exists": index_exists,
        "http_probe": http,
        "evidence": evidence or ["orders status index exists and endpoint responded normally"],
        "recommendation": (
            "create index idx_orders_status_created_at on orders(status, created_at desc)"
            if not index_exists else "no index action required"   # 索引缺失时给出修复建议
        ),
        "suggested_domain": "db_agent",
        "timestamp": _now(),
    }


def check_db_deadlock_result() -> dict[str, Any]:
    """检查数据库是否存在锁等待/死锁。

    查询 pg_stat_activity 视图，找出处于 Lock 等待状态的会话。
    如果有会话在等锁，说明可能存在死锁或锁竞争问题。
    """
    ok, rows = _db_query(
        """
        select pid, wait_event_type, wait_event, state
        from pg_stat_activity
        where wait_event_type = 'Lock';
        """
    )
    lock_waits = rows if ok else []
    return {
        "tool": "check_db_deadlock",
        "status": "lock_waits_detected" if lock_waits else "ok",
        "lock_waits": lock_waits,
        "evidence": [f"{len(lock_waits)} sessions waiting on locks"] if lock_waits else ["no lock waits detected"],
        "error": None if ok else rows,     # 查询失败时 rows 存的是错误信息字符串
        "suggested_domain": "db_agent",
        "timestamp": _now(),
    }


def check_network_ping_result(host: str) -> dict[str, Any]:
    """检查网络连通性（TCP 层面）。

    支持两种输入格式：
    - "host:port" 格式：直接解析并探测指定地址
    - 逻辑名称（如 "nginx"、"app"）：映射到实验环境对应的地址和端口
    - 未知主机名：默认探测其80端口
    """
    if ":" in host:
        # 输入包含端口号，如 "localhost:18080"
        raw_host, raw_port = host.rsplit(":", 1)
        try:
            return {
                "tool": "check_network_ping",
                **_tcp_probe(raw_host or "localhost", int(raw_port)),
                "suggested_domain": "net_agent",
                "timestamp": _now(),
            }
        except ValueError:
            pass    # 端口号解析失败，走下面的逻辑名映射

    # 实验环境中的已知目标映射
    known_targets = {
        "nginx": ("localhost", 18080),
        "app": ("localhost", 18081),
        "postgres": ("localhost", 15432),
        "redis": ("localhost", 16379),
        "localhost": ("localhost", 18080),
    }
    target_host, target_port = known_targets.get(host, (host, 80))  # 未知主机默认探测80端口
    return {
        "tool": "check_network_ping",
        **_tcp_probe(target_host, target_port),
        "suggested_domain": "net_agent",
        "timestamp": _now(),
    }


def check_network_dns_result(domain: str) -> dict[str, Any]:
    """检查 DNS 解析是否正常。

    尝试将域名解析为IP地址，验证DNS服务是否可用。
    """
    try:
        resolved = socket.gethostbyname(domain)
        status = "ok"
        error = None
    except Exception as exc:
        resolved = None
        status = "failed"
        error = str(exc)
    return {
        "tool": "check_network_dns",
        "domain": domain,
        "status": status,
        "resolved_ip": resolved,
        "error": error,
        "suggested_domain": "net_agent",
        "timestamp": _now(),
    }


def check_network_http_route_result(url: str = "http://localhost:18080/health") -> dict[str, Any]:
    """检查 Nginx HTTP 路由是否正常。

    对比两条路径：1) 直接访问应用  2) 通过Nginx访问。
    如果直连正常但Nginx转发失败，说明是Nginx路由配置问题（如 broken_route.conf 场景）。
    """
    nginx = _container_status(LAB_CONTAINERS["nginx"])
    direct = _http_get("http://localhost:18081/health")     # 绕过Nginx直连应用
    via_nginx = _http_get(url)                               # 通过Nginx访问
    route_failed = direct.get("status") == "ok" and via_nginx.get("status") != "ok"
    # 直连OK但Nginx不通 → 路由配置有问题

    evidence = []
    if route_failed:
        evidence.append("direct app health is ok but nginx route failed")  # 典型的路由配置错误
    if not nginx.get("running"):
        evidence.append("nginx container is not running")

    return {
        "tool": "check_network_http_route",
        "status": "route_failed" if route_failed else via_nginx.get("status", "failed"),
        "nginx_container": nginx,
        "direct_app_probe": direct,
        "nginx_probe": via_nginx,
        "evidence": evidence or ["nginx route and direct app probe are consistent"],
        "suggested_domain": "net_agent",
        "timestamp": _now(),
    }


def check_app_process_result(process_name: str) -> dict[str, Any]:
    """检查应用进程是否在运行。

    支持传入逻辑名称（如 "app"、"api"、"fastapi"）或容器名。
    同时获取容器最近20行日志，便于排查问题。
    """
    key = process_name.lower()
    if key in ("app", "api", "fastapi", "main"):
        container_name = LAB_CONTAINERS["app"]      # 逻辑名映射到实际容器名
    else:
        # 如果传入的不是已知逻辑名，优先当作容器名处理
        container_name = process_name if process_name.startswith("srebench-") else LAB_CONTAINERS["app"]

    container = _container_status(container_name)
    logs_ok, logs = _docker("logs", "--tail", "20", container_name)  # 取最近20行日志
    return {
        "tool": "check_app_process",
        "process": process_name,
        "status": "running" if container.get("running") else "stopped",
        "container": container,
        "recent_logs": logs[-1000:] if logs_ok else logs,  # 最多保留1000字符，避免日志过长
        "evidence": (
            [f"container {container_name} is running"]
            if container.get("running")
            else [f"container {container_name} is not running"]
        ),
        "suggested_domain": "app_agent",
        "timestamp": _now(),
    }


def check_app_port_result(port: int) -> dict[str, Any]:
    """检查指定端口是否可达（在本地主机上探测）。"""
    host = "localhost"
    probe = _tcp_probe(host, int(port))
    return {
        "tool": "check_app_port",
        "port": port,
        **probe,
        "suggested_domain": "app_agent",
        "timestamp": _now(),
    }


def check_app_health_result(url: str = "http://localhost:18081/health") -> dict[str, Any]:
    """检查应用健康状态。

    同时检查容器运行状态和 /health 接口的HTTP响应，
    默认直连应用端口（绕过Nginx），确保检测的是应用本身而非网关。
    """
    app_container = _container_status(LAB_CONTAINERS["app"])
    http = _http_get(url)
    return {
        "tool": "check_app_health",
        "status": http.get("status"),
        "container": app_container,
        "http_probe": http,
        "evidence": (
            ["app health endpoint responded successfully"]
            if http.get("status") == "ok"
            else [f"app health failed: {http.get('error') or http.get('http_status')}"]
        ),
        "suggested_domain": "app_agent",
        "timestamp": _now(),
    }


def check_app_redis_connection_result() -> dict[str, Any]:
    """检查 Redis 缓存连接是否正常。

    三层检测：1) Redis容器是否运行  2) TCP端口是否可达  3) 应用的缓存接口(/cache/ping)是否正常。
    三层全部通过才判定为 ok。
    """
    container = _container_status(LAB_CONTAINERS["redis"])
    tcp = _tcp_probe("localhost", 16379)
    http = _http_get("http://localhost:18080/cache/ping")
    status = "ok" if container.get("running") and tcp["status"] == "reachable" and http["status"] == "ok" else "failed"
    evidence = []
    if not container.get("running"):
        evidence.append("redis container is not running")
    if tcp["status"] != "reachable":
        evidence.append(f"redis tcp probe failed: {tcp.get('error')}")
    if http["status"] != "ok":
        evidence.append(f"cache endpoint failed: {http.get('error') or http.get('http_status')}")

    return {
        "tool": "check_app_redis_connection",
        "status": status,
        "container": container,
        "tcp_probe": tcp,
        "cache_endpoint_probe": http,
        "evidence": evidence or ["redis container, port and cache endpoint are healthy"],
        "suggested_domain": "app_agent",
        "timestamp": _now(),
    }
