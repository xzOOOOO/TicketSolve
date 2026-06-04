"""
修复方案的结构化动作 DSL（领域特定语言）。

这套 DSL 的设计思路是：让 LLM 只声明"想做什么动作 + 作用在哪个目标上"，
具体的命令字符串由本地代码根据白名单编译生成。LLM 附带的自由文本命令
仅作为展示/文档用途，实际执行时不会使用。

这样做的好处是安全性大幅提升：LLM 无法注入任意 shell 命令，
只能做预定义的白名单动作。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# ───────────────────────────────────────────────
# 白名单常量：定义允许的动作目标
# ───────────────────────────────────────────────

# 靶场中可恢复的故障类型（对应 chaos.py 的 recover 命令）
LAB_FAULTS = {
    "DB_CONN_FAIL",       # 数据库连接失败
    "APP_PROCESS_DOWN",   # 应用进程宕机
    "REDIS_DOWN",         # Redis 服务宕机
    "NGINX_BAD_ROUTE",    # Nginx 路由配置错误
    "DB_SLOW_QUERY",      # 数据库慢查询（索引缺失）
}

# 允许启动的 Docker 容器（docker start）
STARTABLE_CONTAINERS = {
    "srebench-postgres",  # PostgreSQL 数据库容器
    "srebench-app",       # FastAPI 应用容器
    "srebench-redis",     # Redis 缓存容器
}

# 允许重启的 Docker 容器（docker restart）
RESTARTABLE_CONTAINERS = {
    "srebench-nginx",     # Nginx 反向代理容器
}

# 允许探测的 HTTP 健康检查 URL
HTTP_PROBE_URLS = {
    "http://localhost:18080/health",       # Nginx 入口健康检查
    "http://localhost:18081/health",       # App 直连健康检查
    "http://localhost:18080/cache/ping",   # Redis 缓存接口
    "http://localhost:18080/orders/pending",  # 数据库查询接口
}

# 重建 orders 表索引的 SQL 语句（用于修复 DB_SLOW_QUERY）
ORDERS_INDEX_SQL = (
    "create index if not exists idx_orders_status_created_at "
    "on orders (status, created_at desc);"
)


# ───────────────────────────────────────────────
# 异常与数据结构
# ───────────────────────────────────────────────

class ActionDSLValidationError(ValueError):
    """
    当 action_type + target 组合不在本地白名单内时抛出。
    这是 DSL 的安全防线：任何非法组合都会被拦截。
    """


@dataclass(frozen=True)
class CompiledAction:
    """
    编译后的动作。

    属性：
        action_type: 动作类型，如 RECOVER_FAULT / START_CONTAINER 等
        target: 动作目标，如 APP_PROCESS_DOWN / srebench-app 等
        command: 编译生成的实际可执行命令字符串
    """
    action_type: str
    target: str
    command: str


# ───────────────────────────────────────────────
# 公开 API：从修复步骤中提取和编译动作
# ───────────────────────────────────────────────

def _extract_step_action(step: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    从修复步骤中提取正向动作（执行动作）的定义。

    唯一支持格式：
        step["action_type"] + step["target"] 平铺字段

    参数：
        step: 修复步骤字典

    返回：
        动作定义字典（包含 action_type、target、parameters），
        如果没有定义则返回 None
    """
    if step.get("action_spec") is not None:
        raise ActionDSLValidationError(
            "不支持嵌套字段 action_spec；请使用平铺字段 action_type + target"
        )

    action_type = step.get("action_type")
    target = step.get("target")
    if action_type or target:
        return {
            "action_type": action_type,
            "target": target,
            "parameters": step.get("parameters", {}),
        }

    return None


def _extract_rollback_action(step: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    从修复步骤中提取回滚动作的定义。

    回滚动作用于当正向执行失败时撤销操作。
    唯一支持格式：
        step["rollback_action_type"] + step["rollback_target"] 平铺字段

    参数：
        step: 修复步骤字典

    返回：
        回滚动作定义字典，如果没有定义则返回 None
    """
    if step.get("rollback_action") is not None:
        raise ActionDSLValidationError(
            "不支持嵌套字段 rollback_action；请使用平铺字段 "
            "rollback_action_type + rollback_target"
        )

    action_type = step.get("rollback_action_type")
    target = step.get("rollback_target")
    if action_type or target:
        return {
            "action_type": action_type,
            "target": target,
            "parameters": step.get("rollback_parameters", {}),
        }

    return None


def compile_step_action(step: dict[str, Any]) -> Optional[CompiledAction]:
    """
    编译修复步骤中的正向动作。

    流程：先提取动作定义 → 再调用编译器生成命令。

    参数：
        step: 修复步骤字典

    返回：
        编译后的 CompiledAction，如果步骤没有定义动作则返回 None
    """
    action = _extract_step_action(step)
    return compile_action(action) if action is not None else None


def compile_rollback_action(step: dict[str, Any]) -> Optional[CompiledAction]:
    """
    编译修复步骤中的回滚动作。

    参数：
        step: 修复步骤字典

    返回：
        编译后的 CompiledAction，如果没有定义回滚动作则返回 None
    """
    action = _extract_rollback_action(step)
    return compile_action(action) if action is not None else None


# ───────────────────────────────────────────────
# 核心编译器：action_type + target → 安全命令
# ───────────────────────────────────────────────

def compile_action(action: dict[str, Any]) -> CompiledAction:
    """
    核心编译函数：将 action_type + target 编译成安全的命令字符串。

    这是 DSL 的安全核心。所有命令都是硬编码的模板，LLM 只能选预定义的组合，
    无法注入任意 shell 命令。

    支持的动作类型：
        RECOVER_FAULT      → python lab/chaos.py recover <故障名>
        START_CONTAINER    → docker start <容器名>
        RESTART_CONTAINER  → docker restart <容器名>
        REBUILD_ORDERS_INDEX → docker exec ... psql -c "重建索引SQL"
        HTTP_PROBE         → curl <URL>
        NOOP               → 空命令（无操作）

    参数：
        action: 动作定义字典，必须包含 action_type 和 target

    返回：
        CompiledAction 对象

    抛出：
        ActionDSLValidationError: 当 action_type 或 target 不在白名单内时
    """
    action_type = _normalize_action_type(action.get("action_type"))
    target = _normalize_target(action.get("target"))

    # 根据动作类型选择编译模板，并校验 target 是否在白名单内
    if action_type == "RECOVER_FAULT":
        _require_target(target, LAB_FAULTS, action_type)
        command = f"python lab/chaos.py recover {target}"

    elif action_type == "START_CONTAINER":
        _require_target(target, STARTABLE_CONTAINERS, action_type)
        command = f"docker start {target}"

    elif action_type == "RESTART_CONTAINER":
        _require_target(target, RESTARTABLE_CONTAINERS, action_type)
        command = f"docker restart {target}"

    elif action_type == "REBUILD_ORDERS_INDEX":
        # 重建索引的目标可以是故障名、容器名或索引名
        allowed_targets = {"DB_SLOW_QUERY", "srebench-postgres", "idx_orders_status_created_at"}
        _require_target(target, allowed_targets, action_type)
        command = (
            'docker exec srebench-postgres psql -U labuser -d labdb '
            f'-c "{ORDERS_INDEX_SQL}"'
        )

    elif action_type == "HTTP_PROBE":
        _require_target(target, HTTP_PROBE_URLS, action_type)
        command = f"curl {target}"

    elif action_type == "NOOP":
        # 无操作，用于占位或跳过
        command = ""

    else:
        # 未知的 action_type，列出所有允许的类型
        allowed = ", ".join(sorted(allowed_action_types()))
        raise ActionDSLValidationError(
            f"不支持的 action_type={action_type!r}；允许的类型: {allowed}"
        )

    return CompiledAction(action_type=action_type, target=target, command=command)


def allowed_action_types() -> set[str]:
    """
    返回所有允许的动作类型集合。

    用于错误提示和 Guardrail 校验。
    """
    return {
        "RECOVER_FAULT",
        "START_CONTAINER",
        "RESTART_CONTAINER",
        "REBUILD_ORDERS_INDEX",
        "HTTP_PROBE",
        "NOOP",
    }


def _normalize_action_type(value: Any) -> str:
    """
    规范化动作类型：转大写、去首尾空格。

    参数：
        value: 原始 action_type 值

    返回：
        规范化后的大写字符串

    抛出：
        ActionDSLValidationError: 当 value 为 None 时
    """
    if value is None:
        raise ActionDSLValidationError("缺少 action_type")
    return str(value).strip().upper()


def _normalize_target(value: Any) -> str:
    """
    规范化目标：转字符串、去首尾空格。

    参数：
        value: 原始 target 值

    返回：
        规范化后的字符串，None 则返回空字符串
    """
    if value is None:
        return ""
    return str(value).strip()


def _require_target(target: str, allowed_targets: set[str], action_type: str) -> None:
    """
    校验目标是否在白名单内。

    这是 DSL 的第二道安全防线：即使 action_type 合法，
    target 也必须在对应的允许集合中。

    参数：
        target: 要校验的目标值
        allowed_targets: 该动作类型允许的目标集合
        action_type: 当前动作类型（用于错误提示）

    抛出：
        ActionDSLValidationError: 当 target 不在白名单内时
    """
    if target not in allowed_targets:
        allowed = ", ".join(sorted(allowed_targets))
        raise ActionDSLValidationError(
            f"target={target!r} 不允许用于 action_type={action_type}；"
            f"允许的目标: {allowed}"
        )
