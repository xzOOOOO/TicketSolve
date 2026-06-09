"""
修复方案的结构化动作 DSL（领域特定语言）

这套 DSL 的设计思路是：让 LLM 只声明"想做什么动作 + 作用在哪个目标上"，
具体的命令字符串由本地代码根据白名单编译生成。LLM 附带的自由文本命令
仅作为展示/文档用途，实际执行时不会使用。

这样做的好处是安全性大幅提升：LLM 无法注入任意 shell 命令，
只能做预定义的白名单动作。

架构分层：
1. LLM 层：输出 action_type + target（结构化声明）
2. DSL 层：校验 action_type + target 是否在白名单内（本文件）
3. 执行层：编译成具体命令并执行（executor_v2.py）

如果 LLM 输出的组合不在白名单内，会抛出 ActionDSLValidationError，
该步骤会被标记为失败，不会执行任何命令。
"""

from __future__ import annotations

# dataclass：用于创建轻量级数据类，比 NamedTuple 更灵活，支持默认值和类型注解
from dataclasses import dataclass

# Any：任意类型；Optional：可选类型（X 或 None）
from typing import Any, Optional


# ───────────────────────────────────────────────
# 白名单常量：定义允许的动作目标
# ───────────────────────────────────────────────
# 这些集合是 DSL 的安全边界。任何不在白名单内的 target 都会被拒绝。
# 白名单采用"最小权限原则"：只包含当前靶场环境确实需要的操作。

# LAB_FAULTS：靶场中可恢复的故障类型（对应 chaos.py 的 recover 命令）
# 这些故障名是 chaos.py 脚本预定义的，LLM 只能从中选择
LAB_FAULTS = {
    "DB_CONN_FAIL",       # 数据库连接失败：模拟数据库不可达
    "APP_PROCESS_DOWN",   # 应用进程宕机：模拟 FastAPI 进程崩溃
    "REDIS_DOWN",         # Redis 服务宕机：模拟缓存服务停止
    "NGINX_BAD_ROUTE",    # Nginx 路由配置错误：模拟反向代理配置异常
    "DB_SLOW_QUERY",      # 数据库慢查询（索引缺失）：模拟性能问题
}

# STARTABLE_CONTAINERS：允许启动的 Docker 容器（docker start 命令）
# 限制为靶场环境中确实存在的容器，防止 LLM 尝试启动无关容器
STARTABLE_CONTAINERS = {
    "srebench-postgres",  # PostgreSQL 数据库容器
    "srebench-app",       # FastAPI 应用容器
    "srebench-redis",     # Redis 缓存容器
}

# RESTARTABLE_CONTAINERS：允许重启的 Docker 容器（docker restart 命令）
# 与 STARTABLE_CONTAINERS 分开管理，因为某些容器只允许重启不允许启动（如 nginx）
RESTARTABLE_CONTAINERS = {
    "srebench-nginx",     # Nginx 反向代理容器
}

# HTTP_PROBE_URLS：允许探测的 HTTP 健康检查 URL
# 限制为靶场服务确实暴露的端点，防止 LLM 探测内网其他服务
HTTP_PROBE_URLS = {
    "http://localhost:18080/health",       # Nginx 入口健康检查（对外暴露）
    "http://localhost:18081/health",       # App 直连健康检查（内部暴露）
    "http://localhost:18080/cache/ping",   # Redis 缓存接口（测试缓存连通性）
    "http://localhost:18080/orders/pending",  # 数据库查询接口（测试 DB 连通性）
}

# ORDERS_INDEX_SQL：重建 orders 表索引的 SQL 语句（用于修复 DB_SLOW_QUERY）
# 这是一个固定的、安全的 DDL 语句，创建索引不会丢失数据
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

    这是 DSL 的安全防线：任何非法组合都会被拦截，不会进入执行阶段。
    继承 ValueError 是为了让上层可以用 try/except ValueError 统一捕获。

    触发场景：
    - LLM 输出了未知的 action_type（如 DELETE_CONTAINER）
    - LLM 输出了合法的 action_type 但不合法的 target（如 START_CONTAINER + mysql）
    - 缺少 action_type 或 target 字段
    """


@dataclass(frozen=True)
class CompiledAction:
    """
    编译后的动作。

    这是一个不可变数据类（frozen=True），一旦创建就不能修改，
    防止执行过程中被意外篡改。

    属性：
        action_type: 动作类型，如 RECOVER_FAULT / START_CONTAINER 等
        target: 动作目标，如 APP_PROCESS_DOWN / srebench-app 等
        command: 编译生成的实际可执行命令字符串

    为什么用 dataclass 而不是 dict：
    - 类型安全：IDE 能自动补全和检查类型
    - 不可变性：frozen=True 防止意外修改
    - 可读性：CompiledAction(action_type="START", target="app") 比 dict 更清晰
    """
    action_type: str   # 动作类型，大写字符串，如 "START_CONTAINER"
    target: str        # 动作目标，如 "srebench-app"
    command: str       # 编译后的实际命令，如 "docker start srebench-app"


# ───────────────────────────────────────────────
# 公开 API：从修复步骤中提取和编译动作
# ───────────────────────────────────────────────

def _extract_step_action(step: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    从修复步骤中提取正向动作（执行动作）的定义。

    唯一支持格式：
        step["action_type"] + step["target"] 平铺字段

    为什么不支持嵌套格式：
    早期版本用过 step["action_spec"]["type"] 的嵌套结构，但 LLM 容易搞错层级，
    所以改为平铺字段，降低 LLM 输出难度。

    参数：
        step: 修复步骤字典，通常来自 FixPlan.steps 中的某个元素

    返回：
        动作定义字典（包含 action_type、target、parameters），
        如果步骤没有定义动作则返回 None

    抛出：
        ActionDSLValidationError: 如果检测到不支持的嵌套字段 action_spec
    """
    # 检查是否存在已废弃的嵌套字段，如果存在则报错提示使用平铺字段
    if step.get("action_spec") is not None:
        raise ActionDSLValidationError(
            "不支持嵌套字段 action_spec；请使用平铺字段 action_type + target"
        )

    # 提取动作类型和目标
    action_type = step.get("action_type")
    target = step.get("target")

    # 如果 action_type 或 target 任一存在，就认为是定义了动作
    # 允许其中一个为空，由 compile_action 负责校验完整性
    if action_type or target:
        return {
            "action_type": action_type,           # 动作类型（可能为 None）
            "target": target,                     # 动作目标（可能为 None）
            "parameters": step.get("parameters", {}),  # 可选参数，默认为空字典
        }

    # 既没有 action_type 也没有 target，认为该步骤没有定义结构化动作
    return None


def _extract_rollback_action(step: dict[str, Any]) -> Optional[dict[str, Any]]:
    """
    从修复步骤中提取回滚动作的定义。

    回滚动作用于当正向执行失败时撤销已执行的操作。
    例如：正向动作是 "docker start app"，回滚动作可能是 "docker stop app"。

    唯一支持格式：
        step["rollback_action_type"] + step["rollback_target"] 平铺字段

    参数：
        step: 修复步骤字典

    返回：
        回滚动作定义字典，如果没有定义则返回 None

    抛出：
        ActionDSLValidationError: 如果检测到不支持的嵌套字段 rollback_action
    """
    # 检查是否存在已废弃的嵌套字段
    if step.get("rollback_action") is not None:
        raise ActionDSLValidationError(
            "不支持嵌套字段 rollback_action；请使用平铺字段 "
            "rollback_action_type + rollback_target"
        )

    # 提取回滚动作类型和目标
    action_type = step.get("rollback_action_type")
    target = step.get("rollback_target")

    # 如果任一存在，就认为是定义了回滚动作
    if action_type or target:
        return {
            "action_type": action_type,
            "target": target,
            "parameters": step.get("rollback_parameters", {}),
        }

    # 没有定义回滚动作，返回 None
    return None


def compile_step_action(step: dict[str, Any]) -> Optional[CompiledAction]:
    """
    编译修复步骤中的正向动作。

    流程：先提取动作定义 → 再调用编译器生成命令。

    参数：
        step: 修复步骤字典

    返回：
        编译后的 CompiledAction，如果步骤没有定义动作则返回 None

    使用示例：
        >>> step = {"action_type": "START_CONTAINER", "target": "srebench-app"}
        >>> compiled = compile_step_action(step)
        >>> compiled.command
        'docker start srebench-app'
    """
    action = _extract_step_action(step)
    # 如果提取到动作定义，就调用核心编译器；否则返回 None
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
        NOOP               → 空命令（无操作，用于占位或跳过）

    参数：
        action: 动作定义字典，必须包含 action_type 和 target

    返回：
        CompiledAction 对象，包含编译后的 command 字符串

    抛出：
        ActionDSLValidationError: 当 action_type 或 target 不在白名单内时

    安全机制：
    1. action_type 必须是大写字符串，且在白名单内
    2. target 必须是对应动作类型的允许目标集合内
    3. 命令模板是硬编码的，LLM 无法控制命令的具体格式
    """
    # 规范化输入：转大写、去空格，防止大小写不一致导致的问题
    action_type = _normalize_action_type(action.get("action_type"))
    target = _normalize_target(action.get("target"))

    # 根据动作类型选择编译模板，并校验 target 是否在白名单内
    # 每个分支都调用 _require_target 进行白名单校验

    if action_type == "RECOVER_FAULT":
        # 恢复靶场故障：调用 chaos.py 脚本
        _require_target(target, LAB_FAULTS, action_type)
        command = f"python lab/chaos.py recover {target}"

    elif action_type == "START_CONTAINER":
        # 启动 Docker 容器
        _require_target(target, STARTABLE_CONTAINERS, action_type)
        command = f"docker start {target}"

    elif action_type == "RESTART_CONTAINER":
        # 重启 Docker 容器
        _require_target(target, RESTARTABLE_CONTAINERS, action_type)
        command = f"docker restart {target}"

    elif action_type == "REBUILD_ORDERS_INDEX":
        # 重建 orders 表索引：用于修复 DB_SLOW_QUERY
        # 允许的目标可以是故障名、容器名或索引名，增加灵活性
        allowed_targets = {"DB_SLOW_QUERY", "srebench-postgres", "idx_orders_status_created_at"}
        _require_target(target, allowed_targets, action_type)
        command = (
            'docker exec srebench-postgres psql -U labuser -d labdb '
            f'-c "{ORDERS_INDEX_SQL}"'
        )

    elif action_type == "HTTP_PROBE":
        # HTTP 健康检查探测
        _require_target(target, HTTP_PROBE_URLS, action_type)
        command = f"curl {target}"

    elif action_type == "NOOP":
        # 无操作：用于占位或跳过某些步骤
        command = ""

    else:
        # 未知的 action_type，列出所有允许的类型供调试
        allowed = ", ".join(sorted(allowed_action_types()))
        raise ActionDSLValidationError(
            f"不支持的 action_type={action_type!r}；允许的类型: {allowed}"
        )

    # 返回不可变的编译后动作对象
    return CompiledAction(action_type=action_type, target=target, command=command)


def allowed_action_types() -> set[str]:
    """
    返回所有允许的动作类型集合。

    用于错误提示和 Guardrail 校验。
    当 LLM 输出了未知的 action_type 时，可以在错误信息中列出所有允许的选项。

    返回：
        包含所有合法动作类型的大写字符串集合
    """
    return {
        "RECOVER_FAULT",        # 恢复靶场故障
        "START_CONTAINER",      # 启动 Docker 容器
        "RESTART_CONTAINER",    # 重启 Docker 容器
        "REBUILD_ORDERS_INDEX", # 重建数据库索引
        "HTTP_PROBE",           # HTTP 健康探测
        "NOOP",                 # 无操作
    }


# ───────────────────────────────────────────────
# 内部工具函数：规范化与校验
# ───────────────────────────────────────────────

def _normalize_action_type(value: Any) -> str:
    """
    规范化动作类型：转大写、去首尾空格。

    为什么需要规范化：
    LLM 可能输出 "start_container" 或 "Start_Container"，
    我们需要统一转成 "START_CONTAINER" 再做白名单匹配。

    参数：
        value: 原始 action_type 值，可能是字符串、None 或其他类型

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
        value: 原始 target 值，可能是字符串、None 或其他类型

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

    例如：
    - action_type="START_CONTAINER" 是合法的
    - 但 target="mysql" 不在 STARTABLE_CONTAINERS 中
    - 所以这个组合会被拦截

    参数：
        target: 要校验的目标值
        allowed_targets: 该动作类型允许的目标集合
        action_type: 当前动作类型（用于错误提示）

    抛出：
        ActionDSLValidationError: 当 target 不在白名单内时
    """
    if target not in allowed_targets:
        # 把允许的目标排序后拼接成字符串，方便错误提示
        allowed = ", ".join(sorted(allowed_targets))
        raise ActionDSLValidationError(
            f"target={target!r} 不允许用于 action_type={action_type}；"
            f"允许的目标: {allowed}"
        )
