"""
闭环执行器（Closed-loop Executor）

核心架构：Observe → Decide → Act 循环
不是一次性跑完所有步骤，而是每一步都观察真实结果，根据结果决策下一步。

执行流程：
    执行步骤1 → 观察真实输出(stdout/stderr/exit_code)
             → [成功] → 执行步骤2
             → [失败] → LLM分析错误 → [可重试] → 重试/调整命令
                                      → [不可重试] → 执行回滚 → 报告失败

关键区别（vs 原 Mock 执行器）：
1. 每一步都观察真实结果（命令的 stdout/stderr/exit code）
2. 失败时 LLM 介入分析真实错误信息，决定重试/调整/回滚
3. 状态转换由真实结果驱动，不是 LLM 凭空决定
4. 完整的执行轨迹记录，可追溯每一步的决策过程

Mock 说明：
- MockCommandRunner 模拟命令执行，返回预设的 stdout/stderr/exit_code
- 架构是闭环的，替换为真实 CommandRunner（如 subprocess）即可用于生产
- 面试重点讲架构设计，不是 Mock 本身
"""

import asyncio
import random
import shlex
import subprocess
import sys
import time
from typing import Optional
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from action_dsl import (
    ActionDSLValidationError,
    CompiledAction,
    HTTP_PROBE_URLS,
    LAB_FAULTS,
    ORDERS_INDEX_SQL,
    RESTARTABLE_CONTAINERS,
    STARTABLE_CONTAINERS,
    compile_rollback_action,
    compile_step_action,
)
from schemas import CommandExecutionResult, ErrorAnalysisOutput
from logger import logger


# 项目根目录：当前文件向上退一级（src 的上一级就是项目根目录）
# 用于定位 lab/chaos.py 等靶场脚本
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 靶场故障恢复脚本的绝对路径
# SREBench Lite 环境中，所有故障恢复都通过调用此脚本完成
LAB_CHAOS_SCRIPT = PROJECT_ROOT / "lab" / "chaos.py"


def _is_effective_command(command: Optional[str]) -> bool:
    """判断命令是否真的需要执行，避免把 N/A 当成回滚命令。

    原理：把命令字符串去空白、转小写后，检查是否在"无效命令"黑名单中。
    这是防御性编程——LLM 有时会把"无需回滚"填到 rollback_command 字段里。

    参数：
        command: 命令字符串，可能为 None

    返回：
        True 表示命令有效，需要执行；False 表示是占位符，跳过即可
    """
    if command is None:
        return False
    # 归一化：去首尾空白 + 转小写，确保大小写不敏感匹配
    normalized = command.strip().lower()
    # 无效命令黑名单：空字符串、各种"无"的表达方式
    return normalized not in {"", "n/a", "na", "none", "null", "无", "无需回滚", "不需要"}


def _resolve_step_command(step: dict) -> tuple[str, Optional[CompiledAction]]:
    """
    解析修复步骤中的执行命令。

    优先使用结构化动作 DSL（action_type + target），由本地编译器生成安全命令。
    如果步骤没有定义结构化动作，则回退到兼容模式，使用 step["command"] 自由文本命令。

    参数：
        step: 修复步骤字典

    返回：
        (command, compiled_action): 实际要执行的命令字符串，以及编译后的动作对象（如果有）
    """
    compiled = compile_step_action(step)
    if compiled:
        return compiled.command, compiled
    return step.get("command", "") or "", None


def _resolve_rollback_command(step: dict) -> tuple[str, Optional[CompiledAction]]:
    """
    解析修复步骤中的回滚命令。

    优先使用结构化回滚动作（rollback_action_type + rollback_target），由本地编译器生成。
    如果步骤没有定义结构化回滚动作，则回退到兼容模式，使用 step["rollback_command"] 自由文本命令。

    参数：
        step: 修复步骤字典

    返回：
        (command, compiled_action): 回滚命令字符串，以及编译后的回滚动作对象（如果有）
    """
    compiled = compile_rollback_action(step)
    if compiled:
        return compiled.command, compiled
    return step.get("rollback_command", "") or "", None


def _safe_command_for_log(step: dict) -> str:
    """
    获取用于日志展示的命令字符串。

    优先尝试解析结构化动作，如果 DSL 校验失败则回退到原始 command 字段。
    这个函数不会抛出异常，确保日志记录不会中断。
    """
    try:
        command, _ = _resolve_step_command(step)
        return command
    except ActionDSLValidationError:
        return step.get("command", "") or ""


def _compiled_action_fields(compiled: Optional[CompiledAction]) -> dict:
    """
    将编译后的动作对象转换为字典字段，用于写入执行轨迹。

    这些字段用于追溯：该命令是从哪个 action_type 和 target 编译出来的。
    """
    if not compiled:
        return {}
    return {
        "compiled_from_action_dsl": True,
        "action_type": compiled.action_type,
        "target": compiled.target,
    }


def _compiled_action_from_trace(trace: list[dict]) -> Optional[CompiledAction]:
    """从执行轨迹中还原编译后的动作对象。

    用途：当步骤执行成功后，需要从轨迹里找回该步骤对应的 action_type 和 target，
    写入 executed_steps 以便后续追溯和评测。

    参数：
        trace: 单步的执行轨迹列表，包含多条 trace 记录

    返回：
        如果轨迹中有 DSL 编译标记，则还原为 CompiledAction；否则返回 None
    """
    # 遍历轨迹中的每一条记录，寻找带有 compiled_from_action_dsl 标记的条目
    for item in trace:
        if item.get("compiled_from_action_dsl"):
            return CompiledAction(
                action_type=item.get("action_type", ""),  # 动作类型，如 RECOVER_FAULT
                target=item.get("target", ""),            # 动作目标，如 APP_PROCESS_DOWN
                command=item.get("command", ""),          # 实际执行的命令字符串
            )
    return None


# ============================================================
# 命令执行器抽象接口
# ============================================================

class CommandRunner(ABC):
    """
    命令执行器抽象接口

    设计意图：将"如何执行命令"与"执行流程编排"解耦。
    MockCommandRunner 用于开发测试，SubprocessCommandRunner 用于生产。
    执行器只关心"执行一条命令，返回结果"，不关心业务逻辑。
    """

    @abstractmethod
    async def run(self, command: str, step_id: int, timeout: int = 30) -> CommandExecutionResult:
        """
        执行单条命令并返回结果

        Args:
            command: 要执行的命令字符串
            step_id: 步骤编号（用于日志标识）
            timeout: 超时时间（秒）

        Returns:
            CommandExecutionResult: 包含 exit_code, stdout, stderr, success 等
        """
        ...


# ============================================================
# Mock 命令执行器
# ============================================================

# 预设的 Mock 响应库：根据命令关键词匹配模拟输出
# 这样 Mock 不是简单的"全部成功"，而是能模拟各种真实场景
MOCK_RESPONSES = {
    # 数据库相关命令
    r"\bpg_dump\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="pg_dump: dumping database 'production'...\npg_dump: done",
        stderr="", success=True, execution_time_ms=2500,
    ),
    r"\bpsql\b.*\bALTER\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="ALTER TABLE\nALTER INDEX",
        stderr="", success=True, execution_time_ms=150,
    ),
    r"\bsystemctl\s+stop\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="Job stopped successfully",
        stderr="", success=True, execution_time_ms=800,
    ),
    r"\bsystemctl\s+start\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="Job started successfully",
        stderr="", success=True, execution_time_ms=1200,
    ),
    r"\bsystemctl\s+restart\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="Job restarted successfully",
        stderr="", success=True, execution_time_ms=1500,
    ),
    r"\bsed\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="Configuration updated",
        stderr="", success=True, execution_time_ms=50,
    ),
    r"\bcp\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="copied successfully",
        stderr="", success=True, execution_time_ms=100,
    ),
    r"\bmv\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="moved successfully",
        stderr="", success=True, execution_time_ms=80,
    ),
    # 验证命令
    r"\bping\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="64 bytes from 10.0.0.1: icmp_seq=1 ttl=64 time=0.123 ms\n--- ping statistics ---\n1 packets transmitted, 1 received, 0% packet loss",
        stderr="", success=True, execution_time_ms=1000,
    ),
    r"\bcurl\b.*health": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout='{"status": "healthy", "uptime": 86400}',
        stderr="", success=True, execution_time_ms=200,
    ),
    # 回滚命令
    r"\brestore\b|\brecover\b|\brollback\b": CommandExecutionResult(
        step_id=0, command="", exit_code=0,
        stdout="Rollback completed successfully",
        stderr="", success=True, execution_time_ms=3000,
    ),
}

# 默认成功响应（命令不匹配任何预设模式时使用）
DEFAULT_SUCCESS_RESPONSE = CommandExecutionResult(
    step_id=0, command="", exit_code=0,
    stdout="Command executed successfully",
    stderr="", success=True, execution_time_ms=200,
)

# 模拟失败场景的响应（随机触发，概率约 15%）
# 用于测试闭环执行器的错误处理能力
MOCK_FAILURE_RESPONSES = [
    CommandExecutionResult(
        step_id=0, command="", exit_code=1,
        stdout="",
        stderr="Error: Connection refused. Service not responding on port 5432.",
        success=False, execution_time_ms=5000,
    ),
    CommandExecutionResult(
        step_id=0, command="", exit_code=2,
        stdout="",
        stderr="Permission denied: insufficient privileges to execute this command.",
        success=False, execution_time_ms=100,
    ),
    CommandExecutionResult(
        step_id=0, command="", exit_code=1,
        stdout="",
        stderr="Timeout: Operation did not complete within 30 seconds.",
        success=False, execution_time_ms=30000,
    ),
    CommandExecutionResult(
        step_id=0, command="", exit_code=1,
        stdout="",
        stderr="Error: Resource temporarily unavailable. Another process holds the lock.",
        success=False, execution_time_ms=2000,
    ),
]


class MockCommandRunner(CommandRunner):
    """
    Mock 命令执行器

    模拟真实命令执行，返回预设的 stdout/stderr/exit_code。
    支持两种模式：
    1. 正常模式：根据命令关键词匹配预设响应
    2. 故障注入模式：随机触发失败场景，测试闭环执行器的错误处理

    为什么需要 Mock：
    - 开发阶段不需要真实环境就能测试执行流程
    - 可以稳定复现各种故障场景（连接拒绝、权限不足、超时等）
    - 单元测试速度快，不依赖外部系统

    替换为生产实现只需实现 SubprocessCommandRunner：
        async def run(self, command, step_id, timeout=30):
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return CommandExecutionResult(
                step_id=step_id, command=command,
                exit_code=proc.returncode,
                stdout=stdout.decode(), stderr=stderr.decode(),
                success=proc.returncode == 0,
            )
    """

    def __init__(self, failure_rate: float = 0.15):
        """
        初始化 Mock 执行器。

        参数：
            failure_rate: 故障注入概率，0.0~1.0，默认 0.15（15%）
                          每次执行命令时，有 15% 概率随机返回一个失败结果
        """
        # failure_rate 控制故障注入的频繁程度
        # 设为 0.0 则所有命令都成功（用于测试正常流程）
        # 设为 1.0 则所有命令都失败（用于测试回滚流程）
        self.failure_rate = failure_rate

    async def run(self, command: str, step_id: int, timeout: int = 30) -> CommandExecutionResult:
        """模拟执行一条命令，返回预设结果。

        执行逻辑：
        1. 先按 failure_rate 概率决定是否故障注入
        2. 如果没有故障注入，按正则匹配查找预设响应
        3. 如果都不匹配，返回默认成功响应

        参数：
            command: 要执行的命令字符串
            step_id: 步骤编号，用于标识和日志
            timeout: 超时时间（Mock 中仅用于保持接口一致，不实际计时）

        返回：
            CommandExecutionResult: 包含模拟的 exit_code、stdout、stderr
        """
        logger.debug(f"[MockRunner] 执行步骤 {step_id}: {command}")

        # 故障注入：随机触发失败
        # random.random() 生成 [0, 1) 之间的浮点数
        # 如果小于 failure_rate，就从 MOCK_FAILURE_RESPONSES 中随机选一个失败结果
        if random.random() < self.failure_rate:
            result = random.choice(MOCK_FAILURE_RESPONSES)
            # model_copy(update=...) 是 Pydantic 的复制方法，创建副本并更新指定字段
            result = result.model_copy(update={
                "step_id": step_id,    # 注入当前步骤编号
                "command": command,    # 注入实际命令
            })
            logger.info(f"[MockRunner] 步骤 {step_id} 故障注入: exit_code={result.exit_code}")
            return result

        # 根据命令关键词匹配预设响应
        # re.IGNORECASE 确保匹配不区分大小写
        import re
        for pattern, response in MOCK_RESPONSES.items():
            if re.search(pattern, command, re.IGNORECASE):
                result = response.model_copy(update={
                    "step_id": step_id,
                    "command": command,
                })
                logger.debug(f"[MockRunner] 步骤 {step_id} 匹配模式: {pattern}")
                return result

        # 默认成功响应：命令不匹配任何预设模式时返回
        result = DEFAULT_SUCCESS_RESPONSE.model_copy(update={
            "step_id": step_id,
            "command": command,
        })
        logger.debug(f"[MockRunner] 步骤 {step_id} 使用默认响应")
        return result


# ============================================================
# 安全靶场命令执行器
# ============================================================

class SafeDockerCommandRunner(CommandRunner):
    """
    安全 Docker 靶场执行器

    这个执行器用于 SREBench Lite，不执行任意 shell。
    它只接受白名单命令，并且始终通过 subprocess 参数数组执行，
    不经过 shell 字符串拼接，从源头避免命令注入。

    安全设计原则：
    1. 白名单机制：只有明确允许的命令才能执行
    2. 参数数组执行：使用 subprocess.run(args=list) 而不是 shell=True
    3. 无字符串拼接：命令不会被拼接到 shell 字符串中，避免注入攻击
    4. 只读命令放行：docker ps、inspect、logs 等只读操作允许，但限制查询范围

    支持的动作：
    1. 恢复固定故障：python lab/chaos.py recover <FAULT>
    2. 启动/重启固定容器：docker start/restart srebench-xxx
    3. 重建固定索引：docker exec srebench-postgres psql ... create index ...
    4. 安全 HTTP 验证：curl http://localhost:18080/health 等固定 URL
    """

    # 允许恢复的故障类型集合，从 action_dsl.py 导入
    # 只有这些故障才能通过 chaos.py recover 恢复
    _ALLOWED_LAB_FAULTS = {
        *LAB_FAULTS,
    }

    # 允许的 Docker 命令集合，通过生成器表达式展开
    # 格式：("docker", "start", "srebench-postgres") 等
    _ALLOWED_DOCKER_COMMANDS = {
        *(("docker", "start", container) for container in STARTABLE_CONTAINERS),
        *(("docker", "restart", container) for container in RESTARTABLE_CONTAINERS),
    }

    # 允许操作的容器名称集合（启动 + 重启的容器并集）
    _ALLOWED_CONTAINERS = {
        *STARTABLE_CONTAINERS,
        *RESTARTABLE_CONTAINERS,
    }

    # 允许的 HTTP 探测 URL 集合，从 action_dsl.py 导入
    _ALLOWED_HTTP_URLS = {
        *HTTP_PROBE_URLS,
    }

    # 允许执行的 SQL 语句（重建索引），去掉末尾分号后做精确匹配
    _ALLOWED_INDEX_SQL = (
        ORDERS_INDEX_SQL.rstrip(";")
    )

    async def run(self, command: str, step_id: int, timeout: int = 30) -> CommandExecutionResult:
        """安全执行命令，只允许白名单内的操作。

        执行流程：
        1. 如果是空命令（NOOP），直接返回成功
        2. 解析命令类型（lab_recover / docker / http）
        3. HTTP 探测走 urllib 线程池
        4. 其他命令走 subprocess.run，参数数组执行
        5. 捕获各种异常，统一封装为 CommandExecutionResult

        参数：
            command: 要执行的命令字符串
            step_id: 步骤编号
            timeout: 超时时间（秒）

        返回：
            CommandExecutionResult: 包含执行结果和耗时
        """
        # 记录开始时间，用于计算执行耗时
        started = time.perf_counter()
        try:
            # 步骤 1：检查是否是空命令（如 NOOP）
            if self._is_noop_command(command):
                return CommandExecutionResult(
                    step_id=step_id,
                    command=command,
                    exit_code=0,
                    stdout="跳过空回滚命令",
                    stderr="",
                    success=True,
                    execution_time_ms=0,
                )

            # 步骤 2：解析命令，验证是否在白名单中
            # _resolve_action 会返回 {"type": "process", "args": [...]} 或 {"type": "http", "url": ...}
            action = self._resolve_action(command)

            # 步骤 3：HTTP 探测走独立逻辑（urllib 是同步的，用 asyncio.to_thread 包装）
            if action["type"] == "http":
                return await asyncio.to_thread(
                    self._run_http_probe,
                    command,
                    step_id,
                    action["url"],
                    started,
                    timeout,
                )

            # 步骤 4：进程命令通过 subprocess.run 执行
            # action["args"] 是参数列表，如 ["docker", "start", "srebench-postgres"]
            # cwd=PROJECT_ROOT 确保在正确的工作目录执行
            # check=False 表示不因为非零退出码抛出异常（我们自己处理）
            completed = await asyncio.to_thread(
                subprocess.run,
                action["args"],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            # 计算耗时（毫秒）
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return CommandExecutionResult(
                step_id=step_id,
                command=command,
                exit_code=completed.returncode,
                stdout=(completed.stdout or "").strip(),
                stderr=(completed.stderr or "").strip(),
                success=completed.returncode == 0,
                execution_time_ms=elapsed_ms,
            )
        except ValueError as exc:
            # 白名单校验失败：命令不在允许列表中
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(f"[SafeDockerRunner] 拒绝执行非白名单命令: {command}")
            return CommandExecutionResult(
                step_id=step_id,
                command=command,
                exit_code=126,  # 126 是 bash 中"命令不可执行"的标准退出码
                stdout="",
                stderr=f"命令未在 SREBench Lite 白名单中，已拒绝执行: {exc}",
                success=False,
                execution_time_ms=elapsed_ms,
            )
        except subprocess.TimeoutExpired as exc:
            # 命令执行超时
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return CommandExecutionResult(
                step_id=step_id,
                command=command,
                exit_code=124,  # 124 是 timeout 命令的标准退出码
                stdout=(exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
                stderr=f"命令超时: {timeout}s",
                success=False,
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:
            # 兜底异常捕获：任何其他异常都不应该让执行器崩溃
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return CommandExecutionResult(
                step_id=step_id,
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"执行异常: {exc}",
                success=False,
                execution_time_ms=elapsed_ms,
            )

    def _resolve_action(self, command: str) -> dict:
        """解析命令字符串，确定其类型和参数。

        这是白名单校验的入口：把命令拆分成参数数组，然后依次尝试匹配
        lab_recover、docker、http_probe 三种类型。
        如果都不匹配，抛出 ValueError，外层会捕获并返回 126 错误。

        参数：
            command: 命令字符串，如 "docker start srebench-postgres"

        返回：
            {"type": "process", "args": [...]} 或 {"type": "http", "url": "..."}

        异常：
            ValueError: 命令不在白名单中
        """
        # shlex.split 按 shell 规则拆分字符串，posix=True 确保跨平台一致性
        # 例如：'docker start "my container"' → ['docker', 'start', 'my container']
        parts = shlex.split(command, posix=True)
        if not parts:
            raise ValueError("空命令")

        # 依次尝试三种解析器，优先级：lab_recover > docker > http
        lab_action = self._resolve_lab_recover(parts)
        if lab_action:
            return lab_action

        docker_action = self._resolve_docker(parts)
        if docker_action:
            return docker_action

        http_action = self._resolve_http_probe(parts)
        if http_action:
            return http_action

        # 都不匹配 → 拒绝执行
        raise ValueError(command)

    def _resolve_lab_recover(self, parts: list[str]) -> Optional[dict]:
        """解析 lab/chaos.py recover 命令。

        允许的格式：python lab/chaos.py recover <FAULT>
        其中 <FAULT> 必须在 _ALLOWED_LAB_FAULTS 中。

        参数：
            parts: shlex 拆分后的参数列表

        返回：
            {"type": "process", "args": [...]} 或 None（不匹配时）
        """
        if len(parts) != 4:
            return None
        executable, script, action, fault = parts
        # 统一路径分隔符：Windows 下可能是 \\，转成 / 便于匹配
        script_name = script.replace("\\", "/")
        # 检查解释器是否在白名单中
        if executable not in {"python", "python3", "py", sys.executable}:
            return None
        # 检查脚本路径是否以 lab/chaos.py 结尾
        if not script_name.endswith("lab/chaos.py"):
            return None
        # 检查动作必须是 recover，故障类型必须在允许列表中
        if action != "recover" or fault not in self._ALLOWED_LAB_FAULTS:
            return None
        # 返回标准化参数：使用当前 Python 解释器的绝对路径
        return {
            "type": "process",
            "args": [sys.executable, str(LAB_CHAOS_SCRIPT), "recover", fault],
        }

    def _resolve_docker(self, parts: list[str]) -> Optional[dict]:
        """解析 Docker 相关命令。

        支持三类：
        1. 启动/重启容器（精确匹配 _ALLOWED_DOCKER_COMMANDS）
        2. 只读查询（ps、inspect、logs、pg_isready）
        3. 重建索引（psql create index）

        参数：
            parts: shlex 拆分后的参数列表

        返回：
            {"type": "process", "args": [...]} 或 None
        """
        # 精确匹配启动/重启命令
        if tuple(parts) in self._ALLOWED_DOCKER_COMMANDS:
            return {"type": "process", "args": parts}

        # 只读查询命令
        if self._is_allowed_docker_readonly(parts):
            return {"type": "process", "args": parts}

        # 重建索引命令
        if self._is_allowed_psql_index_command(parts):
            return {"type": "process", "args": parts}

        return None

    def _is_noop_command(self, command: Optional[str]) -> bool:
        """判断命令是否是无效/空命令（NOOP）。"""
        return not _is_effective_command(command)

    def _is_allowed_docker_readonly(self, parts: list[str]) -> bool:
        """检查 Docker 只读命令是否在白名单中。

        允许的只读命令：
        - docker ps [-a] [--filter name=...] [--format ...]
        - docker inspect <ALLOWED_CONTAINER>
        - docker logs [--tail N] <ALLOWED_CONTAINER>
        - docker exec srebench-postgres pg_isready ...

        参数：
            parts: 命令参数列表

        返回：
            True 表示允许执行
        """
        if not parts or parts[0] != "docker":
            return False

        # docker ps（无参数或有安全参数）
        if len(parts) == 2 and parts[1] == "ps":
            return True

        # docker ps 带参数
        if parts[1] == "ps":
            return self._is_allowed_docker_ps(parts[2:])

        # docker inspect <容器名>
        if len(parts) == 3 and parts[1] == "inspect" and parts[2] in self._ALLOWED_CONTAINERS:
            return True

        # docker logs <容器名>
        if parts[1] == "logs":
            return self._is_allowed_docker_logs(parts[2:])

        # docker exec srebench-postgres pg_isready ...
        if self._is_allowed_pg_ready_command(parts):
            return True

        return False

    def _is_allowed_docker_ps(self, args: list[str]) -> bool:
        """检查 docker ps 的参数是否安全。

        只允许以下参数：
        - -a, --all
        - --filter name=<ALLOWED_CONTAINER>
        - --format <任意值>

        参数：
            args: docker ps 后面的参数列表

        返回：
            True 表示参数安全
        """
        index = 0
        while index < len(args):
            arg = args[index]
            if arg in {"-a", "--all"}:
                index += 1
                continue
            if arg == "--filter":
                if index + 1 >= len(args):
                    return False
                filter_arg = args[index + 1]
                if not self._is_allowed_name_filter(filter_arg):
                    return False
                index += 2
                continue
            if arg.startswith("--filter="):
                if not self._is_allowed_name_filter(arg.split("=", 1)[1]):
                    return False
                index += 1
                continue
            if arg == "--format":
                if index + 1 >= len(args):
                    return False
                index += 2
                continue
            if arg.startswith("--format="):
                index += 1
                continue
            # 遇到不认识的参数，拒绝执行
            return False
        return True

    def _is_allowed_name_filter(self, filter_arg: str) -> bool:
        """检查 --filter name=... 中的容器名是否在白名单中。"""
        if not filter_arg.startswith("name="):
            return False
        name = filter_arg.split("=", 1)[1]
        return name in self._ALLOWED_CONTAINERS

    def _is_allowed_docker_logs(self, args: list[str]) -> bool:
        """检查 docker logs 的参数是否安全。

        只允许：
        - docker logs <ALLOWED_CONTAINER>
        - docker logs --tail N <ALLOWED_CONTAINER>

        参数：
            args: docker logs 后面的参数列表

        返回：
            True 表示参数安全
        """
        if not args:
            return False

        # 最后一个参数必须是容器名
        container = args[-1]
        if container not in self._ALLOWED_CONTAINERS:
            return False

        # 检查前面的参数（除了最后一个）
        index = 0
        while index < len(args) - 1:
            arg = args[index]
            if arg == "--tail":
                if index + 1 >= len(args) - 1:
                    return False
                return args[index + 1].isdigit()
            if arg.startswith("--tail="):
                return arg.split("=", 1)[1].isdigit()
            return False

        return True

    def _is_allowed_pg_ready_command(self, parts: list[str]) -> bool:
        """检查是否是 pg_isready 健康检查命令。

        允许格式：docker exec srebench-postgres pg_isready [-U labuser] [-d labdb]

        参数：
            parts: 命令参数列表

        返回：
            True 表示是允许的健康检查命令
        """
        if len(parts) < 5:
            return False
        if parts[:4] != ["docker", "exec", "srebench-postgres", "pg_isready"]:
            return False
        # 只允许 -U labuser 和 -d labdb 作为附加参数
        allowed_args = {"-U", "labuser", "-d", "labdb"}
        return all(part in allowed_args for part in parts[4:])

    def _is_allowed_psql_index_command(self, parts: list[str]) -> bool:
        """检查是否是重建索引的 psql 命令。

        允许格式：docker exec srebench-postgres psql -U labuser -d labdb -c "CREATE INDEX ..."
        SQL 内容必须与 _ALLOWED_INDEX_SQL 完全一致（忽略大小写和多余空格）。

        参数：
            parts: 命令参数列表

        返回：
            True 表示是允许的索引重建命令
        """
        if len(parts) < 10:
            return False
        prefix = ["docker", "exec", "srebench-postgres", "psql"]
        if parts[:4] != prefix:
            return False
        # 必须包含 -U、-d、-c 三个参数
        if "-U" not in parts or "-d" not in parts or "-c" not in parts:
            return False
        try:
            # 提取用户、数据库、SQL 语句
            user = parts[parts.index("-U") + 1]
            database = parts[parts.index("-d") + 1]
            sql = parts[parts.index("-c") + 1]
        except IndexError:
            return False
        # 规范化 SQL：去空白、转小写、去末尾分号，用于精确匹配
        normalized_sql = " ".join(sql.strip().rstrip(";").lower().split())
        allowed_sql = " ".join(self._ALLOWED_INDEX_SQL.lower().split())
        return user == "labuser" and database == "labdb" and normalized_sql == allowed_sql

    def _resolve_http_probe(self, parts: list[str]) -> Optional[dict]:
        """解析 HTTP 探测命令。

        允许格式：curl <ALLOWED_URL>
        只有 _ALLOWED_HTTP_URLS 中的 URL 才能被探测。

        参数：
            parts: 命令参数列表

        返回：
            {"type": "http", "url": "..."} 或 None
        """
        if len(parts) == 2 and parts[0] == "curl" and parts[1] in self._ALLOWED_HTTP_URLS:
            return {"type": "http", "url": parts[1]}
        return None

    def _run_http_probe(
        self,
        command: str,
        step_id: int,
        url: str,
        started: float,
        timeout: int,
    ) -> CommandExecutionResult:
        """执行 HTTP 探测请求。

        使用 urllib 发送 GET 请求，检查服务健康状态。
        因为 urllib 是同步的，所以这个方法会被 asyncio.to_thread 包装后调用。

        参数：
            command: 原始命令字符串（用于日志和结果记录）
            step_id: 步骤编号
            url: 要探测的 URL
            started: 开始时间戳（time.perf_counter()）
            timeout: 超时时间（秒）

        返回：
            CommandExecutionResult: HTTP 响应结果
        """
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=timeout) as resp:
                # 读取响应体，限制最多 1000 字符，避免大响应拖慢系统
                body = resp.read().decode("utf-8", errors="replace")
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                return CommandExecutionResult(
                    step_id=step_id,
                    command=command,
                    exit_code=0 if 200 <= resp.status < 300 else resp.status,
                    stdout=body[:1000],
                    stderr="",
                    success=200 <= resp.status < 300,
                    execution_time_ms=elapsed_ms,
                )
        except HTTPError as exc:
            # HTTP 错误（4xx、5xx）
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return CommandExecutionResult(
                step_id=step_id,
                command=command,
                exit_code=exc.code,
                stdout="",
                stderr=f"HTTP {exc.code}: {exc.reason}",
                success=False,
                execution_time_ms=elapsed_ms,
            )
        except URLError as exc:
            # 网络错误（连接失败、DNS 解析失败等）
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return CommandExecutionResult(
                step_id=step_id,
                command=command,
                exit_code=1,
                stdout="",
                stderr=str(exc.reason),
                success=False,
                execution_time_ms=elapsed_ms,
            )


# ============================================================
# 闭环执行器核心
# ============================================================

class ClosedLoopExecutor:
    """
    闭环执行器：Observe → Decide → Act

    核心循环：
    1. 执行当前步骤（通过 CommandRunner）
    2. 观察执行结果（exit_code, stdout, stderr）
    3. 如果成功 → 继续下一步
    4. 如果失败 → LLM 分析错误 → 决定重试/调整/回滚
    5. 重复直到所有步骤完成或触发回滚

    这才是真正的 Agent 架构——在真实环境中行动、观察、适应。
    """

    def __init__(
        self,
        command_runner: CommandRunner,
        llm=None,
        max_retries_per_step: int = 2,
    ):
        """
        Args:
            command_runner: 命令执行器（Mock 或真实）
            llm: LLM 实例，用于失败时的错误分析（可选，无 LLM 则直接回滚）
            max_retries_per_step: 每步最大重试次数
        """
        self.runner = command_runner
        self.llm = llm
        self.max_retries = max_retries_per_step

    async def execute_plan(
        self,
        fix_plan: dict,
        error_analyzer=None,
    ) -> dict:
        """
        执行完整修复方案（闭环）

        流程：
        1. 遍历 fix_plan.steps
        2. 对每个步骤调用 _execute_step_with_retry
        3. 收集执行轨迹（execution_trace）
        4. 如果某步彻底失败且无法回滚 → 整体失败
        5. 汇总结果返回

        Args:
            fix_plan: 修复方案字典，含 steps 列表
            error_analyzer: 错误分析链（Prompt | LLM），可选

        Returns:
            {
                "execution_result": {...},
                "execution_trace": [...],
            }
        """
        steps = fix_plan.get("steps", [])
        plan_id = fix_plan.get("plan_id", "UNKNOWN")
        executed_steps = []
        execution_trace = []
        overall_status = "success"

        logger.info(
            f"[ClosedLoopExecutor] 开始执行方案 {plan_id}, "
            f"共 {len(steps)} 个步骤"
        )

        for step in steps:
            step_id = step.get("step_id", 0)
            command = _safe_command_for_log(step)

            # 闭环执行单步（含重试和 LLM 决策）
            step_result, step_trace = await self._execute_step_with_retry(
                step=step,
                error_analyzer=error_analyzer,
            )

            execution_trace.extend(step_trace)

            if step_result.success:
                executed_step = {
                    "step_id": step_id,
                    "action": step.get("action", ""),
                    "command": step_result.command or command,
                    "status": "success",
                    "exit_code": step_result.exit_code,
                    "stdout": step_result.stdout,
                    "stderr": step_result.stderr,
                    "attempts": len([t for t in step_trace if t.get("trace_type") == "execute"]),
                }
                executed_step.update(_compiled_action_fields(_compiled_action_from_trace(step_trace)))
                executed_steps.append({
                    **executed_step,
                })
            else:
                # 步骤彻底失败，尝试回滚
                overall_status = "failed"
                failed_step = {
                    "step_id": step_id,
                    "action": step.get("action", ""),
                    "command": step_result.command or command,
                    "status": "failed",
                    "exit_code": step_result.exit_code,
                    "stdout": step_result.stdout,
                    "stderr": step_result.stderr,
                }
                failed_step.update(_compiled_action_fields(_compiled_action_from_trace(step_trace)))
                executed_steps.append(failed_step)

                # 执行回滚
                rollback_invalid = False
                try:
                    rollback_command, rollback_action = _resolve_rollback_command(step)
                except ActionDSLValidationError as exc:
                    rollback_command = ""
                    rollback_action = None
                    rollback_invalid = True
                    execution_trace.append({
                        "trace_type": "rollback",
                        "step_id": step_id,
                        "exit_code": 126,
                        "success": False,
                        "stdout": "",
                        "stderr": f"Invalid rollback action DSL: {exc}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                if _is_effective_command(rollback_command):
                    logger.info(f"[ClosedLoopExecutor] 步骤 {step_id} 失败，执行回滚: {rollback_command}")
                    rollback_result = await self.runner.run(rollback_command, step_id=-step_id)
                    execution_trace.append({
                        "trace_type": "rollback",
                        "step_id": step_id,
                        "command": rollback_command,
                        "exit_code": rollback_result.exit_code,
                        "success": rollback_result.success,
                        "stdout": rollback_result.stdout,
                        "stderr": rollback_result.stderr,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **_compiled_action_fields(rollback_action),
                    })
                    rollback_step = {
                        "step_id": step_id,
                        "action": f"回滚步骤 {step_id}",
                        "command": rollback_command,
                        "status": "rollback_success" if rollback_result.success else "rollback_failed",
                        "exit_code": rollback_result.exit_code,
                        "stdout": rollback_result.stdout,
                        "stderr": rollback_result.stderr,
                    }
                    rollback_step.update(_compiled_action_fields(rollback_action))
                    executed_steps.append(rollback_step)
                elif not rollback_invalid:
                    logger.warning(f"[ClosedLoopExecutor] 步骤 {step_id} 失败且无回滚命令")
                    execution_trace.append({
                        "trace_type": "rollback_skipped",
                        "step_id": step_id,
                        "reason": "无回滚命令",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                # 失败后不再继续执行后续步骤
                break

        result = {
            "plan_id": plan_id,
            "executed_steps": executed_steps,
            "overall_status": overall_status,
            "summary": f"执行{'完成' if overall_status == 'success' else '失败'}，"
                       f"共执行 {len(executed_steps)} 个步骤",
            "total_steps": len(steps),
            "completed_steps": len([s for s in executed_steps if s.get("status") == "success"]),
        }

        logger.info(
            f"[ClosedLoopExecutor] 方案 {plan_id} 执行结束: "
            f"status={overall_status}, "
            f"completed={result['completed_steps']}/{result['total_steps']}"
        )

        return {
            "execution_result": result,
            "execution_trace": execution_trace,
        }

    async def _execute_step_with_retry(
        self,
        step: dict,
        error_analyzer=None,
    ) -> tuple[CommandExecutionResult, list[dict]]:
        """
        执行单步骤（含重试和 LLM 错误分析）

        流程：
        1. 执行命令 → 观察结果
        2. 成功 → 返回
        3. 失败 → LLM 分析错误 → 决定 retry/adjust/rollback
        4. retry → 重新执行（最多 max_retries 次）
        5. adjust → 用 LLM 调整后的命令重新执行
        6. rollback → 返回失败，由上层处理回滚

        Args:
            step: 步骤字典，含 step_id, command, action 等
            error_analyzer: 错误分析链（Prompt | LLM）

        Returns:
            (最终执行结果, 执行轨迹列表)
        """
        step_id = step.get("step_id", 0)
        trace = []
        try:
            command, compiled_action = _resolve_step_command(step)
        except ActionDSLValidationError as exc:
            result = CommandExecutionResult(
                step_id=step_id,
                command=step.get("command", "") or "",
                exit_code=126,
                stdout="",
                stderr=f"Invalid action DSL: {exc}",
                success=False,
                execution_time_ms=0,
            )
            trace.append({
                "trace_type": "execute",
                "step_id": step_id,
                "attempt": 1,
                "command": result.command,
                "exit_code": result.exit_code,
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "invalid_action_dsl": True,
            })
            return result, trace

        current_command = command

        for attempt in range(1, self.max_retries + 1):
            # 执行命令
            logger.info(
                f"[ClosedLoopExecutor] 步骤 {step_id} 第 {attempt} 次执行: "
                f"{current_command[:80]}"
            )

            result = await self.runner.run(current_command, step_id)

            trace.append({
                "trace_type": "execute",
                "step_id": step_id,
                "attempt": attempt,
                "command": current_command,
                "exit_code": result.exit_code,
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **_compiled_action_fields(compiled_action),
            })

            if result.success:
                logger.info(f"[ClosedLoopExecutor] 步骤 {step_id} 第 {attempt} 次执行成功")
                return result, trace

            # 执行失败，尝试 LLM 错误分析
            logger.warning(
                f"[ClosedLoopExecutor] 步骤 {step_id} 第 {attempt} 次执行失败: "
                f"exit_code={result.exit_code}, stderr={result.stderr[:100]}"
            )

            if error_analyzer and attempt < self.max_retries:
                # LLM 分析错误，决定下一步动作
                decision = await self._analyze_error(
                    step=step,
                    command=current_command,
                    result=result,
                    attempt=attempt,
                    error_analyzer=error_analyzer,
                )

                trace.append({
                    "trace_type": "llm_decision",
                    "step_id": step_id,
                    "attempt": attempt,
                    "action": decision.action,
                    "reasoning": decision.reasoning,
                    "adjusted_command": decision.adjusted_command,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                if decision.action == "retry":
                    logger.info(f"[ClosedLoopExecutor] LLM 决策: 重试步骤 {step_id}")
                    continue
                elif decision.action == "adjust" and decision.adjusted_command:
                    if compiled_action:
                        # 关键安全设计：如果当前步骤使用了结构化动作 DSL（action_type + target），
                        # 则忽略 LLM 调整后的自由文本命令，继续重试本地编译的白名单命令。
                        # 这防止了 LLM 通过 "adjust" 决策绕过 DSL 安全限制。
                        logger.info(
                            f"[ClosedLoopExecutor] 结构化动作步骤 {step_id} 忽略 LLM 调整命令，"
                            "继续重试已编译的白名单命令"
                        )
                        continue
                    # 兼容旧模式：没有结构化动作时，允许 LLM 调整自由文本命令
                    logger.info(
                        f"[ClosedLoopExecutor] LLM 决策: 调整命令 "
                        f"{current_command[:50]} → {decision.adjusted_command[:50]}"
                    )
                    current_command = decision.adjusted_command
                    continue
                elif decision.action == "rollback":
                    logger.info(f"[ClosedLoopExecutor] LLM 决策: 回滚步骤 {step_id}")
                    return result, trace
                elif decision.action == "skip":
                    logger.info(f"[ClosedLoopExecutor] LLM 决策: 跳过步骤 {step_id}")
                    # 跳过视为成功（LLM 判断该步骤非必要）
                    return CommandExecutionResult(
                        step_id=step_id,
                        command=current_command,
                        exit_code=0,
                        stdout="[SKIPPED] LLM 判断该步骤可跳过",
                        stderr="",
                        success=True,
                        execution_time_ms=0,
                    ), trace
            else:
                # 无 LLM 或已达最大重试次数
                if attempt < self.max_retries:
                    logger.info(f"[ClosedLoopExecutor] 无 LLM 分析器，简单重试步骤 {step_id}")
                    continue
                else:
                    logger.warning(
                        f"[ClosedLoopExecutor] 步骤 {step_id} 达到最大重试次数 "
                        f"{self.max_retries}，放弃"
                    )

        # 所有重试都失败
        return result, trace

    async def _analyze_error(
        self,
        step: dict,
        command: str,
        result: CommandExecutionResult,
        attempt: int,
        error_analyzer,
    ) -> ErrorAnalysisOutput:
        """
        LLM 分析执行错误，决定下一步动作

        这是闭环执行器的核心决策点：
        - 输入：步骤信息 + 执行结果（真实的 stdout/stderr/exit_code）
        - 输出：retry / adjust / rollback / skip
        - 由真实错误驱动决策，不是 LLM 凭空决定

        Args:
            step: 步骤信息
            command: 执行的命令
            result: 执行结果
            attempt: 当前重试次数
            error_analyzer: 错误分析链（Prompt | LLM）

        Returns:
            ErrorAnalysisOutput: LLM 的决策
        """
        try:
            decision = await error_analyzer.ainvoke({
                "step_id": step.get("step_id"),
                "action": step.get("action", ""),
                "command": command,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "attempt": attempt,
                "max_retries": self.max_retries,
                "risk_level": step.get("risk_level", "low"),
            })

            if decision is None:
                return ErrorAnalysisOutput(
                    action="rollback",
                    reasoning="LLM 分析返回空结果，默认回滚",
                    estimated_fix_probability=0.0,
                )

            logger.info(
                f"[ClosedLoopExecutor] LLM 错误分析: action={decision.action}, "
                f"reasoning={decision.reasoning[:80]}"
            )
            return decision

        except Exception as e:
            logger.error(f"[ClosedLoopExecutor] LLM 错误分析失败: {e}")
            return ErrorAnalysisOutput(
                action="rollback",
                reasoning=f"LLM 分析异常: {str(e)}，默认回滚",
                estimated_fix_probability=0.0,
            )
