# ============================================================
# case_library.py：案例库模块（Agent 记忆系统）
#
# 作用：
#   存储已解决的故障案例，在新工单到达时检索相似案例，
#   为 SupervisorAgent 和 FixAgent 提供历史经验参考。
#
# 核心设计：
#   - 使用简单的确定性评分器（非神经网络），行为透明、可审计
#   - 案例以 JSON 文件形式存储，无需数据库，部署简单
#   - 自动从工作流状态中提取成功案例并入库
#
# 为什么不用向量数据库？
#   1. 项目规模小，JSON 文件足够
#   2. 确定性评分可解释（能告诉用户为什么匹配这个案例）
#   3. 零外部依赖，降低部署复杂度
# ============================================================

# 模块文档字符串：给外部工具（如 Sphinx、IDE）看的模块说明
"""Agent memory / case library.

The library stores resolved incidents and retrieves similar cases for new
tickets before Supervisor/FixAgent make decisions. It intentionally uses a
small deterministic scorer so the behavior is transparent and easy to audit.
"""

# from __future__ import annotations：启用 PEP 563，支持 dict[str, Any] 等类型注解
from __future__ import annotations

# json：读写 JSON 格式的案例库文件
import json
# re：正则表达式，用于分词提取 token
import re
# datetime/timezone：生成案例的创建/更新时间戳
from datetime import datetime, timezone
# Path：面向对象的路径操作，比字符串拼接更安全
from pathlib import Path
# typing.Any：任意类型，用于兼容各种输入
from typing import Any


# ═══════════════════════════════════════════════════════════
# 一、全局常量定义
# ═══════════════════════════════════════════════════════════

# PROJECT_ROOT：项目根目录路径
# __file__ 是当前文件路径，resolve() 取绝对路径，parents[1] 往上退两级
# 如 .../src/case_library.py → parents[0]=src → parents[1]=项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DEFAULT_CASE_LIBRARY_PATH：默认案例库文件路径
# 存放在项目根目录的 eval/case_library.json，方便评测时读取
DEFAULT_CASE_LIBRARY_PATH = PROJECT_ROOT / "eval" / "case_library.json"

# CASE_KEYWORDS：关键词映射表，用于增强分词效果
# 键是"标准化词"，值是"同义词列表"
# 作用：当症状中出现"数据库"时，自动关联 "db"、"postgres" 等词，提高匹配率
CASE_KEYWORDS = {
    "db": ["db", "database", "postgres", "postgresql", "psql", "数据库", "连接", "查询", "慢查询"],
    "app": ["app", "application", "process", "fastapi", "应用", "进程", "健康", "不可用"],
    "redis": ["redis", "cache", "缓存"],
    "nginx": ["nginx", "route", "路由", "网关", "反向代理"],
    "slow": ["slow", "latency", "timeout", "超时", "慢", "延迟"],
    "health": ["health", "健康", "探活"],
    "orders": ["orders", "pending", "订单"],
}


# ═══════════════════════════════════════════════════════════
# 二、案例库 CRUD 操作
# ═══════════════════════════════════════════════════════════

def load_cases(path: Path | str = DEFAULT_CASE_LIBRARY_PATH) -> list[dict[str, Any]]:
    """加载案例库文件。

    参数：
        path：案例库文件路径，默认 DEFAULT_CASE_LIBRARY_PATH

    返回：
        案例字典列表。如果文件不存在或解析失败，返回空列表。

    容错设计：
    - 文件不存在 → 返回 []（首次运行时自动创建）
    - JSON 解析失败 → 返回 []（防止损坏文件导致系统崩溃）
    - 根元素不是列表 → 返回 []
    - 过滤掉非字典元素（防止格式异常的数据）
    """
    # case_path：把字符串路径转成 Path 对象，统一操作接口
    case_path = Path(path)
    # 文件不存在时直接返回空列表，不报错
    if not case_path.exists():
        return []
    try:
        # read_text(encoding="utf-8")：以 UTF-8 编码读取文件内容
        # json.loads：把 JSON 字符串解析成 Python 对象
        data = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # JSON 格式损坏（如手动编辑时漏了逗号），返回空列表
        return []
    # 根元素必须是列表，否则返回空列表
    if not isinstance(data, list):
        return []
    # 过滤：只保留字典类型的元素，忽略可能的异常数据
    return [case for case in data if isinstance(case, dict)]


def save_cases(cases: list[dict[str, Any]], path: Path | str = DEFAULT_CASE_LIBRARY_PATH) -> None:
    """保存案例列表到文件。

    参数：
        cases：案例字典列表
        path：保存路径，默认 DEFAULT_CASE_LIBRARY_PATH

    副作用：
        会覆盖原文件内容。如果父目录不存在，自动创建。
    """
    # case_path：统一转成 Path 对象
    case_path = Path(path)
    # mkdir(parents=True, exist_ok=True)：递归创建父目录，已存在时不报错
    case_path.parent.mkdir(parents=True, exist_ok=True)
    # json.dumps：把 Python 对象序列化为 JSON 字符串
    # ensure_ascii=False：允许中文直接输出，不转义成 \uXXXX
    # indent=2：格式化缩进，方便人工查看和 diff
    # + "\n"：文件末尾加换行符，符合 Unix 惯例
    case_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════
# 三、相似案例检索
# ═══════════════════════════════════════════════════════════

def retrieve_similar_cases(
    symptom: str,
    *,
    limit: int = 3,
    path: Path | str = DEFAULT_CASE_LIBRARY_PATH,
) -> list[dict[str, Any]]:
    """根据故障症状检索相似历史案例。

    检索流程：
    1. 对 symptom 分词得到 query_tokens
    2. 遍历案例库中的每个案例，计算相似度得分
    3. 按得分降序排序，返回前 limit 个

    评分逻辑（_score_case）：
    - 基础分：query 和 case 的 token 交集数量
    - 症状完全包含 bonus：+3.0（如果 case 的某个症状是 query 的子串或超集）
    - 修复目标匹配 bonus：+2.0（如果 case 的修复目标出现在 query 中）
    - 额外加权：交集数量 × 0.25，封顶 1.5

    参数：
        symptom：故障症状描述字符串
        limit：最多返回几条案例，默认 3
        path：案例库文件路径

    返回：
        相似案例列表，每个案例多了 score（得分）和 matched_terms（匹配到的词）字段
    """
    # query_tokens：对症状描述分词，得到一组标准化 token
    query_tokens = _tokenize(symptom)
    # 如果分词结果为空（如 symptom 是空字符串或纯标点），直接返回空
    if not query_tokens:
        return []

    # ranked：存储所有得分大于 0 的案例
    ranked = []
    for case in load_cases(path):
        # score：相似度得分；matched_terms：具体匹配到哪些 token
        score, matched_terms = _score_case(case, query_tokens, symptom)
        # 得分小于等于 0 表示完全不相关，跳过
        if score <= 0:
            continue
        # 把得分和匹配词追加到案例字典中，方便下游使用
        ranked.append({
            **case,
            "score": round(score, 3),  # 保留 3 位小数
            "matched_terms": sorted(matched_terms),  # 排序后输出，保证确定性
        })

    # 排序：先按 score 降序，score 相同按 updated_at 降序（最新的优先）
    ranked.sort(key=lambda item: (item["score"], item.get("updated_at", "")), reverse=True)
    # 只返回前 limit 个
    return ranked[:limit]


def format_case_context(cases: list[dict[str, Any]]) -> str:
    """把相似案例列表格式化成字符串，用于插入 LLM prompt。

    参数：
        cases：retrieve_similar_cases 返回的案例列表

    返回：
        人可读的案例描述字符串，如果为空返回 "无相似历史案例。"

    输出格式示例：
        案例 1: Postgres 连接超时
        - 症状: 应用报错 database connection failed
        - 工具证据: check_db_connection: 连接被拒绝
        - 根因: PostgreSQL 容器停止
        - 成功修复动作: RECOVER_FAULT / DB_CONN_FAIL
        - 验证方式: curl http://localhost:18080/health
        - 匹配: score=5.5, terms=['db', 'postgres', '连接']
    """
    # 空列表时返回固定提示，避免 LLM 困惑
    if not cases:
        return "无相似历史案例。"

    # blocks：存储每个案例的格式化文本块
    blocks = []
    for index, case in enumerate(cases, start=1):
        # repair：成功案例的修复动作信息
        repair = case.get("successful_repair_action", {})
        # verification：验证方式信息
        verification = case.get("verification", {})
        # action_text：把修复动作格式化成字符串
        action_text = _format_repair_action(repair)
        # commands：验证命令列表，最多取前 3 条
        commands = verification.get("commands", [])
        if isinstance(commands, list):
            verification_text = ", ".join(str(command) for command in commands[:3])
        else:
            verification_text = str(commands)

        # 拼接成多行文本块
        blocks.append(
            "\n".join([
                f"案例 {index}: {case.get('title') or case.get('case_id', 'unknown')}",
                f"- 症状: {_join_list(case.get('symptoms'))}",
                f"- 工具证据: {_join_list(case.get('tool_evidence'))}",
                f"- 根因: {case.get('root_cause', '未知')}",
                f"- 成功修复动作: {action_text}",
                f"- 验证方式: {verification_text or verification.get('expected_result', '未记录')}",
                f"- 匹配: score={case.get('score')}, terms={case.get('matched_terms', [])}",
            ])
        )
    # 用两个换行符分隔不同案例
    return "\n\n".join(blocks)


# ═══════════════════════════════════════════════════════════
# 四、从工作流状态自动构建案例
# ═══════════════════════════════════════════════════════════

def build_case_from_state(state: Any) -> dict[str, Any] | None:
    """从工作流状态中提取信息，构建一个新的案例字典。

    使用场景：工单处理完成后，如果验证通过，自动把本次经验入库。

    参数：
        state：工作流状态对象（SystemState 或字典）

    返回：
        案例字典，如果验证未通过返回 None（不保存失败案例）
    """
    # verification_result：验证结果，必须 verified=True 才入库
    verification_result = _getattr_or_key(state, "verification_result") or {}
    if not verification_result.get("verified"):
        return None

    # 从状态中提取基本信息
    ticket_id = _getattr_or_key(state, "ticket_id") or "unknown"
    symptom = _getattr_or_key(state, "symptom") or ""
    fix_plan = _to_dict(_getattr_or_key(state, "fix_plan") or {})
    # diagnosis：优先取聚合诊断，没有则取各 Agent 的诊断结果
    diagnosis = _best_diagnosis(state)
    # tool_evidence：收集所有 Agent 的工具执行结果作为证据
    tool_evidence = _collect_tool_evidence(state)
    # successful_actions：从修复计划中提取所有步骤作为成功动作
    successful_actions = _collect_successful_actions(fix_plan)

    # 构造案例字典，字段设计兼容 retrieve_similar_cases 的评分逻辑
    return {
        "case_id": f"ticket:{ticket_id}",  # 案例 ID，格式 ticket:工单号
        "title": f"Resolved ticket {ticket_id}",  # 标题
        "symptoms": [symptom],  # 症状列表（当前只有一个）
        "tool_evidence": tool_evidence,  # 工具证据列表
        "root_cause": diagnosis.get("diagnosis", "未记录"),  # 根因诊断
        "possible_causes": diagnosis.get("possible_causes", []),  # 可能原因
        "successful_repair_action": successful_actions[0] if successful_actions else {},  # 首个成功动作
        "successful_repair_actions": successful_actions,  # 所有成功动作
        "verification": {  # 验证信息
            "commands": [
                probe.get("url")
                for probe in verification_result.get("verification_probe", [])
                if probe.get("url")
            ],
            "expected_result": "所有恢复探针 HTTP 2xx",
            "probe_result": verification_result,
        },
        "source_ticket_id": ticket_id,  # 来源工单号
        "created_at": datetime.now(timezone.utc).isoformat(),  # 创建时间
        "updated_at": datetime.now(timezone.utc).isoformat(),  # 更新时间
    }


def upsert_case(case: dict[str, Any], path: Path | str = DEFAULT_CASE_LIBRARY_PATH) -> dict[str, Any]:
    """插入或更新单个案例。

    如果 case_id 已存在，合并新数据并更新 updated_at；
    如果不存在，追加到列表末尾。

    参数：
        case：案例字典，必须包含 case_id（没有会自动生成）
        path：案例库文件路径

    返回：
        最终保存的案例字典（可能包含自动生成的 case_id）
    """
    # 加载现有案例列表
    cases = load_cases(path)
    # case_id：案例唯一标识
    case_id = case.get("case_id")
    if not case_id:
        # 如果没有 case_id，自动生成一个（格式 case:序号）
        case_id = f"case:{len(cases) + 1}"
        case["case_id"] = case_id

    # replaced：标记是否找到并替换了已有案例
    replaced = False
    for index, existing in enumerate(cases):
        if existing.get("case_id") == case_id:
            # 合并：保留旧字段，用新字段覆盖，更新 updated_at
            merged = {**existing, **case, "updated_at": datetime.now(timezone.utc).isoformat()}
            cases[index] = merged
            case = merged
            replaced = True
            break

    # 如果没找到相同 case_id，追加为新案例
    if not replaced:
        cases.append(case)

    # 保存回文件
    save_cases(cases, path)
    return case


def upsert_case_from_state(
    state: Any,
    path: Path | str = DEFAULT_CASE_LIBRARY_PATH,
) -> dict[str, Any] | None:
    """从工作流状态自动构建案例并入库。

    这是 build_case_from_state + upsert_case 的组合便捷函数。

    参数：
        state：工作流状态对象
        path：案例库文件路径

    返回：
        保存后的案例字典，如果验证未通过返回 None
    """
    case = build_case_from_state(state)
    if not case:
        return None
    return upsert_case(case, path)


# ═══════════════════════════════════════════════════════════
# 五、内部辅助函数（以下划线开头，模块私有）
# ═══════════════════════════════════════════════════════════

def _score_case(case: dict[str, Any], query_tokens: set[str], raw_query: str) -> tuple[float, set[str]]:
    """计算单个案例与查询的相似度得分。

    评分规则（可解释、确定性）：
    1. 基础分 = query_tokens 与 case_tokens 的交集数量
    2. 症状包含 bonus：如果 case 的某个症状是 raw_query 的子串或超集，+3.0
    3. 修复目标匹配 bonus：如果 case 的修复目标出现在 raw_query 中，+2.0
    4. 额外加权：min(交集数量, 6) × 0.25（封顶 1.5）

    参数：
        case：单个案例字典
        query_tokens：查询分词后的 token 集合
        raw_query：原始查询字符串（小写）

    返回：
        (score, matched_terms) 元组
    """
    # case_text：把案例的所有文本字段拼接成一个大字符串
    case_text = " ".join([
        str(case.get("title", "")),
        _join_list(case.get("symptoms")),
        _join_list(case.get("tool_evidence")),
        str(case.get("root_cause", "")),
        _join_list(case.get("possible_causes")),
        # 把修复动作字典转成 JSON 字符串参与匹配
        json.dumps(case.get("successful_repair_action", {}), ensure_ascii=False),
    ])
    # case_tokens：对案例文本分词
    case_tokens = _tokenize(case_text)
    # matched_terms：交集，即同时出现在 query 和 case 中的 token
    matched_terms = query_tokens & case_tokens
    # 基础分：交集数量
    score = float(len(matched_terms))

    # bonus 1：症状完全包含匹配
    raw_query_lower = raw_query.lower()
    for symptom in case.get("symptoms", []) or []:
        symptom_text = str(symptom).lower()
        # 双向包含：case 症状是 query 的子串，或 query 是 case 症状的子串
        if symptom_text and (symptom_text in raw_query_lower or raw_query_lower in symptom_text):
            score += 3.0

    # bonus 2：修复目标匹配
    target = str((case.get("successful_repair_action") or {}).get("target", "")).lower()
    if target and target in raw_query_lower:
        score += 2.0

    # bonus 3：额外加权，封顶 1.5（避免 token 过多时分数爆炸）
    score += min(len(matched_terms), 6) * 0.25
    return score, matched_terms


def _tokenize(text: str) -> set[str]:
    """对文本进行分词，提取有意义的 token。

    分词流程：
    1. 转小写
    2. 用正则提取字母/数字/下划线/冒号/斜杠/点号组成的词
    3. 根据 CASE_KEYWORDS 映射表，把同义词加入 token 集合
    4. 过滤掉长度小于等于 1 的 token

    参数：
        text：任意文本字符串

    返回：
        token 集合（去重）
    """
    # normalized：统一转小写，避免大小写影响匹配
    normalized = str(text).lower()
    # re.findall(r"[a-z0-9_:/.-]+")：提取技术词汇（如 localhost:15432、DB_CONN_FAIL）
    tokens = set(re.findall(r"[a-z0-9_:/.-]+", normalized))
    # 根据关键词映射表扩展 token
    for canonical, keywords in CASE_KEYWORDS.items():
        # 如果文本中包含某个关键词类别的任意词，加入该类别的主词
        if any(keyword.lower() in normalized for keyword in keywords):
            tokens.add(canonical)
            # 同时加入所有匹配到的具体关键词
            tokens.update(keyword.lower() for keyword in keywords if keyword.lower() in normalized)
    # 过滤：长度大于 1 的 token 才保留（去掉 "a"、"1" 等无意义单字符）
    return {token for token in tokens if len(token) > 1}


def _format_repair_action(repair: Any) -> str:
    """把修复动作格式化成人可读的字符串。

    参数：
        repair：修复动作字典或任意类型

    返回：
        如 "RECOVER_FAULT / DB_CONN_FAIL" 或 "未记录"
    """
    if isinstance(repair, dict):
        action_type = repair.get("action_type", "")
        target = repair.get("target", "")
        command = repair.get("command", "")
        # 用 " / " 连接非空字段
        return " / ".join(part for part in [action_type, target, command] if part) or "未记录"
    return str(repair or "未记录")


def _join_list(value: Any) -> str:
    """把列表值用 "；" 连接成字符串，非列表直接转字符串。

    参数：
        value：任意值

    返回：
        字符串，None 返回空字符串
    """
    if isinstance(value, list):
        return "；".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _getattr_or_key(obj: Any, key: str) -> Any:
    """兼容字典和对象的属性读取。

    参数：
        obj：字典或任意对象
        key：属性名或字典键

    返回：
        属性值或 None

    为什么需要这个函数？
    state 可能是 SystemState 对象（用 getattr）或字典（用 .get()），
    这个函数统一两种访问方式。
    """
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _to_dict(value: Any) -> dict[str, Any]:
    """把任意值转成字典。

    支持：
    - Pydantic 模型（有 model_dump 方法）
    - 字典（直接返回）
    - 其他类型（返回空字典）
    """
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _best_diagnosis(state: Any) -> dict[str, Any]:
    """从状态中提取最佳诊断结果。

    优先级：aggregated_diagnosis > db_agent_result > net_agent_result > app_agent_result
    因为聚合诊断综合了多个 Agent 的意见，最可靠。
    """
    for key in ["aggregated_diagnosis", "db_agent_result", "net_agent_result", "app_agent_result"]:
        value = _getattr_or_key(state, key)
        if value:
            return _to_dict(value)
    return {}


def _collect_tool_evidence(state: Any) -> list[str]:
    """从状态中提取所有 Agent 的工具执行结果作为证据。

    返回：
        证据字符串列表，最多 10 条（防止过长）
    """
    evidence = []
    for key in ["db_agent_result", "net_agent_result", "app_agent_result"]:
        result = _to_dict(_getattr_or_key(state, key) or {})
        for tool_result in result.get("tool_results", []) or []:
            if isinstance(tool_result, dict):
                tool = tool_result.get("tool", "unknown_tool")
                # 截取前 500 字符，防止单条证据过长
                evidence.append(f"{tool}: {str(tool_result.get('result'))[:500]}")
    return evidence[:10]


def _collect_successful_actions(fix_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """从修复计划中提取所有步骤作为成功动作记录。

    参数：
        fix_plan：修复计划字典

    返回：
        动作字典列表，过滤掉值为 None 的字段
    """
    actions = []
    for step in fix_plan.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        action = {
            "step_id": step.get("step_id"),
            "action": step.get("action"),
            "action_type": step.get("action_type"),
            "target": step.get("target"),
            "command": step.get("command"),
        }
        # 过滤 None 值，减少存储体积
        actions.append({k: v for k, v in action.items() if v is not None})
    return actions
