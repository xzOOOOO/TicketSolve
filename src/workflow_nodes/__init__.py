"""
工作流节点包。

本包按职责拆分原 nodes.py 中的节点工厂函数，并通过懒加载方式导出。
懒加载可以避免只测试某个轻量节点时提前导入 LangChain Prompt 等重依赖。
"""

# import_module：按需导入具体节点模块，避免包初始化时加载全部依赖
from importlib import import_module
# Any：用于标注 __getattr__ 的返回值类型
from typing import Any


# NODE_EXPORTS：节点工厂函数名称到模块路径的映射
NODE_EXPORTS = {
    "create_case_memory_node": "workflow_nodes.case_memory",
    "create_dispatch_node": "workflow_nodes.dispatch",
    "create_dynamic_check_node": "workflow_nodes.dynamic_check",
    "create_aggregate_node": "workflow_nodes.aggregate",
    "create_repair_planner_node": "workflow_nodes.repair_planner",
    "create_guardrail_node": "workflow_nodes.guardrail_node",
    "create_human_approval_node": "workflow_nodes.approval",
    "create_executor_node": "workflow_nodes.executor_node",
    "create_replanner_node": "workflow_nodes.replanner_node",
    "create_verify_node": "workflow_nodes.verify",
    "create_save_node": "workflow_nodes.save",
    "create_other_handler_node": "workflow_nodes.approval",
}


def __getattr__(name: str) -> Any:
    """
    按需加载节点工厂函数。

    参数：
        name: 需要读取的导出名称

    返回：
        对应节点模块中的工厂函数

    异常说明：
        当 name 不在 NODE_EXPORTS 中时抛出 AttributeError。
    """
    # 如果请求的名称不是节点导出，则交给 Python 标准属性错误处理
    if name not in NODE_EXPORTS:
        raise AttributeError(f"module 'workflow_nodes' has no attribute {name!r}")

    # module_path：节点工厂函数所在模块路径
    module_path = NODE_EXPORTS[name]
    # module：按需导入的节点模块
    module = import_module(module_path)
    # exported：具体节点工厂函数对象
    exported = getattr(module, name)
    # globals 缓存：后续访问同名节点时不再重复导入
    globals()[name] = exported
    return exported


# __all__：控制 from workflow_nodes import * 时导出的节点工厂函数
__all__ = list(NODE_EXPORTS)
