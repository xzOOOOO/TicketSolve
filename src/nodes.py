"""
所有节点定义 - 整合 router, agents, fix, approval, executor
"""
import json
from state import SystemState, DiagnosisType, ApprovalStatus
from langchain_core.tools import tool
from prompts import (
    ROUTER_PROMPT, DB_PROMPT, DB_DIAGNOSIS_PROMPT,
    NET_PROMPT, NET_DIAGNOSIS_PROMPT,
    APP_PROMPT, APP_DIAGNOSIS_PROMPT,
    FIX_PROMPT
)
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from database import AsyncSessionLocal, save_ticket
from logger import logger

def parse_json_content(content: str) -> dict:
    """解析JSON内容，支持多种格式"""
    result = None
    try:
        result = json.loads(content)
        logger.debug("JSON解析成功")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON解析失败,尝试提取：{e}")
        try:
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                result = json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON解析失败,提取失败：{e}")
    return result

@tool
def check_db_connection() -> str:
    """检查数据库连接状态"""
    return json.dumps({"status": "timeout", "error": "Connection timed out after 30s", "possible_issue": "数据库连接池耗尽或网络阻塞"})

@tool
def check_db_slow_query() -> str:
    """检查数据库慢查询"""
    return json.dumps({"slow_queries": [{"sql": "SELECT * FROM orders WHERE status='pending'", "duration": "15s"}], "possible_issue": "存在多条慢查询，疑似缺少索引"})

@tool
def check_db_deadlock() -> str:
    """检查数据库死锁"""
    return json.dumps({"deadlocks": [], "possible_issue": "未检测到死锁"})

@tool
def check_network_ping(host: str) -> str:
    """检查网络连通性"""
    return json.dumps({"target": host, "status": "unreachable", "latency": None, "packet_loss": "100%", "possible_issue": "网络不通或目标主机不可达"})

@tool
def check_network_dns(domain: str) -> str:
    """检查DNS解析"""
    return json.dumps({"domain": domain, "resolved_ip": "10.0.0.1", "dns_status": "ok", "possible_issue": "DNS解析正常"})

@tool
def check_app_process(process_name: str) -> str:
    """检查应用进程状态"""
    return json.dumps({"process": process_name, "status": "running", "pid": 12345, "cpu": "85%", "memory": "92%", "possible_issue": "进程CPU和内存使用率过高"})

@tool
def check_app_port(port: int) -> str:
    """检查应用端口状态"""
    return json.dumps({"port": port, "status": "listening", "connection_count": 150, "possible_issue": "连接数正常"})

def create_router_node(llm):
    chain = ROUTER_PROMPT | llm

    async def router_node(state: SystemState) -> dict:
        """路由节点：分析故障现象并分类"""
        try:
            logger.info(f"Router开始分析: symptom={state.symptom[:50]}...")
            response = await chain.ainvoke({"symptom": state.symptom})
            result = parse_json_content(response.content) or {"diagnosis_type": "other", "urgency": "medium"}
            
            diagnosis_type = result.get("diagnosis_type", "other")
            urgency = result.get("urgency", "medium")
            
            logger.info(f"Router分析完成: diagnosis_type={diagnosis_type}, urgency={urgency}")
            
            return {
                "diagnosis_type": diagnosis_type,
                "urgency": urgency,
                "messages": [f"Router: 诊断={diagnosis_type}, 紧急度={urgency}"]
            }
        except Exception as e:
            logger.exception(f"Router节点执行失败: {e}")
            return {
                "diagnosis_type": "other",
                "urgency": "medium",
                "messages": [f"Router: 分析失败，使用默认值"]
            }
    return router_node

def create_db_agent_node(llm):
    tools = [check_db_connection, check_db_slow_query, check_db_deadlock]
    chain = DB_PROMPT | llm.bind_tools(tools)

    async def db_agent_node(state: SystemState) -> dict:
        """数据库诊断Agent节点"""
        try:
            logger.info(f"DB Agent开始诊断: symptom={state.symptom[:50]}...")
            response = await chain.ainvoke({"symptom": state.symptom})  
            
            tool_results = []
            for tool_call in response.tool_calls:
                logger.debug(f"调用工具: {tool_call['name']}, args={tool_call['args']}")
                for t in tools:
                    if t.name == tool_call["name"]:
                        result = await t.ainvoke(tool_call["args"])
                        tool_results.append({"tool": tool_call["name"], "result": result})
                        logger.debug(f"工具 {tool_call['name']} 返回成功")
                        break
            
            diagnosis_response = DB_DIAGNOSIS_PROMPT | llm
            diagnosis = await diagnosis_response.ainvoke({
                "symptom": state.symptom, 
                "tool_calls": str(response.tool_calls), 
                "tool_results": str(tool_results)
            })
            result = parse_json_content(diagnosis.content) or {"diagnosis": "无法解析", "possible_causes": []}
            
            logger.info(f"DB Agent诊断完成: diagnosis={result.get('diagnosis')}")
            
            return {
                "db_agent_result": {**result, "tool_results": tool_results}, 
                "messages": [f"DB Agent: {result.get('diagnosis')}"]
            }
        except Exception as e:
            logger.exception(f"DB Agent节点执行失败: {e}")
            return {
                "db_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"DB Agent: 诊断失败 - {str(e)}"]
            }
    return db_agent_node

def create_net_agent_node(llm):
    tools = [check_network_ping, check_network_dns]
    chain = NET_PROMPT | llm.bind_tools(tools)

    async def net_agent_node(state: SystemState) -> dict:
        """网络诊断Agent节点"""
        try:
            logger.info(f"Net Agent开始诊断: symptom={state.symptom[:50]}...")
            response = await chain.ainvoke({"symptom": state.symptom})  
            
            tool_results = []
            for tool_call in response.tool_calls:
                logger.debug(f"调用工具: {tool_call['name']}, args={tool_call['args']}")
                for t in tools:
                    if t.name == tool_call["name"]:
                        result = await t.ainvoke(tool_call["args"])
                        tool_results.append({"tool": tool_call["name"], "result": result})
                        logger.debug(f"工具 {tool_call['name']} 返回成功")
                        break
            
            diagnosis_response = NET_DIAGNOSIS_PROMPT | llm
            diagnosis = await diagnosis_response.ainvoke({
                "symptom": state.symptom, 
                "tool_calls": str(response.tool_calls), 
                "tool_results": str(tool_results)
            })
            result = parse_json_content(diagnosis.content) or {"diagnosis": "无法解析", "possible_causes": []}
            
            logger.info(f"Net Agent诊断完成: diagnosis={result.get('diagnosis')}")
            
            return {
                "net_agent_result": {**result, "tool_results": tool_results}, 
                "messages": [f"Net Agent: {result.get('diagnosis')}"]
            }
        except Exception as e:
            logger.exception(f"Net Agent节点执行失败: {e}")
            return {
                "net_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"Net Agent: 诊断失败 - {str(e)}"]
            }
    return net_agent_node

def create_app_agent_node(llm):
    tools = [check_app_process, check_app_port]
    chain = APP_PROMPT | llm.bind_tools(tools)

    async def app_agent_node(state: SystemState) -> dict:
        """应用诊断Agent节点"""
        try:
            logger.info(f"App Agent开始诊断: symptom={state.symptom[:50]}...")
            response = await chain.ainvoke({"symptom": state.symptom})      
            
            tool_results = []
            for tool_call in response.tool_calls:
                logger.debug(f"调用工具: {tool_call['name']}, args={tool_call['args']}")
                for t in tools:
                    if t.name == tool_call["name"]:
                        result = await t.ainvoke(tool_call["args"])
                        tool_results.append({"tool": tool_call["name"], "result": result})
                        logger.debug(f"工具 {tool_call['name']} 返回成功")
                        break
            
            diagnosis_response = APP_DIAGNOSIS_PROMPT | llm
            diagnosis = await diagnosis_response.ainvoke({
                "symptom": state.symptom, 
                "tool_calls": str(response.tool_calls), 
                "tool_results": str(tool_results)
            })
            result = parse_json_content(diagnosis.content) or {"diagnosis": "无法解析", "possible_causes": []}
            
            logger.info(f"App Agent诊断完成: diagnosis={result.get('diagnosis')}")
            
            return {
                "app_agent_result": {**result, "tool_results": tool_results}, 
                "messages": [f"App Agent: {result.get('diagnosis')}"]
            }
        except Exception as e:
            logger.exception(f"App Agent节点执行失败: {e}")
            return {
                "app_agent_result": {"diagnosis": "诊断失败", "possible_causes": [str(e)]},
                "messages": [f"App Agent: 诊断失败 - {str(e)}"]
            }
    return app_agent_node

def create_fix_agent_node(llm):
    chain = FIX_PROMPT | llm

    async def fix_agent_node(state: SystemState) -> dict: 
        """修复方案生成Agent节点"""
        try:
            diagnosis_type = state.diagnosis_type
            if diagnosis_type == "db":
                diagnosis_result = state.db_agent_result
            elif diagnosis_type == "net":
                diagnosis_result = state.net_agent_result
            elif diagnosis_type == "app":
                diagnosis_result = state.app_agent_result
            else:
                diagnosis_result = {}
            
            logger.info(f"Fix Agent开始生成修复方案: diagnosis_type={diagnosis_type}")
            
            response = await chain.ainvoke({
                "diagnosis_type": diagnosis_type, 
                "diagnosis_result": str(diagnosis_result)
            })
            result = parse_json_content(response.content) or {
                "plan_id": "PLAN-ERROR", 
                "description": "无法生成方案", 
                "risk_level": "unknown", 
                "prerequisites": [], 
                "steps": [], 
                "verification": {"commands": [], "expected_result": ""}, 
                "estimated_time": "0"
            }
            
            logger.info(f"Fix Agent方案生成完成: plan_id={result.get('plan_id')}, risk_level={result.get('risk_level')}")
            
            return {
                "fix_plan": result, 
                "messages": [f"Fix Agent: 生成修复方案 {result.get('plan_id')} - 风险等级: {result.get('risk_level')}"]
            }
        except Exception as e:
            logger.exception(f"Fix Agent节点执行失败: {e}")
            return {
                "fix_plan": {
                    "plan_id": "PLAN-ERROR",
                    "description": "方案生成失败",
                    "risk_level": "unknown",
                    "prerequisites": [],
                    "steps": [],
                    "verification": {"commands": [], "expected_result": ""},
                    "estimated_time": "0"
                },
                "messages": [f"Fix Agent: 方案生成失败 - {str(e)}"]
            }
    return fix_agent_node

def create_human_approval_node():
    async def human_approval_node(state: SystemState) -> dict:    
        """人工审批节点"""
        try:
            logger.info(f"审批节点: 请求审批工单 {state.ticket_id}")
            
            approval = interrupt({
                "type": "approval_required", 
                "ticket_id": state.ticket_id, 
                "fix_plan": state.fix_plan, 
                "message": f"请审批修复方案: {state.fix_plan.plan_id}"
            })
            
            if approval.get("approved", False):
                logger.info(f"审批节点: 工单 {state.ticket_id} 已审批通过, 备注: {approval.get('comments', '')}")
                return {
                    "approval_status": ApprovalStatus.APPROVED, 
                    "approver_comments": approval.get("comments", ""), 
                    "messages": [f"人工审批: 已批准 - {approval.get('comments', '')}"]
                }
            else:
                logger.info(f"审批节点: 工单 {state.ticket_id} 已拒绝, 备注: {approval.get('comments', '')}")
                return {
                    "approval_status": ApprovalStatus.REJECTED, 
                    "approver_comments": approval.get("comments", ""), 
                    "messages": [f"人工审批: 已拒绝 - {approval.get('comments', '')}"]
                }
        except GraphInterrupt:
            raise
        except Exception as e:
            logger.exception(f"审批节点执行失败: {e}")
            return {
                "approval_status": ApprovalStatus.REJECTED,
                "approver_comments": f"审批异常: {str(e)}",
                "messages": [f"人工审批: 异常 - {str(e)}"]
            }
    return human_approval_node

def create_other_handler_node():
    async def other_handler_node(state: SystemState) -> dict:
        """处理other类型工单：记录日志并保存工单"""
        async with AsyncSessionLocal() as db:
            try:
                logger.info(f"Other Handler: 工单 {state.ticket_id} 被分类为other类型，记录并归档")
                
                result = {
                    "messages": [
                        f"Other Handler: 工单 {state.ticket_id} 被分类为other类型",
                        f"Other Handler: 症状: {state.symptom}",
                        f"Other Handler: 紧急程度: {state.urgency}",
                        f"Other Handler: 已记录并归档，无需进一步处理"
                    ]
                }
                
                merged_state = {**state.__dict__, **result}
                merged_state["messages"] = state.messages + result["messages"]
                
                ticket = await save_ticket(db, merged_state)
                result["messages"].append(f"归档: 工单 {ticket.ticket_id} 已保存")
                
                logger.info(f"Other Handler: 工单 {ticket.ticket_id} 已保存")
                
                return result
            except Exception as e:
                logger.exception(f"Other Handler节点执行失败: {e}")
                return {
                    "messages": [f"Other Handler: 保存工单失败 - {str(e)}"]
                }
            finally:
                db.close()
                logger.debug("Other Handler: 数据库会话已关闭")
    return other_handler_node

def create_executor_node():
    async def executor_node(state: SystemState) -> dict:
        """执行节点：执行修复方案并保存工单"""
        async with AsyncSessionLocal() as db:
            try:
                fix_plan = state.fix_plan
                steps = fix_plan.steps if fix_plan else []
                executed_steps = []
                
                logger.info(f"执行节点: 开始执行修复方案 {fix_plan.plan_id if fix_plan else '未知'}, 共 {len(steps)} 个步骤")
                
                for step in steps:
                    logger.debug(f"执行步骤 {step.step_id}: {step.action}")
                    executed_steps.append({
                        "step_id": step.step_id, 
                        "action": step.action, 
                        "command": step.command, 
                        "status": "success", 
                        "output": f"Mock执行成功: {step.command}"
                    })
                    logger.debug(f"步骤 {step.step_id} 执行完成")
                
                result = {
                    "execution_result": {
                        "plan_id": fix_plan.plan_id if fix_plan else None, 
                        "executed_steps": executed_steps, 
                        "overall_status": "success", 
                        "summary": f"执行完成，共 {len(executed_steps)} 个步骤"
                    }, 
                    "messages": [f"执行节点: 完成修复方案执行 - {len(executed_steps)} 个步骤"]
                }
                
                merged_state = {**state.__dict__, **result}
                merged_state["messages"] = state.messages + result["messages"]
                
                ticket = await save_ticket(db, merged_state)
                result["messages"].append(f"归档: 工单 {ticket.ticket_id} 已保存")
                
                logger.info(f"执行节点: 修复方案执行完成，工单 {ticket.ticket_id} 已保存")
                
                return result
            except Exception as e:
                logger.exception(f"执行节点执行失败: {e}")
                return {
                    "execution_result": {
                        "plan_id": fix_plan.plan_id if fix_plan else None,
                        "executed_steps": [],
                        "overall_status": "failed",
                        "summary": f"执行失败: {str(e)}"
                    },
                    "messages": [f"执行节点: 执行失败 - {str(e)}"]
                }
            finally:
                db.close()
                logger.debug("执行节点: 数据库会话已关闭")
    return executor_node
