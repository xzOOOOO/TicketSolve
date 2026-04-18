import asyncio
from langchain_openai import ChatOpenAI
from workflow import create_async_workflow
from database import init_db
from langgraph.types import Command
from logger import logger
import sys
from config import settings
async def main():
    try:
        logger.info("=" * 50)
        logger.info("工单系统启动")
        logger.info("=" * 50)
        
        logger.info("初始化数据库...")
        await init_db()
        logger.info("数据库初始化完成")
        
        logger.info("创建LLM实例...")
        llm_config = settings.get_llm_config()
        llm = ChatOpenAI(**llm_config)
        logger.info("LLM实例创建完成")
        
        logger.info("创建工作流...")
        app = create_async_workflow(llm)
        logger.info("工作流创建完成")
        
        config = {"configurable": {"thread_id": "test-thread-001"}}
        
        initial_state = {
            "ticket_id": "TKT-004",
            "symptom": "数据读不出来"
        }
        
        logger.info("第一次调用工作流...")
        result = await app.ainvoke(initial_state, config=config)
        logger.info("第一次调用完成，到达中断点")
        
        logger.info("发送审批指令...")
        result = await app.ainvoke(Command(resume={"approved": True, "comments": "同意执行"}), config=config)
        logger.info("第二次调用完成")
        
        logger.info(f"最终结果: {result}")
        print(result)
        
        logger.info("工单系统执行完成")
        
    except KeyboardInterrupt:
        logger.warning("用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())