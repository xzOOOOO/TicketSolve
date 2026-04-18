"""
LLM 请求重试机制（简化版）
实现指数退避重试，提升系统容错能力
"""
import asyncio
from logger import logger
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Optional


class LLMRetryWrapper(Runnable):
    """
    LLM 重试包装器（简化版）
    
    用法：
    llm = ChatOpenAI(...)
    retry_llm = LLMRetryWrapper(llm, max_retries=3, base_delay=1.0)
    然后使用 retry_llm 替代原来的 llm
    """
    
    def __init__(self, llm, max_retries: int = 3, base_delay: float = 1.0):
        self.llm = llm
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    async def ainvoke(self, input_data, config=None, **kwargs):
        """异步调用 LLM（带重试）"""
        self.total_calls += 1
        node_name = config.get("node_name", "unknown") if config else "unknown"
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    logger.warning(f"[{node_name}] 重试第 {attempt} 次，等待 {delay} 秒")
                    await asyncio.sleep(delay)
                
                result = await self.llm.ainvoke(input_data, config=config, **kwargs)
                self.successful_calls += 1
                return result
                
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"[{node_name}] 重试 {self.max_retries} 次后失败: {e}")
                    self.failed_calls += 1
                    raise
                logger.warning(f"[{node_name}] 第 {attempt + 1} 次失败: {e}")
    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Any:
        """同步调用（Runnable 要求实现）"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.ainvoke(input, config=config, **kwargs))
    
    def bind_tools(self, tools):
        """绑定工具（返回带重试的版本）"""
        bound_llm = self.llm.bind_tools(tools)
        wrapper = self
        
        class BoundRetryLLM(Runnable):
            async def ainvoke(inner_self, input_data, config=None, **kwargs):
                node_name = config.get("node_name", "unknown") if config else "unknown"
                
                for attempt in range(wrapper.max_retries + 1):
                    try:
                        if attempt > 0:
                            delay = wrapper.base_delay * (2 ** (attempt - 1))
                            logger.warning(f"[{node_name}] 工具调用重试第 {attempt} 次，等待 {delay} 秒")
                            await asyncio.sleep(delay)
                        
                        result = await bound_llm.ainvoke(input_data, config=config, **kwargs)
                        wrapper.successful_calls += 1
                        return result
                    except Exception as e:
                        if attempt == wrapper.max_retries:
                            logger.error(f"[{node_name}] 工具调用重试 {wrapper.max_retries} 次后失败: {e}")
                            wrapper.failed_calls += 1
                            raise
                        logger.warning(f"[{node_name}] 工具调用第 {attempt + 1} 次失败: {e}")
            
            def invoke(
                self,
                input: Any,
                config: Optional[RunnableConfig] = None,
                **kwargs: Any
            ) -> Any:
                """同步调用（Runnable 要求实现）"""
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(self.ainvoke(input, config=config, **kwargs))
        
        return BoundRetryLLM()
    
    def __getattr__(self, name):
        """代理其他属性访问到原始 LLM"""
        return getattr(self.llm, name)
