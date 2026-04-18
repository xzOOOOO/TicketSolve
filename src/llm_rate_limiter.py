"""
LLM 请求限流器
实现令牌桶算法，控制并发请求数和每分钟请求数
"""
import asyncio
import time
from logger import logger


class LLMRateLimiter:
    """
    LLM 请求限流器
    
    功能：
    - 控制并发请求数（防止同时发出过多请求）
    - 控制每分钟请求数（RPM 限制）
    - 自动等待和排队
    """
    
    def __init__(self, max_concurrent: int = 5, rpm_limit: int = 60):
        """
        初始化限流器
        
        Args:
            max_concurrent: 最大并发请求数
            rpm_limit: 每分钟最大请求数 (Requests Per Minute)
        """
        self.max_concurrent = max_concurrent
        self.rpm_limit = rpm_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_timestamps = []
        self._lock = asyncio.Lock()
        
        logger.info(f"LLM限流器初始化: 最大并发={max_concurrent}, RPM限制={rpm_limit}")
    
    async def acquire(self, node_name: str = "unknown"):
        """
        获取请求许可（在调用 LLM 前调用）
        
        Args:
            node_name: 调用节点名称（用于日志）
        """
        logger.debug(f"[{node_name}] 等待获取LLM请求许可...")
        
        await self.semaphore.acquire()
        
        async with self._lock:
            now = time.time()
            
            self.request_timestamps = [
                ts for ts in self.request_timestamps 
                if ts > now - 60
            ]
            
            if len(self.request_timestamps) >= self.rpm_limit:
                wait_time = 60 - (now - self.request_timestamps[0])
                if wait_time > 0:
                    logger.warning(
                        f"[{node_name}] RPM限制达到，等待 {wait_time:.2f} 秒 "
                        f"(当前窗口内请求数: {len(self.request_timestamps)})"
                    )
                    await asyncio.sleep(wait_time)
            
            self.request_timestamps.append(time.time())
        
        logger.debug(f"[{node_name}] 获取LLM请求许可成功")
    
    def release(self, node_name: str = "unknown"):
        """
        释放请求许可（在 LLM 调用完成后调用）
        
        Args:
            node_name: 调用节点名称（用于日志）
        """
        self.semaphore.release()
        logger.debug(f"[{node_name}] 释放LLM请求许可")
    
    def get_stats(self) -> dict:
        """获取限流器当前状态"""
        now = time.time()
        active_requests = len([
            ts for ts in self.request_timestamps 
            if ts > now - 60
        ])
        
        return {
            "max_concurrent": self.max_concurrent,
            "rpm_limit": self.rpm_limit,
            "current_rpm": active_requests,
            "available_capacity": self.rpm_limit - active_requests
        }


class RateLimitedLLM:
    """
    限流 LLM 包装器
    
    用法：
    llm = ChatOpenAI(...)
    rate_limiter = LLMRateLimiter()
    limited_llm = RateLimitedLLM(llm, rate_limiter)
    
    然后使用 limited_llm 替代原来的 llm
    """
    
    def __init__(self, llm, rate_limiter: LLMRateLimiter):
        """
        初始化限流 LLM 包装器
        
        Args:
            llm: 原始 LLM 实例
            rate_limiter: 限流器实例
        """
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.request_count = 0
        self.error_count = 0
    
    async def ainvoke(self, input_data, config=None, **kwargs):
        """异步调用 LLM（带限流）"""
        node_name = config.get("node_name", "unknown") if config else "unknown"
        
        await self.rate_limiter.acquire(node_name)
        
        try:
            self.request_count += 1
            result = await self.llm.ainvoke(input_data, config=config, **kwargs)
            return result
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{node_name}] LLM调用失败 (第{self.request_count}次): {e}")
            raise
        finally:
            self.rate_limiter.release(node_name)
    
    def bind_tools(self, tools):
        """绑定工具（返回限流版本的工具调用）"""
        bound_llm = self.llm.bind_tools(tools)
        
        class BoundRateLimitedLLM:
            def __init__(self, bound_llm, rate_limiter):
                self.bound_llm = bound_llm
                self.rate_limiter = rate_limiter
            
            async def ainvoke(self, input_data, config=None, **kwargs):
                node_name = config.get("node_name", "unknown") if config else "unknown"
                
                await self.rate_limiter.acquire(node_name)
                
                try:
                    self.request_count = getattr(self, 'request_count', 0) + 1
                    result = await self.bound_llm.ainvoke(input_data, config=config, **kwargs)
                    return result
                except Exception as e:
                    logger.error(f"[{node_name}] LLM工具调用失败: {e}")
                    raise
                finally:
                    self.rate_limiter.release(node_name)
        
        return BoundRateLimitedLLM(bound_llm, self.rate_limiter)
    
    def __getattr__(self, name):
        """代理其他属性访问到原始 LLM"""
        return getattr(self.llm, name)
