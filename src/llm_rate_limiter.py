"""
LLM 请求限流器
实现令牌桶算法，控制并发请求数和每分钟请求数

使用 LangChain 回调机制（AsyncCallbackHandler）实现限流，
无需包装 LLM 对象，with_structured_output / bind_tools 天然可用。
"""
import asyncio
import time
from langchain_core.callbacks import AsyncCallbackHandler
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
        self.max_concurrent = max_concurrent
        self.rpm_limit = rpm_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_timestamps = []
        self._lock = asyncio.Lock()
        logger.info(f"LLM限流器初始化: 最大并发={max_concurrent}, RPM限制={rpm_limit}")

    async def acquire(self, node_name: str = "unknown"):
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
        self.semaphore.release()
        logger.debug(f"[{node_name}] 释放LLM请求许可")

    def get_stats(self) -> dict:
        now = time.time()
        active_requests = len([
            ts for ts in self.request_timestamps
            if ts > now - 60
        ])
        return {
            "max_concurrent": self.max_concurrent,
            "rpm_limit": self.rpm_limit,
            "current_rpm": active_requests,
            "available_capacity": self.rpm_limit - active_requests,
        }


class RateLimitCallback(AsyncCallbackHandler):
    """
    限流回调处理器

    挂载到 ChatOpenAI 的 callbacks 参数上即可，
    每次 LLM 调用前自动限流，调用后自动释放。
    with_structured_output / bind_tools 产生的内部调用也会被拦截。

    用法：
        llm = ChatOpenAI(callbacks=[RateLimitCallback(rate_limiter)], **config)
    """

    def __init__(self, rate_limiter: LLMRateLimiter):
        self.rate_limiter = rate_limiter

    async def on_llm_start(self, serialized, prompts, **kwargs):
        await self.rate_limiter.acquire("llm")

    async def on_llm_end(self, response, **kwargs):
        self.rate_limiter.release("llm")

    async def on_llm_error(self, error, **kwargs):
        self.rate_limiter.release("llm")
