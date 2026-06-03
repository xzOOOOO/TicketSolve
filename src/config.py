import os
from pathlib import Path
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, APIConnectionError, RateLimitError

load_dotenv()

class Settings:
    """应用配置类"""
    
    # LLM 配置
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5-flash")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # LLM 限流配置
    LLM_MAX_CONCURRENT: int = int(os.getenv("LLM_MAX_CONCURRENT", "5"))
    LLM_RPM_LIMIT: int = int(os.getenv("LLM_RPM_LIMIT", "60"))
    
    # LLM 重试配置
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_RETRY_EXPONENTIAL_JITTER: bool = os.getenv("LLM_RETRY_EXPONENTIAL_JITTER", "true").lower() == "true"
    
    # 数据库配置
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "xxxxxx")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "tickets")
    DB_ECHO: bool = os.getenv("DB_ECHO", "true").lower() == "true"
    
    # 服务配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # 执行器配置
    # mock: 只模拟命令执行；docker_lab: 在 SREBench Lite 靶场中执行白名单命令
    EXECUTOR_MODE: str = os.getenv("EXECUTOR_MODE", "mock")
    
    @classmethod
    def get_database_url(cls) -> str:
        """组装数据库连接URL"""
        return f"postgresql+asyncpg://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """获取LLM配置字典"""
        return {
            "model": cls.LLM_MODEL,
            "base_url": cls.LLM_BASE_URL,
            "api_key": cls.LLM_API_KEY
        }
    
    @classmethod
    def get_retry_config(cls) -> dict:
        """获取LLM重试配置字典（用于 Runnable.with_retry）"""
        return {
            "stop_after_attempt": cls.LLM_MAX_RETRIES + 1,
            "wait_exponential_jitter": cls.LLM_RETRY_EXPONENTIAL_JITTER,
            "retry_if_exception_type": (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                APIError,
            ),
        }

settings = Settings()
