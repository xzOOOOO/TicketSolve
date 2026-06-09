# os：用于读取环境变量
import os
# load_dotenv：从 .env 文件加载环境变量
from dotenv import load_dotenv
# OpenAI 异常类，用于重试配置
from openai import APIError, APITimeoutError, APIConnectionError, RateLimitError

# 加载 .env 文件中的环境变量（如果存在）
load_dotenv()

class Settings:
    """
    应用配置类

    所有配置项从环境变量读取，支持 .env 文件。
    提供默认值，确保即使不配置环境变量也能运行（虽然某些功能可能受限）。
    """

    # ═══════════════════════════════════════════════
    # LLM 配置
    # ═══════════════════════════════════════════════
    # LLM_API_KEY：大模型 API 密钥，用于调用 LLM 服务
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    # LLM_MODEL：使用的模型名称，默认通义千问 qwen3.5-flash
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.5-flash")
    # LLM_BASE_URL：LLM API 的基础 URL，默认阿里云 DashScope
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    # ═══════════════════════════════════════════════
    # LLM 限流配置
    # ═══════════════════════════════════════════════
    # LLM_MAX_CONCURRENT：最大并发请求数，防止同时发送过多请求
    LLM_MAX_CONCURRENT: int = int(os.getenv("LLM_MAX_CONCURRENT", "5"))
    # LLM_RPM_LIMIT：每分钟最大请求数（Rate Per Minute）
    LLM_RPM_LIMIT: int = int(os.getenv("LLM_RPM_LIMIT", "60"))

    # ═══════════════════════════════════════════════
    # LLM 重试配置
    # ═══════════════════════════════════════════════
    # LLM_MAX_RETRIES：最大重试次数（不含首次请求）
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    # LLM_RETRY_EXPONENTIAL_JITTER：是否使用指数退避+抖动策略
    LLM_RETRY_EXPONENTIAL_JITTER: bool = os.getenv("LLM_RETRY_EXPONENTIAL_JITTER", "true").lower() == "true"

    # ═══════════════════════════════════════════════
    # 数据库配置
    # ═══════════════════════════════════════════════
    # DB_USER：数据库用户名
    DB_USER: str = os.getenv("DB_USER", "postgres")
    # DB_PASSWORD：数据库密码
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "xxxxxx")
    # DB_HOST：数据库主机地址
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    # DB_PORT：数据库端口
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    # DB_NAME：数据库名称
    DB_NAME: str = os.getenv("DB_NAME", "tickets")
    # DB_ECHO：是否打印 SQL 语句（调试用）
    DB_ECHO: bool = os.getenv("DB_ECHO", "true").lower() == "true"

    # ═══════════════════════════════════════════════
    # 服务配置
    # ═══════════════════════════════════════════════
    # HOST：API 服务监听地址，0.0.0.0 表示监听所有网卡
    HOST: str = os.getenv("HOST", "0.0.0.0")
    # PORT：API 服务监听端口
    PORT: int = int(os.getenv("PORT", "8000"))
    # DEBUG：是否开启调试模式
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # ═══════════════════════════════════════════════
    # 执行器配置
    # ═══════════════════════════════════════════════
    # EXECUTOR_MODE：执行器模式
    # - mock：只模拟命令执行（默认，用于测试）
    # - docker_lab：在 SREBench Lite 靶场中执行白名单命令
    EXECUTOR_MODE: str = os.getenv("EXECUTOR_MODE", "mock")
    
    @classmethod
    def get_database_url(cls) -> str:
        """
        组装数据库连接 URL

        使用 asyncpg 驱动，支持异步数据库操作。
        格式：postgresql+asyncpg://user:password@host:port/dbname
        """
        return f"postgresql+asyncpg://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

    @classmethod
    def get_llm_config(cls) -> dict:
        """
        获取 LLM 配置字典

        返回的字典可直接传给 LangChain 的 ChatOpenAI 等模型初始化。
        """
        return {
            "model": cls.LLM_MODEL,
            "base_url": cls.LLM_BASE_URL,
            "api_key": cls.LLM_API_KEY
        }

    @classmethod
    def get_retry_config(cls) -> dict:
        """
        获取 LLM 重试配置字典（用于 Runnable.with_retry）

        配置说明：
        - stop_after_attempt：最大尝试次数（含首次）
        - wait_exponential_jitter：指数退避+抖动，避免所有请求同时重试
        - retry_if_exception_type：只在特定异常时重试（网络/超时/限流/API错误）
        """
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

# settings：全局配置单例，项目中通过 from config import settings 使用
settings = Settings()
