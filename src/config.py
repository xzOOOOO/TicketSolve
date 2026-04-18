import os
from pathlib import Path
from dotenv import load_dotenv

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

settings = Settings()