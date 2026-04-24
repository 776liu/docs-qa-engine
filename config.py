import os

from pydantic_settings import  BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置管理"""

    API_TITLE: str = "RAG API"
    API_VERSION: str = "1.0"

    API_TOKEN: str = os.getenv("API_TOKEN")

    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH")
    DOCS_DIR: str = os.getenv("DOCS_DIR")

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "deepseek-chat"
    LLM_API_BASE: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 3
    MAX_CACHE_SIZE: int = 500
    MAX_HISTORY_LENGTH: int = 20

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


# 单例模式，全局使用
settings = Settings()