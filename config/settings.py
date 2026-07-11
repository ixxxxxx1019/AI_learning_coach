"""
Pydantic Settings —— 统一的配置管理入口。

所有环境变量在此集中定义、验证、文档化。
使用 BaseSettings 自动从 .env 和环境变量加载。

Usage:
    from config.settings import get_settings
    settings = get_settings()
    api_key = settings.deepseek_api_key.get_secret_value()
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """AI学习教练 全局配置。

    所有字段可通过环境变量或 .env 文件设置。
    SecretStr 类型字段在日志中自动隐藏。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- DeepSeek / LLM ---
    deepseek_api_key: SecretStr = Field(
        default=SecretStr("sk-placeholder"),
        description="DeepSeek API Key — https://platform.deepseek.com",
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API 地址",
    )
    default_model: str = Field(
        default="deepseek-chat",
        description="默认模型名称",
    )
    llm_temperature_creative: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="创意输出（Tutor）的 temperature",
    )
    llm_temperature_structured: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="结构化输出（Planner/Evaluator）的 temperature",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        le=8192,
        description="单次 LLM 调用最大 token 数",
    )

    # --- LangSmith 追踪 ---
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API Key — https://smith.langchain.com",
    )
    langsmith_project: str = Field(
        default="ai-coach-langchain",
        description="LangSmith 项目名",
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API 端点",
    )
    langchain_tracing_v2: bool = Field(
        default=False,
        description="是否启用 LangChain 追踪 v2",
    )

    # --- Redis / 缓存 ---
    redis_url: str | None = Field(
        default=None,
        description="Redis 连接 URL，如 redis://localhost:6379/0",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="LLM 响应缓存 TTL（秒）",
    )

    # --- 持久化 ---
    checkpointer_type: str = Field(
        default="memory",
        pattern="^(memory|sqlite|postgres)$",
        description="LangGraph checkpointer 类型",
    )
    sqlite_db_path: str = Field(
        default="data/checkpoints.db",
        description="SQLite checkpointer 数据库路径",
    )
    postgres_url: str | None = Field(
        default=None,
        description="PostgreSQL 连接 URL",
    )

    # --- 日志 ---
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="日志级别",
    )
    log_format: str = Field(
        default="console",
        pattern="^(json|console)$",
        description="日志格式：json（生产）或 console（开发）",
    )

    # --- 应用 ---
    app_name: str = Field(
        default="AI学习教练",
        description="应用显示名称",
    )
    app_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Streamlit 服务端口",
    )


@lru_cache
def get_settings() -> Settings:
    """获取 Settings 单例（带缓存）。

    使用 @lru_cache 确保整个进程中只创建一次 Settings 实例。
    这样每次调用 get_settings() 不会重复读取 .env 文件。
    """
    return Settings()
