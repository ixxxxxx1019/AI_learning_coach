"""
config 包 —— 应用配置层。

提供：
- settings: Pydantic Settings，统一的配置入口
- logging_config: structlog 结构化日志配置
"""

from config.logging_config import get_logger, setup_logging
from config.settings import Settings, get_settings

__all__ = ["Settings", "get_logger", "get_settings", "setup_logging"]
