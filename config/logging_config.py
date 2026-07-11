"""
Structlog 结构化日志配置。

提供统一的日志工厂，支持两种输出模式：
- console: 彩色、人类可读（开发）
- json:    JSON 行格式（生产，可接入 ELK/Loki）

Usage:
    from config.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("user_action", action="start_learning", subject="CET6")
"""

import logging
import sys
from typing import Any

import structlog

from config.settings import get_settings


def _get_processors(log_format: str) -> list[Any]:
    """根据日志格式选择处理器链。"""
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        # 生产：JSON 格式 + 自动添加额外字段
        shared_processors.append(structlog.processors.dict_tracebacks)
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        # 开发：彩色 Console
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    shared_processors.append(
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter
    )

    return shared_processors, renderer


def setup_logging() -> None:
    """全局初始化 structlog。

    应在 app 入口或最外层模块调用一次。
    幂等：多次调用不会重复配置。
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    processors, renderer = _get_processors(settings.log_format)

    # 配置标准库 logging → structlog 桥接
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 设置 root logger 的输出
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # 降低第三方库的噪声日志
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """获取绑定上下文的 structlog logger。

    Args:
        name: 通常传 __name__，会被添加到日志上下文

    Returns:
        绑定了 app_name 的 structlog logger
    """
    settings = get_settings()
    return structlog.get_logger(name).bind(app=settings.app_name)
