"""
LLM 调用韧性层 —— 重试 + 熔断。

提供生产级 API 调用保障：
- 自动重试临时性错误（网络抖动、限流、超时）
- 熔断器防止雪崩（连续失败时快速失败而非无限等待）

Usage:
    from agent.resilience import retryable_invoke, CircuitBreaker

    result = retryable_invoke(chain, {"user_input": "..."})

    cb = CircuitBreaker()
    result = cb.call(chain.invoke, {"user_input": "..."})
"""

import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.logging_config import get_logger

logger = get_logger(__name__)

# ---- 可重试的异常类型 ----
RETRYABLE_EXCEPTIONS = (
    RateLimitError,       # 429 — API 限流
    APITimeoutError,      # 请求超时
    APIConnectionError,   # 网络连接错误
    APIError,             # 其他 OpenAI API 错误（5xx）
)


# ============================================================
# Retry 装饰器
# ============================================================

def retryable_invoke(
    chain,
    input_dict: dict[str, Any],
    max_retries: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
) -> Any:
    """对 LangChain chain 的 invoke 调用添加自动重试。

    重试策略：
    - 仅在临时性错误时重试（RateLimit、Timeout、Connection）
    - Auth 错误不重试（API Key 错误重试无意义）
    - 等待时间指数增长：1s → 2s → 4s → 8s → ...

    Args:
        chain:       LangChain Runnable
        input_dict:  传给 chain.invoke() 的 dict
        max_retries: 最大重试次数
        min_wait:    首次等待秒数
        max_wait:    最大等待秒数

    Returns:
        chain.invoke() 的结果

    Raises:
        原始异常（如果所有重试都失败）
    """

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        stop=stop_after_attempt(max_retries + 1),  # 1 次原始 + N 次重试
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _invoke_with_retry():
        return chain.invoke(input_dict)

    return _invoke_with_retry()


# ============================================================
# Circuit Breaker（熔断器）
# ============================================================

class CircuitBreakerOpenError(Exception):
    """熔断器打开时抛出的异常。"""

    def __init__(self, message: str = "Circuit breaker is OPEN"):
        super().__init__(message)


class CircuitBreaker:
    """熔断器 —— 防止级联失败。

    三种状态：
        CLOSED    → 正常调用
        OPEN      → 熔断中，快速失败（抛出 CircuitBreakerOpenError）
        HALF_OPEN → 探测恢复，允许一次试调用

    状态转换：
        CLOSED ──[连续失败 N 次]──→ OPEN
        OPEN   ──[超时后]────────→ HALF_OPEN
        HALF_OPEN ──[试调用成功]──→ CLOSED
        HALF_OPEN ──[试调用失败]──→ OPEN

    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        result = cb.call(chain.invoke, {"user_input": "..."})
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default",
    ):
        """
        Args:
            failure_threshold: 连续失败多少次后熔断
            recovery_timeout:  熔断后多少秒进入半开状态
            name:              熔断器名称（用于日志区分）
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name

        self._failure_count = 0
        self._last_failure_time: datetime | None = None
        self._state = "CLOSED"
        self._lock = threading.Lock()

    # ---- 属性 ----
    @property
    def state(self) -> str:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    # ---- 核心方法 ----
    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """通过熔断器调用函数。

        Args:
            fn: 要调用的函数
            *args, **kwargs: 传递给 fn 的参数

        Returns:
            fn 的返回值

        Raises:
            CircuitBreakerOpenError: 熔断器打开时
            函数原始异常: 非熔断导致的失败
        """
        with self._lock:
            if self._state == "OPEN":
                if self._should_try_recovery():
                    self._state = "HALF_OPEN"
                    logger.warning(
                        "circuit_breaker_half_open",
                        name=self.name,
                        failure_count=self._failure_count,
                    )
                else:
                    elapsed = (
                        datetime.now() - self._last_failure_time
                    ).total_seconds() if self._last_failure_time else 0
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN "
                        f"(failures={self._failure_count}, "
                        f"elapsed={elapsed:.1f}s, "
                        f"retry_after={self.recovery_timeout - elapsed:.1f}s)"
                    )

            elif self._state == "HALF_OPEN":
                logger.info("circuit_breaker_probing", name=self.name)

        # 执行调用
        try:
            result = fn(*args, **kwargs)
        except RETRYABLE_EXCEPTIONS:
            # 注意：这里只捕获 RETRYABLE_EXCEPTIONS
            # 其他异常（如 ValueError）不应该触发熔断
            self._on_failure()
            raise
        # 注意：非 RETRYABLE 异常不触发熔断，直接向上抛
        except Exception:
            raise

        # 成功 → 重置
        self._on_success()
        return result

    # ---- 内部方法 ----
    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
                logger.error(
                    "circuit_breaker_open",
                    name=self.name,
                    failure_count=self._failure_count,
                    recovery_timeout=self.recovery_timeout,
                )

    def _on_success(self):
        with self._lock:
            if self._state == "HALF_OPEN":
                logger.info(
                    "circuit_breaker_recovered",
                    name=self.name,
                )
            self._state = "CLOSED"
            self._failure_count = 0
            self._last_failure_time = None

    def _should_try_recovery(self) -> bool:
        """判断是否应该尝试恢复（进入 HALF_OPEN）。"""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def reset(self):
        """手动重置熔断器（用于测试或手动恢复）。"""
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._last_failure_time = None
            logger.info("circuit_breaker_reset", name=self.name)
