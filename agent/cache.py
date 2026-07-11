"""
LLM 响应缓存 —— 减少重复 API 调用，降低成本。

双层缓存架构：
- L1（内存）：functools.lru_cache — 同一进程相同请求即时返回
- L2（磁盘）：diskcache — 跨进程/跨重启缓存，TTL 自动过期

Usage:
    from agent.cache import LLMCache

    cache = LLMCache()
    result = cache.get_or_invoke("planner", {"user_input": "..."}, chain.invoke)
"""

import contextlib
import hashlib
import json
import threading
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)

# ---- 磁盘缓存（可选） ----
_disk_cache: Any | None = None
_disk_cache_lock = threading.Lock()


def _get_disk_cache():
    """延迟初始化 diskcache（避免强制依赖）。"""
    global _disk_cache
    if _disk_cache is None:
        with _disk_cache_lock:
            if _disk_cache is None:
                try:
                    import diskcache

                    cache_dir = Path(__file__).parent.parent / "data" / ".llm_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    _disk_cache = diskcache.Cache(str(cache_dir))
                    logger.info("disk_cache_initialized", dir=str(cache_dir))
                except ImportError:
                    logger.info("disk_cache_unavailable", reason="diskcache not installed")
                    _disk_cache = False  # 标记为不可用
                except Exception as e:
                    logger.warning("disk_cache_init_failed", error=str(e))
                    _disk_cache = False
    return _disk_cache if _disk_cache is not False else None


# ============================================================
# 缓存 Key 生成
# ============================================================


def _make_cache_key(agent_name: str, input_dict: dict[str, Any]) -> str:
    """生成稳定的缓存 Key。

    使用 SHA256 哈希确保：
    - 相同输入 → 相同 Key
    - 不同输入 → 不同 Key（碰撞概率可忽略）
    """
    # 排序确保 dict 的 key 顺序不影响哈希
    canonical = json.dumps(
        {"agent": agent_name, "input": input_dict},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================
# LLMCache 主类
# ============================================================


class LLMCache:
    """LLM 响应双层缓存。

    Usage:
        cache = LLMCache()
        result = cache.get_or_invoke(
            "planner",
            {"user_input": "学科：CET6词汇..."},
            planner_chain.invoke,
        )
    """

    def __init__(self, ttl_seconds: int | None = None):
        """
        Args:
            ttl_seconds: 缓存 TTL（秒），None 则使用 Settings 中的配置
        """
        settings = get_settings()
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        # 命中/未命中统计
        self.hits = 0
        self.misses = 0

    def get_or_invoke(
        self,
        agent_name: str,
        input_dict: dict[str, Any],
        invoke_fn: Callable[[dict[str, Any]], Any],
        skip_cache: bool = False,
    ) -> Any:
        """从缓存获取或调用 LLM。

        Args:
            agent_name: Agent 名称（planner/tutor/quiz/grading/diagnosis）
            input_dict: 传给 chain.invoke() 的 dict
            invoke_fn:  实际的 LLM 调用函数
            skip_cache: 是否跳过缓存（强制调用 LLM）

        Returns:
            LLM 响应（可能是缓存的）
        """
        if skip_cache:
            return invoke_fn(input_dict)

        cache_key = _make_cache_key(agent_name, input_dict)

        # L1: 内存缓存
        result = self._memory_cache_get(cache_key)
        if result is not None:
            self.hits += 1
            logger.debug("cache_hit", agent=agent_name, tier="L1_memory")
            return result

        # L2: 磁盘缓存
        result = self._disk_cache_get(cache_key)
        if result is not None:
            self.hits += 1
            logger.debug("cache_hit", agent=agent_name, tier="L2_disk")
            # 回填 L1
            self._memory_cache_set(cache_key, result)
            return result

        # 未命中 → 调用 LLM
        self.misses += 1
        logger.debug("cache_miss", agent=agent_name)
        result = invoke_fn(input_dict)

        # 写入双层缓存
        self._memory_cache_set(cache_key, result)
        self._disk_cache_set(cache_key, result)

        return result

    # ---- L1 内存缓存 ----
    @staticmethod
    @lru_cache(maxsize=128)
    def _memory_cache_get(cache_key: str) -> Any | None:
        """L1 内存缓存查询（lru_cache 自动管理淘汰）。"""
        # lru_cache 装饰的函数实际上是一个包装器
        # 这里使用 sentinel 模式
        return _MemoryCache.get(cache_key)

    @staticmethod
    def _memory_cache_set(cache_key: str, value: Any):
        """L1 内存缓存写入。"""
        _MemoryCache.set(cache_key, value)
        # 更新 lru_cache 包装
        LLMCache._memory_cache_get.__wrapped__(cache_key)

    # ---- L2 磁盘缓存 ----
    def _disk_cache_get(self, cache_key: str) -> Any | None:
        """L2 磁盘缓存查询（带 TTL 过期检查）。"""
        cache = _get_disk_cache()
        if cache is None:
            return None
        try:
            value = cache.get(cache_key, default=None)
            if value is not None:
                # 检查 TTL
                expire_key = f"{cache_key}_expire"
                expire_time = cache.get(expire_key, default=None)
                if expire_time:
                    import time

                    if time.time() > expire_time:
                        # 已过期
                        cache.delete(cache_key)
                        cache.delete(expire_key)
                        return None
                return value
        except Exception as e:
            logger.warning("disk_cache_read_error", error=str(e))
        return None

    def _disk_cache_set(self, cache_key: str, value: Any):
        """L2 磁盘缓存写入。"""
        cache = _get_disk_cache()
        if cache is None:
            return
        try:
            import time

            cache.set(cache_key, value)
            cache.set(f"{cache_key}_expire", time.time() + self._ttl)
        except Exception as e:
            logger.warning("disk_cache_write_error", error=str(e))

    def clear(self):
        """清除所有缓存。"""
        self._memory_cache_get.cache_clear()
        _MemoryCache.clear()
        cache = _get_disk_cache()
        if cache:
            with contextlib.suppress(Exception):
                cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("cache_cleared")

    @property
    def hit_rate(self) -> float:
        """缓存命中率。"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


# ---- L1 内存缓存的简单字典实现（与 lru_cache 配合） ----
class _MemoryCache:
    """线程安全的内存缓存。"""

    _store: ClassVar[dict[str, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, key: str) -> Any | None:
        with cls._lock:
            return cls._store.get(key)

    @classmethod
    def set(cls, key: str, value: Any):
        with cls._lock:
            cls._store[key] = value

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._store.clear()
