"""
会话级锁：同一 thread_id（会话）的并发请求串行化，避免图状态、待确认 SQL、人工中断等被并发写坏。

- 按 thread_id 分桶，每桶一个 asyncio.Lock，桶数可配置，避免无限创建锁。
- 不同 thread_id 可能落在同桶（哈希碰撞），仅会轻微串行化，不影响正确性。
- 可通过配置关闭（conversation_lock_enabled=False）。
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)

_locks: list[asyncio.Lock] = []
_initialized = False


def _ensure_locks() -> list[asyncio.Lock]:
    global _locks, _initialized
    if _initialized:
        return _locks
    n = max(1, min(4096, getattr(get_settings(), "conversation_lock_buckets", 1024)))
    _locks = [asyncio.Lock() for _ in range(n)]
    _initialized = True
    logger.info("会话锁已初始化: buckets=%d", n)
    return _locks


def _lock_for(thread_id: str) -> asyncio.Lock:
    locks = _ensure_locks()
    idx = hash(thread_id) % len(locks)
    return locks[idx]


@asynccontextmanager
async def conversation_lock(thread_id: str):
    """
    获取该会话的锁，在上下文内串行化对同一 thread_id 的操作。
    若 conversation_lock_enabled 为 False，则直接 yield，不加锁。
    """
    if not getattr(get_settings(), "conversation_lock_enabled", True):
        yield
        return
    lock = _lock_for(thread_id)
    async with lock:
        yield
