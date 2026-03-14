"""
多 worker 共享状态：解析缓存、待确认 SQL、人工中断会话。

当 shared_state_redis_url 配置且可用时，状态存 Redis，多进程/多 worker 共享；
否则使用进程内内存（单 worker 或无需共享）。
"""
import asyncio
import json
import logging
from typing import Any, Optional

from config import get_settings

logger = logging.getLogger(__name__)

_PREFIX_PARSE = "kb:parse:v1:"
_PREFIX_PENDING_SQL = "kb:pending_sql:v1:"
_KEY_INTERRUPTED = "kb:interrupted:v1"
_TTL_PARSE = 3600
_TTL_PENDING = 86400

# 未配置 Redis 时的进程内存储
_memory_parse: dict[str, Any] = {}
_memory_pending_sql: dict[str, str] = {}
_memory_interrupted: set[str] = set()
_memory_lock = asyncio.Lock()

_shared_redis: Any = None
_shared_redis_lock = asyncio.Lock()


async def _get_shared_redis():
    """懒加载共享状态用 Redis（仅当 shared_state_redis_url 配置时）。"""
    global _shared_redis
    url = (getattr(get_settings(), "shared_state_redis_url", None) or "").strip()
    if not url:
        return None
    if _shared_redis is not None:
        return _shared_redis
    async with _shared_redis_lock:
        if _shared_redis is not None:
            return _shared_redis
        try:
            from redis.asyncio import Redis
            _shared_redis = Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=5.0,
            )
            await _shared_redis.ping()
            logger.info("共享状态 Redis 已连接（多 worker 共享生效）")
            return _shared_redis
        except Exception as e:
            logger.warning("共享状态 Redis 连接失败，将使用进程内内存: %s", e)
            return None


async def close_shared_state_redis():
    """关闭共享状态 Redis 连接（lifespan 退出时调用）。"""
    global _shared_redis
    async with _shared_redis_lock:
        if _shared_redis is not None:
            try:
                await _shared_redis.aclose()
            except Exception:
                pass
            _shared_redis = None


# ---------- 解析缓存（ParseResult） ----------

async def get_parse_result(cache_key: str):
    """
    获取解析缓存。返回 ParseResult 或 None。
    需在调用方从 src.doc.mineru_client 导入 ParseResult 做反序列化。
    """
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            return _memory_parse.get(cache_key)
    try:
        raw = await redis.get(_PREFIX_PARSE + cache_key)
        if raw is None:
            return None
        data = json.loads(raw)
        from src.doc.mineru_client import ParseResult
        return ParseResult.model_validate(data)
    except Exception as e:
        logger.debug("get_parse_result Redis 读失败: %s", e)
        return None


async def set_parse_result(cache_key: str, value: Any) -> None:
    """写入解析缓存。value 需为 ParseResult（或含 model_dump 的对象）。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            _memory_parse[cache_key] = value
        return
    try:
        data = value.model_dump() if hasattr(value, "model_dump") else value
        await redis.set(
            _PREFIX_PARSE + cache_key,
            json.dumps(data, ensure_ascii=False),
            ex=_TTL_PARSE,
        )
    except Exception as e:
        logger.warning("set_parse_result Redis 写失败: %s", e)


async def delete_parse_result(cache_key: str) -> None:
    """删除解析缓存项。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            _memory_parse.pop(cache_key, None)
        return
    try:
        await redis.delete(_PREFIX_PARSE + cache_key)
    except Exception:
        pass


async def has_parse_result(cache_key: str) -> bool:
    """是否存在该解析缓存。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            return cache_key in _memory_parse
    try:
        return await redis.exists(_PREFIX_PARSE + cache_key) > 0
    except Exception:
        return False


# ---------- 待确认 SQL ----------

async def get_pending_sql(thread_id: str) -> Optional[str]:
    """获取该会话的待确认 SQL。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            return _memory_pending_sql.get(thread_id)
    try:
        return await redis.get(_PREFIX_PENDING_SQL + thread_id)
    except Exception:
        return None


async def set_pending_sql(thread_id: str, sql: str) -> None:
    """设置该会话的待确认 SQL。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            _memory_pending_sql[thread_id] = sql
        return
    try:
        await redis.set(_PREFIX_PENDING_SQL + thread_id, sql, ex=_TTL_PENDING)
    except Exception as e:
        logger.warning("set_pending_sql Redis 写失败: %s", e)


async def pop_pending_sql(thread_id: str) -> Optional[str]:
    """取出并删除该会话的待确认 SQL。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            return _memory_pending_sql.pop(thread_id, None)
    try:
        key = _PREFIX_PENDING_SQL + thread_id
        sql = await redis.get(key)
        if sql is not None:
            await redis.delete(key)
        return sql
    except Exception:
        return None


# ---------- 人工中断会话 ----------

async def is_interrupted(thread_id: str) -> bool:
    """该会话是否处于人工介入中断状态。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            return thread_id in _memory_interrupted
    try:
        return await redis.sismember(_KEY_INTERRUPTED, thread_id)
    except Exception:
        return False


async def add_interrupted(thread_id: str) -> None:
    """标记会话为人工介入中断。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            _memory_interrupted.add(thread_id)
        return
    try:
        await redis.sadd(_KEY_INTERRUPTED, thread_id)
    except Exception as e:
        logger.warning("add_interrupted Redis 写失败: %s", e)


async def discard_interrupted(thread_id: str) -> None:
    """取消会话的人工介入中断标记。"""
    redis = await _get_shared_redis()
    if redis is None:
        async with _memory_lock:
            _memory_interrupted.discard(thread_id)
        return
    try:
        await redis.srem(_KEY_INTERRUPTED, thread_id)
    except Exception:
        pass
