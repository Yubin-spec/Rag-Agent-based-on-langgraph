"""
用户问答缓存：将「问题 → 答案」存入 Redis，后续相同/归一化后相似问题直接命中缓存返回。

优化要点：
- 连接池：多协程共享连接池，可配置 max_connections、超时、健康检查间隔。
- 断线重连：连接/读写失败时置空客户端，下次调用时懒加载重建。
- 单条价值长度限制：超长答案不写入，避免占满内存。
- 生命周期：应用关闭时主动关闭连接池，避免连接泄漏。
"""
import asyncio
import hashlib
import logging
import re
from typing import Any, Optional

from config import get_settings

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "kb:answer:v1:"
_MAX_QUESTION_LEN = 500

_redis: Any = None
_redis_lock = asyncio.Lock()


def _normalize_question(question: str) -> str:
    """归一化问题：去首尾空白、连续空白折叠为单空格、小写、截断至 _MAX_QUESTION_LEN，用于缓存 key 匹配。"""
    s = (question or "").strip()
    s = re.sub(r"\s+", " ", s)[:_MAX_QUESTION_LEN]
    return s.lower().strip()


def _cache_key(question: str) -> str:
    """对归一化后的问题做 SHA256，取前 32 位，加上前缀得到 Redis key（kb:answer:v1:...）。"""
    norm = _normalize_question(question)
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]
    return f"{_CACHE_KEY_PREFIX}{h}"


def _is_redis_connection_error(e: Exception) -> bool:
    """是否为连接/超时类错误，需要置空客户端以便下次重连。"""
    if e is None:
        return False
    name = type(e).__name__
    msg = (str(e) or "").lower()
    if name in ("ConnectionError", "TimeoutError", "OSError"):
        return True
    if "redis" in name or "connection" in msg or "timeout" in msg or "closed" in msg:
        return True
    return False


async def _get_redis():
    """
    懒加载全局 Redis 连接（redis.asyncio + 连接池）。
    answer_cache_enabled 为 False 或 redis_url 为空时返回 None。
    使用 Lock 保证并发下只建一次池。
    """
    global _redis
    if _redis is not None:
        return _redis
    settings = get_settings()
    if not getattr(settings, "answer_cache_enabled", True) or not getattr(settings, "redis_url", "").strip():
        return None
    async with _redis_lock:
        if _redis is not None:
            return _redis
        try:
            from redis.asyncio import Redis
            _redis = Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=getattr(settings, "redis_socket_connect_timeout", 2.0),
                socket_timeout=getattr(settings, "redis_socket_timeout", 5.0),
                health_check_interval=getattr(settings, "redis_health_check_interval", 30) or 0,
                max_connections=getattr(settings, "redis_max_connections", 10),
            )
            return _redis
        except Exception as e:
            logger.warning("Redis 连接失败，问答缓存将不可用: %s", e)
            return None


def _invalidate_redis() -> None:
    """置空全局客户端，下次调用时将重新建连（用于断线后重连）。"""
    global _redis
    _redis = None


async def close_redis_connection() -> None:
    """关闭 Redis 连接池，供应用 lifespan 退出时调用，避免连接泄漏。"""
    global _redis
    async with _redis_lock:
        if _redis is not None:
            try:
                await _redis.aclose()
            except Exception as e:
                logger.debug("关闭 Redis 连接时异常: %s", e)
            _redis = None


async def get_cached_answer(question: str) -> Optional[str]:
    """
    根据归一化问题查缓存，命中则返回答案，否则返回 None。
    Redis 不可用或断线时返回 None，不抛错；断线时会置空客户端以便下次重连。
    """
    if not (question or "").strip():
        return None
    client = await _get_redis()
    if client is None:
        return None
    try:
        key = _cache_key(question)
        value = await client.get(key)
        return value
    except Exception as e:
        if _is_redis_connection_error(e):
            _invalidate_redis()
            logger.warning("Redis 读缓存失败（已置空客户端以便重连）: %s", e)
        else:
            logger.debug("Redis get 异常: %s", e)
        return None


async def set_cached_answer(question: str, answer: str) -> None:
    """
    将问题→答案写入 Redis，TTL 使用配置的 answer_cache_ttl_seconds。
    若配置了 answer_cache_max_value_bytes 且答案超长则跳过写入。
    Redis 不可用或断线时不写入、不抛错；断线时会置空客户端以便下次重连。
    """
    if not (question or "").strip() or not (answer or "").strip():
        return
    settings = get_settings()
    max_bytes = max(0, getattr(settings, "answer_cache_max_value_bytes", 0))
    if max_bytes > 0:
        answer_bytes = answer.encode("utf-8")
        if len(answer_bytes) > max_bytes:
            logger.debug("答案长度 %s 超过 answer_cache_max_value_bytes=%s，跳过写入缓存", len(answer_bytes), max_bytes)
            return
    client = await _get_redis()
    if client is None:
        return
    try:
        ttl = max(60, getattr(settings, "answer_cache_ttl_seconds", 86400))
        key = _cache_key(question)
        await client.set(key, answer, ex=ttl)
    except Exception as e:
        if _is_redis_connection_error(e):
            _invalidate_redis()
            logger.warning("Redis 写缓存失败（已置空客户端以便重连）: %s", e)
        else:
            logger.debug("Redis set 异常: %s", e)
