# src/answer_cache.py
"""
用户问答缓存：将「问题 → 答案」存入 Redis，后续相同/相似问题直接命中缓存返回。
问题经归一化（去空格、小写、截长）后做 key，便于匹配历史问题。
"""
import hashlib
import re
from typing import Any, Optional

from config import get_settings

_CACHE_KEY_PREFIX = "kb:answer:v1:"
_MAX_QUESTION_LEN = 500
_redis: Any = None  # redis.asyncio.Redis when available


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


def _get_redis():
    """懒加载全局 Redis 连接（redis.asyncio）；answer_cache_enabled 为 False 或 redis_url 为空时返回 None。"""
    global _redis
    if _redis is not None:
        return _redis
    settings = get_settings()
    if not getattr(settings, "answer_cache_enabled", True) or not getattr(settings, "redis_url", "").strip():
        return None
    try:
        from redis.asyncio import Redis
        _redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
        return _redis
    except Exception:
        return None


async def get_cached_answer(question: str) -> Optional[str]:
    """
    根据归一化问题查缓存，命中则返回答案，否则返回 None。
    Redis 不可用时返回 None，不抛错。
    """
    if not (question or "").strip():
        return None
    client = _get_redis()
    if client is None:
        return None
    try:
        key = _cache_key(question)
        value = await client.get(key)
        return value
    except Exception:
        return None


async def set_cached_answer(question: str, answer: str) -> None:
    """
    将问题→答案写入 Redis，TTL 使用配置的 answer_cache_ttl_seconds。
    Redis 不可用或 answer 为空时不写入，不抛错。
    """
    if not (question or "").strip() or not (answer or "").strip():
        return
    client = _get_redis()
    if client is None:
        return
    try:
        settings = get_settings()
        ttl = max(60, getattr(settings, "answer_cache_ttl_seconds", 86400))
        key = _cache_key(question)
        await client.set(key, answer, ex=ttl)
    except Exception:
        pass
