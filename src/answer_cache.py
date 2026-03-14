"""
用户问答缓存：将「问题 → 答案」存入 Redis，后续相同/归一化后相似问题直接命中缓存返回。

优化要点：
- 连接池：多协程共享连接池，可配置 max_connections、超时、健康检查间隔。
- 断线重连：连接/读写失败时置空客户端，下次调用时懒加载重建。
- 单条价值长度限制：超长答案不写入，避免占满内存。
- 生命周期：应用关闭时主动关闭连接池，避免连接泄漏。
- 热 key：可选进程内 LRU 本地缓存，命中则不再打 Redis，降低热 key 压力。
- 防击穿：同一问题缓存未命中时，仅一个协程回源计算，其余等待后复用结果（单飞锁）。
"""
import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from config import get_settings

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "kb:answer:v1:"
_MAX_QUESTION_LEN = 500

_redis: Any = None
_redis_lock = asyncio.Lock()

# 本地热 key 缓存：key -> (value, expiry_ts)，按访问顺序做 LRU，容量与 TTL 可配置
_local_cache: Optional[OrderedDict] = None
_local_cache_lock = asyncio.Lock()

# 防击穿：按 key 分桶的锁，同一问题只有一个协程回源，其余等待后读缓存
_single_flight_locks: List[asyncio.Lock] = []
_single_flight_initialized = False


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


def _ensure_single_flight_locks() -> List[asyncio.Lock]:
    """按配置初始化防击穿分桶锁（仅首次调用时创建）。"""
    global _single_flight_locks, _single_flight_initialized
    if _single_flight_initialized:
        return _single_flight_locks
    n = max(1, min(4096, getattr(get_settings(), "answer_cache_single_flight_buckets", 256)))
    _single_flight_locks = [asyncio.Lock() for _ in range(n)]
    _single_flight_initialized = True
    return _single_flight_locks


def _get_single_flight_lock(question: str) -> asyncio.Lock:
    """根据问题得到对应的单飞锁（同一问题同一把锁，不同问题可能同桶）。"""
    key = _cache_key(question)
    locks = _ensure_single_flight_locks()
    return locks[hash(key) % len(locks)]


async def _get_local_cache() -> Optional[OrderedDict]:
    """若启用本地热 key 缓存，返回 LRU OrderedDict（key -> (value, expiry_ts)），否则返回 None。"""
    global _local_cache
    settings = get_settings()
    max_entries = max(0, getattr(settings, "answer_cache_local_max_entries", 0))
    if max_entries <= 0:
        return None
    async with _local_cache_lock:
        if _local_cache is None:
            _local_cache = OrderedDict()
        return _local_cache


def _local_cache_ttl_seconds() -> int:
    return max(1, getattr(get_settings(), "answer_cache_local_ttl_seconds", 60))


def _local_cache_max_entries() -> int:
    return max(0, getattr(get_settings(), "answer_cache_local_max_entries", 0))


async def _local_get(key: str) -> Optional[str]:
    """从本地缓存读取，过期或不存在返回 None。"""
    cache = await _get_local_cache()
    if cache is None:
        return None
    async with _local_cache_lock:
        entry = cache.get(key)
        if entry is None:
            return None
        value, expiry_ts = entry
        if time.time() > expiry_ts:
            cache.pop(key, None)
            return None
        cache.move_to_end(key)  # LRU
        return value


async def _local_set(key: str, value: str) -> None:
    """写入本地缓存，并做容量与过期清理。"""
    cache = await _get_local_cache()
    if cache is None:
        return
    ttl = _local_cache_ttl_seconds()
    max_entries = _local_cache_max_entries()
    async with _local_cache_lock:
        now = time.time()
        # 先删过期
        to_del = [k for k, (_, exp) in cache.items() if exp < now]
        for k in to_del:
            cache.pop(k, None)
        cache[key] = (value, now + ttl)
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)


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


def _parse_sentinel_nodes(nodes_str: str):
    """解析 redis_sentinel_nodes 字符串为 [(host, port), ...]。"""
    out = []
    for part in (nodes_str or "").strip().split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            host, _, port_str = part.rpartition(":")
            try:
                out.append((host.strip(), int(port_str.strip())))
            except ValueError:
                continue
        else:
            out.append((part, 26379))
    return out


async def _get_redis():
    """
    懒加载全局 Redis 连接（redis.asyncio + 连接池）。
    当配置 redis_sentinel_service_name 与 redis_sentinel_nodes 时，通过 Sentinel 获取 master；
    否则使用 redis_url 直连。
    answer_cache_enabled 为 False 且未配 Sentinel 时返回 None。
    """
    global _redis
    if _redis is not None:
        return _redis
    settings = get_settings()
    if not getattr(settings, "answer_cache_enabled", True):
        return None
    sentinel_name = (getattr(settings, "redis_sentinel_service_name", None) or "").strip()
    sentinel_nodes_str = (getattr(settings, "redis_sentinel_nodes", None) or "").strip()
    use_sentinel = sentinel_name and sentinel_nodes_str
    if not use_sentinel and not (getattr(settings, "redis_url", "") or "").strip():
        return None
    async with _redis_lock:
        if _redis is not None:
            return _redis
        try:
            from redis.asyncio import Redis
            if use_sentinel:
                from redis.asyncio.sentinel import Sentinel
                nodes = _parse_sentinel_nodes(sentinel_nodes_str)
                if not nodes:
                    raise ValueError("redis_sentinel_nodes 解析后为空")
                sentinel = Sentinel(
                    nodes,
                    socket_timeout=getattr(settings, "redis_socket_connect_timeout", 2.0),
                )
                _redis = sentinel.master_for(
                    sentinel_name,
                    decode_responses=True,
                    socket_timeout=getattr(settings, "redis_socket_timeout", 5.0),
                )
                logger.info("Redis 已通过 Sentinel 连接: service=%s", sentinel_name)
            else:
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
            logger.exception("Redis 连接失败，问答缓存将不可用: %s", e)
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


async def redis_ping() -> bool:
    """探活：若 Redis 已连接则执行 PING，返回是否成功。未配置或未连接时返回 False。"""
    client = await _get_redis()
    if client is None:
        return False
    try:
        await client.ping()
        return True
    except Exception:
        return False


async def get_cached_answer(question: str) -> Optional[str]:
    """
    根据归一化问题查缓存，命中则返回答案，否则返回 None。
    若启用本地热 key 缓存，先查本地再查 Redis；本地命中可减轻 Redis 热 key 压力。
    Redis 不可用或断线时返回 None，不抛错；断线时会置空客户端以便下次重连。
    """
    if not (question or "").strip():
        return None
    key = _cache_key(question)
    local_val = await _local_get(key)
    if local_val is not None:
        return local_val
    client = await _get_redis()
    if client is None:
        return None
    try:
        value = await client.get(key)
        if value is not None and _local_cache_max_entries() > 0:
            await _local_set(key, value)
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
    将问题→答案写入 Redis（及可选本地热 key 缓存），TTL 使用配置的 answer_cache_ttl_seconds。
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
    key = _cache_key(question)
    if _local_cache_max_entries() > 0:
        await _local_set(key, answer)
    client = await _get_redis()
    if client is None:
        return
    try:
        ttl = max(60, getattr(settings, "answer_cache_ttl_seconds", 86400))
        await client.set(key, answer, ex=ttl)
    except Exception as e:
        if _is_redis_connection_error(e):
            _invalidate_redis()
            logger.warning("Redis 写缓存失败（已置空客户端以便重连）: %s", e)
        else:
            logger.debug("Redis set 异常: %s", e)


@asynccontextmanager
async def answer_lock(question: str):
    """
    防击穿：同一问题缓存未命中时，仅持锁的协程回源计算并写缓存，其余协程等待后再次读缓存。
    用法：在 get_cached_answer 未命中后，用 async with answer_lock(question): 包裹「再查一次缓存 + 计算 + set_cached_answer」。
    """
    lock = _get_single_flight_lock(question)
    async with lock:
        yield
