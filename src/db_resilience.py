# src/db_resilience.py
"""
数据库连接韧性层：连接池复用、重试、熔断、高并发限流、降级。

覆盖范围：
- PostgreSQL / SQLite（SQLAlchemy 同步 + asyncpg 异步）
- Milvus 向量数据库（pymilvus）

设计目标：
- 所有数据库连接统一走此模块，避免各处散落的连接创建。
- 连接失败时自动重试（指数退避），连续失败后熔断（短时拒绝，避免雪崩）。
- 熔断期间对非关键路径返回降级结果而非抛异常。
- Milvus 支持懒重连：连接断开后下次操作自动尝试重连。
"""
import asyncio
import logging
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Optional

from config import get_settings

logger = logging.getLogger(__name__)


def _db_config():
    """从配置读取连接池与熔断参数。"""
    s = get_settings()
    return {
        "pool_size": max(1, getattr(s, "postgresql_pool_size", 5)),
        "max_overflow": max(0, getattr(s, "postgresql_max_overflow", 10)),
        "pool_timeout": max(1, getattr(s, "postgresql_pool_timeout", 10)),
        "cb_threshold": max(1, getattr(s, "db_circuit_breaker_threshold", 5)),
        "cb_recovery": max(5.0, float(getattr(s, "db_circuit_breaker_recovery_seconds", 30))),
        "statement_timeout_ms": max(0, getattr(s, "postgresql_statement_timeout_ms", 0)),
    }


# ---------- 熔断器 ----------

class CircuitBreaker:
    """
    简易熔断器：连续失败达到阈值后进入 OPEN 状态，拒绝请求；
    经过冷却期后进入 HALF_OPEN，放行一次探测；探测成功则恢复 CLOSED。
    """

    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 30.0, name: str = ""):
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._name = name or "db"
        self._failures = 0
        self._last_failure_time = 0.0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "OPEN":
                if time.monotonic() - self._last_failure_time >= self._recovery_seconds:
                    self._state = "HALF_OPEN"
            return self._state

    def allow_request(self) -> bool:
        s = self.state
        return s in ("CLOSED", "HALF_OPEN")

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state != "CLOSED":
                logger.info("[%s] 熔断器恢复 CLOSED", self._name)
            self._state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= self._failure_threshold and self._state == "CLOSED":
                self._state = "OPEN"
                logger.warning(
                    "[%s] 连续 %d 次失败，熔断器 OPEN（%ds 后半开探测）",
                    self._name, self._failures, int(self._recovery_seconds),
                )
            elif self._state == "HALF_OPEN":
                self._state = "OPEN"
                logger.warning("[%s] 半开探测失败，熔断器重新 OPEN", self._name)

    def status(self) -> dict:
        return {
            "name": self._name,
            "state": self.state,
            "failures": self._failures,
            "threshold": self._failure_threshold,
            "recovery_seconds": self._recovery_seconds,
        }


# ---------- 同步连接池管理 ----------

_engines: Dict[str, Any] = {}
_breakers: Dict[str, CircuitBreaker] = {}
_engine_lock = threading.Lock()


def _get_breaker(uri: str) -> CircuitBreaker:
    cfg = _db_config()
    key = (uri or "").strip()
    if key not in _breakers:
        with _engine_lock:
            if key not in _breakers:
                _breakers[key] = CircuitBreaker(
                    failure_threshold=cfg["cb_threshold"],
                    recovery_seconds=cfg["cb_recovery"],
                    name=f"db:{key[:40]}",
                )
    return _breakers[key]


def get_engine(
    uri: str,
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 10,
    pool_recycle: int = 1800,
):
    """
    按 URI 复用 SQLAlchemy Engine。

    参数：
    - pool_size: 连接池常驻连接数。
    - max_overflow: 超出 pool_size 后允许的临时连接数。
    - pool_timeout: 从池中获取连接的最大等待秒数（高并发时排队上限）。
    - pool_recycle: 连接回收周期（秒），避免数据库侧超时断连。
    """
    key = (uri or "").strip()
    if not key:
        raise ValueError("database URI is empty")
    if key not in _engines:
        with _engine_lock:
            if key not in _engines:
                from sqlalchemy import create_engine
                is_sqlite = key.startswith("sqlite")
                kwargs = {"pool_pre_ping": True}
                if not is_sqlite:
                    kwargs.update(
                        pool_size=pool_size,
                        max_overflow=max_overflow,
                        pool_timeout=pool_timeout,
                        pool_recycle=pool_recycle,
                    )
                    cfg = _db_config()
                    if cfg["statement_timeout_ms"] > 0:
                        kwargs["connect_args"] = {
                            "options": f"-c statement_timeout={cfg['statement_timeout_ms']}",
                        }
                _engines[key] = create_engine(key, **kwargs)
                logger.info("创建 DB Engine: %s (pool=%d, overflow=%d)", key[:60], pool_size, max_overflow)
    return _engines[key]


def dispose_all_engines() -> None:
    """释放所有缓存的 Engine（应用关闭时调用）。"""
    with _engine_lock:
        for uri, eng in list(_engines.items()):
            try:
                eng.dispose()
            except Exception as e:
                logger.debug("dispose engine %s: %s", uri[:50], e)
        _engines.clear()


@contextmanager
def safe_connection(
    uri: str,
    *,
    retries: int = 2,
    retry_delay: float = 0.5,
    critical: bool = True,
    pool_size: Optional[int] = None,
    max_overflow: Optional[int] = None,
    pool_timeout: Optional[int] = None,
):
    """
    安全获取数据库连接的上下文管理器，内置重试 + 熔断。
    连接池参数未显式传入时从配置读取（postgresql_pool_size 等）。
    """
    breaker = _get_breaker(uri)

    if not breaker.allow_request():
        if critical:
            raise ConnectionError(f"数据库熔断中（{breaker.status()['state']}），暂时拒绝连接")
        logger.warning("数据库熔断中，降级处理")
        yield None
        return

    cfg = _db_config()
    ps = pool_size if pool_size is not None else cfg["pool_size"]
    mo = max_overflow if max_overflow is not None else cfg["max_overflow"]
    pt = pool_timeout if pool_timeout is not None else cfg["pool_timeout"]
    engine = get_engine(uri, pool_size=ps, max_overflow=mo, pool_timeout=pt)
    last_err = None

    for attempt in range(1, retries + 2):
        try:
            with engine.connect() as conn:
                breaker.record_success()
                yield conn
                return
        except Exception as e:
            last_err = e
            breaker.record_failure()
            if attempt <= retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "数据库连接失败（第 %d/%d 次），%.1fs 后重试: %s",
                    attempt, retries + 1, delay, e,
                )
                time.sleep(delay)

    if critical:
        raise ConnectionError(f"数据库连接失败（已重试 {retries} 次）: {last_err}") from last_err
    logger.error("数据库连接失败（已重试），降级处理: %s", last_err)
    yield None


# ---------- 异步连接池管理（asyncpg） ----------

_async_pools: Dict[str, Any] = {}
_async_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None


async def get_async_pool(
    uri: str,
    *,
    min_size: int = 2,
    max_size: int = 10,
    command_timeout: int = 15,
    retries: int = 2,
    retry_delay: float = 1.0,
):
    """
    按 URI 复用 asyncpg 连接池，创建失败时重试。

    参数：
    - min_size / max_size: 池大小范围。
    - command_timeout: 单条 SQL 超时。
    - retries: 创建池失败时重试次数。
    """
    key = (uri or "").strip()
    if not key:
        raise ValueError("database URI is empty")
    if key in _async_pools:
        pool = _async_pools[key]
        if not pool._closed:
            return pool

    lock = _async_lock or asyncio.Lock()
    async with lock:
        if key in _async_pools:
            pool = _async_pools[key]
            if not pool._closed:
                return pool

        import asyncpg
        last_err = None
        for attempt in range(1, retries + 2):
            try:
                pool = await asyncpg.create_pool(
                    key,
                    min_size=min_size,
                    max_size=max_size,
                    command_timeout=command_timeout,
                )
                _async_pools[key] = pool
                logger.info("创建 asyncpg 池: %s (min=%d, max=%d)", key[:60], min_size, max_size)
                return pool
            except Exception as e:
                last_err = e
                if attempt <= retries:
                    delay = retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "asyncpg 池创建失败（第 %d/%d 次），%.1fs 后重试: %s",
                        attempt, retries + 1, delay, e,
                    )
                    await asyncio.sleep(delay)
        raise ConnectionError(f"asyncpg 池创建失败（已重试 {retries} 次）: {last_err}") from last_err


@asynccontextmanager
async def safe_async_connection(
    uri: str,
    *,
    critical: bool = True,
    retries: int = 2,
    retry_delay: float = 0.5,
):
    """
    异步安全获取 asyncpg 连接，内置重试 + 熔断。

    critical=False 时熔断/失败后 yield None（降级）。
    """
    breaker = _get_breaker(uri)

    if not breaker.allow_request():
        if critical:
            raise ConnectionError(f"数据库熔断中（{breaker.status()['state']}），暂时拒绝连接")
        logger.warning("数据库熔断中（async），降级处理")
        yield None
        return

    last_err = None
    for attempt in range(1, retries + 2):
        try:
            pool = await get_async_pool(uri)
            async with pool.acquire() as conn:
                breaker.record_success()
                yield conn
                return
        except Exception as e:
            last_err = e
            breaker.record_failure()
            if attempt <= retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "异步数据库连接失败（第 %d/%d 次），%.1fs 后重试: %s",
                    attempt, retries + 1, delay, e,
                )
                await asyncio.sleep(delay)

    if critical:
        raise ConnectionError(f"异步数据库连接失败（已重试 {retries} 次）: {last_err}") from last_err
    logger.error("异步数据库连接失败（已重试），降级处理: %s", last_err)
    yield None


async def dispose_all_async_pools() -> None:
    """关闭所有 asyncpg 连接池。"""
    for key, pool in list(_async_pools.items()):
        try:
            await pool.close()
        except Exception as e:
            logger.debug("关闭 asyncpg 池 %s: %s", key[:50], e)
    _async_pools.clear()


# ---------- Milvus 韧性层 ----------

_milvus_collections: Dict[str, Any] = {}
_milvus_lock = threading.Lock()


def _get_milvus_breaker(uri: str) -> CircuitBreaker:
    cfg = _db_config()
    key = f"milvus:{(uri or '').strip()[:40]}"
    if key not in _breakers:
        with _engine_lock:
            if key not in _breakers:
                _breakers[key] = CircuitBreaker(
                    failure_threshold=min(3, cfg["cb_threshold"]),
                    recovery_seconds=cfg["cb_recovery"],
                    name=key,
                )
    return _breakers[key]


def _milvus_connect(uri: str) -> None:
    """建立 Milvus 连接（幂等，pymilvus 内部会复用）。"""
    from pymilvus import connections
    alias = "default"
    connections.connect(alias=alias, uri=uri)


def get_milvus_collection(
    uri: str,
    collection_name: str,
    *,
    retries: int = 2,
    retry_delay: float = 1.0,
    create_if_missing: bool = False,
    schema_builder=None,
):
    """
    获取 Milvus Collection，内置重试 + 熔断 + 懒重连。

    参数：
    - uri: Milvus 连接地址。
    - collection_name: 目标 collection 名称。
    - retries: 连接失败时重试次数。
    - create_if_missing: True 时若 collection 不存在则通过 schema_builder 创建。
    - schema_builder: 可调用对象，签名 schema_builder(collection_name) -> Collection，
                      仅在 create_if_missing=True 且 collection 不存在时调用。

    返回 Collection 对象，熔断或失败时返回 None。
    """
    breaker = _get_milvus_breaker(uri)
    cache_key = f"{uri}|{collection_name}"

    if not breaker.allow_request():
        logger.warning("Milvus 熔断中，跳过连接: %s/%s", uri[:40], collection_name)
        return None

    cached = _milvus_collections.get(cache_key)
    if cached is not None:
        try:
            cached.num_entities  # lightweight ping
            breaker.record_success()
            return cached
        except Exception:
            with _milvus_lock:
                _milvus_collections.pop(cache_key, None)

    last_err = None
    for attempt in range(1, retries + 2):
        try:
            _milvus_connect(uri)
            from pymilvus import Collection, utility
            if utility.has_collection(collection_name):
                coll = Collection(collection_name)
                coll.load()
            elif create_if_missing and schema_builder:
                coll = schema_builder(collection_name)
            else:
                logger.warning("Milvus collection '%s' 不存在", collection_name)
                breaker.record_success()
                return None

            with _milvus_lock:
                _milvus_collections[cache_key] = coll
            breaker.record_success()
            logger.info("Milvus 连接成功: %s/%s", uri[:40], collection_name)
            return coll
        except Exception as e:
            last_err = e
            breaker.record_failure()
            if attempt <= retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Milvus 连接失败（第 %d/%d 次），%.1fs 后重试: %s",
                    attempt, retries + 1, delay, e,
                )
                time.sleep(delay)

    logger.error("Milvus 连接失败（已重试 %d 次）: %s", retries, last_err)
    return None


def milvus_ping(uri: str) -> bool:
    """探活：尝试连接 Milvus 并执行轻量操作，不触发熔断。"""
    try:
        _milvus_connect(uri)
        from pymilvus import utility
        utility.list_collections()  # 轻量操作
        return True
    except Exception:
        return False


def invalidate_milvus_collection(uri: str, collection_name: str) -> None:
    """使缓存的 Collection 失效，下次 get 时会重连。"""
    cache_key = f"{uri}|{collection_name}"
    with _milvus_lock:
        _milvus_collections.pop(cache_key, None)


def milvus_operation_with_retry(
    uri: str,
    collection_name: str,
    operation,
    *,
    retries: int = 2,
    retry_delay: float = 0.5,
    critical: bool = False,
    default=None,
):
    """
    执行 Milvus 操作（search / insert / flush 等），失败时自动重连重试。

    参数：
    - operation: 可调用对象，签名 operation(collection) -> result。
    - critical: True 时失败抛异常；False 时返回 default。
    - default: 降级时的默认返回值。

    用法：
        results = milvus_operation_with_retry(
            uri, coll_name,
            lambda coll: coll.search(...),
            default=[],
        )
    """
    breaker = _get_milvus_breaker(uri)

    if not breaker.allow_request():
        if critical:
            raise ConnectionError(f"Milvus 熔断中: {uri[:40]}/{collection_name}")
        logger.warning("Milvus 熔断中，降级处理: %s/%s", uri[:40], collection_name)
        return default

    last_err = None
    for attempt in range(1, retries + 2):
        coll = get_milvus_collection(uri, collection_name)
        if coll is None:
            if critical:
                raise ConnectionError(f"Milvus collection 不可用: {collection_name}")
            return default
        try:
            result = operation(coll)
            breaker.record_success()
            return result
        except Exception as e:
            last_err = e
            breaker.record_failure()
            invalidate_milvus_collection(uri, collection_name)
            if attempt <= retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Milvus 操作失败（第 %d/%d 次），%.1fs 后重试: %s",
                    attempt, retries + 1, delay, e,
                )
                time.sleep(delay)

    if critical:
        raise ConnectionError(f"Milvus 操作失败（已重试 {retries} 次）: {last_err}") from last_err
    logger.error("Milvus 操作失败（已重试），降级处理: %s", last_err)
    return default


# ---------- 状态查询 ----------

def get_all_breaker_status() -> list:
    """返回所有熔断器状态（含 PostgreSQL 和 Milvus），供健康检查 API 使用。"""
    return [b.status() for b in _breakers.values()]
