# src/chat_history.py
"""
对话长期记忆：将多轮对话历史与运行时状态写入 PostgreSQL，进程重启后可恢复。

- `chat_history`：按 thread_id 追加存储对话消息；
- `chat_runtime_state`：存储待确认 SQL、人工介入暂停态等流程状态；
- 请求开始时若内存无该会话状态则从 DB 加载并注入；
- 每轮结束后将本轮 user+assistant 追加写入 DB，并同步更新运行时状态。
- 连接池：按 uri 复用 Engine，避免每次 create_engine/dispose，提升并发下 DB 访问性能。
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

_TABLE_NAME = "chat_history"
_STATE_TABLE_NAME = "chat_runtime_state"
_SESSION_TABLE_NAME = "chat_sessions"

_engines: Dict[str, Any] = {}


def _get_engine(uri: str):
    """按 uri 复用 SQLAlchemy Engine，避免每次建连与销毁。"""
    global _engines
    key = (uri or "").strip()
    if not key:
        raise ValueError("chat_history uri is empty")
    if key not in _engines:
        from sqlalchemy import create_engine
        _engines[key] = create_engine(key, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return _engines[key]


def dispose_engines() -> None:
    """释放所有缓存的 Engine（应用关闭时调用，避免连接泄漏）。"""
    global _engines
    for uri, eng in list(_engines.items()):
        try:
            eng.dispose()
        except Exception as e:
            logger.debug("dispose chat_history engine %s: %s", uri[:50], e)
    _engines.clear()


def _ensure_table(uri: str) -> None:
    """创建长期记忆所需的 PostgreSQL 表（若不存在）。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id         SERIAL PRIMARY KEY,
                thread_id  VARCHAR(512) NOT NULL,
                role       VARCHAR(32)  NOT NULL,
                content    TEXT         NOT NULL,
                created_at TIMESTAMPTZ  DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chat_history_thread_id ON chat_history(thread_id)
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_runtime_state (
                thread_id   VARCHAR(512) PRIMARY KEY,
                pending_sql TEXT,
                interrupted BOOLEAN      NOT NULL DEFAULT FALSE,
                updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_SESSION_TABLE_NAME} (
                thread_id       VARCHAR(512) PRIMARY KEY,
                conversation_id VARCHAR(128) NOT NULL,
                user_id         VARCHAR(255),
                title           VARCHAR(255) NOT NULL,
                title_manual    BOOLEAN      NOT NULL DEFAULT FALSE,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.execute(text(f"""
            ALTER TABLE {_SESSION_TABLE_NAME}
            ADD COLUMN IF NOT EXISTS title_manual BOOLEAN NOT NULL DEFAULT FALSE
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{_SESSION_TABLE_NAME}_user_updated
            ON {_SESSION_TABLE_NAME}(user_id, updated_at DESC)
        """))
        conn.commit()


def load_messages(uri: str, thread_id: str, limit: int = 0) -> List[BaseMessage]:
    """
    从 PostgreSQL 按 thread_id 读取历史消息，按 id 升序，转为 HumanMessage/AIMessage 列表。
    limit > 0 时只取最近 limit 条（按 id 升序），用于减少长会话加载量。
    """
    from sqlalchemy import text
    engine = _get_engine(uri)
    out: List[BaseMessage] = []
    try:
        with engine.connect() as conn:
            if limit > 0:
                rows = conn.execute(
                    text("SELECT role, content FROM chat_history WHERE thread_id = :tid ORDER BY id DESC LIMIT :lim"),
                    {"tid": thread_id[:512], "lim": limit},
                ).fetchall()
                rows = list(reversed(rows))  # 恢复为按 id 升序（时间顺序）
            else:
                rows = conn.execute(
                    text("SELECT role, content FROM chat_history WHERE thread_id = :tid ORDER BY id"),
                    {"tid": thread_id[:512]},
                ).fetchall()
        for role, content in rows:
            content = (content or "").strip()
            if role == "user" or role == "human":
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
    except Exception as e:
        logger.warning("加载对话历史失败: %s", e)
    return out


def append_messages(uri: str, thread_id: str, messages: List[BaseMessage]) -> None:
    """
    将本轮的若干条消息追加写入 chat_history（INSERT），不覆盖已有记录。
    若配置了 chat_history_max_content_chars > 0，单条 content 超过则截断后写入。
    """
    if not messages:
        return
    from sqlalchemy import text
    from config import get_settings
    engine = _get_engine(uri)
    max_chars = max(0, getattr(get_settings(), "chat_history_max_content_chars", 0))
    cap = min(65535, max_chars) if max_chars else 65535  # DB 兼容
    payloads: list[tuple[str, str]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            continue
        content = (m.content if hasattr(m, "content") else str(m)) or ""
        if cap < 65535 and len(content) > cap:
            content = content[:cap].rstrip() + "…"
        payloads.append((role, content[:65535]))
    if not payloads:
        return

    try:
        with engine.connect() as conn:
            # 最近一轮重复写入抑制：若最后 N 条消息与当前待写入内容完全一致，则认为是重试/重复提交，跳过写入。
            recent_rows = conn.execute(
                text(
                    "SELECT role, content FROM chat_history WHERE thread_id = :tid ORDER BY id DESC LIMIT :n"
                ),
                {"tid": thread_id[:512], "n": len(payloads)},
            ).fetchall()
            recent_payloads = list(reversed([(r[0], r[1]) for r in recent_rows]))
            if recent_payloads == payloads:
                return
            for role, content in payloads:
                conn.execute(
                    text(
                        "INSERT INTO chat_history (thread_id, role, content) VALUES (:tid, :role, :content)"
                    ),
                    {"tid": thread_id[:512], "role": role, "content": content},
                )
            conn.commit()
    except Exception as e:
        logger.warning("写入对话历史失败: %s", e)


def load_runtime_state(uri: str, thread_id: str) -> dict:
    """加载会话运行时状态：待确认 SQL、人工介入暂停态。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT pending_sql, interrupted FROM {_STATE_TABLE_NAME} WHERE thread_id = :tid"
                ),
                {"tid": thread_id[:512]},
            ).fetchone()
        if not row:
            return {"pending_sql": None, "interrupted": False}
        return {
            "pending_sql": row[0],
            "interrupted": bool(row[1]),
        }
    except Exception as e:
        logger.warning("加载会话运行时状态失败: %s", e)
        return {"pending_sql": None, "interrupted": False}


def save_runtime_state(
    uri: str,
    thread_id: str,
    pending_sql: Optional[str],
    interrupted: bool,
) -> None:
    """保存会话运行时状态，确保服务重启后仍可恢复待确认 SQL 与 interrupt 状态。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    try:
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {_STATE_TABLE_NAME} (thread_id, pending_sql, interrupted, updated_at)
                    VALUES (:tid, :pending_sql, :interrupted, NOW())
                    ON CONFLICT (thread_id)
                    DO UPDATE SET
                        pending_sql = EXCLUDED.pending_sql,
                        interrupted = EXCLUDED.interrupted,
                        updated_at = NOW()
                """),
                {
                    "tid": thread_id[:512],
                    "pending_sql": pending_sql,
                    "interrupted": interrupted,
                },
            )
            conn.commit()
    except Exception as e:
        logger.warning("保存会话运行时状态失败: %s", e)


def upsert_conversation_session(
    uri: str,
    thread_id: str,
    conversation_id: str,
    user_id: Optional[str],
    title: str,
) -> None:
    """创建或更新会话元数据，供前端左侧会话列表展示。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    safe_title = (title or "新对话").strip()[:255] or "新对话"
    try:
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {_SESSION_TABLE_NAME} (thread_id, conversation_id, user_id, title, updated_at)
                    VALUES (:thread_id, :conversation_id, :user_id, :title, NOW())
                    ON CONFLICT (thread_id)
                    DO UPDATE SET
                        conversation_id = EXCLUDED.conversation_id,
                        user_id = EXCLUDED.user_id,
                        title = CASE
                            WHEN {_SESSION_TABLE_NAME}.title_manual = TRUE
                                THEN {_SESSION_TABLE_NAME}.title
                            WHEN {_SESSION_TABLE_NAME}.title IS NULL OR {_SESSION_TABLE_NAME}.title = '' OR {_SESSION_TABLE_NAME}.title = '新对话'
                                THEN EXCLUDED.title
                            ELSE {_SESSION_TABLE_NAME}.title
                        END,
                        updated_at = NOW()
                """),
                {
                    "thread_id": thread_id[:512],
                    "conversation_id": conversation_id[:128],
                    "user_id": user_id,
                    "title": safe_title,
                },
            )
            conn.commit()
    except Exception as e:
        logger.warning("保存会话元数据失败: %s", e)


def list_conversation_sessions(uri: str, user_id: Optional[str]) -> List[dict]:
    """按更新时间倒序列出某用户的会话列表。未传 user_id 时仅返回匿名会话。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    out: List[dict] = []
    try:
        with engine.connect() as conn:
            if user_id is None:
                rows = conn.execute(
                    text(f"""
                        SELECT conversation_id, title, updated_at
                        FROM {_SESSION_TABLE_NAME}
                        WHERE user_id IS NULL
                        ORDER BY updated_at DESC
                    """)
                ).fetchall()
            else:
                rows = conn.execute(
                    text(f"""
                        SELECT conversation_id, title, updated_at
                        FROM {_SESSION_TABLE_NAME}
                        WHERE user_id = :user_id
                        ORDER BY updated_at DESC
                    """),
                    {"user_id": user_id},
                ).fetchall()
        for row in rows:
            out.append(
                {
                    "conversation_id": row[0],
                    "title": row[1] or "新对话",
                    "updated_at": row[2].isoformat() if row[2] is not None else "",
                }
            )
    except Exception as e:
        logger.warning("读取会话列表失败: %s", e)
    return out


def get_conversation_session(uri: str, thread_id: str) -> Optional[dict]:
    """按 thread_id 读取单个会话元数据。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f"""
                    SELECT conversation_id, user_id, title, updated_at
                    FROM {_SESSION_TABLE_NAME}
                    WHERE thread_id = :thread_id
                """),
                {"thread_id": thread_id[:512]},
            ).fetchone()
        if not row:
            return None
        return {
            "conversation_id": row[0],
            "user_id": row[1],
            "title": row[2],
            "updated_at": row[3].isoformat() if row[3] is not None else "",
        }
    except Exception as e:
        logger.warning("读取单个会话元数据失败: %s", e)
        return None


def rename_conversation_session(uri: str, thread_id: str, title: str) -> bool:
    """重命名会话标题。成功返回 True。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    safe_title = (title or "").strip()[:255]
    if not safe_title:
        return False
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                    UPDATE {_SESSION_TABLE_NAME}
                    SET title = :title, title_manual = TRUE, updated_at = NOW()
                    WHERE thread_id = :thread_id
                """),
                {"title": safe_title, "thread_id": thread_id[:512]},
            )
            conn.commit()
            return result.rowcount > 0
    except Exception as e:
        logger.warning("重命名会话失败: %s", e)
        return False


def delete_conversation_session(uri: str, thread_id: str) -> None:
    """删除某个会话的历史消息、运行时状态与会话元数据。"""
    from sqlalchemy import text
    engine = _get_engine(uri)
    try:
        with engine.connect() as conn:
            conn.execute(text(f"DELETE FROM {_TABLE_NAME} WHERE thread_id = :thread_id"), {"thread_id": thread_id[:512]})
            conn.execute(text(f"DELETE FROM {_STATE_TABLE_NAME} WHERE thread_id = :thread_id"), {"thread_id": thread_id[:512]})
            conn.execute(text(f"DELETE FROM {_SESSION_TABLE_NAME} WHERE thread_id = :thread_id"), {"thread_id": thread_id[:512]})
            conn.commit()
    except Exception as e:
        logger.warning("删除会话失败: %s", e)


# ---------- 异步长期记忆（asyncpg，可选） ----------
_async_pools: Dict[str, Any] = {}


async def _get_async_pool(uri: str):
    """按 uri 复用 asyncpg 连接池。"""
    global _async_pools
    key = (uri or "").strip()
    if not key:
        raise ValueError("chat_history uri is empty")
    if key not in _async_pools:
        try:
            import asyncpg
            _async_pools[key] = await asyncpg.create_pool(key, min_size=1, max_size=5, command_timeout=10)
        except Exception as e:
            logger.warning("asyncpg 连接池创建失败，将回退同步: %s", e)
            raise
    return _async_pools[key]


async def load_messages_async(uri: str, thread_id: str, limit: int = 0) -> List[BaseMessage]:
    """异步从 PostgreSQL 加载历史消息，语义同 load_messages。"""
    pool = await _get_async_pool(uri)
    out: List[BaseMessage] = []
    try:
        if limit > 0:
            rows = await pool.fetch(
                "SELECT role, content FROM chat_history WHERE thread_id = $1 ORDER BY id DESC LIMIT $2",
                thread_id[:512], limit,
            )
            rows = list(reversed(rows))
        else:
            rows = await pool.fetch(
                "SELECT role, content FROM chat_history WHERE thread_id = $1 ORDER BY id",
                thread_id[:512],
            )
        for row in rows:
            role, content = row["role"], (row["content"] or "").strip()
            if role in ("user", "human"):
                out.append(HumanMessage(content=content))
            else:
                out.append(AIMessage(content=content))
    except Exception as e:
        logger.warning("加载对话历史失败(async): %s", e)
    return out


async def append_messages_async(uri: str, thread_id: str, messages: List[BaseMessage]) -> None:
    """异步追加消息到 chat_history，语义同 append_messages（含 content 长度与重复抑制）。"""
    if not messages:
        return
    from config import get_settings
    max_chars = max(0, getattr(get_settings(), "chat_history_max_content_chars", 0))
    cap = min(65535, max_chars) if max_chars else 65535
    payloads: list[tuple[str, str]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            continue
        content = (getattr(m, "content", None) or "") or ""
        if cap < 65535 and len(content) > cap:
            content = content[:cap].rstrip() + "…"
        payloads.append((role, content[:65535]))
    if not payloads:
        return
    pool = await _get_async_pool(uri)
    try:
        async with pool.acquire() as conn:
            recent = await conn.fetch(
                "SELECT role, content FROM chat_history WHERE thread_id = $1 ORDER BY id DESC LIMIT $2",
                thread_id[:512], len(payloads),
            )
            recent_payloads = list(reversed([(r["role"], r["content"]) for r in recent]))
            if recent_payloads == payloads:
                return
            for role, content in payloads:
                await conn.execute(
                    "INSERT INTO chat_history (thread_id, role, content) VALUES ($1, $2, $3)",
                    thread_id[:512], role, content,
                )
    except Exception as e:
        logger.warning("写入对话历史失败(async): %s", e)


async def load_runtime_state_async(uri: str, thread_id: str) -> dict:
    """异步加载运行时状态，语义同 load_runtime_state。"""
    pool = await _get_async_pool(uri)
    try:
        row = await pool.fetchrow(
            f"SELECT pending_sql, interrupted FROM {_STATE_TABLE_NAME} WHERE thread_id = $1",
            thread_id[:512],
        )
        if not row:
            return {"pending_sql": None, "interrupted": False}
        return {"pending_sql": row["pending_sql"], "interrupted": bool(row["interrupted"])}
    except Exception as e:
        logger.warning("加载会话运行时状态失败(async): %s", e)
        return {"pending_sql": None, "interrupted": False}


async def save_runtime_state_async(
    uri: str, thread_id: str, pending_sql: Optional[str], interrupted: bool,
) -> None:
    """异步保存运行时状态，语义同 save_runtime_state。"""
    pool = await _get_async_pool(uri)
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {_STATE_TABLE_NAME} (thread_id, pending_sql, interrupted, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET pending_sql = EXCLUDED.pending_sql, interrupted = EXCLUDED.interrupted, updated_at = NOW()
                """,
                thread_id[:512], pending_sql, interrupted,
            )
    except Exception as e:
        logger.warning("保存会话运行时状态失败(async): %s", e)


async def dispose_async_pools() -> None:
    """关闭所有 asyncpg 连接池（应用关闭时调用）。"""
    global _async_pools
    for key, pool in list(_async_pools.items()):
        try:
            await pool.close()
        except Exception as e:
            logger.debug("关闭 asyncpg 池 %s: %s", key[:50], e)
    _async_pools.clear()


def ensure_table_if_configured(uri: Optional[str]) -> None:
    """当配置了 chat_history_postgresql_uri 时在首次使用时建表。"""
    if not (uri or "").strip() or "postgresql" not in (uri or "").lower():
        return
    try:
        _ensure_table(uri.strip())
    except Exception as e:
        logger.warning("创建 chat_history 表失败: %s", e)
