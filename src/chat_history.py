# src/chat_history.py
"""
对话长期记忆：将多轮对话历史与运行时状态写入 PostgreSQL，进程重启后可恢复。

- `chat_history`：按 thread_id 追加存储对话消息；
- `chat_runtime_state`：存储待确认 SQL、人工介入暂停态等流程状态；
- 请求开始时若内存无该会话状态则从 DB 加载并注入；
- 每轮结束后将本轮 user+assistant 追加写入 DB，并同步更新运行时状态。
- 连接韧性：通过 db_resilience 统一管理连接池复用、重试、熔断与降级。
"""
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.db_resilience import (
    get_engine,
    safe_connection,
    dispose_all_engines,
    get_async_pool,
    safe_async_connection,
    dispose_all_async_pools,
)

logger = logging.getLogger(__name__)

_TABLE_NAME = "chat_history"
_STATE_TABLE_NAME = "chat_runtime_state"
_SESSION_TABLE_NAME = "chat_sessions"


def _get_engine(uri: str):
    """按 uri 复用 SQLAlchemy Engine（委托 db_resilience）。"""
    return get_engine(uri)


def dispose_engines() -> None:
    """释放所有缓存的 Engine（应用关闭时调用，避免连接泄漏）。"""
    dispose_all_engines()


def _ensure_table(uri: str) -> None:
    """创建长期记忆所需的 PostgreSQL 表（若不存在）。"""
    from sqlalchemy import text
    with safe_connection(uri, retries=3, retry_delay=1.0, critical=True) as conn:
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
    连接失败时降级返回空列表（不阻断对话流程）。
    """
    from sqlalchemy import text
    out: List[BaseMessage] = []
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("加载对话历史降级：数据库不可用，返回空历史")
                return out
            if limit > 0:
                rows = conn.execute(
                    text("SELECT role, content FROM chat_history WHERE thread_id = :tid ORDER BY id DESC LIMIT :lim"),
                    {"tid": thread_id[:512], "lim": limit},
                ).fetchall()
                rows = list(reversed(rows))
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
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("写入对话历史降级：数据库不可用，本轮消息未持久化")
                return
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
    """加载会话运行时状态：待确认 SQL、人工介入暂停态。连接失败时降级返回默认值。"""
    from sqlalchemy import text
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                return {"pending_sql": None, "interrupted": False}
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
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("保存运行时状态降级：数据库不可用")
                return
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
    safe_title = (title or "新对话").strip()[:255] or "新对话"
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("保存会话元数据降级：数据库不可用")
                return
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
    """按更新时间倒序列出某用户的会话列表。连接失败时降级返回空列表。"""
    from sqlalchemy import text
    out: List[dict] = []
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                return out
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
    """按 thread_id 读取单个会话元数据。连接失败时降级返回 None。"""
    from sqlalchemy import text
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                return None
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
    """重命名会话标题。成功返回 True。连接失败时降级返回 False。"""
    from sqlalchemy import text
    safe_title = (title or "").strip()[:255]
    if not safe_title:
        return False
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                return False
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
    try:
        with safe_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("删除会话降级：数据库不可用")
                return
            conn.execute(text(f"DELETE FROM {_TABLE_NAME} WHERE thread_id = :thread_id"), {"thread_id": thread_id[:512]})
            conn.execute(text(f"DELETE FROM {_STATE_TABLE_NAME} WHERE thread_id = :thread_id"), {"thread_id": thread_id[:512]})
            conn.execute(text(f"DELETE FROM {_SESSION_TABLE_NAME} WHERE thread_id = :thread_id"), {"thread_id": thread_id[:512]})
            conn.commit()
    except Exception as e:
        logger.warning("删除会话失败: %s", e)


# ---------- 异步长期记忆（asyncpg，可选，通过 db_resilience 管理连接池） ----------


async def _get_async_pool(uri: str):
    """按 uri 复用 asyncpg 连接池（委托 db_resilience，含重试）。"""
    return await get_async_pool(uri)


async def load_messages_async(uri: str, thread_id: str, limit: int = 0) -> List[BaseMessage]:
    """异步从 PostgreSQL 加载历史消息。连接失败时降级返回空列表。"""
    out: List[BaseMessage] = []
    try:
        async with safe_async_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("加载对话历史降级(async)：数据库不可用")
                return out
            if limit > 0:
                rows = await conn.fetch(
                    "SELECT role, content FROM chat_history WHERE thread_id = $1 ORDER BY id DESC LIMIT $2",
                    thread_id[:512], limit,
                )
                rows = list(reversed(rows))
            else:
                rows = await conn.fetch(
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
    """异步追加消息到 chat_history。连接失败时降级跳过（不阻断对话）。"""
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
    try:
        async with safe_async_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("写入对话历史降级(async)：数据库不可用")
                return
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
    """异步加载运行时状态。连接失败时降级返回默认值。"""
    try:
        async with safe_async_connection(uri, critical=False) as conn:
            if conn is None:
                return {"pending_sql": None, "interrupted": False}
            row = await conn.fetchrow(
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
    """异步保存运行时状态。连接失败时降级跳过。"""
    try:
        async with safe_async_connection(uri, critical=False) as conn:
            if conn is None:
                logger.warning("保存运行时状态降级(async)：数据库不可用")
                return
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
    await dispose_all_async_pools()


def ensure_table_if_configured(uri: Optional[str]) -> None:
    """当配置了 chat_history_postgresql_uri 时在首次使用时建表。"""
    if not (uri or "").strip() or "postgresql" not in (uri or "").lower():
        return
    try:
        _ensure_table(uri.strip())
    except Exception as e:
        logger.warning("创建 chat_history 表失败: %s", e)
