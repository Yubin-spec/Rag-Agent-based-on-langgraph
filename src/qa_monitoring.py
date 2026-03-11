"""
问答效果监控与用户反馈分析。

功能：
- 记录每次问答的基础观测数据（question/answer/route/耗时/质量标签等）
- 记录 RAG 过程追踪（检索得分、grounding score、重生成次数、命中的文档/切片等）
- 记录用户反馈（点赞/点踩、标签、多行补充说明）
- 提供汇总统计、差评案例与反馈标签分析，供 API / 前端看板展示
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_OBS_TABLE = "qa_observation"
_TRACE_TABLE = "qa_rag_trace"
_FEEDBACK_TABLE = "qa_feedback"


def _ensure_table(uri: str) -> None:
    """创建问答监控与反馈分析所需的 PostgreSQL 表。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_OBS_TABLE} (
                observation_id    VARCHAR(64) PRIMARY KEY,
                thread_id         VARCHAR(512) NOT NULL,
                conversation_id   VARCHAR(128) NOT NULL,
                user_id           VARCHAR(255),
                question          TEXT NOT NULL,
                answer            TEXT,
                route             VARCHAR(64) NOT NULL,
                response_mode     VARCHAR(16) NOT NULL DEFAULT 'sync',
                success           BOOLEAN NOT NULL DEFAULT TRUE,
                used_cache        BOOLEAN NOT NULL DEFAULT FALSE,
                pending_sql       BOOLEAN NOT NULL DEFAULT FALSE,
                latency_ms        INTEGER NOT NULL DEFAULT 0,
                quality_label     VARCHAR(64) NOT NULL DEFAULT 'unknown',
                fallback_reason   VARCHAR(255),
                llm_model         VARCHAR(128),
                llm_endpoint_name VARCHAR(128),
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{_OBS_TABLE}_conversation_created
            ON {_OBS_TABLE}(conversation_id, created_at DESC)
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{_OBS_TABLE}_user_created
            ON {_OBS_TABLE}(user_id, created_at DESC)
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_TRACE_TABLE} (
                id                    SERIAL PRIMARY KEY,
                observation_id        VARCHAR(64) NOT NULL,
                final_status          VARCHAR(64) NOT NULL DEFAULT '',
                retrieve_attempt      INTEGER NOT NULL DEFAULT 0,
                top_match_score       DOUBLE PRECISION NOT NULL DEFAULT 0,
                top_normalized_score  DOUBLE PRECISION NOT NULL DEFAULT 0,
                grounding_score       DOUBLE PRECISION NOT NULL DEFAULT 0,
                regenerate_count      INTEGER NOT NULL DEFAULT 0,
                has_evidence_citations BOOLEAN NOT NULL DEFAULT FALSE,
                source_count          INTEGER NOT NULL DEFAULT 0,
                scenario_templates    JSONB NOT NULL DEFAULT '[]'::jsonb,
                retrieved_doc_ids     JSONB NOT NULL DEFAULT '[]'::jsonb,
                retrieved_chunk_ids   JSONB NOT NULL DEFAULT '[]'::jsonb,
                extra                 JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{_TRACE_TABLE}_observation_id
            ON {_TRACE_TABLE}(observation_id)
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_FEEDBACK_TABLE} (
                id              SERIAL PRIMARY KEY,
                observation_id  VARCHAR(64) NOT NULL,
                conversation_id VARCHAR(128) NOT NULL,
                user_id         VARCHAR(255),
                actor_key       VARCHAR(512) NOT NULL,
                rating          VARCHAR(16) NOT NULL,
                tags            JSONB NOT NULL DEFAULT '[]'::jsonb,
                free_text       TEXT NOT NULL DEFAULT '',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(actor_key, observation_id)
            )
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{_FEEDBACK_TABLE}_conversation_created
            ON {_FEEDBACK_TABLE}(conversation_id, created_at DESC)
        """))
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{_FEEDBACK_TABLE}_rating_created
            ON {_FEEDBACK_TABLE}(rating, created_at DESC)
        """))
        conn.commit()
    engine.dispose()


def ensure_table_if_configured(uri: Optional[str]) -> None:
    """当配置了 PostgreSQL URI 时创建问答监控相关表。"""

    if not (uri or "").strip() or "postgresql" not in (uri or "").lower():
        return
    try:
        _ensure_table((uri or "").strip())
    except Exception as e:
        logger.warning("创建问答监控表失败: %s", e)


def save_observation(uri: str, observation: dict[str, Any], trace: Optional[dict[str, Any]] = None) -> None:
    """保存一条问答观测记录，并可选写入对应的 RAG 追踪信息。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    obs = dict(observation or {})
    observation_id = str(obs.get("observation_id") or "").strip()
    if not observation_id:
        engine.dispose()
        return
    try:
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {_OBS_TABLE} (
                        observation_id, thread_id, conversation_id, user_id, question, answer,
                        route, response_mode, success, used_cache, pending_sql, latency_ms,
                        quality_label, fallback_reason, llm_model, llm_endpoint_name, created_at
                    )
                    VALUES (
                        :observation_id, :thread_id, :conversation_id, :user_id, :question, :answer,
                        :route, :response_mode, :success, :used_cache, :pending_sql, :latency_ms,
                        :quality_label, :fallback_reason, :llm_model, :llm_endpoint_name, NOW()
                    )
                    ON CONFLICT (observation_id)
                    DO UPDATE SET
                        answer = EXCLUDED.answer,
                        route = EXCLUDED.route,
                        response_mode = EXCLUDED.response_mode,
                        success = EXCLUDED.success,
                        used_cache = EXCLUDED.used_cache,
                        pending_sql = EXCLUDED.pending_sql,
                        latency_ms = EXCLUDED.latency_ms,
                        quality_label = EXCLUDED.quality_label,
                        fallback_reason = EXCLUDED.fallback_reason,
                        llm_model = EXCLUDED.llm_model,
                        llm_endpoint_name = EXCLUDED.llm_endpoint_name
                """),
                {
                    "observation_id": observation_id[:64],
                    "thread_id": str(obs.get("thread_id") or "")[:512],
                    "conversation_id": str(obs.get("conversation_id") or "")[:128],
                    "user_id": obs.get("user_id"),
                    "question": str(obs.get("question") or "")[:20000],
                    "answer": str(obs.get("answer") or "")[:65535],
                    "route": str(obs.get("route") or "unknown")[:64],
                    "response_mode": str(obs.get("response_mode") or "sync")[:16],
                    "success": bool(obs.get("success", True)),
                    "used_cache": bool(obs.get("used_cache", False)),
                    "pending_sql": bool(obs.get("pending_sql", False)),
                    "latency_ms": int(obs.get("latency_ms") or 0),
                    "quality_label": str(obs.get("quality_label") or "unknown")[:64],
                    "fallback_reason": str(obs.get("fallback_reason") or "")[:255] or None,
                    "llm_model": str(obs.get("llm_model") or "")[:128] or None,
                    "llm_endpoint_name": str(obs.get("llm_endpoint_name") or "")[:128] or None,
                },
            )
            if trace is not None:
                conn.execute(text(f"DELETE FROM {_TRACE_TABLE} WHERE observation_id = :observation_id"), {"observation_id": observation_id[:64]})
                conn.execute(
                    text(f"""
                        INSERT INTO {_TRACE_TABLE} (
                            observation_id, final_status, retrieve_attempt, top_match_score, top_normalized_score,
                            grounding_score, regenerate_count, has_evidence_citations, source_count,
                            scenario_templates, retrieved_doc_ids, retrieved_chunk_ids, extra, created_at
                        )
                        VALUES (
                            :observation_id, :final_status, :retrieve_attempt, :top_match_score, :top_normalized_score,
                            :grounding_score, :regenerate_count, :has_evidence_citations, :source_count,
                            CAST(:scenario_templates AS JSONB), CAST(:retrieved_doc_ids AS JSONB),
                            CAST(:retrieved_chunk_ids AS JSONB), CAST(:extra AS JSONB), NOW()
                        )
                    """),
                    {
                        "observation_id": observation_id[:64],
                        "final_status": str(trace.get("final_status") or "")[:64],
                        "retrieve_attempt": int(trace.get("retrieve_attempt") or 0),
                        "top_match_score": float(trace.get("top_match_score") or 0),
                        "top_normalized_score": float(trace.get("top_normalized_score") or 0),
                        "grounding_score": float(trace.get("grounding_score") or 0),
                        "regenerate_count": int(trace.get("regenerate_count") or 0),
                        "has_evidence_citations": bool(trace.get("has_evidence_citations", False)),
                        "source_count": int(trace.get("source_count") or 0),
                        "scenario_templates": json.dumps(trace.get("scenario_templates") or [], ensure_ascii=False),
                        "retrieved_doc_ids": json.dumps(trace.get("retrieved_doc_ids") or [], ensure_ascii=False),
                        "retrieved_chunk_ids": json.dumps(trace.get("retrieved_chunk_ids") or [], ensure_ascii=False),
                        "extra": json.dumps(trace, ensure_ascii=False),
                    },
                )
            conn.commit()
    except Exception as e:
        logger.warning("保存问答观测失败: %s", e)
    finally:
        engine.dispose()


def list_conversation_observations(uri: str, conversation_id: str, user_id: Optional[str]) -> list[dict[str, Any]]:
    """按时间顺序列出某会话下的观测记录，用于将 observation_id 映射回历史消息。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    out: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            if user_id is None:
                rows = conn.execute(
                    text(f"""
                        SELECT observation_id, route, answer, created_at
                        FROM {_OBS_TABLE}
                        WHERE conversation_id = :conversation_id AND user_id IS NULL
                          AND success = TRUE
                          AND COALESCE(answer, '') <> ''
                        ORDER BY created_at ASC
                    """),
                    {"conversation_id": conversation_id[:128]},
                ).fetchall()
            else:
                rows = conn.execute(
                    text(f"""
                        SELECT observation_id, route, answer, created_at
                        FROM {_OBS_TABLE}
                        WHERE conversation_id = :conversation_id AND user_id = :user_id
                          AND success = TRUE
                          AND COALESCE(answer, '') <> ''
                        ORDER BY created_at ASC
                    """),
                    {"conversation_id": conversation_id[:128], "user_id": user_id},
                ).fetchall()
        for row in rows:
            out.append({
                "observation_id": row[0],
                "route": row[1] or "",
                "answer": row[2] or "",
                "created_at": row[3].isoformat() if row[3] is not None else "",
            })
    except Exception as e:
        logger.warning("读取会话观测记录失败: %s", e)
    finally:
        engine.dispose()
    return out


def upsert_feedback(
    uri: str,
    *,
    observation_id: str,
    conversation_id: str,
    user_id: Optional[str],
    rating: str,
    tags: list[str],
    free_text: str,
) -> dict[str, Any]:
    """按 observation_id + actor_key 记录或更新反馈。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    actor_key = str(user_id or f"anon:{conversation_id}")[:512]
    safe_tags = [str(tag).strip()[:64] for tag in (tags or []) if str(tag).strip()]
    try:
        with engine.connect() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {_FEEDBACK_TABLE} (
                        observation_id, conversation_id, user_id, actor_key, rating, tags, free_text, created_at, updated_at
                    )
                    VALUES (
                        :observation_id, :conversation_id, :user_id, :actor_key, :rating,
                        CAST(:tags AS JSONB), :free_text, NOW(), NOW()
                    )
                    ON CONFLICT (actor_key, observation_id)
                    DO UPDATE SET
                        rating = EXCLUDED.rating,
                        tags = EXCLUDED.tags,
                        free_text = EXCLUDED.free_text,
                        updated_at = NOW()
                """),
                {
                    "observation_id": observation_id[:64],
                    "conversation_id": conversation_id[:128],
                    "user_id": user_id,
                    "actor_key": actor_key,
                    "rating": rating[:16],
                    "tags": json.dumps(safe_tags, ensure_ascii=False),
                    "free_text": (free_text or "")[:5000],
                },
            )
            conn.commit()
    except Exception as e:
        logger.warning("保存问答反馈失败: %s", e)
    finally:
        engine.dispose()
    return {
        "observation_id": observation_id[:64],
        "conversation_id": conversation_id[:128],
        "user_id": user_id,
        "rating": rating[:16],
        "tags": safe_tags,
        "free_text": (free_text or "")[:5000],
    }


def get_feedback_map(uri: str, conversation_id: str, user_id: Optional[str]) -> dict[str, dict[str, Any]]:
    """读取某会话下的反馈映射：observation_id -> feedback。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    out: dict[str, dict[str, Any]] = {}
    try:
        with engine.connect() as conn:
            if user_id is None:
                rows = conn.execute(
                    text(f"""
                        SELECT observation_id, rating, tags, free_text, updated_at
                        FROM {_FEEDBACK_TABLE}
                        WHERE conversation_id = :conversation_id AND user_id IS NULL
                    """),
                    {"conversation_id": conversation_id[:128]},
                ).fetchall()
            else:
                rows = conn.execute(
                    text(f"""
                        SELECT observation_id, rating, tags, free_text, updated_at
                        FROM {_FEEDBACK_TABLE}
                        WHERE conversation_id = :conversation_id AND user_id = :user_id
                    """),
                    {"conversation_id": conversation_id[:128], "user_id": user_id},
                ).fetchall()
        for row in rows:
            out[row[0]] = {
                "rating": row[1] or "",
                "tags": list(row[2] or []),
                "free_text": row[3] or "",
                "updated_at": row[4].isoformat() if row[4] is not None else "",
            }
    except Exception as e:
        logger.warning("读取反馈映射失败: %s", e)
    finally:
        engine.dispose()
    return out


def get_summary(uri: str, *, days: int = 7, user_id: Optional[str] = None) -> dict[str, Any]:
    """聚合问答效果总览，供看板概览展示。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    days = max(1, int(days))
    result = {
        "days": days,
        "total_questions": 0,
        "success_count": 0,
        "cache_hit_count": 0,
        "knowledge_count": 0,
        "negative_feedback_count": 0,
        "positive_feedback_count": 0,
        "avg_latency_ms": 0.0,
        "low_grounding_count": 0,
        "regenerated_count": 0,
        "fallback_count": 0,
    }
    try:
        with engine.connect() as conn:
            params = {"days": days}
            user_filter = "o.user_id IS NULL" if user_id is None else "o.user_id = :user_id"
            if user_id is not None:
                params["user_id"] = user_id
            row = conn.execute(
                text(f"""
                    SELECT
                        COUNT(*) AS total_questions,
                        COALESCE(SUM(CASE WHEN o.success THEN 1 ELSE 0 END), 0) AS success_count,
                        COALESCE(SUM(CASE WHEN o.used_cache THEN 1 ELSE 0 END), 0) AS cache_hit_count,
                        COALESCE(SUM(CASE WHEN o.route IN ('knowledge', 'qa', 'text2sql', 'rag', 'cache') THEN 1 ELSE 0 END), 0) AS knowledge_count,
                        COALESCE(AVG(o.latency_ms), 0) AS avg_latency_ms,
                        COALESCE(SUM(CASE WHEN t.grounding_score > 0 AND t.grounding_score < 0.18 THEN 1 ELSE 0 END), 0) AS low_grounding_count,
                        COALESCE(SUM(CASE WHEN t.regenerate_count > 0 THEN 1 ELSE 0 END), 0) AS regenerated_count,
                        COALESCE(SUM(CASE WHEN o.quality_label IN ('fallback', 'low_grounding', 'needs_human', 'no_retrieval_hit') THEN 1 ELSE 0 END), 0) AS fallback_count
                    FROM {_OBS_TABLE} o
                    LEFT JOIN {_TRACE_TABLE} t ON o.observation_id = t.observation_id
                    WHERE {user_filter}
                      AND o.created_at >= NOW() - (:days || ' days')::interval
                """),
                params,
            ).fetchone()
            if row:
                result.update({
                    "total_questions": int(row[0] or 0),
                    "success_count": int(row[1] or 0),
                    "cache_hit_count": int(row[2] or 0),
                    "knowledge_count": int(row[3] or 0),
                    "avg_latency_ms": round(float(row[4] or 0), 2),
                    "low_grounding_count": int(row[5] or 0),
                    "regenerated_count": int(row[6] or 0),
                    "fallback_count": int(row[7] or 0),
                })
            fb_filter = "user_id IS NULL" if user_id is None else "user_id = :user_id"
            fb_row = conn.execute(
                text(f"""
                    SELECT
                        COALESCE(SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END), 0)
                    FROM {_FEEDBACK_TABLE}
                    WHERE {fb_filter}
                      AND created_at >= NOW() - (:days || ' days')::interval
                """),
                params,
            ).fetchone()
            if fb_row:
                result["negative_feedback_count"] = int(fb_row[0] or 0)
                result["positive_feedback_count"] = int(fb_row[1] or 0)
    except Exception as e:
        logger.warning("读取问答监控汇总失败: %s", e)
    finally:
        engine.dispose()
    return result


def list_bad_cases(uri: str, *, days: int = 7, user_id: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    """列出近期疑似效果不佳的案例：差评、低 grounding、兜底回答优先。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    days = max(1, int(days))
    limit = max(1, min(int(limit), 100))
    out: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            params = {"days": days, "limit": limit}
            user_filter = "o.user_id IS NULL" if user_id is None else "o.user_id = :user_id"
            if user_id is not None:
                params["user_id"] = user_id
            rows = conn.execute(
                text(f"""
                    SELECT
                        o.observation_id, o.conversation_id, o.question, o.answer, o.route,
                        o.quality_label, o.latency_ms, o.created_at,
                        COALESCE(t.grounding_score, 0) AS grounding_score,
                        COALESCE(t.regenerate_count, 0) AS regenerate_count,
                        COALESCE(f.rating, '') AS rating,
                        COALESCE(f.tags, '[]'::jsonb) AS tags
                    FROM {_OBS_TABLE} o
                    LEFT JOIN {_TRACE_TABLE} t ON o.observation_id = t.observation_id
                    LEFT JOIN {_FEEDBACK_TABLE} f ON o.observation_id = f.observation_id
                    WHERE {user_filter}
                      AND o.created_at >= NOW() - (:days || ' days')::interval
                      AND (
                        COALESCE(f.rating, '') = 'down'
                        OR COALESCE(t.grounding_score, 1) < 0.18
                        OR o.quality_label IN ('fallback', 'low_grounding', 'needs_human', 'no_retrieval_hit')
                      )
                    ORDER BY
                        CASE WHEN COALESCE(f.rating, '') = 'down' THEN 0 ELSE 1 END,
                        COALESCE(t.grounding_score, 1) ASC,
                        o.created_at DESC
                    LIMIT :limit
                """),
                params,
            ).fetchall()
        for row in rows:
            out.append({
                "observation_id": row[0],
                "conversation_id": row[1],
                "question": row[2] or "",
                "answer": row[3] or "",
                "route": row[4] or "",
                "quality_label": row[5] or "",
                "latency_ms": int(row[6] or 0),
                "created_at": row[7].isoformat() if row[7] is not None else "",
                "grounding_score": round(float(row[8] or 0), 4),
                "regenerate_count": int(row[9] or 0),
                "rating": row[10] or "",
                "tags": list(row[11] or []),
            })
    except Exception as e:
        logger.warning("读取差评案例失败: %s", e)
    finally:
        engine.dispose()
    return out


def list_feedback_tag_stats(uri: str, *, days: int = 7, user_id: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    """统计近 N 天反馈标签分布。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    days = max(1, int(days))
    limit = max(1, min(int(limit), 100))
    out: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            params = {"days": days, "limit": limit}
            user_filter = "user_id IS NULL" if user_id is None else "user_id = :user_id"
            if user_id is not None:
                params["user_id"] = user_id
            rows = conn.execute(
                text(f"""
                    SELECT tag, COUNT(*) AS cnt
                    FROM (
                        SELECT jsonb_array_elements_text(tags) AS tag
                        FROM {_FEEDBACK_TABLE}
                        WHERE {user_filter}
                          AND created_at >= NOW() - (:days || ' days')::interval
                    ) x
                    GROUP BY tag
                    ORDER BY cnt DESC, tag ASC
                    LIMIT :limit
                """),
                params,
            ).fetchall()
        for row in rows:
            out.append({"tag": row[0] or "", "count": int(row[1] or 0)})
    except Exception as e:
        logger.warning("读取反馈标签统计失败: %s", e)
    finally:
        engine.dispose()
    return out


def list_scenario_stats(uri: str, *, days: int = 7, user_id: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    """统计近 N 天命中的场景模板分布。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    days = max(1, int(days))
    limit = max(1, min(int(limit), 100))
    out: list[dict[str, Any]] = []
    try:
        with engine.connect() as conn:
            params = {"days": days, "limit": limit}
            user_filter = "o.user_id IS NULL" if user_id is None else "o.user_id = :user_id"
            if user_id is not None:
                params["user_id"] = user_id
            rows = conn.execute(
                text(f"""
                    SELECT scenario_name, COUNT(*) AS cnt
                    FROM (
                        SELECT jsonb_array_elements_text(t.scenario_templates) AS scenario_name
                        FROM {_OBS_TABLE} o
                        JOIN {_TRACE_TABLE} t ON o.observation_id = t.observation_id
                        WHERE {user_filter}
                          AND o.created_at >= NOW() - (:days || ' days')::interval
                    ) s
                    GROUP BY scenario_name
                    ORDER BY cnt DESC, scenario_name ASC
                    LIMIT :limit
                """),
                params,
            ).fetchall()
        for row in rows:
            out.append({"scenario": row[0] or "", "count": int(row[1] or 0)})
    except Exception as e:
        logger.warning("读取场景模板统计失败: %s", e)
    finally:
        engine.dispose()
    return out


def delete_conversation_data(uri: str, conversation_id: str, user_id: Optional[str]) -> None:
    """删除某个会话下的问答观测、追踪与反馈数据。"""

    from sqlalchemy import create_engine, text

    engine = create_engine(uri, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            if user_id is None:
                rows = conn.execute(
                    text(f"SELECT observation_id FROM {_OBS_TABLE} WHERE conversation_id = :conversation_id AND user_id IS NULL"),
                    {"conversation_id": conversation_id[:128]},
                ).fetchall()
                conn.execute(
                    text(f"DELETE FROM {_FEEDBACK_TABLE} WHERE conversation_id = :conversation_id AND user_id IS NULL"),
                    {"conversation_id": conversation_id[:128]},
                )
                conn.execute(
                    text(f"DELETE FROM {_OBS_TABLE} WHERE conversation_id = :conversation_id AND user_id IS NULL"),
                    {"conversation_id": conversation_id[:128]},
                )
            else:
                rows = conn.execute(
                    text(f"SELECT observation_id FROM {_OBS_TABLE} WHERE conversation_id = :conversation_id AND user_id = :user_id"),
                    {"conversation_id": conversation_id[:128], "user_id": user_id},
                ).fetchall()
                conn.execute(
                    text(f"DELETE FROM {_FEEDBACK_TABLE} WHERE conversation_id = :conversation_id AND user_id = :user_id"),
                    {"conversation_id": conversation_id[:128], "user_id": user_id},
                )
                conn.execute(
                    text(f"DELETE FROM {_OBS_TABLE} WHERE conversation_id = :conversation_id AND user_id = :user_id"),
                    {"conversation_id": conversation_id[:128], "user_id": user_id},
                )
            for row in rows:
                conn.execute(text(f"DELETE FROM {_TRACE_TABLE} WHERE observation_id = :observation_id"), {"observation_id": row[0]})
            conn.commit()
    except Exception as e:
        logger.warning("删除会话监控数据失败: %s", e)
    finally:
        engine.dispose()
