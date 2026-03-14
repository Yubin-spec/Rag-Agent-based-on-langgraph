# api/main.py
"""
知识库智能体 HTTP API：文档上传/解析/确认上传、多智能体对话。

主要能力：
- 文档：POST /doc/upload 上传并解析（MinerU 或占位），POST /doc/confirm_upload 确认后写入 Milvus。
- 对话：POST /chat 非流式、POST /chat/stream 流式（先出首字）；总控路由到闲聊或知识库，内部仅 DeepSeek。
- Text2SQL：GET/PUT /text2sql/schema 表结构审核，POST /text2sql/confirm_execute 人工确认执行删除/修改类 SQL。

隔离与限制：
- 会话隔离：conversation_id 作为 LangGraph thread_id，同会话共享历史；单对话最多 15 轮。
- 用户隔离：请求头 X-User-Id 或 body.user_id 存在时，会话/待确认 SQL/解析缓存 均按 user_id 隔离。
- 超时与打断：单次请求可配置超时；流式支持客户端断开时 CancelledError 处理。
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

# 将项目根目录加入 path，保证 config、src 可导入
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FRONTEND_DIR = ROOT / "frontend"

import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid

from config import get_settings
from src.doc.mineru_client import MinerUClient, ParseResult, ChunkItem
from src.doc.milvus_upload import MilvusUploader
from src.doc.validation import validate_parse_result
from src.graph.app import get_graph
from src.agents.supervisor import supervisor_node_async
from src.agents.chat_agent import chat_agent_stream_async
from src.kb.engine import KnowledgeEngine
from src.kb.schema_loader import read_db_schema, load_schema_overrides, save_schema_overrides
from src.answer_cache import answer_lock, get_cached_answer, set_cached_answer, close_redis_connection, redis_ping
from src.chat_history import (
    load_messages as chat_history_load,
    append_messages as chat_history_append,
    dispose_engines as chat_history_dispose_engines,
    dispose_async_pools as chat_history_dispose_async_pools,
    ensure_table_if_configured as chat_history_ensure_table,
    load_messages_async as chat_history_load_messages_async,
    append_messages_async as chat_history_append_messages_async,
    load_runtime_state as chat_history_load_runtime_state,
    load_runtime_state_async as chat_history_load_runtime_state_async,
    save_runtime_state as chat_history_save_runtime_state,
    save_runtime_state_async as chat_history_save_runtime_state_async,
    upsert_conversation_session as chat_history_upsert_conversation_session,
    list_conversation_sessions as chat_history_list_conversation_sessions,
    get_conversation_session as chat_history_get_conversation_session,
    rename_conversation_session as chat_history_rename_conversation_session,
    delete_conversation_session as chat_history_delete_conversation_session,
)
from src.qa_monitoring import (
    ensure_table_if_configured as qa_monitoring_ensure_table,
    save_observation as qa_monitoring_save_observation,
    upsert_feedback as qa_monitoring_upsert_feedback,
    get_feedback_map as qa_monitoring_get_feedback_map,
    list_conversation_observations as qa_monitoring_list_conversation_observations,
    get_summary as qa_monitoring_get_summary,
    list_bad_cases as qa_monitoring_list_bad_cases,
    list_feedback_tag_stats as qa_monitoring_list_feedback_tag_stats,
    list_scenario_stats as qa_monitoring_list_scenario_stats,
    delete_conversation_data as qa_monitoring_delete_conversation_data,
)
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from src.llm import get_deepseek_llm, get_deepseek_router_status, get_last_deepseek_endpoint_name
from src.db_resilience import get_all_breaker_status, milvus_ping
from src import shared_state as shared_state_module
from src.shared_state import close_shared_state_redis

logger = logging.getLogger(__name__)


def _qa_monitoring_uri(settings=None) -> str:
    """返回问答监控 PostgreSQL 连接串；未单独配置时回退到会话长期记忆库。"""
    settings = settings or get_settings()
    return (
        (getattr(settings, "qa_monitoring_postgresql_uri", None) or "").strip()
        or (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    )


async def _save_qa_observation_async(observation: dict, trace: Optional[dict] = None) -> None:
    """异步落库一条问答观测记录；未启用 PostgreSQL 时静默跳过。"""
    settings = get_settings()
    uri = _qa_monitoring_uri(settings)
    if not uri or "postgresql" not in uri.lower():
        return
    await asyncio.to_thread(qa_monitoring_save_observation, uri, observation, trace)


def _postgresql_ping(uri: str) -> None:
    """同步执行一次 PostgreSQL 探活：连接并执行 SELECT 1。"""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(uri, pool_pre_ping=True, pool_size=1, max_overflow=0)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as e:
        logger.warning("PostgreSQL 探活失败（下次探活会重试）: %s", e)


def _postgresql_ping_returns_bool(uri: str) -> bool:
    """PostgreSQL 探活，返回是否成功（供健康检查使用）。"""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(uri, pool_pre_ping=True, pool_size=1, max_overflow=0)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


async def _postgresql_keepalive_loop() -> None:
    """后台任务：每分钟对配置的 PostgreSQL 执行一次探活，保持数据库在线可用。"""
    settings = get_settings()
    uri = (getattr(settings, "postgresql_keepalive_uri", None) or "").strip()
    interval = max(30, getattr(settings, "postgresql_keepalive_interval_seconds", 60))
    if not uri or "postgresql" not in uri.lower():
        return
    while True:
        try:
            await asyncio.sleep(interval)
            await asyncio.to_thread(_postgresql_ping, uri)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("PostgreSQL 探活任务异常（下次探活会重试）: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建对话历史表（若配置）、可选设置线程池大小、启动 PostgreSQL 探活任务、API 限流，关闭时取消。"""
    keepalive_task = None
    settings = get_settings()
    limit = max(0, getattr(settings, "api_max_concurrent_requests", 0))
    app.state.api_semaphore = asyncio.Semaphore(limit) if limit > 0 else None
    if limit > 0:
        logger.info("API 全局限流已启用: max_concurrent=%d", limit)
    # 异步高并发：可选扩大 to_thread 使用的线程池，避免大量并发时排队
    n = getattr(settings, "asyncio_thread_pool_workers", 0)
    if n > 0:
        from concurrent.futures import ThreadPoolExecutor
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=n))
        logger.info("asyncio 默认线程池已设置为 max_workers=%s", n)
    chat_uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if chat_uri and "postgresql" in chat_uri.lower():
        await asyncio.to_thread(chat_history_ensure_table, chat_uri)
        logger.info("对话长期记忆已启用（PostgreSQL）")
    qa_monitor_uri = _qa_monitoring_uri(settings)
    if qa_monitor_uri and "postgresql" in qa_monitor_uri.lower():
        await asyncio.to_thread(qa_monitoring_ensure_table, qa_monitor_uri)
        logger.info("问答监控与用户反馈分析已启用（PostgreSQL）")
    uri = (getattr(settings, "postgresql_keepalive_uri", None) or "").strip()
    if uri and "postgresql" in uri.lower():
        keepalive_task = asyncio.create_task(_postgresql_keepalive_loop())
        logger.info("PostgreSQL 探活已启动，间隔 %s 秒", getattr(settings, "postgresql_keepalive_interval_seconds", 60))
    yield
    await close_redis_connection()
    await close_shared_state_redis()
    chat_history_dispose_engines()
    await chat_history_dispose_async_pools()
    if keepalive_task and not keepalive_task.done():
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="知识库智能体 API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_concurrency_limit_middleware(request: Request, call_next):
    """API 全局限流：超过 api_max_concurrent_requests 时排队等待；若配置为 0 则不限流。"""
    sem = getattr(app.state, "api_semaphore", None)
    if sem is None:
        return await call_next(request)
    async with sem:
        return await call_next(request)

# 解析缓存、待确认 SQL、人工中断状态：由 shared_state 统一提供（内存或 Redis，见 shared_state_redis_url）


async def _ensure_state_from_db(graph, config: dict, thread_id: str) -> None:
    """
    长期记忆：若配置了 chat_history_postgresql_uri，从 PostgreSQL 加载该会话历史；
    若当前图状态（MemorySaver）中无消息，则用 DB 中的消息注入状态，便于进程重启后恢复。
    """
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        return
    use_asyncpg = getattr(settings, "chat_history_use_asyncpg", False)
    load_limit = max(0, getattr(settings, "chat_history_load_max_messages", 0))
    try:
        if use_asyncpg:
            runtime_state = await chat_history_load_runtime_state_async(uri, thread_id)
            db_messages = await chat_history_load_messages_async(uri, thread_id, load_limit)
        else:
            runtime_state = await asyncio.to_thread(chat_history_load_runtime_state, uri, thread_id)
            db_messages = await asyncio.to_thread(chat_history_load, uri, thread_id, load_limit)
    except Exception:
        runtime_state = await asyncio.to_thread(chat_history_load_runtime_state, uri, thread_id)
        db_messages = await asyncio.to_thread(chat_history_load, uri, thread_id, load_limit)
    pending_sql = (runtime_state.get("pending_sql") or "").strip()
    if pending_sql:
        await shared_state_module.set_pending_sql(thread_id, pending_sql)
    else:
        await shared_state_module.pop_pending_sql(thread_id)
    if runtime_state.get("interrupted"):
        await shared_state_module.add_interrupted(thread_id)
    else:
        await shared_state_module.discard_interrupted(thread_id)
    if not db_messages:
        return
    try:
        snapshot = await graph.aget_state(config) if hasattr(graph, "aget_state") else await asyncio.to_thread(graph.get_state, config)
        current = (snapshot.values or {}).get("messages") or []
        if len(current) == 0:
            if hasattr(graph, "aupdate_state"):
                await graph.aupdate_state(config, {"messages": db_messages})
            else:
                await asyncio.to_thread(graph.update_state, config, {"messages": db_messages})
    except Exception:
        pass


def _persist_chat_messages_sync(thread_id: str, messages: list) -> None:
    """将本轮新增的若干条消息追加写入 PostgreSQL（同步，供 to_thread 调用）。"""
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower() or not messages:
        return
    chat_history_append(uri, thread_id, messages)


async def _persist_chat_messages(thread_id: str, messages: list) -> None:
    """将本轮消息写入长期记忆；若启用 asyncpg 则异步写入，否则 to_thread 同步。"""
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower() or not messages:
        return
    if getattr(settings, "chat_history_use_asyncpg", False):
        try:
            await chat_history_append_messages_async(uri, thread_id, messages)
        except Exception:
            await asyncio.to_thread(chat_history_append, uri, thread_id, messages)
    else:
        await asyncio.to_thread(chat_history_append, uri, thread_id, messages)


async def _persist_runtime_state_sync(thread_id: str) -> None:
    """将待确认 SQL 与 interrupt 暂停态同步保存到 PostgreSQL（需先 await 获取共享状态）。"""
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        return
    pending_sql = await shared_state_module.get_pending_sql(thread_id)
    interrupted = await shared_state_module.is_interrupted(thread_id)
    await asyncio.to_thread(chat_history_save_runtime_state, uri, thread_id, pending_sql, interrupted)


async def _persist_runtime_state(thread_id: str) -> None:
    """将待确认 SQL 与 interrupt 暂停态写入长期记忆；若启用 asyncpg 则异步写入。"""
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        return
    pending_sql = await shared_state_module.get_pending_sql(thread_id)
    interrupted = await shared_state_module.is_interrupted(thread_id)
    if getattr(settings, "chat_history_use_asyncpg", False):
        try:
            await chat_history_save_runtime_state_async(uri, thread_id, pending_sql, interrupted)
        except Exception:
            await asyncio.to_thread(chat_history_save_runtime_state, uri, thread_id, pending_sql, interrupted)
    else:
        await asyncio.to_thread(chat_history_save_runtime_state, uri, thread_id, pending_sql, interrupted)


async def _resume_if_interrupted(graph, config: dict, thread_id: str, timeout_seconds: float) -> None:
    """若会话处于人工介入暂停态，则先 resume；成功后同时清理内存与 PostgreSQL 中的暂停标记。"""
    if not await shared_state_module.is_interrupted(thread_id):
        return
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(Command(resume=True), config),
            timeout=timeout_seconds,
        )
        resumed_messages = result.get("messages") or []
        last_ai = None
        for m in resumed_messages:
            if isinstance(m, AIMessage):
                last_ai = m
        if last_ai is not None:
            await _persist_chat_messages( thread_id, [last_ai])
        await shared_state_module.discard_interrupted(thread_id)
        await _persist_runtime_state(thread_id)
    except Exception as e:
        logger.warning("恢复人工介入会话失败: %s", e)


def _user_id_from_request(request: Request, body_user_id: Optional[str] = None) -> Optional[str]:
    """
    从请求中解析用户 ID，用于用户隔离。
    优先级：body 中的 user_id > 请求头 X-User-Id（或 x-user-id）。
    Returns:
        非空字符串或 None；None 表示未传用户标识，走会话级隔离（兼容旧客户端）。
    """
    if body_user_id and str(body_user_id).strip():
        return str(body_user_id).strip()
    h = request.headers.get("X-User-Id") or request.headers.get("x-user-id")
    return h.strip() if h and h.strip() else None


def _thread_id(user_id: Optional[str], conversation_id: str) -> str:
    """
    生成 LangGraph 与内存缓存的会话键。
    有 user_id 时返回 "{user_id}:{conversation_id}"，实现用户维度隔离；
    无 user_id 时直接返回 conversation_id，与旧行为一致。
    """
    if user_id:
        return f"{user_id}:{conversation_id}"
    return conversation_id


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None  # 对话 ID，同 ID 即同会话（数据隔离），不传则新开对话
    user_id: Optional[str] = None  # 可选，与 X-User-Id 二选一，用于用户隔离


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str  # 本对话 ID，后续请求带上以实现多轮
    observation_id: Optional[str] = None


class ChatHistoryItem(BaseModel):
    role: str
    content: str
    observation_id: Optional[str] = None
    feedback_rating: str = ""
    feedback_tags: List[str] = []
    feedback_text: str = ""


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[ChatHistoryItem]


class ConversationListItem(BaseModel):
    conversation_id: str
    title: str
    updated_at: str = ""


class ConversationListResponse(BaseModel):
    conversations: List[ConversationListItem]


class RenameConversationRequest(BaseModel):
    conversation_id: str
    title: str
    user_id: Optional[str] = None


class QAFeedbackRequest(BaseModel):
    observation_id: str
    conversation_id: str
    rating: str
    tags: List[str] = []
    free_text: str = ""
    user_id: Optional[str] = None


class QAFeedbackItem(BaseModel):
    tag: str
    count: int


class QAScenarioStatItem(BaseModel):
    scenario: str
    count: int


class QABadCaseItem(BaseModel):
    observation_id: str
    conversation_id: str
    question: str
    answer: str
    route: str
    quality_label: str
    latency_ms: int
    created_at: str
    grounding_score: float = 0.0
    regenerate_count: int = 0
    rating: str = ""
    tags: List[str] = []


class QAMonitorSummaryResponse(BaseModel):
    days: int
    total_questions: int
    success_count: int
    cache_hit_count: int
    knowledge_count: int
    negative_feedback_count: int
    positive_feedback_count: int
    avg_latency_ms: float
    low_grounding_count: int
    regenerated_count: int
    fallback_count: int


class QAMonitorAnalyticsResponse(BaseModel):
    summary: QAMonitorSummaryResponse
    bad_cases: List[QABadCaseItem]
    feedback_tags: List[QAFeedbackItem]
    scenarios: List[QAScenarioStatItem]


class LLMEndpointStatusItem(BaseModel):
    name: str
    base_url: str
    weight: int
    inflight: int
    failures: int
    circuit_open: bool
    seconds_until_retry: float
    model_client_count: int


class LLMRouterStatusResponse(BaseModel):
    router: str
    endpoint_count: int
    healthy_endpoint_count: int
    all_circuits_open: bool
    timestamp: float
    endpoints: List[LLMEndpointStatusItem]


class ConfirmUploadRequest(BaseModel):
    task_id: str
    user_id: Optional[str] = None  # 可选，与 X-User-Id 二选一，用于用户隔离
    chunks: Optional[List[ChunkItem]] = None  # 用户自定义后的 chunks，若不传则用原解析结果


class SchemaColumn(BaseModel):
    name: str
    dtype: str
    comment: str = ""


class SchemaTable(BaseModel):
    name: str
    comment: str = ""
    columns: List[SchemaColumn] = []


class SchemaRelation(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str


class SchemaOverridesBody(BaseModel):
    """人工审核：表含义、列含义、表间关联（仅人工整理的关联会喂给大模型）。"""
    table_comments: dict = {}
    column_comments: dict = {}  # { "表名": { "列名": "含义" } }
    relations: List[SchemaRelation] = []


async def _health_probe_details() -> dict:
    """探测各依赖健康状态，供 /health 与 /health/ready 使用。"""
    settings = get_settings()
    details = {}
    # PostgreSQL（对话历史）
    chat_uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if chat_uri and "postgresql" in chat_uri.lower():
        ok = await asyncio.to_thread(_postgresql_ping_returns_bool, chat_uri)
        details["postgresql_chat"] = "ok" if ok else "unreachable"
    # PostgreSQL（问答监控，若与 chat 不同则单独探测）
    qa_uri = _qa_monitoring_uri(settings)
    if qa_uri and "postgresql" in qa_uri.lower() and qa_uri != chat_uri:
        ok = await asyncio.to_thread(_postgresql_ping_returns_bool, qa_uri)
        details["postgresql_qa_monitoring"] = "ok" if ok else "unreachable"
    # Redis
    if getattr(settings, "answer_cache_enabled", True) and (getattr(settings, "redis_url", "") or "").strip():
        ok = await redis_ping()
        details["redis"] = "ok" if ok else "unreachable"
    # Milvus
    milvus_uri = (getattr(settings, "milvus_uri", "") or "").strip()
    if milvus_uri:
        ok = milvus_ping(milvus_uri)
        details["milvus"] = "ok" if ok else "unreachable"
    # LLM（至少一个 endpoint 可用）
    router_status = get_deepseek_router_status()
    healthy = router_status.get("healthy_endpoint_count", 0)
    all_open = router_status.get("all_circuits_open", False)
    details["llm"] = "ok" if healthy > 0 else ("degraded" if all_open else "no_endpoints")
    return details


@app.get("/health")
async def health():
    """
    健康检查，供负载均衡或容器 liveness 探针使用。
    探测 DB/Redis/Milvus/LLM，返回 status: ok | degraded 及 details。
    """
    details = await _health_probe_details()
    failures = [k for k, v in details.items() if v not in ("ok",)]
    status = "degraded" if failures else "ok"
    result = {"status": status, "details": details}
    if failures:
        result["unhealthy"] = {k: details[k] for k in failures}
    result["breakers"] = get_all_breaker_status()
    return result


@app.get("/health/ready")
async def health_ready():
    """
    深度就绪检查，供 K8s readiness 探针使用。
    仅对已配置的依赖做检查；任一已配置依赖不可用时返回 503。
    """
    details = await _health_probe_details()
    unhealthy = [k for k, v in details.items() if v not in ("ok",)]
    if unhealthy:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "unhealthy": unhealthy, "details": details},
        )
    return {"status": "ready", "details": details}


@app.get("/llm/router/status", response_model=LLMRouterStatusResponse)
def llm_router_status():
    """查看当前 DeepSeek 路由器状态：节点并发、失败次数、熔断状态等。"""
    return get_deepseek_router_status()


@app.get("/")
def index():
    """根路径：若存在 frontend/index.html 则返回该页面，否则返回 API 说明 JSON。"""
    if FRONTEND_DIR.exists():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return {"message": "Knowledge Base Agent API", "docs": "/docs"}


def _parse_cache_key(user_id: Optional[str], task_id: str) -> str:
    """
    生成文档解析缓存的键，用于 _parse_cache。
    有 user_id 时按用户隔离，无则使用 'anon'，避免不同用户互相覆盖解析结果。
    """
    return f"{user_id or 'anon'}:{task_id}"


@app.post("/doc/upload", response_model=dict)
async def doc_upload(request: Request, file: UploadFile = File(...)):
    """
    上传文档并解析（MinerU 或本地占位）。
    将文件写入 upload_dir，调用解析器得到全文与 chunks，存入 _parse_cache（key 含 user_id）。
    返回 task_id、原文摘要、chunks 等供前端对比；确认后由 /doc/confirm_upload 写入 Milvus。
    """
    user_id = _user_id_from_request(request, None)
    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    path = os.path.join(settings.upload_dir, file.filename or "upload")
    content = await file.read()
    def _write(path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)
    await asyncio.to_thread(_write, path, content)
    client = MinerUClient()
    result = await client.parse_file_async(file.filename or path, content)
    report = await asyncio.to_thread(
        validate_parse_result, result.full_text, result.chunks,
        file_path=result.original_path,
    )
    cache_key = _parse_cache_key(user_id, result.task_id)
    await shared_state_module.set_parse_result(cache_key, result)
    return {
        "task_id": result.task_id,
        "original_path": result.original_path,
        "doc_name": result.doc_name,
        "full_text": result.full_text[:5000],
        "chunks": [c.model_dump() for c in result.chunks],
        "raw_markdown": (result.raw_markdown or "")[:5000],
        "validation": {
            "passed": report.passed,
            "summary": report.summary(),
            "errors": [{"category": i.category, "message": i.message, "chunk_index": i.chunk_index} for i in report.errors],
            "warnings": [{"category": i.category, "message": i.message, "chunk_index": i.chunk_index} for i in report.warnings],
        },
    }


@app.post("/doc/confirm_upload")
async def confirm_upload(request: Request, body: ConfirmUploadRequest):
    """
    用户确认上传：从 _parse_cache 取出对应 task_id 的解析结果（按 user_id 隔离），
    可选使用 body.chunks 覆盖原 chunks，然后调用 MilvusUploader 写入向量库。
    确认成功后从缓存删除该条，防止重复确认。
    """
    user_id = _user_id_from_request(request, body.user_id)
    cache_key = _parse_cache_key(user_id, body.task_id)
    base = await shared_state_module.get_parse_result(cache_key)
    if base is None:
        raise HTTPException(status_code=404, detail="task_id 不存在或已过期")
    if body.chunks is not None and len(body.chunks) > 0:
        result = ParseResult(
            task_id=base.task_id,
            original_path=base.original_path,
            full_text=base.full_text,
            chunks=body.chunks,
            raw_markdown=base.raw_markdown,
            doc_name=base.doc_name,
        )
    else:
        result = base
    report = await asyncio.to_thread(
        validate_parse_result, result.full_text, result.chunks,
        file_path=result.original_path,
    )
    if not report.passed:
        return {
            "uploaded": 0,
            "task_id": body.task_id,
            "validation": {
                "passed": False,
                "summary": report.summary(),
                "errors": [{"category": i.category, "message": i.message, "chunk_index": i.chunk_index} for i in report.errors],
                "warnings": [{"category": i.category, "message": i.message, "chunk_index": i.chunk_index} for i in report.warnings],
            },
        }
    uploader = MilvusUploader()
    count = await asyncio.to_thread(uploader.upload_parse_result, result)
    await shared_state_module.delete_parse_result(cache_key)
    return {
        "uploaded": count,
        "task_id": body.task_id,
        "validation": {
            "passed": True,
            "summary": report.summary(),
            "warnings": [{"category": i.category, "message": i.message, "chunk_index": i.chunk_index} for i in report.warnings],
        },
    }


@app.get("/text2sql/schema")
async def get_text2sql_schema():
    """
    获取 Text2SQL 使用的数据库表结构及人工审核内容。
    从 schema_loader 读取 DB 表/列与 overrides JSON，合并后返回 tables（含 comment）、relations。
    供前端「Text2SQL 表结构审核」页展示与编辑，编辑后通过 PUT /text2sql/schema 保存。
    """
    try:
        overrides = await asyncio.to_thread(load_schema_overrides)
        table_comments = overrides.get("table_comments") or {}
        column_comments = overrides.get("column_comments") or {}
        tables, _ = await asyncio.to_thread(
            read_db_schema,
            table_comments=table_comments,
            column_comments=column_comments,
        )
        tables_out = [
            {
                "name": t.name,
                "comment": table_comments.get(t.name, t.comment or ""),
                "columns": [
                    {"name": c.name, "dtype": c.dtype, "comment": column_comments.get(t.name, {}).get(c.name, c.comment or "")}
                    for c in t.columns
                ],
            }
            for t in tables
        ]
        relations = overrides.get("relations") or []
        return {"tables": tables_out, "relations": relations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/text2sql/schema")
async def put_text2sql_schema(body: SchemaOverridesBody):
    """
    保存人工审核的表/列含义及表间关联到 text2sql_schema_overrides_path（JSON）。
    Text2SQL 生成 SQL 时会使用该文件中的表间关联做多表 JOIN 校验，避免笛卡尔积。
    写入操作在线程池执行，不阻塞事件循环。
    """
    try:
        table_comments = body.table_comments or {}
        column_comments = body.column_comments or {}
        relations = [
            {"left_table": r.left_table, "left_column": r.left_column, "right_table": r.right_table, "right_column": r.right_column}
            for r in (body.relations or [])
        ]
        await asyncio.to_thread(save_schema_overrides, table_comments, column_comments, relations)
        return {"ok": True, "message": "已保存，表间关联将仅使用人工整理的内容喂给大模型。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _count_user_turns(messages: list) -> int:
    """
    统计当前对话中用户消息条数，即「已完成的轮数」。
    用于与 max_conversation_turns 比较，超过则提示新开对话。
    """
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def _conversation_title_from_message(message: str) -> str:
    """用首条用户消息生成会话标题，风格接近常见 AI 对话产品。"""
    text = " ".join((message or "").strip().split())
    if not text:
        return "新对话"
    return text[:24] + ("..." if len(text) > 24 else "")


def _serialize_chat_messages(
    messages: list,
    assistant_metadata: Optional[List[dict]] = None,
    feedback_map: Optional[dict[str, dict]] = None,
) -> List[ChatHistoryItem]:
    """将图状态中的消息序列化为前端可消费的历史消息列表，并尽量补齐 observation_id/反馈信息。"""
    items: List[ChatHistoryItem] = []
    assistant_metadata = assistant_metadata or []
    feedback_map = feedback_map or {}
    assistant_idx = 0
    for m in messages or []:
        if isinstance(m, HumanMessage):
            items.append(ChatHistoryItem(role="user", content=(m.content or "").strip()))
        elif isinstance(m, AIMessage):
            meta = assistant_metadata[assistant_idx] if assistant_idx < len(assistant_metadata) else {}
            assistant_idx += 1
            observation_id = (meta.get("observation_id") or "").strip() or None
            feedback = feedback_map.get(observation_id or "", {})
            items.append(
                ChatHistoryItem(
                    role="assistant",
                    content=(m.content or "").strip(),
                    observation_id=observation_id,
                    feedback_rating=(feedback.get("rating") or "").strip(),
                    feedback_tags=list(feedback.get("tags") or []),
                    feedback_text=(feedback.get("free_text") or "").strip(),
                )
            )
    return items


def _quality_label_from_trace(trace: Optional[dict], *, success: bool, route: str, fallback_reason: str = "") -> str:
    """根据 route 与知识库 trace 生成可聚合的质量标签。"""
    if not success:
        return "failed"
    if route == "chat":
        return "chat_response"
    if route == "cache":
        return "cache_hit"
    if route == "text2sql_confirm":
        return "text2sql_confirmed"
    trace = trace or {}
    final_status = (trace.get("final_status") or "").strip()
    if final_status in ("qa_hit", "text2sql_answer", "text2sql_pending", "rag_grounded", "rag_regenerated"):
        return final_status
    if final_status in ("rag_no_hit", "rag_no_context"):
        return "no_retrieval_hit"
    if final_status == "rag_fallback_unconfirmed":
        return "fallback"
    if fallback_reason:
        return fallback_reason[:64]
    return "unknown"


def _build_observation_payload(
    *,
    observation_id: str,
    thread_id: str,
    conversation_id: str,
    user_id: Optional[str],
    question: str,
    answer: str,
    route: str,
    response_mode: str,
    success: bool,
    latency_ms: int,
    used_cache: bool = False,
    pending_sql: bool = False,
    trace: Optional[dict] = None,
    fallback_reason: str = "",
) -> dict:
    """构造统一的问答观测记录。"""
    settings = get_settings()
    llm_model = getattr(settings, "llm_model", "")
    endpoint_name = get_last_deepseek_endpoint_name()
    return {
        "observation_id": observation_id,
        "thread_id": thread_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "route": route,
        "response_mode": response_mode,
        "success": success,
        "used_cache": used_cache,
        "pending_sql": pending_sql or bool((trace or {}).get("pending_sql")),
        "latency_ms": max(0, int(latency_ms)),
        "quality_label": _quality_label_from_trace(trace, success=success, route=route, fallback_reason=fallback_reason),
        "fallback_reason": fallback_reason or (trace or {}).get("fallback_reason") or "",
        "llm_model": llm_model,
        "llm_endpoint_name": endpoint_name,
    }


def _persist_conversation_session_sync(
    thread_id: str,
    conversation_id: str,
    user_id: Optional[str],
    title: str,
) -> None:
    """保存或刷新会话元数据，供前端左侧列表展示。"""
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        return
    chat_history_upsert_conversation_session(
        uri,
        thread_id,
        conversation_id,
        user_id,
        title,
    )


async def _generate_conversation_title_async(message: str) -> str:
    """用大模型为新会话生成简短自然标题；失败时回退到首句截断。"""
    fallback = _conversation_title_from_message(message)
    text = (message or "").strip()
    if not text:
        return fallback
    try:
        llm = get_deepseek_llm(temperature=0.2)
        prompt = (
            "请基于用户的第一条消息生成一个简短中文对话标题，8到16个字，"
            "不要带引号、句号、书名号、冒号，只输出标题本身。\n\n"
            f"用户消息：{text[:500]}"
        )
        out = await llm.ainvoke(prompt)
        title = " ".join((out.content or "").strip().split())
        if not title:
            return fallback
        title = title.strip("《》“”\"'：:。.!?？")
        return title[:24] if title else fallback
    except Exception:
        return fallback


async def _persist_conversation_session(
    thread_id: str,
    conversation_id: str,
    user_id: Optional[str],
    title_seed: str,
) -> None:
    """创建或刷新会话元数据；若是新会话则先生成更自然的标题。"""
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        return
    existing = await asyncio.to_thread(chat_history_get_conversation_session, uri, thread_id)
    if existing and (existing.get("title") or "").strip():
        title = existing["title"].strip()
    else:
        title = await _generate_conversation_title_async(title_seed)
    await asyncio.to_thread(
        _persist_conversation_session_sync,
        thread_id,
        conversation_id,
        user_id,
        title,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, req: ChatRequest):
    """
    多智能体对话（非流式）：按 thread_id（含 user_id）隔离会话，单对话最多 15 轮。
    流程：resume 中断会话 → 检查轮数 → 问答缓存 → 「确认执行」处理 → 图 invoke → 处理 interrupt/pending_sql → 写缓存。
    """
    settings = get_settings()
    user_id = _user_id_from_request(request, req.user_id)
    conversation_id = req.conversation_id or str(uuid.uuid4())
    thread_id = _thread_id(user_id, conversation_id)
    started_at = time.perf_counter()
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # 长期记忆：若配置了 PostgreSQL，且当前进程内无该会话状态，则从 DB 加载并注入
    await _ensure_state_from_db(graph, config, thread_id)

    # 若该会话曾因 human_agent 调用 interrupt() 而暂停，需先 resume 再处理新消息
    await _resume_if_interrupted(
        graph,
        config,
        thread_id,
        float(getattr(settings, "agent_request_timeout_seconds", 120)),
    )

    # 先取当前状态，检查是否已达 15 轮（异步不阻塞）
    try:
        snapshot = await graph.aget_state(config) if hasattr(graph, "aget_state") else await asyncio.to_thread(graph.get_state, config)
        if snapshot and snapshot.values:
            current_messages = snapshot.values.get("messages") or []
            n = _count_user_turns(current_messages)
            if n >= settings.max_conversation_turns:
                observation_id = str(uuid.uuid4())
                reply_text = "本轮对话已达 15 轮，建议您新开对话窗口以获得更好体验。"
                await _save_qa_observation_async(
                    _build_observation_payload(
                        observation_id=observation_id,
                        thread_id=thread_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        question=req.message,
                        answer=reply_text,
                        route="chat_limit",
                        response_mode="sync",
                        success=False,
                        latency_ms=int((time.perf_counter() - started_at) * 1000),
                        fallback_reason="conversation_turn_limit",
                    )
                )
                return ChatResponse(
                    reply=reply_text,
                    conversation_id=conversation_id,
                    observation_id=observation_id,
                )
    except Exception:
        pass

    # 问答缓存：若历史问题命中缓存，直接返回并写入本轮对话状态（缓存异常不影响主流程）
    if getattr(settings, "answer_cache_enabled", True):
        try:
            cached = await get_cached_answer(req.message)
        except Exception:
            cached = None
        if cached is not None:
            observation_id = str(uuid.uuid4())
            if hasattr(graph, "aupdate_state"):
                await graph.aupdate_state(
                    config,
                    {"messages": [HumanMessage(content=req.message), AIMessage(content=cached)]},
                )
            else:
                await asyncio.to_thread(
                    graph.update_state,
                    config,
                    {"messages": [HumanMessage(content=req.message), AIMessage(content=cached)]},
                )
            await _persist_conversation_session(thread_id, conversation_id, user_id, req.message)
            await _persist_chat_messages( thread_id, [HumanMessage(content=req.message), AIMessage(content=cached)])
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=req.message,
                    answer=cached,
                    route="cache",
                    response_mode="sync",
                    success=True,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    used_cache=True,
                )
            )
            return ChatResponse(reply=cached, conversation_id=conversation_id, observation_id=observation_id)

    # 对话内「确认执行」：从共享状态取出该会话的待执行 SQL 并执行，结果写入图状态后返回
    if req.message.strip() in ("确认执行", "确认"):
        sql = await shared_state_module.pop_pending_sql(thread_id)
        await _persist_runtime_state(thread_id)
        if sql:
            observation_id = str(uuid.uuid4())
            try:
                from src.kb.text2sql import _execute_sql
                ex = await asyncio.to_thread(_execute_sql, sql, settings.text2sql_database_uri)
            except Exception as e:
                await shared_state_module.set_pending_sql(thread_id, sql)  # 放回以便重试
                await _persist_runtime_state(thread_id)
                reply_text = f"执行失败：{e}。请重试或转人工。"
                await _save_qa_observation_async(
                    _build_observation_payload(
                        observation_id=observation_id,
                        thread_id=thread_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        question=req.message,
                        answer=reply_text,
                        route="text2sql_confirm",
                        response_mode="sync",
                        success=False,
                        latency_ms=int((time.perf_counter() - started_at) * 1000),
                        fallback_reason="text2sql_confirm_failed",
                    )
                )
                return ChatResponse(reply=reply_text, conversation_id=conversation_id, observation_id=observation_id)
            if not ex.ok:
                await shared_state_module.set_pending_sql(thread_id, sql)
                await _persist_runtime_state(thread_id)
                reply_text = f"执行失败：{ex.error_message}。请重试或转人工。"
                await _save_qa_observation_async(
                    _build_observation_payload(
                        observation_id=observation_id,
                        thread_id=thread_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        question=req.message,
                        answer=reply_text,
                        route="text2sql_confirm",
                        response_mode="sync",
                        success=False,
                        latency_ms=int((time.perf_counter() - started_at) * 1000),
                        fallback_reason="text2sql_confirm_failed",
                    )
                )
                return ChatResponse(reply=reply_text, conversation_id=conversation_id, observation_id=observation_id)
            n = len(ex.rows) if ex.rows else 0
            reply_exec = "已按您确认执行，操作完成。" + (f" 影响 {n} 行。" if n else " 影响 0 行。")
            # 将本轮对话写入图状态，保持历史一致
            if hasattr(graph, "aupdate_state"):
                await graph.aupdate_state(
                    config,
                    {"messages": [HumanMessage(content=req.message), AIMessage(content=reply_exec)]},
                )
            else:
                await asyncio.to_thread(
                    graph.update_state,
                    config,
                    {"messages": [HumanMessage(content=req.message), AIMessage(content=reply_exec)]},
                )
            await _persist_conversation_session(thread_id, conversation_id, user_id, req.message)
            await _persist_chat_messages( thread_id, [HumanMessage(content=req.message), AIMessage(content=reply_exec)])
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=req.message,
                    answer=reply_exec,
                    route="text2sql_confirm",
                    response_mode="sync",
                    success=True,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                )
            )
            return ChatResponse(reply=reply_exec, conversation_id=conversation_id, observation_id=observation_id)

    # 防击穿：同一问题仅一个协程回源，持锁后再次查缓存再决定是否走图
    async with answer_lock(req.message):
        cached = await get_cached_answer(req.message)
        if cached is not None:
            observation_id = str(uuid.uuid4())
            if hasattr(graph, "aupdate_state"):
                await graph.aupdate_state(
                    config,
                    {"messages": [HumanMessage(content=req.message), AIMessage(content=cached)]},
                )
            else:
                await asyncio.to_thread(
                    graph.update_state,
                    config,
                    {"messages": [HumanMessage(content=req.message), AIMessage(content=cached)]},
                )
            await _persist_conversation_session(thread_id, conversation_id, user_id, req.message)
            await _persist_chat_messages( thread_id, [HumanMessage(content=req.message), AIMessage(content=cached)])
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=req.message,
                    answer=cached,
                    route="cache",
                    response_mode="sync",
                    success=True,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    used_cache=True,
                )
            )
            return ChatResponse(reply=cached, conversation_id=conversation_id, observation_id=observation_id)
        messages = [HumanMessage(content=req.message)]
        try:
            result = await asyncio.wait_for(
                graph.ainvoke({"messages": messages}, config),
                timeout=float(getattr(settings, "agent_request_timeout_seconds", 120)),
            )
        except asyncio.TimeoutError:
            observation_id = str(uuid.uuid4())
            reply_text = "请求处理超时，您可以重新发送或转人工客服。"
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=req.message,
                    answer=reply_text,
                    route="timeout",
                    response_mode="sync",
                    success=False,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    fallback_reason="timeout",
                )
            )
            return ChatResponse(
                reply=reply_text,
                conversation_id=conversation_id,
                observation_id=observation_id,
            )
        except asyncio.CancelledError:
            observation_id = str(uuid.uuid4())
            reply_text = "请求已取消。"
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=req.message,
                    answer=reply_text,
                    route="cancelled",
                    response_mode="sync",
                    success=False,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    fallback_reason="cancelled",
                )
            )
            return ChatResponse(
                reply=reply_text,
                conversation_id=conversation_id,
                observation_id=observation_id,
            )
        # 图返回 __interrupt__ 表示进入人工节点：把提示语返回用户，并记录 thread_id 以便下次先 resume
        if result.get("__interrupt__"):
            interrupts = result["__interrupt__"]
            if interrupts:
                observation_id = str(uuid.uuid4())
                payload = getattr(interrupts[0], "value", interrupts[0])
                reply = payload.get("human_message", payload) if isinstance(payload, dict) else str(payload)
                await shared_state_module.add_interrupted(thread_id)
                await _persist_runtime_state(thread_id)
                reply_text = reply or "已转人工客服，请稍候。"
                await _save_qa_observation_async(
                    _build_observation_payload(
                        observation_id=observation_id,
                        thread_id=thread_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        question=req.message,
                        answer=reply_text,
                        route="human",
                        response_mode="sync",
                        success=False,
                        latency_ms=int((time.perf_counter() - started_at) * 1000),
                        fallback_reason="human_handoff",
                    )
                )
                return ChatResponse(reply=reply_text, conversation_id=conversation_id, observation_id=observation_id)
        # 知识库节点若生成了删除/修改类 SQL 且未执行，会放入 state.pending_sql，这里写入共享状态
        if result.get("pending_sql"):
            await shared_state_module.set_pending_sql(thread_id, result["pending_sql"])
            await _persist_runtime_state(thread_id)
        last_msg = None
        for m in result.get("messages", []):
            if isinstance(m, AIMessage):
                last_msg = m
        reply = (last_msg.content if last_msg else "抱歉，未生成回复。").strip()
        observation_id = str(uuid.uuid4())
        route = (result.get("route") or ("knowledge" if result.get("qa_trace") else "chat")).strip()
        qa_trace = result.get("qa_trace") or None
        if getattr(settings, "answer_cache_enabled", True) and reply:
            try:
                await set_cached_answer(req.message, reply)
            except Exception:
                pass
        await _persist_conversation_session(thread_id, conversation_id, user_id, req.message)
        # 长期记忆：本轮 user + assistant 追加写入 PostgreSQL
        last_two = result.get("messages") or []
        if len(last_two) >= 2:
            await _persist_chat_messages( thread_id, last_two[-2:])
        await _save_qa_observation_async(
            _build_observation_payload(
                observation_id=observation_id,
                thread_id=thread_id,
                conversation_id=conversation_id,
                user_id=user_id,
                question=req.message,
                answer=reply,
                route=route,
                response_mode="sync",
                success=bool(reply),
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                pending_sql=bool(result.get("pending_sql")),
                trace=qa_trace,
                fallback_reason=(qa_trace or {}).get("fallback_reason", ""),
            ),
            trace=qa_trace,
        )
        return ChatResponse(reply=reply, conversation_id=conversation_id, observation_id=observation_id)


async def _chat_stream_generator(
    thread_id: str,
    conversation_id: str,
    message: str,
    user_id: Optional[str] = None,
    started_at: Optional[float] = None,
):
    """
    流式对话的 SSE 生成器：先出首字再逐 chunk；路由与 LLM 均异步，不阻塞事件循环。
    thread_id：图状态与共享状态（待确认 SQL）的键（含 user_id）；conversation_id 仅用于 SSE 的 done 事件给前端。
    顺序：确认执行分支 → 缓存命中 → 总控路由 → chat/knowledge 流式输出 → 写状态与缓存 → done。
    """
    settings_stream = get_settings()
    started_at = started_at or time.perf_counter()
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    # 流式下的「确认执行」：执行待确认 SQL 并流式返回结果，然后发送 done
    pending_sql_val = await shared_state_module.get_pending_sql(thread_id)
    if message.strip() in ("确认执行", "确认") and pending_sql_val:
        sql = await shared_state_module.pop_pending_sql(thread_id)
        observation_id = str(uuid.uuid4())
        await _persist_runtime_state(thread_id)
        try:
            from src.kb.text2sql import _execute_sql
            ex = await asyncio.to_thread(_execute_sql, sql, settings_stream.text2sql_database_uri)
        except Exception as e:
            await shared_state_module.set_pending_sql(thread_id, sql)
            await _persist_runtime_state(thread_id)
            reply_text = f"执行失败：{e}。请重试或转人工。"
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=message,
                    answer=reply_text,
                    route="text2sql_confirm",
                    response_mode="stream",
                    success=False,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    fallback_reason="text2sql_confirm_failed",
                )
            )
            yield f"data: {json.dumps({'text': reply_text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
            return
        if not ex.ok:
            await shared_state_module.set_pending_sql(thread_id, sql)
            await _persist_runtime_state(thread_id)
            reply_text = f"执行失败：{ex.error_message}。请重试或转人工。"
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=message,
                    answer=reply_text,
                    route="text2sql_confirm",
                    response_mode="stream",
                    success=False,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    fallback_reason="text2sql_confirm_failed",
                )
            )
            yield f"data: {json.dumps({'text': reply_text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
            return
        n = len(ex.rows) if ex.rows else 0
        reply_exec = "已按您确认执行，操作完成。" + (f" 影响 {n} 行。" if n else " 影响 0 行。")
        if hasattr(graph, "aupdate_state"):
            await graph.aupdate_state(config, {"messages": [HumanMessage(content=message), AIMessage(content=reply_exec)]})
        else:
            await asyncio.to_thread(graph.update_state, config, {"messages": [HumanMessage(content=message), AIMessage(content=reply_exec)]})
        await _persist_conversation_session(thread_id, conversation_id, user_id, message)
        await _persist_chat_messages( thread_id, [HumanMessage(content=message), AIMessage(content=reply_exec)])
        await _save_qa_observation_async(
            _build_observation_payload(
                observation_id=observation_id,
                thread_id=thread_id,
                conversation_id=conversation_id,
                user_id=user_id,
                question=message,
                answer=reply_exec,
                route="text2sql_confirm",
                response_mode="stream",
                success=True,
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        )
        yield f"data: {json.dumps({'text': reply_exec}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
        return

    # 问答缓存：问题归一化后查 Redis，命中则先流式输出首字再按 chunk 输出剩余，写状态后 done
    if getattr(settings_stream, "answer_cache_enabled", True):
        try:
            cached = await get_cached_answer(message)
        except Exception:
            cached = None
        if cached is not None:
            observation_id = str(uuid.uuid4())
            if cached:
                yield f"data: {json.dumps({'text': cached[0:1]}, ensure_ascii=False)}\n\n"
                rest = cached[1:]
                chunk_size = 64
                for i in range(0, len(rest), chunk_size):
                    yield f"data: {json.dumps({'text': rest[i:i + chunk_size]}, ensure_ascii=False)}\n\n"
            if hasattr(graph, "aupdate_state"):
                await graph.aupdate_state(config, {"messages": [HumanMessage(content=message), AIMessage(content=cached)]})
            else:
                await asyncio.to_thread(graph.update_state, config, {"messages": [HumanMessage(content=message), AIMessage(content=cached)]})
            await _persist_conversation_session(thread_id, conversation_id, user_id, message)
            await _persist_chat_messages( thread_id, [HumanMessage(content=message), AIMessage(content=cached)])
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=message,
                    answer=cached,
                    route="cache",
                    response_mode="stream",
                    success=True,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    used_cache=True,
                )
            )
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
            return

    # 防击穿：同一问题仅一个协程回源，持锁后再次查缓存再决定是否走图
    async with answer_lock(message):
        cached = await get_cached_answer(message)
        if cached is not None:
            observation_id = str(uuid.uuid4())
            if cached:
                yield f"data: {json.dumps({'text': cached[0:1]}, ensure_ascii=False)}\n\n"
                rest = cached[1:]
                chunk_size = 64
                for i in range(0, len(rest), chunk_size):
                    yield f"data: {json.dumps({'text': rest[i:i + chunk_size]}, ensure_ascii=False)}\n\n"
            if hasattr(graph, "aupdate_state"):
                await graph.aupdate_state(config, {"messages": [HumanMessage(content=message), AIMessage(content=cached)]})
            else:
                await asyncio.to_thread(graph.update_state, config, {"messages": [HumanMessage(content=message), AIMessage(content=cached)]})
            await _persist_conversation_session(thread_id, conversation_id, user_id, message)
            await _persist_chat_messages( thread_id, [HumanMessage(content=message), AIMessage(content=cached)])
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=message,
                    answer=cached,
                    route="cache",
                    response_mode="stream",
                    success=True,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    used_cache=True,
                )
            )
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
            return

        try:
            snapshot = await graph.aget_state(config) if hasattr(graph, "aget_state") else await asyncio.to_thread(graph.get_state, config)
            current = (snapshot.values or {}).get("messages") or []
        except Exception:
            current = []
        new_messages = list(current) + [HumanMessage(content=message)]
        new_state = {"messages": new_messages}

        # 总控路由：仅当明确为 knowledge 时走知识库流式，否则走闲聊流式
        next_action = (await supervisor_node_async(new_state)).get("next", "chat")
        if next_action != "knowledge":
            next_action = "chat"

        full_chunks: list = []
        try:
            qa_trace = None
            if next_action == "chat":
                async for chunk in chat_agent_stream_async(new_state):
                    full_chunks.append(chunk)
                    yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            else:
                engine = KnowledgeEngine()
                pending_holder: list = []
                async for chunk in engine.aquery_stream(message, pending_sql_out=pending_holder):
                    full_chunks.append(chunk)
                    yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
                qa_trace = engine.get_last_trace()
                # 知识库若生成了待确认 SQL，通过 pending_holder 带回，写入共享状态供对话内确认
                if pending_holder:
                    await shared_state_module.set_pending_sql(thread_id, pending_holder[0])
                    await _persist_runtime_state(thread_id)
            full_reply = "".join(full_chunks) if full_chunks else ""
            observation_id = str(uuid.uuid4())
            if full_reply:
                if hasattr(graph, "aupdate_state"):
                    await graph.aupdate_state(
                        config,
                        {"messages": [HumanMessage(content=message), AIMessage(content=full_reply)]},
                    )
                else:
                    await asyncio.to_thread(
                        graph.update_state,
                        config,
                        {"messages": [HumanMessage(content=message), AIMessage(content=full_reply)]},
                    )
                if getattr(settings_stream, "answer_cache_enabled", True):
                    try:
                        await set_cached_answer(message, full_reply)
                    except Exception:
                        pass
                await _persist_conversation_session(thread_id, conversation_id, user_id, message)
                await _persist_chat_messages( thread_id, [HumanMessage(content=message), AIMessage(content=full_reply)])
            route = "chat" if next_action == "chat" else "knowledge"
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=message,
                    answer=full_reply,
                    route=route,
                    response_mode="stream",
                    success=bool(full_reply),
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    pending_sql=bool(qa_trace and qa_trace.get("pending_sql")),
                    trace=qa_trace,
                    fallback_reason=(qa_trace or {}).get("fallback_reason", ""),
                ),
                trace=qa_trace,
            )
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            observation_id = str(uuid.uuid4())
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=message,
                    answer="请求已取消。",
                    route="cancelled",
                    response_mode="stream",
                    success=False,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    fallback_reason="cancelled",
                )
            )
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'cancelled': True, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            observation_id = str(uuid.uuid4())
            await _save_qa_observation_async(
                _build_observation_payload(
                    observation_id=observation_id,
                    thread_id=thread_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    question=message,
                    answer=str(e),
                    route="stream_error",
                    response_mode="stream",
                    success=False,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    fallback_reason="stream_exception",
                )
            )
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


class ConfirmSqlRequest(BaseModel):
    """确认执行待执行的 SQL（删除/修改类）。"""
    conversation_id: str
    user_id: Optional[str] = None  # 可选，与 X-User-Id 二选一，须与发起该会话的用户一致


@app.post("/text2sql/confirm_execute", response_model=dict)
async def text2sql_confirm_execute(request: Request, body: ConfirmSqlRequest):
    """
    人工确认后执行该会话下待执行的 SQL（删除/修改类）。
    带 X-User-Id 或 body.user_id 时，仅能确认该用户自己的待执行 SQL。
    """
    user_id = _user_id_from_request(request, body.user_id)
    thread_id = _thread_id(user_id, body.conversation_id)
    await _ensure_state_from_db(get_graph(), {"configurable": {"thread_id": thread_id}}, thread_id)
    sql = await shared_state_module.pop_pending_sql(thread_id)
    await _persist_runtime_state(thread_id)
    if not sql:
        raise HTTPException(
            status_code=400,
            detail="当前会话没有待确认的 SQL，或已执行/已过期。请先在对话中发起删除或修改类请求。",
        )
    settings = get_settings()
    uri = settings.text2sql_database_uri
    try:
        from src.kb.text2sql import _execute_sql
        ex = await asyncio.to_thread(_execute_sql, sql, uri)
    except Exception as e:
        await shared_state_module.set_pending_sql(thread_id, sql)
        await _persist_runtime_state(thread_id)
        raise HTTPException(status_code=500, detail=f"执行失败：{e}")
    if not ex.ok:
        await shared_state_module.set_pending_sql(thread_id, sql)
        await _persist_runtime_state(thread_id)
        raise HTTPException(status_code=400, detail=ex.error_message or "SQL 执行失败")
    if ex.rows is None or len(ex.rows) == 0:
        return {"ok": True, "message": "已执行，影响 0 行。", "rows_affected": 0}
    return {"ok": True, "message": "已执行。", "rows_affected": len(ex.rows)}


@app.post("/chat/stream")
async def chat_stream(request: Request, req: ChatRequest):
    """
    流式对话接口：SSE 流，先出首字再逐段输出；thread_id 含 user_id 时按用户隔离。
    返回：Content-Type text/event-stream；每条 data 为 JSON：text（内容片段）、done（结束）、conversation_id、error（异常）。
    若会话处于 interrupt 状态会先 resume，再走 _chat_stream_generator。
    """
    settings = get_settings()
    user_id = _user_id_from_request(request, req.user_id)
    conversation_id = req.conversation_id or str(uuid.uuid4())
    thread_id = _thread_id(user_id, conversation_id)
    started_at = time.perf_counter()
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    await _ensure_state_from_db(graph, config, thread_id)
    await _resume_if_interrupted(
        graph,
        config,
        thread_id,
        float(getattr(settings, "agent_request_timeout_seconds", 120)),
    )

    try:
        snapshot = await graph.aget_state(config) if hasattr(graph, "aget_state") else await asyncio.to_thread(graph.get_state, config)
        if snapshot and snapshot.values:
            current_messages = snapshot.values.get("messages") or []
            n = _count_user_turns(current_messages)
            if n >= settings.max_conversation_turns:
                observation_id = str(uuid.uuid4())
                reply_text = "本轮对话已达 15 轮，建议您新开对话窗口以获得更好体验。"
                await _save_qa_observation_async(
                    _build_observation_payload(
                        observation_id=observation_id,
                        thread_id=thread_id,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        question=req.message,
                        answer=reply_text,
                        route="chat_limit",
                        response_mode="stream",
                        success=False,
                        latency_ms=int((time.perf_counter() - started_at) * 1000),
                        fallback_reason="conversation_turn_limit",
                    )
                )
                async def over_limit():
                    yield f"data: {json.dumps({'text': reply_text}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id, 'observation_id': observation_id}, ensure_ascii=False)}\n\n"
                return StreamingResponse(
                    over_limit(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
    except Exception:
        pass

    return StreamingResponse(
        _chat_stream_generator(thread_id, conversation_id, req.message, user_id, started_at),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(request: Request, conversation_id: str, user_id: Optional[str] = None):
    """
    获取某个会话的完整历史消息，供前端在刷新页面或重新进入会话时恢复展示。
    会先按 thread_id 恢复 PostgreSQL 长期记忆到图状态，再从当前状态中返回 user/assistant 消息。
    """
    actual_user_id = _user_id_from_request(request, user_id)
    thread_id = _thread_id(actual_user_id, conversation_id)
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    await _ensure_state_from_db(graph, config, thread_id)

    try:
        snapshot = await graph.aget_state(config) if hasattr(graph, "aget_state") else await asyncio.to_thread(graph.get_state, config)
        current_messages = (snapshot.values or {}).get("messages") or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取会话历史失败：{e}")

    qa_monitor_uri = _qa_monitoring_uri(get_settings())
    assistant_observations: List[dict] = []
    feedback_map: dict[str, dict] = {}
    if qa_monitor_uri and "postgresql" in qa_monitor_uri.lower():
        assistant_observations = await asyncio.to_thread(
            qa_monitoring_list_conversation_observations,
            qa_monitor_uri,
            conversation_id,
            actual_user_id,
        )
        feedback_map = await asyncio.to_thread(
            qa_monitoring_get_feedback_map,
            qa_monitor_uri,
            conversation_id,
            actual_user_id,
        )

    return ChatHistoryResponse(
        conversation_id=conversation_id,
        messages=_serialize_chat_messages(current_messages, assistant_metadata=assistant_observations, feedback_map=feedback_map),
    )


@app.get("/chat/conversations", response_model=ConversationListResponse)
async def chat_conversations(request: Request, user_id: Optional[str] = None):
    """按更新时间倒序返回当前用户的会话列表，供前端左侧会话栏展示。"""
    actual_user_id = _user_id_from_request(request, user_id)
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        return ConversationListResponse(conversations=[])
    conversations = await asyncio.to_thread(chat_history_list_conversation_sessions, uri, actual_user_id)
    return ConversationListResponse(
        conversations=[ConversationListItem(**item) for item in conversations],
    )


@app.put("/chat/conversations/title", response_model=dict)
async def rename_chat_conversation(request: Request, body: RenameConversationRequest):
    """重命名某个会话标题。"""
    actual_user_id = _user_id_from_request(request, body.user_id)
    thread_id = _thread_id(actual_user_id, body.conversation_id)
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        raise HTTPException(status_code=400, detail="未启用 PostgreSQL 会话持久化，暂不支持会话标题管理。")
    ok = await asyncio.to_thread(chat_history_rename_conversation_session, uri, thread_id, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或标题无效。")
    return {"ok": True, "conversation_id": body.conversation_id, "title": body.title.strip()}


@app.delete("/chat/conversations/{conversation_id}", response_model=dict)
async def delete_chat_conversation(request: Request, conversation_id: str, user_id: Optional[str] = None):
    """删除某个会话的长期记忆与当前进程内状态。"""
    actual_user_id = _user_id_from_request(request, user_id)
    thread_id = _thread_id(actual_user_id, conversation_id)
    settings = get_settings()
    uri = (getattr(settings, "chat_history_postgresql_uri", None) or "").strip()
    if not uri or "postgresql" not in uri.lower():
        raise HTTPException(status_code=400, detail="未启用 PostgreSQL 会话持久化，暂不支持删除会话。")
    await asyncio.to_thread(chat_history_delete_conversation_session, uri, thread_id)
    qa_monitor_uri = _qa_monitoring_uri(settings)
    if qa_monitor_uri and "postgresql" in qa_monitor_uri.lower():
        await asyncio.to_thread(qa_monitoring_delete_conversation_data, qa_monitor_uri, conversation_id, actual_user_id)
    await shared_state_module.pop_pending_sql(thread_id)
    await shared_state_module.discard_interrupted(thread_id)
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        if hasattr(graph, "aupdate_state"):
            await graph.aupdate_state(config, {"messages": []})
        else:
            await asyncio.to_thread(graph.update_state, config, {"messages": []})
    except Exception:
        pass
    return {"ok": True, "conversation_id": conversation_id}


@app.post("/qa/feedback", response_model=dict)
async def qa_feedback(request: Request, body: QAFeedbackRequest):
    """记录或更新用户对某条回答的反馈（点赞/点踩 + 标签 + 补充说明）。"""
    rating = (body.rating or "").strip().lower()
    if rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating 仅支持 up 或 down。")
    actual_user_id = _user_id_from_request(request, body.user_id)
    uri = _qa_monitoring_uri(get_settings())
    if not uri or "postgresql" not in uri.lower():
        raise HTTPException(status_code=400, detail="未启用问答监控 PostgreSQL，暂不支持反馈分析。")
    payload = await asyncio.to_thread(
        qa_monitoring_upsert_feedback,
        uri,
        observation_id=body.observation_id,
        conversation_id=body.conversation_id,
        user_id=actual_user_id,
        rating=rating,
        tags=list(body.tags or []),
        free_text=(body.free_text or "").strip(),
    )
    return {"ok": True, **payload}


@app.get("/qa/analytics", response_model=QAMonitorAnalyticsResponse)
async def qa_analytics(request: Request, days: int = 7, limit: int = 20, user_id: Optional[str] = None):
    """返回问答效果监控总览、差评案例、反馈标签与场景统计。"""
    actual_user_id = _user_id_from_request(request, user_id)
    uri = _qa_monitoring_uri(get_settings())
    if not uri or "postgresql" not in uri.lower():
        return QAMonitorAnalyticsResponse(
            summary=QAMonitorSummaryResponse(
                days=max(1, days),
                total_questions=0,
                success_count=0,
                cache_hit_count=0,
                knowledge_count=0,
                negative_feedback_count=0,
                positive_feedback_count=0,
                avg_latency_ms=0.0,
                low_grounding_count=0,
                regenerated_count=0,
                fallback_count=0,
            ),
            bad_cases=[],
            feedback_tags=[],
            scenarios=[],
        )
    summary = await asyncio.to_thread(qa_monitoring_get_summary, uri, days=max(1, days), user_id=actual_user_id)
    bad_cases = await asyncio.to_thread(qa_monitoring_list_bad_cases, uri, days=max(1, days), user_id=actual_user_id, limit=max(1, limit))
    feedback_tags = await asyncio.to_thread(qa_monitoring_list_feedback_tag_stats, uri, days=max(1, days), user_id=actual_user_id, limit=max(1, limit))
    scenarios = await asyncio.to_thread(qa_monitoring_list_scenario_stats, uri, days=max(1, days), user_id=actual_user_id, limit=max(1, limit))
    return QAMonitorAnalyticsResponse(
        summary=QAMonitorSummaryResponse(**summary),
        bad_cases=[QABadCaseItem(**item) for item in bad_cases],
        feedback_tags=[QAFeedbackItem(**item) for item in feedback_tags],
        scenarios=[QAScenarioStatItem(**item) for item in scenarios],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
