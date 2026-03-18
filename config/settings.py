# config/settings.py
"""
项目全局配置。
注意：本项目仅允许使用以下模型，禁止调用 OpenAI 自有模型：
  - 大模型：DeepSeek R1（或 deepseek-chat），通过 DeepSeek API 调用；
  - 向量模型：BGE-M3（BAAI/bge-m3）；
  - 重排模型：BGE Reranker Large（BAAI/bge-reranker-large）。
"""

from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache


# 允许使用的大模型名称（仅 DeepSeek，禁止 OpenAI 的 gpt-4 等）
ALLOWED_LLM_PREFIXES = ("deepseek-", "deepseek_reasoner")
# 允许的向量/重排模型：仅 BGE 系列
ALLOWED_EMBEDDING_MODELS = ("BAAI/bge-m3", "bge-m3")
ALLOWED_RERANKER_MODELS = ("BAAI/bge-reranker-large", "bge-reranker-large")


class Settings(BaseSettings):
    """
    配置项说明：
    - 大模型：仅使用 DeepSeek，通过兼容 OpenAI 的 API 调用（base_url 必须指向 DeepSeek）。
    - 向量与重排：仅使用 BGE-M3、BGE Reranker Large，本地加载。
    """

    # ---------- 大模型（仅 DeepSeek，禁止 OpenAI） ----------
    # 环境变量 OPENAI_API_BASE：必须为 DeepSeek API 地址，如 https://api.deepseek.com/v1
    openai_api_base: str = "https://api.deepseek.com/v1"
    # 环境变量 OPENAI_API_KEY：DeepSeek API Key
    openai_api_key: str = ""
    # 模型名：仅允许 deepseek-chat、deepseek-reasoner 等，禁止 gpt-4 等 OpenAI 模型
    llm_model: str = "deepseek-chat"
    # 多 endpoint 配置，格式：base_url|api_key|weight|name，多项可用逗号或换行分隔；为空时退回单节点配置
    deepseek_api_endpoints: str = ""
    deepseek_circuit_breaker_failures: int = 3  # 同一节点连续失败达到该次数后熔断
    deepseek_circuit_breaker_open_seconds: int = 30  # 熔断持续时间，到期后自动半开探测

    # ---------- 向量模型（仅 BGE-M3，本地部署/本地加载） ----------
    # 可为 HuggingFace 模型名（如 BAAI/bge-m3）或本地路径（如 /path/to/bge-m3），本地加载不调用远程 API
    bge_embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # ---------- 重排模型（仅 BGE Reranker Large，本地部署/本地加载） ----------
    # 可为 HuggingFace 模型名或本地路径，本地加载不调用远程 API
    bge_reranker_model: str = "BAAI/bge-reranker-large"

    # ---------- Milvus 向量库 ----------
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "kb_chunks"
    milvus_dim: int = 1024  # BGE-M3 向量维度

    # ---------- RAG 切片与检索 ----------
    rag_chunk_sizes: List[int] = [256, 384, 512, 768]
    rag_parent_overlap: int = 150
    rag_bm25_ratio: float = 0.3
    rag_vector_ratio: float = 0.7
    rag_min_match_score: float = 0.3
    rag_max_retrieve_attempts: int = 3
    rag_default_chunk_size: int = 512
    # 默认：父子分段 + 切点句/段边界对齐（避免断句、不过碎）。True 时仅用固定长度无对齐（兼容旧行为）
    rag_use_legacy_fixed_chunking: bool = False
    rag_answer_grounding_min_score: float = 0.18  # 最终答案与检索文档的最低关联度，低于该值视为可能幻觉
    rag_answer_max_regenerate_times: int = 3  # 最终答案与文档关联度低时，最多重生成 3 次
    # 注入生成阶段的检索上下文总字符数上限，0 表示不限制；超出则从末尾丢弃 chunk，控制 token
    rag_max_context_chars: int = 0
    # RAG 检索结果缓存：进程内 LRU 条数，0 表示关闭；相同 query 命中则跳过 Milvus+BM25
    rag_retrieval_cache_max_entries: int = 0
    rag_retrieval_cache_ttl_seconds: int = 300  # 仅当 max_entries>0 时有效
    # RRF 融合：True 时 BM25 与向量两路用 RRF 合并排序，否则按比例取条数
    rag_use_rrf: bool = True
    rag_rrf_k: int = 60  # RRF 公式 1/(k+rank) 的 k
    # 重排前规则过滤：True 时丢弃与 query 无任何词/字重叠的 chunk，减少送入重排的噪音
    rag_pre_rerank_require_query_overlap: bool = True
    # 重排后多样性：保留重排得分最高的前 N 条作为必选证据（保证 top 可引用），其余名额用 MMR 选多样性
    rag_rerank_anchor_count: int = 3  # 必选证据条数，保证最终 top3 可引用
    rag_use_diversity_after_rerank: bool = True
    rag_diversity_mmr_lambda: float = 0.8  # MMR 中相关性权重，越高越偏向重排分
    # Query 规则改写：配置表路径非空且启用时，检索前对 query 做归一化/纠错/同义词替换（规则需自行梳理，见 docs/QUERY_REWRITE_RULES.md）
    rag_use_query_rewrite_by_rules: bool = False
    rag_query_rewrite_rules_path: str = "data/query_rewrite_rules.json"  # 相对项目根；示例见 data/query_rewrite_rules.example.json
    # 二次 RAG：首轮 RAG+生成效果不达标时，是否尝试基于 Q2（改写/细化后的问题）再检索一次
    rag_enable_second_round: bool = False
    # 触发二次 RAG 的 grounding 下限；若为 0 则默认使用 rag_answer_grounding_min_score 的一半
    rag_second_round_min_grounding: float = 0.1
    # 二次 RAG 时放大的 top_k 系数（>1 表示扩大召回范围）
    rag_second_round_top_k_factor: float = 1.5

    # ---------- Text2SQL ----------
    text2sql_database_uri: str = "sqlite:///./data/kb.db"
    text2sql_schema_refresh_interval_seconds: int = 3600  # 1 小时检测 schema 是否变更并重新扫描
    text2sql_schema_overrides_path: str = "./data/text2sql_schema_overrides.json"  # 人工审核的表/列含义及表间关联
    text2sql_default_limit: int = 500  # SELECT 无 LIMIT 时自动追加的上限，避免大表全表扫；0 表示不自动加

    # ---------- knowledge 二级路由（QA miss 后：Text2SQL vs RAG） ----------
    # True：当规则判别不确定时，调用 DeepSeek 做一次二分类推理（更准，略慢）
    knowledge_router_use_llm_when_uncertain: bool = True
    # 规则判别命中时的优先级策略：True 表示规则直接决定不再调用 LLM；False 表示始终调用 LLM 覆盖规则（更准更慢）
    knowledge_router_rules_short_circuit: bool = True
    # LLM 路由结果进程内缓存（LRU）条数；0 表示不缓存
    knowledge_router_cache_max_entries: int = 256
    # 低置信度时是否发起澄清（避免硬分流误判）
    knowledge_router_clarify_on_low_confidence: bool = True
    # 低置信度阈值（0~1），低于该值时优先澄清
    knowledge_router_low_confidence_threshold: float = 0.6

    # ---------- MinerU 文档解析（本地部署版） ----------
    # MinerU 本地部署时填写本地服务地址，如 http://127.0.0.1:8001；为空则使用占位解析（仅文本+切块）
    mineru_api_url: str = ""
    # 本地部署时通常不需要 token，留空即可；若服务端要求鉴权再填写
    mineru_api_token: str = ""
    # True：使用本地 MinerU 时配置 mineru_api_url 即可；url 为空则使用占位解析
    mineru_use_local: bool = True
    # 并发控制：同时调用 MinerU API 的最大数量，超出时排队等待，避免打满 MinerU
    mineru_concurrency_limit: int = 5
    # MinerU API 单次请求超时（秒）
    mineru_timeout_seconds: int = 120
    # MinerU 请求失败时重试次数（5xx/超时/连接错误），0 表示不重试
    mineru_retry_times: int = 2

    # ---------- 路径 ----------
    upload_dir: str = "./data/uploads"
    qa_data_path: str = "./data/high_freq_qa.json"
    # 高频 QA 匹配策略：
    # - exact/alias：强精确匹配（最高精度）
    # - semantic：轻量相似度（用于兜底召回，需阈值控制）
    # - contains：历史包含匹配（兼容旧行为，误命中风险更高，建议逐步关闭）
    qa_enable_semantic_match: bool = True
    # 默认值偏保守但可命中常见改写；上线后建议用线上弱反馈分布再收紧/放宽
    qa_semantic_min_score: float = 0.55
    qa_semantic_min_query_coverage: float = 0.35
    qa_semantic_require_any_overlap: bool = True  # True 时先做字符重叠过滤，再计算分数
    qa_semantic_top_k: int = 30  # 召回候选数量（越大越不漏，越慢）
    qa_semantic_min_margin: float = 0.03  # top1-top2 分差，小于该值认为歧义，避免误命中
    # 语义匹配性能优化：先用便宜的 ngram 重叠做候选预筛，再对少量候选做 evaluate_retrieval 精排
    qa_semantic_ngram_size: int = 2  # 中文场景用 2-gram 通常性价比最好
    qa_semantic_prefilter_top_n: int = 80  # 预筛后最多进入精排的候选数
    qa_match_cache_max_entries: int = 512  # 高频 QA 匹配结果 LRU 缓存（按归一化 query）
    # 验证器：防止“分数最高但其实不相关”
    qa_semantic_max_irrelevant_ratio: float = 0.75  # 无关信息比例上限；越低越严格
    qa_semantic_min_ngram_overlap_count: int = 3  # query 与候选 ngram 重叠计数下限
    qa_enable_legacy_contains_match: bool = True

    # ---------- 对话与上下文（性能与隔离） ----------
    max_conversation_turns: int = 15  # 单对话最多轮数，超过后提示新开对话
    llm_context_window_turns: int = 10  # 喂给大模型的最近轮数（节省 token），窗口内全量保留不压缩
    llm_context_summarize_old: bool = True  # 是否将窗口外的旧对话压缩为摘要后一并喂给大模型（避免早期信息丢失）
    # 单条消息字符上限（约等于 token 控制）：历史消息超过则截断，当前轮用户输入不截断
    llm_context_max_chars_per_message_old: int = 600  # 历史轮每条消息最大字符数，0 表示不截断
    llm_context_max_chars_per_message_latest: int = 0   # 当前轮用户消息最大字符数，0 表示不截断
    # 旧对话摘要时送入 LLM 的对话文本最大字符数，0 表示不限制（默认 8000）
    llm_context_summary_input_max_chars: int = 8000

    # ---------- 异常与人工介入 ----------
    agent_request_timeout_seconds: int = 120  # 单次请求超时，超时后可提示用户打断或转人工
    agent_llm_retry_times: int = 2  # LLM 调用失败时重试次数（如超时、5xx）
    agent_need_human_reply: str = "当前服务暂时异常，请稍后重试或转人工客服。"  # 节点异常时统一提示

    # ---------- 异步高并发 ----------
    # 线程池大小：asyncio.to_thread 使用的默认 executor 的 max_workers；0 表示使用 Python 默认（约 min(32, cpu+4)）
    asyncio_thread_pool_workers: int = 0
    # Uvicorn 进程 worker 数；仅在生产启动（无 reload）时生效，默认 1。>1 时需配合负载均衡「会话保持」使用（见 docs/ASYNC_CONCURRENCY.md）
    uvicorn_workers: int = 7
    # API 全局限流：同时处理的请求数上限，0 表示不限制；超限返回 503
    api_max_concurrent_requests: int = 0
    # 优雅关闭：SIGTERM 后等待进行中请求完成的秒数
    graceful_shutdown_timeout_seconds: int = 30

    # ---------- 问答缓存（Redis） ----------
    redis_url: str = "redis://localhost:6379/0"  # 为空或连接失败时不使用缓存
    answer_cache_ttl_seconds: int = 86400  # 缓存 TTL，默认 24 小时
    answer_cache_enabled: bool = True  # 是否启用问答缓存
    answer_cache_max_value_bytes: int = 0  # 单条答案最大缓存字节数，0 表示不限制；超长不写入避免占满内存
    # 热 key：进程内 LRU 本地缓存条数，0 表示关闭；命中则不打 Redis，减轻热 key 压力
    answer_cache_local_max_entries: int = 0
    answer_cache_local_ttl_seconds: int = 60  # 本地缓存 TTL（秒），仅当 local_max_entries>0 时有效
    # 防击穿：单飞锁分桶数，同一问题仅一个协程回源，其余等待后读缓存；建议 256～1024
    answer_cache_single_flight_buckets: int = 256
    redis_max_connections: int = 10  # 连接池最大连接数，高并发时可调大
    redis_socket_connect_timeout: float = 2.0  # 建连超时（秒）
    redis_socket_timeout: float = 5.0  # 读写超时（秒）
    redis_health_check_interval: int = 30  # 连接池健康检查间隔（秒），0 表示不检查
    # Redis 哨兵（可选）：当两者均配置时，通过 Sentinel 获取 master 连接，忽略 redis_url 直连
    redis_sentinel_service_name: str = ""  # 如 mymaster
    redis_sentinel_nodes: str = ""  # 如 127.0.0.1:26379,127.0.0.1:26380

    # ---------- 会话锁 ----------
    # 同一会话（thread_id）的并发请求是否串行化，避免图状态/待确认 SQL 被并发写坏
    conversation_lock_enabled: bool = True
    conversation_lock_buckets: int = 1024  # 锁分桶数，thread_id 哈希取模；不同会话可能同桶

    # ---------- 多 worker 共享状态（P3） ----------
    # 当配置且 Redis 可用时，解析缓存、待确认 SQL、人工中断状态存 Redis，多进程/多 worker 共享
    shared_state_redis_url: str = ""  # 为空则使用进程内内存（单 worker 或无需共享）

    # ---------- 数据库连接池与熔断 ----------
    postgresql_pool_size: int = 5
    postgresql_max_overflow: int = 10
    postgresql_pool_timeout: int = 10
    postgresql_statement_timeout_ms: int = 0  # 单条 SQL 超时（毫秒），0 表示不限制
    db_circuit_breaker_threshold: int = 5  # 连续失败达该次数后熔断
    db_circuit_breaker_recovery_seconds: float = 30.0  # 熔断持续时间，到期后半开探测

    # ---------- PostgreSQL 探活 ----------
    postgresql_keepalive_uri: str = ""  # 为空则不探活；设为 postgresql://... 时每分钟执行 SELECT 1 保持连接可用
    postgresql_keepalive_interval_seconds: int = 60  # 探活间隔，默认 60 秒

    # ---------- 对话长期记忆（PostgreSQL） ----------
    chat_history_postgresql_uri: str = ""  # 为空则仅用内存（MemorySaver）；设为 postgresql://... 时对话历史落库，进程重启后可恢复
    qa_monitoring_postgresql_uri: str = ""  # 问答效果监控与用户反馈分析库；为空时回退到 chat_history_postgresql_uri
    # 加载时只取最近 N 条消息注入短期记忆，0 表示全量加载；可减少长会话的加载量与反序列化
    chat_history_load_max_messages: int = 0
    # 写入 chat_history 时单条 content 最大字符数，0 表示不截断（仍受 DB TEXT 限制）
    chat_history_max_content_chars: int = 0
    # 短期记忆：Redis checkpointer URL（需 Redis 带 RedisJSON/RediSearch，如 Redis 8+ 或 Stack）；为空则用进程内 MemorySaver
    chat_checkpointer_redis_url: str = ""
    # 长期记忆：True 时使用 asyncpg 异步读写（不占线程池），需安装 asyncpg；False 时用同步 SQLAlchemy + to_thread
    chat_history_use_asyncpg: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"

    @field_validator("openai_api_base")
    @classmethod
    def must_be_deepseek_api(cls, v: str) -> str:
        """禁止使用 OpenAI 官方 API，仅允许 DeepSeek。"""
        if not v:
            return v
        v_lower = v.lower()
        if "openai.com" in v_lower and "deepseek" not in v_lower:
            raise ValueError(
                "本项目禁止调用 OpenAI 模型。请将 OPENAI_API_BASE 设为 DeepSeek API 地址，例如 https://api.deepseek.com/v1"
            )
        return v

    @field_validator("deepseek_api_endpoints")
    @classmethod
    def deepseek_endpoints_must_be_compatible(cls, v: str) -> str:
        """多 endpoint 也只允许 DeepSeek 或其兼容网关，禁止 OpenAI 官方地址。"""
        raw = (v or "").strip()
        if not raw:
            return raw
        for item in [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]:
            base_url = item.split("|")[0].strip().lower()
            if "openai.com" in base_url and "deepseek" not in base_url:
                raise ValueError("DEEPSEEK_API_ENDPOINTS 中禁止填写 OpenAI 官方地址，请改为 DeepSeek 或其兼容网关。")
        return raw

    @field_validator("llm_model")
    @classmethod
    def llm_must_be_deepseek(cls, v: str) -> str:
        """大模型仅允许 DeepSeek 系列。"""
        if not v:
            return "deepseek-chat"
        v_lower = v.lower()
        if "deepseek" not in v_lower:
            raise ValueError(
                f"本项目仅允许使用 DeepSeek 大模型（如 deepseek-chat、deepseek-reasoner），禁止使用 OpenAI 等其它模型。当前值: {v}"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """
    获取全局配置单例；使用 lru_cache 避免重复解析环境变量与 .env。
    进程内首次调用后返回同一 Settings 实例。
    """
    return Settings()
