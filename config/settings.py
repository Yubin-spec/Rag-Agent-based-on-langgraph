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
    rag_answer_grounding_min_score: float = 0.18  # 最终答案与检索文档的最低关联度，低于该值视为可能幻觉
    rag_answer_max_regenerate_times: int = 3  # 最终答案与文档关联度低时，最多重生成 3 次

    # ---------- Text2SQL ----------
    text2sql_database_uri: str = "sqlite:///./data/kb.db"
    text2sql_schema_refresh_interval_seconds: int = 3600  # 1 小时检测 schema 是否变更并重新扫描
    text2sql_schema_overrides_path: str = "./data/text2sql_schema_overrides.json"  # 人工审核的表/列含义及表间关联

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

    # ---------- 路径 ----------
    upload_dir: str = "./data/uploads"
    qa_data_path: str = "./data/high_freq_qa.json"

    # ---------- 对话与上下文（性能与隔离） ----------
    max_conversation_turns: int = 15  # 单对话最多轮数，超过后提示新开对话
    llm_context_window_turns: int = 10  # 喂给大模型的最近轮数（节省 token），窗口内全量保留不压缩
    llm_context_summarize_old: bool = True  # 是否将窗口外的旧对话压缩为摘要后一并喂给大模型（避免早期信息丢失）

    # ---------- 异常与人工介入 ----------
    agent_request_timeout_seconds: int = 120  # 单次请求超时，超时后可提示用户打断或转人工
    agent_llm_retry_times: int = 2  # LLM 调用失败时重试次数（如超时、5xx）
    agent_need_human_reply: str = "当前服务暂时异常，请稍后重试或转人工客服。"  # 节点异常时统一提示

    # ---------- 问答缓存（Redis） ----------
    redis_url: str = "redis://localhost:6379/0"  # 为空或连接失败时不使用缓存
    answer_cache_ttl_seconds: int = 86400  # 缓存 TTL，默认 24 小时
    answer_cache_enabled: bool = True  # 是否启用问答缓存

    # ---------- PostgreSQL 探活 ----------
    postgresql_keepalive_uri: str = ""  # 为空则不探活；设为 postgresql://... 时每分钟执行 SELECT 1 保持连接可用
    postgresql_keepalive_interval_seconds: int = 60  # 探活间隔，默认 60 秒

    # ---------- 对话长期记忆（PostgreSQL） ----------
    chat_history_postgresql_uri: str = ""  # 为空则仅用内存（MemorySaver）；设为 postgresql://... 时对话历史落库，进程重启后可恢复
    qa_monitoring_postgresql_uri: str = ""  # 问答效果监控与用户反馈分析库；为空时回退到 chat_history_postgresql_uri

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
