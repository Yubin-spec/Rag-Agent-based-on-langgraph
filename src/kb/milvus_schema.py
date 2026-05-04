# src/kb/milvus_schema.py
"""
Milvus Collection Schema 统一管理。

提供：
- kb_chunks Schema（HNSW + 标量索引，分仓字段）
- faq_chunks Schema（HNSW + 标量索引）
- 索引参数统一从 settings 读取
- 索引信息查询（行数、索引状态）

索引设计文档见 docs/Milvus索引结构设计.md
"""
import logging
from typing import Any, Dict, List, Optional

from config import get_settings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

logger = logging.getLogger(__name__)


def _hnsw_index_params() -> Dict[str, Any]:
    """从配置读取 HNSW 索引参数字典。"""
    s = get_settings()
    return {
        "metric_type": "IP",  # 内积；BGE-M3 归一化后 IP 等价于 Cosine
        "index_type": "HNSW",
        "params": {
            "M": getattr(s, "milvus_hnsw_m", 16),
            "efConstruction": getattr(s, "milvus_hnsw_ef_construction", 256),
        },
    }


def _scalar_index_params() -> Dict[str, Any]:
    """标量索引参数（STL_SORT）。"""
    return {}


# ---------- kb_chunks Schema ----------

KB_CHUNKS_FIELDS: List[Dict[str, Any]] = [
    # 主键与溯源
    {"name": "id",            "dtype": DataType.VARCHAR, "max_length": 256,  "is_primary": True},
    {"name": "doc_id",        "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "chunk_id",      "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "doc_name",      "dtype": DataType.VARCHAR, "max_length": 256},
    # 内容层级
    {"name": "page",          "dtype": DataType.INT64},
    {"name": "parent_block",  "dtype": DataType.INT64},
    {"name": "child_block",   "dtype": DataType.INT64},
    {"name": "content",       "dtype": DataType.VARCHAR, "max_length": 65535},
    {"name": "parent_content","dtype": DataType.VARCHAR, "max_length": 65535},
    # 分仓字段
    {"name": "kb_tier",       "dtype": DataType.VARCHAR, "max_length": 32},
    {"name": "org_id",        "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "owner_user_id", "dtype": DataType.VARCHAR, "max_length": 256},
    {"name": "doc_source",    "dtype": DataType.VARCHAR, "max_length": 64},
    # 向量
    {"name": "embedding",     "dtype": DataType.FLOAT_VECTOR, "dim": 1024},
]

KB_CHUNKS_SCALAR_INDEX_FIELDS: List[str] = [
    "kb_tier", "org_id", "owner_user_id",
]


def build_kb_chunks_schema(collection_name: str) -> Collection:
    """
    创建 kb_chunks Collection：向量 HNSW 索引 + 分仓标量索引。

    索引参数从 settings 读取（milvus_hnsw_m / ef_construction / ef_search）。
    chunk_id 格式：「文档名-p页码-b父块编号-c子块编号」
    """
    s = get_settings()
    dim = s.milvus_dim

    fields = [
        FieldSchema(name=f["name"], dtype=f["dtype"], **{
            k: v for k, v in f.items() if k not in ("name", "dtype")
        })
        for f in KB_CHUNKS_FIELDS
    ]
    schema = CollectionSchema(
        fields=fields,
        description="kb chunks with scope metadata",
    )
    coll = Collection(name=collection_name, schema=schema)

    # 向量索引
    coll.create_index(
        field_name="embedding",
        index_params=_hnsw_index_params(),
    )

    # 标量索引
    for fname in KB_CHUNKS_SCALAR_INDEX_FIELDS:
        try:
            coll.create_index(field_name=fname, index_params=_scalar_index_params())
        except Exception as e:
            logger.debug("kb_chunks 标量索引可选跳过 %s: %s", fname, e)

    coll.load()
    return coll


# ---------- faq_chunks Schema ----------

FAQ_CHUNKS_FIELDS: List[Dict[str, Any]] = [
    # 主键
    {"name": "id",               "dtype": DataType.VARCHAR, "max_length": 128,  "is_primary": True},
    # 问句归一化
    {"name": "question_norm",     "dtype": DataType.VARCHAR, "max_length": 1024},
    # 内容
    {"name": "candidate_text",   "dtype": DataType.VARCHAR, "max_length": 4096},
    {"name": "matched_question", "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "matched_alias",    "dtype": DataType.VARCHAR, "max_length": 2048},
    {"name": "answer",           "dtype": DataType.VARCHAR, "max_length": 65535},
    # 元信息
    {"name": "match_type",       "dtype": DataType.VARCHAR, "max_length": 32},
    # 向量
    {"name": "embedding",        "dtype": DataType.FLOAT_VECTOR, "dim": 1024},
]

FAQ_CHUNKS_SCALAR_INDEX_FIELDS: List[str] = [
    "question_norm",
]


def build_faq_chunks_schema(collection_name: str) -> Collection:
    """
    创建 faq_chunks Collection：向量 HNSW 索引 + question_norm 标量索引。

    用于高频 QA 精确匹配与语义召回。
    """
    s = get_settings()
    dim = s.milvus_dim

    fields = [
        FieldSchema(name=f["name"], dtype=f["dtype"], **{
            k: v for k, v in f.items() if k not in ("name", "dtype")
        })
        for f in FAQ_CHUNKS_FIELDS
    ]
    schema = CollectionSchema(
        fields=fields,
        description="faq chunks (exact/alias + semantic)",
    )
    coll = Collection(name=collection_name, schema=schema)

    # 向量索引
    coll.create_index(
        field_name="embedding",
        index_params=_hnsw_index_params(),
    )

    # 标量索引
    for fname in FAQ_CHUNKS_SCALAR_INDEX_FIELDS:
        try:
            coll.create_index(field_name=fname, index_params=_scalar_index_params())
        except Exception as e:
            logger.debug("faq_chunks 标量索引可选跳过 %s: %s", fname, e)

    coll.load()
    return coll


# ---------- 索引信息查询 ----------

def get_collection_stats(uri: str, collection_name: str) -> Optional[Dict[str, Any]]:
    """
    查询 Collection 统计信息：行数、索引状态、字段列表。
    失败时返回 None。
    """
    from src.db_resilience import milvus_operation_with_retry

    def _get_info(coll: Collection) -> Dict[str, Any]:
        try:
            indexes = {}
            for idx in coll.indexes:
                indexes[idx.field] = {
                    "index_type": idx.params.get("index_type", "UNKNOWN"),
                    "metric_type": idx.params.get("metric_type", "UNKNOWN"),
                    "params": idx.params.get("params", {}),
                }
            return {
                "collection_name": coll.name,
                "num_entities": coll.num_entities,
                "indexes": indexes,
            }
        except Exception as e:
            logger.warning("获取 collection 信息失败: %s", e)
            return {}

    return milvus_operation_with_retry(
        uri,
        collection_name,
        _get_info,
        retries=1,
        critical=False,
        default=None,
    )
