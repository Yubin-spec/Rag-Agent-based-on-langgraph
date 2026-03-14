# src/doc/milvus_upload.py
"""
将用户确认后的解析结果写入 Milvus，支持父子块与向量索引。
向量模型仅使用 BGE-M3（config 中 bge_embedding_model），不调用 OpenAI 或其它 embedding 服务。

存储格式：每条记录包含结构化元数据（doc_name、page、parent_block、child_block），
chunk_id 格式为「文档名-p页码-b父块编号-c子块编号」，便于按文档/页码/层级检索与溯源。

连接韧性：通过 db_resilience 管理 Milvus 连接，支持重试、熔断与懒重连。
"""
import logging
from typing import List, Optional

from config import get_settings
from src.kb.embedding_loader import get_bge_embedding
from src.db_resilience import get_milvus_collection, milvus_operation_with_retry
from .mineru_client import ChunkItem, ParseResult

logger = logging.getLogger(__name__)


def _build_collection_schema(collection_name: str):
    """创建 Milvus collection 并建 HNSW 索引。"""
    from pymilvus import Collection, FieldSchema, CollectionSchema, DataType
    dim = get_settings().milvus_dim
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="parent_block", dtype=DataType.INT64),
        FieldSchema(name="child_block", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="parent_content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="kb chunks with structured metadata")
    coll = Collection(name=collection_name, schema=schema)
    coll.create_index(
        field_name="embedding",
        index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}},
    )
    coll.load()
    return coll


class MilvusUploader:
    """
    将确认后的解析结果（ParseResult）写入 Milvus：BGE-M3 编码 content/parent_content，
    写入 id、doc_id、chunk_id、doc_name、page、parent_block、child_block、content、parent_content、embedding；
    无 collection 时自动创建。连接失败时自动重试，熔断后降级返回 0。
    """

    def __init__(self):
        self.settings = get_settings()
        self._embed = get_bge_embedding()

    def _get_collection(self):
        """通过 db_resilience 获取 Collection，支持重试 + 熔断 + 懒重连。"""
        return get_milvus_collection(
            self.settings.milvus_uri,
            self.settings.milvus_collection,
            create_if_missing=True,
            schema_builder=_build_collection_schema,
        )

    def upload_parse_result(
        self,
        parse_result: ParseResult,
        doc_id: Optional[str] = None,
    ) -> int:
        """将解析结果中的 chunks 向量化并写入 Milvus。返回写入条数。Milvus 不可用时返回 0。"""
        if self._embed is None:
            logger.warning("BGE-M3 未加载，无法写入 Milvus")
            return 0
        doc_id = doc_id or parse_result.task_id
        doc_name_default = getattr(parse_result, "doc_name", "") or ""
        ids: List[str] = []
        doc_ids: List[str] = []
        chunk_ids: List[str] = []
        doc_names: List[str] = []
        pages: List[int] = []
        parent_blocks: List[int] = []
        child_blocks: List[int] = []
        contents: List[str] = []
        parent_contents: List[str] = []
        texts_to_embed: List[str] = []
        for c in parse_result.chunks:
            chunk_id = c.chunk_id or f"{doc_id}_{len(ids)}"
            ids.append(chunk_id)
            doc_ids.append(doc_id)
            chunk_ids.append(chunk_id)
            doc_names.append(getattr(c, "doc_name", "") or doc_name_default)
            pages.append(getattr(c, "page", 0) or 0)
            parent_blocks.append(getattr(c, "parent_block", 0) or 0)
            child_blocks.append(getattr(c, "child_block", 0) or 0)
            contents.append((c.content or "")[:65530])
            parent_contents.append((c.parent_content or "")[:65530])
            texts_to_embed.append(
                (c.parent_content or "") + "\n" + (c.content or "")
                if c.parent_content else (c.content or "")
            )
        if not texts_to_embed:
            return 0
        embeddings = self._embed.encode(texts_to_embed).tolist()
        entities = [
            ids, doc_ids, chunk_ids, doc_names,
            pages, parent_blocks, child_blocks,
            contents, parent_contents, embeddings,
        ]

        def _do_insert(coll):
            coll.insert(entities)
            coll.flush()
            return len(ids)

        count = milvus_operation_with_retry(
            self.settings.milvus_uri,
            self.settings.milvus_collection,
            _do_insert,
            retries=2,
            critical=False,
            default=0,
        )
        if count > 0:
            logger.info(
                "Milvus 写入 %d 条 chunk，doc_id=%s, doc_name=%s",
                count, doc_id, doc_name_default,
            )
        return count
