# src/doc/milvus_upload.py
"""
将用户确认后的解析结果写入 Milvus，支持父子块与向量索引。
向量模型仅使用 BGE-M3（config 中 bge_embedding_model），不调用 OpenAI 或其它 embedding 服务。
"""
from typing import List, Optional
from config import get_settings
from .mineru_client import ChunkItem, ParseResult


def _get_embedding_model():
    """加载 BGE-M3 用于生成 chunk 向量（仅允许的 embedding 模型）。"""
    try:
        from FlagEmbedding import FlagModel
        s = get_settings()
        return FlagModel(
            s.bge_embedding_model,
            use_fp16=False,
            device=s.embedding_device,
        )
    except Exception:
        return None


class MilvusUploader:
    """
    将确认后的解析结果（ParseResult）写入 Milvus：BGE-M3 编码 content/parent_content，
    写入 id、doc_id、chunk_id、content、parent_content、embedding；无 collection 时自动创建。
    """

    def __init__(self):
        """加载配置与 BGE-M3，并确保目标 collection 存在且已 load。"""
        self.settings = get_settings()
        self._embed = _get_embedding_model()
        self._collection = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """确保 Milvus 中已存在目标 collection（有则 load，无则创建并建 HNSW 索引）。"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            connections.connect(uri=self.settings.milvus_uri)
            coll_name = self.settings.milvus_collection
            if utility.has_collection(coll_name):
                self._collection = Collection(coll_name)
                self._collection.load()
                return
            dim = self.settings.milvus_dim
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="parent_content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields=fields, description="kb chunks")
            self._collection = Collection(name=coll_name, schema=schema)
            self._collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}},
            )
            self._collection.load()
        except Exception:
            try:
                from pymilvus import connections, Collection
                connections.connect(uri=self.settings.milvus_uri)
                self._collection = Collection(self.settings.milvus_collection)
                self._collection.load()
            except Exception:
                self._collection = None

    def upload_parse_result(
        self,
        parse_result: ParseResult,
        doc_id: Optional[str] = None,
    ) -> int:
        """将解析结果中的 chunks 向量化并写入 Milvus。返回写入条数。"""
        if self._collection is None or self._embed is None:
            return 0
        doc_id = doc_id or parse_result.task_id
        ids = []
        doc_ids = []
        chunk_ids = []
        contents = []
        parent_contents = []
        texts_to_embed = []
        for c in parse_result.chunks:
            chunk_id = c.chunk_id or f"{doc_id}_{len(ids)}"
            ids.append(chunk_id)
            doc_ids.append(doc_id)
            chunk_ids.append(chunk_id)
            contents.append((c.content or "")[:65530])
            parent_contents.append((c.parent_content or "")[:65530])
            # 有父块时用「父块+子块」一起编码，提高检索上下文
            texts_to_embed.append((c.parent_content or "") + "\n" + (c.content or "") if c.parent_content else (c.content or ""))
        if not texts_to_embed:
            return 0
        embeddings = self._embed.encode(texts_to_embed).tolist()
        entities = [ids, doc_ids, chunk_ids, contents, parent_contents, embeddings]
        self._collection.insert(entities)
        self._collection.flush()
        return len(ids)
