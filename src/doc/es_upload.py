"""
将用户确认后的解析结果写入 Elasticsearch，支持父子块与向量索引。
向量模型仅使用 BGE-M3（config 中 bge_embedding_model），不调用 OpenAI 或其它 embedding 服务。
"""
from pathlib import Path
from typing import Optional

from elasticsearch import Elasticsearch

from config import get_settings
from src.kb.embedding_loader import get_bge_embedding
from .mineru_client import ParseResult


class ESUploader:
    """
    将确认后的解析结果（ParseResult）写入 Elasticsearch：BGE-M3 编码 content/parent_content，
    写入 doc_id、chunk_id、content、parent_content、embedding；无索引时自动创建 dense_vector 索引。
    """

    def __init__(self):
        """使用单例 BGE-M3（与 RAG 复用），并确保目标索引存在。"""
        self.settings = get_settings()
        self._embed = get_bge_embedding()
        self._client = None
        self._index = self.settings.es_index
        self._ensure_index()

    def _es_auth(self) -> Optional[tuple[str, str]]:
        """ES 开启安全认证时返回 basic_auth，否则 None。"""
        username = (getattr(self.settings, "es_username", "") or "").strip()
        password = (getattr(self.settings, "es_password", "") or "").strip()
        return (username, password) if username and password else None

    def _ensure_index(self) -> None:
        """确保 Elasticsearch 中已存在目标索引（无则创建 dense_vector 映射）。"""
        try:
            self._client = Elasticsearch(
                self.settings.es_uri,
                basic_auth=self._es_auth(),
                request_timeout=getattr(self.settings, "es_timeout_seconds", 30),
            )
            if not self._client.indices.exists(index=self._index):
                dim = self.settings.es_dim
                self._client.indices.create(
                    index=self._index,
                    mappings={
                        "properties": {
                            "doc_id": {"type": "keyword"},
                            "chunk_id": {"type": "keyword"},
                            "doc_name": {"type": "keyword"},
                            "category": {"type": "keyword"},
                            "doc_path": {"type": "keyword"},
                            "content": {"type": "text"},
                            "parent_content": {"type": "text"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": dim,
                                "index": True,
                                "similarity": "cosine",
                            },
                        }
                    },
                )
            else:
                # 老索引升级：补充文档级过滤字段，便于先按文档/分类/路径缩小检索范围
                self._client.indices.put_mapping(
                    index=self._index,
                    properties={
                        "doc_name": {"type": "keyword"},
                        "category": {"type": "keyword"},
                        "doc_path": {"type": "keyword"},
                    },
                )
        except Exception:
            self._client = None

    def upload_parse_result(
        self,
        parse_result: ParseResult,
        doc_id: Optional[str] = None,
        category: Optional[str] = None,
        doc_path: Optional[str] = None,
    ) -> int:
        """将解析结果中的 chunks 向量化并写入 Elasticsearch。返回写入条数。"""
        if self._client is None or self._embed is None:
            return 0
        doc_id = doc_id or parse_result.task_id
        doc_name = Path(parse_result.original_path or "").name
        doc_path = (doc_path or "").strip() or (parse_result.original_path or "").strip()
        category = (category or "").strip()
        actions = []
        for idx, c in enumerate(parse_result.chunks):
            chunk_id = c.chunk_id or f"{doc_id}_{idx}"
            content = (c.content or "")[:65530]
            parent_content = (c.parent_content or "")[:65530]
            # 有父块时用「父块+子块」一起编码，提高检索上下文
            text_to_embed = (parent_content + "\n" + content) if parent_content else content
            actions.append({"index": {"_index": self._index, "_id": chunk_id}})
            actions.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "doc_name": doc_name,
                    "category": category,
                    "doc_path": doc_path,
                    "content": content,
                    "parent_content": parent_content,
                    "embedding": text_to_embed,  # 占位，下方替换为真实向量
                }
            )
        if not actions:
            return 0
        texts = [actions[i + 1]["embedding"] for i in range(0, len(actions), 2)]
        embeddings = self._embed.encode(texts).tolist()
        for i in range(0, len(actions), 2):
            actions[i + 1]["embedding"] = embeddings[i // 2]
        resp = self._client.bulk(operations=actions, refresh="wait_for")
        return len(embeddings) if not resp.get("errors") else 0
