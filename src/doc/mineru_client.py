# src/doc/mineru_client.py
"""
文档解析：对接 MinerU，基础版支持占位实现与真实 MinerU API 调用。
占位实现使用 kb.chunking 做父子切片（可配置切块大小、父块重叠约 150），不调用任何大模型或 OpenAI。
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from config import get_settings


class ChunkItem(BaseModel):
    """单段解析内容（子块），可选 parent_content 用于父子分段。"""
    content: str
    parent_content: Optional[str] = None
    chunk_id: str = ""


class ParseResult(BaseModel):
    """解析结果：任务 id、原始路径、全文、分段 chunks，供前端对比与自定义后上传 Milvus。"""
    task_id: str
    original_path: str
    full_text: str
    chunks: List[ChunkItem] = []
    raw_markdown: Optional[str] = None


class MinerUClient:
    """
    文档解析客户端：mineru_use_local 或无 API 时用占位（chunking 父子切片）；
    有 mineru_api_url + mineru_api_token 时调用远程 MinerU API。
    """

    def __init__(self):
        """从 config 读取 mineru_use_local、mineru_api_url、mineru_api_token 等。"""
        self.settings = get_settings()

    def parse_file(self, file_path: str, file_bytes: Optional[bytes] = None) -> ParseResult:
        """解析单个文档；file_path 为原始文件名，file_bytes 可选（上传内容）。返回 task_id、全文、chunks。"""
        task_id = str(uuid.uuid4())
        original_path = file_path or "unknown"
        if self.settings.mineru_use_local and not self.settings.mineru_api_url:
            return self._parse_placeholder(task_id, original_path, file_bytes)
        if self.settings.mineru_api_url and self.settings.mineru_api_token:
            return self._parse_via_api(task_id, original_path, file_bytes)
        return self._parse_placeholder(task_id, original_path, file_bytes)

    def _parse_placeholder(
        self, task_id: str, original_path: str, file_bytes: Optional[bytes]
    ) -> ParseResult:
        """占位实现：优先使用父子切片（可配置大小、父块重叠约 150），否则按行/段切。"""
        if file_bytes:
            try:
                full_text = file_bytes.decode("utf-8", errors="replace")
            except Exception:
                full_text = file_bytes.decode("latin-1", errors="replace")
        else:
            p = Path(original_path)
            if p.exists():
                full_text = p.read_text(encoding="utf-8", errors="replace")
            else:
                full_text = f"[占位] 未读取到文件内容: {original_path}\n请配置 MinerU 或上传文件内容。"
        chunks = self._chunk_with_strategy(task_id, full_text)
        return ParseResult(
            task_id=task_id,
            original_path=original_path,
            full_text=full_text,
            chunks=chunks,
            raw_markdown=full_text,
        )

    def _chunk_with_strategy(self, task_id: str, full_text: str) -> List[ChunkItem]:
        """使用配置的切块大小与父块重叠（约 150）做父子切片。"""
        try:
            from src.kb.chunking import chunk_text
            chunk_size = getattr(self.settings, "rag_default_chunk_size", 512)
            parent_overlap = getattr(self.settings, "rag_parent_overlap", 150)
            parent_ctx = min(chunk_size * 2, 768)
            with_parent = chunk_text(
                full_text,
                child_chunk_size=chunk_size,
                parent_overlap=parent_overlap,
                parent_ctx_len=parent_ctx,
                doc_id=task_id,
            )
            return [
                ChunkItem(
                    content=c.content,
                    parent_content=c.parent_content or None,
                    chunk_id=c.chunk_id,
                )
                for c in with_parent
            ]
        except Exception:
            pass
        # 占位兜底：按行/空行与长度切块，无父子结构
        raw_lines = full_text.splitlines()
        chunks: List[ChunkItem] = []
        current: List[str] = []
        for line in raw_lines:
            current.append(line)
            if line.strip() == "" or len("".join(current)) > 500:
                block = "\n".join(current).strip()
                if block:
                    chunks.append(
                        ChunkItem(
                            content=block,
                            parent_content=None,
                            chunk_id=f"{task_id}_{len(chunks)}",
                        )
                    )
                current = []
        if current:
            block = "\n".join(current).strip()
            if block:
                chunks.append(
                    ChunkItem(
                        content=block,
                        parent_content=None,
                        chunk_id=f"{task_id}_{len(chunks)}",
                    )
                )
        if not chunks:
            chunks = [
                ChunkItem(
                    content=full_text[:2000] or "(无内容)",
                    chunk_id=f"{task_id}_0",
                )
            ]
        return chunks

    def _parse_via_api(
        self, task_id: str, original_path: str, file_bytes: Optional[bytes]
    ) -> ParseResult:
        """调用 MinerU 云 API（需上传文件到可访问 URL 或使用 API 支持的方式）。"""
        import requests
        url = self.settings.mineru_api_url
        headers = {"Authorization": f"Bearer {self.settings.mineru_api_token}"}
        # 若 API 接受 file_url，可先传文件到存储再传 URL；这里简化
        if file_bytes:
            files = {"file": (Path(original_path).name, file_bytes)}
            r = requests.post(url, headers=headers, files=files, timeout=120)
        else:
            r = requests.post(
                url,
                headers=headers,
                json={"path": original_path},
                timeout=120,
            )
        if r.status_code != 200:
            # API 失败时回退到占位解析，避免直接报错
            return self._parse_placeholder(
                task_id, original_path, r.content[:5000] if r.content else None
            )
        data = r.json()
        full_text = data.get("markdown") or data.get("text") or ""
        chunks = []
        for i, seg in enumerate(data.get("chunks", []) or [full_text]):
            if isinstance(seg, dict):
                content = seg.get("content") or seg.get("text") or ""
                parent = seg.get("parent_content")
            else:
                content = str(seg)
                parent = None
            chunks.append(
                ChunkItem(
                    content=content,
                    parent_content=parent,
                    chunk_id=f"{task_id}_{i}",
                )
            )
        if not chunks and full_text:
            chunks = [ChunkItem(content=full_text[:2000], chunk_id=f"{task_id}_0")]
        return ParseResult(
            task_id=task_id,
            original_path=original_path,
            full_text=full_text,
            chunks=chunks,
            raw_markdown=full_text,
        )
