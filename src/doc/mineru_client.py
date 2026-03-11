# src/doc/mineru_client.py
"""
文档解析：对接 MinerU 本地部署服务或占位实现。
- 本地部署版：mineru_api_url 指向本地 MinerU 服务（如 http://127.0.0.1:8001），token 可选。
- 占位实现：当未配置 mineru_api_url 时使用 kb.chunking 做父子切片，不调用 MinerU。
- 并发：多请求同时走 MinerU 时使用异步 httpx + 信号量限制并发数，不阻塞事件循环。
"""
import asyncio
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from config import get_settings

# 多请求同时调 MinerU 时用信号量限流，避免打满 MinerU 服务
_mineru_semaphore: Optional[asyncio.Semaphore] = None
_mineru_semaphore_lock = threading.Lock()


def _get_mineru_semaphore() -> asyncio.Semaphore:
    global _mineru_semaphore
    if _mineru_semaphore is None:
        with _mineru_semaphore_lock:
            if _mineru_semaphore is None:
                limit = max(1, getattr(get_settings(), "mineru_concurrency_limit", 5))
                _mineru_semaphore = asyncio.Semaphore(limit)
    return _mineru_semaphore


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
    文档解析客户端（支持 MinerU 本地部署）：
    - 当 mineru_api_url 非空时，调用该地址的 MinerU 服务（本地或远程），mineru_api_token 可选；
    - 当 mineru_api_url 为空时，使用占位解析（chunking 父子切片）。
    """

    def __init__(self):
        """从 config 读取 mineru_use_local、mineru_api_url、mineru_api_token 等。"""
        self.settings = get_settings()

    def parse_file(self, file_path: str, file_bytes: Optional[bytes] = None) -> ParseResult:
        """同步解析（供兼容）；并发场景请用 parse_file_async。"""
        task_id = str(uuid.uuid4())
        original_path = file_path or "unknown"
        api_url = (self.settings.mineru_api_url or "").strip()
        if api_url:
            return self._parse_via_api(task_id, original_path, file_bytes)
        return self._parse_placeholder(task_id, original_path, file_bytes)

    async def parse_file_async(
        self, file_path: str, file_bytes: Optional[bytes] = None
    ) -> ParseResult:
        """
        异步解析：多请求并发时使用。
        - 走 MinerU API 时：httpx 异步请求 + 信号量限流，不占线程；
        - 占位解析时：CPU 切块放到线程池，不阻塞事件循环。
        """
        task_id = str(uuid.uuid4())
        original_path = file_path or "unknown"
        api_url = (self.settings.mineru_api_url or "").strip()
        if not api_url:
            return await asyncio.to_thread(
                self._parse_placeholder, task_id, original_path, file_bytes
            )
        timeout = max(10, getattr(self.settings, "mineru_timeout_seconds", 120))
        sem = _get_mineru_semaphore()
        async with sem:
            return await self._parse_via_api_async(
                task_id, original_path, file_bytes, timeout=timeout
            )

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
        """调用 MinerU API（本地部署或远程）；本地部署时 token 可为空。"""
        import requests
        url = (self.settings.mineru_api_url or "").strip().rstrip("/")
        token = (self.settings.mineru_api_token or "").strip()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
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
            # API 失败时返回明确错误信息，不将错误响应体当作文档解析
            err_msg = f"[MinerU 解析失败] HTTP {r.status_code}"
            if r.content and len(r.content) < 500:
                try:
                    err_msg += ": " + r.content.decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
            return ParseResult(
                task_id=task_id,
                original_path=original_path,
                full_text=err_msg,
                chunks=[ChunkItem(content=err_msg, chunk_id=f"{task_id}_0")],
                raw_markdown=None,
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

    async def _parse_via_api_async(
        self,
        task_id: str,
        original_path: str,
        file_bytes: Optional[bytes],
        *,
        timeout: int = 120,
    ) -> ParseResult:
        """异步调用 MinerU API（httpx），与 _parse_via_api 响应格式一致。"""
        import httpx
        url = (self.settings.mineru_api_url or "").strip().rstrip("/")
        token = (self.settings.mineru_api_token or "").strip()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=float(timeout)) as client:
                if file_bytes:
                    files = {"file": (Path(original_path).name, file_bytes)}
                    r = await client.post(url, headers=headers, files=files)
                else:
                    r = await client.post(
                        url, headers=headers, json={"path": original_path}
                    )
        except Exception as e:
            err_msg = f"[MinerU 解析失败] 请求异常: {e!s}"
            return ParseResult(
                task_id=task_id,
                original_path=original_path,
                full_text=err_msg,
                chunks=[ChunkItem(content=err_msg, chunk_id=f"{task_id}_0")],
                raw_markdown=None,
            )
        if r.status_code != 200:
            err_msg = f"[MinerU 解析失败] HTTP {r.status_code}"
            if r.content and len(r.content) < 500:
                try:
                    err_msg += ": " + r.content.decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
            return ParseResult(
                task_id=task_id,
                original_path=original_path,
                full_text=err_msg,
                chunks=[ChunkItem(content=err_msg, chunk_id=f"{task_id}_0")],
                raw_markdown=None,
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
