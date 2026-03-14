# src/doc/mineru_client.py
"""
文档解析：对接 MinerU 本地部署服务或占位实现。
- 本地部署版：mineru_api_url 指向本地 MinerU 服务（如 http://127.0.0.1:8001），token 可选。
- 占位实现：当未配置 mineru_api_url 时使用 kb.chunking 做父子切片，不调用 MinerU。
- 并发：多请求同时走 MinerU 时使用异步 httpx + 信号量限制并发数，不阻塞事件循环。
"""
import asyncio
import re
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from config import get_settings


def _safe_doc_name(path: str) -> str:
    """从文件路径提取安全的文档名（去后缀、替换特殊字符）。"""
    name = Path(path).stem if path else "doc"
    name = re.sub(r"[^\w\u4e00-\u9fff-]", "_", name)
    return name[:80] or "doc"


_PAGE_SPLIT_RE = re.compile(
    r"(?:^|\n)(?:---\s*)?(?:第\s*(\d+)\s*页|Page\s*(\d+)|p\.?\s*(\d+))",
    re.IGNORECASE,
)


def _detect_pages(text: str) -> List[tuple]:
    """
    检测文本中的页码标记，返回 [(page_number, start_offset), ...]。
    支持「第N页」、「Page N」、「p.N」等格式。
    """
    pages = []
    for m in _PAGE_SPLIT_RE.finditer(text):
        num = m.group(1) or m.group(2) or m.group(3)
        if num:
            pages.append((int(num), m.start()))
    return pages


def _build_structured_chunk_id(doc_name: str, page: int, parent_block: int, child_block: int) -> str:
    """生成结构化 chunk_id：文档名-p页码-b父块编号-c子块编号。"""
    return f"{doc_name}-p{page}-b{parent_block}-c{child_block}"

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
    doc_name: str = ""
    page: int = 0
    parent_block: int = 0
    child_block: int = 0


class ParseResult(BaseModel):
    """解析结果：任务 id、原始路径、全文、分段 chunks，供前端对比与自定义后上传 Milvus。"""
    task_id: str
    original_path: str
    full_text: str
    chunks: List[ChunkItem] = []
    raw_markdown: Optional[str] = None
    doc_name: str = ""


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
        doc_name = _safe_doc_name(original_path)
        chunks = self._chunk_with_strategy(task_id, full_text, doc_name)
        return ParseResult(
            task_id=task_id,
            original_path=original_path,
            full_text=full_text,
            chunks=chunks,
            raw_markdown=full_text,
            doc_name=doc_name,
        )

    def _chunk_with_strategy(self, task_id: str, full_text: str, doc_name: str = "") -> List[ChunkItem]:
        """使用配置的切块大小与父块重叠（约 150）做父子切片，生成结构化 chunk_id。"""
        doc_name = doc_name or _safe_doc_name("")
        pages = _detect_pages(full_text)

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
            return self._assign_structured_ids(with_parent, full_text, doc_name, pages)
        except Exception:
            pass
        raw_lines = full_text.splitlines()
        chunks: List[ChunkItem] = []
        current: List[str] = []
        offset = 0
        for line in raw_lines:
            current.append(line)
            if line.strip() == "" or len("".join(current)) > 500:
                block = "\n".join(current).strip()
                if block:
                    page = self._page_for_offset(offset, pages)
                    parent_block = len(chunks)
                    cid = _build_structured_chunk_id(doc_name, page, parent_block, 0)
                    chunks.append(ChunkItem(
                        content=block,
                        parent_content=None,
                        chunk_id=cid,
                        doc_name=doc_name,
                        page=page,
                        parent_block=parent_block,
                        child_block=0,
                    ))
                offset += sum(len(l) + 1 for l in current)
                current = []
        if current:
            block = "\n".join(current).strip()
            if block:
                page = self._page_for_offset(offset, pages)
                parent_block = len(chunks)
                cid = _build_structured_chunk_id(doc_name, page, parent_block, 0)
                chunks.append(ChunkItem(
                    content=block,
                    parent_content=None,
                    chunk_id=cid,
                    doc_name=doc_name,
                    page=page,
                    parent_block=parent_block,
                    child_block=0,
                ))
        if not chunks:
            cid = _build_structured_chunk_id(doc_name, 1, 0, 0)
            chunks = [ChunkItem(
                content=full_text[:2000] or "(无内容)",
                chunk_id=cid,
                doc_name=doc_name,
                page=1,
                parent_block=0,
                child_block=0,
            )]
        return chunks

    def _assign_structured_ids(
        self,
        with_parent: list,
        full_text: str,
        doc_name: str,
        pages: List[tuple],
    ) -> List[ChunkItem]:
        """为 chunk_text() 产出的 ChunkWithParent 列表分配结构化 chunk_id 和页码。"""
        parent_block_counter: dict = {}
        child_counter: dict = {}
        items: List[ChunkItem] = []

        for c in with_parent:
            start = full_text.find(c.content)
            page = self._page_for_offset(max(0, start), pages)

            page_key = page
            if page_key not in parent_block_counter:
                parent_block_counter[page_key] = 0
                child_counter[page_key] = {}

            parent_text = (c.parent_content or "").strip()
            if parent_text not in child_counter[page_key]:
                child_counter[page_key][parent_text] = 0
                pb = parent_block_counter[page_key]
                parent_block_counter[page_key] = pb + 1
            else:
                pb = -1
                for pt, _ in child_counter[page_key].items():
                    pb += 1
                    if pt == parent_text:
                        break

            cb = child_counter[page_key][parent_text]
            child_counter[page_key][parent_text] = cb + 1

            cid = _build_structured_chunk_id(doc_name, page, pb, cb)
            items.append(ChunkItem(
                content=c.content,
                parent_content=c.parent_content or None,
                chunk_id=cid,
                doc_name=doc_name,
                page=page,
                parent_block=pb,
                child_block=cb,
            ))
        return items

    @staticmethod
    def _page_for_offset(offset: int, pages: List[tuple]) -> int:
        """根据字符偏移量确定所在页码。"""
        if not pages:
            return 1
        page = 1
        for pnum, pstart in pages:
            if offset >= pstart:
                page = pnum
            else:
                break
        return page

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
        doc_name = _safe_doc_name(original_path)
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
                doc_name=doc_name,
            )
        data = r.json()
        full_text = data.get("markdown") or data.get("text") or ""
        chunks = self._build_chunks_from_api_data(data, task_id, doc_name, full_text)
        return ParseResult(
            task_id=task_id,
            original_path=original_path,
            full_text=full_text,
            chunks=chunks,
            raw_markdown=full_text,
            doc_name=doc_name,
        )

    def _build_chunks_from_api_data(
        self, data: dict, task_id: str, doc_name: str, full_text: str
    ) -> List[ChunkItem]:
        """从 MinerU API 响应构建带结构化编号的 ChunkItem 列表。"""
        pages = _detect_pages(full_text)
        raw_chunks = data.get("chunks", []) or [full_text]
        chunks: List[ChunkItem] = []
        parent_block_per_page: dict = {}
        child_per_parent: dict = {}
        offset = 0
        for seg in raw_chunks:
            if isinstance(seg, dict):
                content = seg.get("content") or seg.get("text") or ""
                parent = seg.get("parent_content")
                seg_page = seg.get("page")
            else:
                content = str(seg)
                parent = None
                seg_page = None
            if seg_page is not None:
                page = int(seg_page)
            else:
                start = full_text.find(content, offset) if content else offset
                if start < 0:
                    start = offset
                page = self._page_for_offset(start, pages)
                offset = start + len(content)

            pk = page
            if pk not in parent_block_per_page:
                parent_block_per_page[pk] = 0
                child_per_parent[pk] = {}
            parent_key = (parent or "").strip()
            if parent_key not in child_per_parent[pk]:
                child_per_parent[pk][parent_key] = 0
                pb = parent_block_per_page[pk]
                parent_block_per_page[pk] = pb + 1
            else:
                pb = list(child_per_parent[pk].keys()).index(parent_key)
            cb = child_per_parent[pk][parent_key]
            child_per_parent[pk][parent_key] = cb + 1

            cid = _build_structured_chunk_id(doc_name, page, pb, cb)
            chunks.append(ChunkItem(
                content=content,
                parent_content=parent,
                chunk_id=cid,
                doc_name=doc_name,
                page=page,
                parent_block=pb,
                child_block=cb,
            ))
        if not chunks and full_text:
            cid = _build_structured_chunk_id(doc_name, 1, 0, 0)
            chunks = [ChunkItem(
                content=full_text[:2000], chunk_id=cid,
                doc_name=doc_name, page=1, parent_block=0, child_block=0,
            )]
        return chunks

    def _is_mineru_retryable(self, exc: Exception, status_code: int) -> bool:
        """判断是否可重试：5xx、429、超时、连接错误。"""
        if status_code >= 500 or status_code == 429:
            return True
        if exc is None:
            return False
        msg = (str(exc) or "").lower()
        return any(kw in msg for kw in ("timeout", "timed out", "connection", "connect", "reset", "refused"))

    async def _parse_via_api_async(
        self,
        task_id: str,
        original_path: str,
        file_bytes: Optional[bytes],
        *,
        timeout: int = 120,
    ) -> ParseResult:
        """异步调用 MinerU API（httpx），支持 5xx/超时/连接错误重试。"""
        import httpx
        url = (self.settings.mineru_api_url or "").strip().rstrip("/")
        token = (self.settings.mineru_api_token or "").strip()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        doc_name = _safe_doc_name(original_path)
        retries = max(0, getattr(self.settings, "mineru_retry_times", 0))
        last_err_msg: Optional[str] = None
        last_status = 0

        for attempt in range(retries + 1):
            exc_occurred: Optional[Exception] = None
            r = None
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
                exc_occurred = e
                last_err_msg = str(e)
                last_status = 0
                if attempt < retries and self._is_mineru_retryable(e, 0):
                    await asyncio.sleep(1.0 * (2 ** attempt))
                    continue
                err_msg = f"[MinerU 解析失败] 请求异常: {e!s}"
                return ParseResult(
                    task_id=task_id,
                    original_path=original_path,
                    full_text=err_msg,
                    chunks=[ChunkItem(content=err_msg, chunk_id=f"{task_id}_0")],
                    raw_markdown=None,
                    doc_name=doc_name,
                )

            if r is None:
                continue
            last_status = r.status_code
            if r.status_code == 200:
                data = r.json()
                full_text = data.get("markdown") or data.get("text") or ""
                chunks = self._build_chunks_from_api_data(data, task_id, doc_name, full_text)
                return ParseResult(
                    task_id=task_id,
                    original_path=original_path,
                    full_text=full_text,
                    chunks=chunks,
                    raw_markdown=full_text,
                    doc_name=doc_name,
                )
            last_err_msg = f"HTTP {r.status_code}"
            if r.content and len(r.content) < 500:
                try:
                    last_err_msg += ": " + r.content.decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
            if attempt < retries and self._is_mineru_retryable(exc_occurred or Exception(), r.status_code):
                await asyncio.sleep(1.0 * (2 ** attempt))
                continue
            err_msg = f"[MinerU 解析失败] {last_err_msg}"
            return ParseResult(
                task_id=task_id,
                original_path=original_path,
                full_text=err_msg,
                chunks=[ChunkItem(content=err_msg, chunk_id=f"{task_id}_0")],
                raw_markdown=None,
                doc_name=doc_name,
            )

        err_msg = f"[MinerU 解析失败] {last_err_msg or '未知错误'}"
        return ParseResult(
            task_id=task_id,
            original_path=original_path,
            full_text=err_msg,
            chunks=[ChunkItem(content=err_msg, chunk_id=f"{task_id}_0")],
            raw_markdown=None,
            doc_name=doc_name,
        )
