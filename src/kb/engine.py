# src/kb/engine.py
"""
知识库统一入口：优先 QA 精准匹配 → Text2SQL → RAG 检索生成；RAG 答案展示依据来源与依赖切片。
大模型仅使用 DeepSeek（src.llm.get_deepseek_llm），不调用 OpenAI。
支持同步 query/query_stream 与异步 aquery_stream，便于高性能并发。
"""
import asyncio
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from src.llm import get_deepseek_llm
from .prompt_templates import render_prompt_template_guidance, select_prompt_templates
from .qa_store import QAStore
from .text2sql import Text2SQL, Text2SQLConfirmRequired
from .rag import get_rag_retriever, RAGRetrieveResult
from .retrieval_eval import evaluate_retrieval


def _get_llm():
    """获取 DeepSeek 对话模型（仅此项目允许的大模型，非 OpenAI）。"""
    return get_deepseek_llm(temperature=0.3)


_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一名海关政策与业务知识助手。你必须严格根据给定的参考片段回答，禁止使用参考内容之外的常识、猜测、补充知识或臆断。

要求：
1. 只能依据参考内容回答；若参考内容不足，请明确说明“根据当前检索结果无法确认”，并建议用户补充问题或转人工。
2. 不得编造数字、时间、政策条款、流程、名单、结论。
3. 回答采用结构化输出：优先使用“结论：”“依据：”“提示：”这样的结构；简单问题可只保留必要部分。
4. 任何事实性结论后都必须附上证据编号，例如 [证据1]、[证据2]；不得写“根据经验”“通常来说”之类无证据表述。
5. 简单问题简短答，复杂问题分阶段答（先结论再分点补充）。
6. 若收到“重试说明”，表示上一版答案与参考内容关联度不足；你必须更加严格贴合参考内容重新生成。"""),
    ("human", """参考内容：
{context}

用户问题：{question}

场景化补充要求：
{scenario_guidance}

重试说明：
{retry_note}

请基于参考内容回答："""),
])


def _format_sources(rag_result: RAGRetrieveResult) -> str:
    """
    将 RAG 检索到的 chunks 格式化为「依据来源」展示文案。
    若有 parent_content 则展示「背景+片段」，每段截断避免过长；供用户核对答案依据。
    """
    if not rag_result.chunks:
        return ""
    lines = ["【依据来源】以下内容来自知识库检索，供您核对："]
    for i, c in enumerate(rag_result.chunks, 1):
        part = (c.parent_content or "").strip()
        if part and part != (c.content or "").strip():
            lines.append(f"\n--- 证据{i}（背景+片段） ---")
            lines.append(part[:400] + ("…" if len(part) > 400 else ""))
        content = (c.content or "").strip()
        if content:
            lines.append(content[:600] + ("…" if len(content) > 600 else ""))
    return "\n".join(lines)


@dataclass
class KnowledgeQueryTrace:
    """记录一次知识库问答的质量追踪信息，供监控与反馈分析使用。"""

    route: str = "knowledge"
    final_status: str = "unknown"
    retrieve_attempt: int = 0
    top_match_score: float = 0.0
    top_normalized_score: float = 0.0
    grounding_score: float = 0.0
    regenerate_count: int = 0
    has_evidence_citations: bool = False
    source_count: int = 0
    scenario_templates: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    fallback_reason: str = ""
    pending_sql: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化 dict。"""

        return {
            "route": self.route,
            "final_status": self.final_status,
            "retrieve_attempt": self.retrieve_attempt,
            "top_match_score": self.top_match_score,
            "top_normalized_score": self.top_normalized_score,
            "grounding_score": self.grounding_score,
            "regenerate_count": self.regenerate_count,
            "has_evidence_citations": self.has_evidence_citations,
            "source_count": self.source_count,
            "scenario_templates": list(self.scenario_templates),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "fallback_reason": self.fallback_reason,
            "pending_sql": self.pending_sql,
        }


class KnowledgeEngine:
    """
    知识库问答统一入口：按顺序尝试 QA 精准匹配 → Text2SQL → RAG。
    其中 RAG 生成阶段使用的 LLM 仅限 DeepSeek（_get_llm），不调用 OpenAI。
    """

    def __init__(self):
        """初始化 QA 库、Text2SQL、RAG 检索器及 RAG 用 LLM 链（仅 DeepSeek）。"""
        self.qa = QAStore()
        self.text2sql = Text2SQL()
        self.rag = get_rag_retriever()
        self._llm = _get_llm()  # 仅 DeepSeek，禁止 OpenAI
        self._rag_chain = _RAG_PROMPT | self._llm
        self._last_trace = KnowledgeQueryTrace(final_status="init")

    def _reset_trace(self) -> KnowledgeQueryTrace:
        """开始一次新的知识库问答前重置 trace。"""

        self._last_trace = KnowledgeQueryTrace()
        return self._last_trace

    def get_last_trace(self) -> dict[str, Any]:
        """返回最近一次 query / aquery / aquery_stream 的追踪信息。"""

        return deepcopy(self._last_trace.to_dict())

    def _build_rag_context(self, rag_result: RAGRetrieveResult) -> str:
        """将检索到的父/子块拼成带编号的生成上下文，便于模型引用 [证据N]。"""
        context_parts = []
        for i, c in enumerate(rag_result.chunks, 1):
            parent = (c.parent_content or "").strip()
            content = (c.content or "").strip()
            if parent and parent != content:
                context_parts.append(f"[证据{i}]\n[背景]\n{parent}\n[片段]\n{content}")
            else:
                context_parts.append(f"[证据{i}]\n{content}")
        return "\n\n---\n\n".join(context_parts)

    def _normalize_answer_for_eval(self, answer_text: str) -> str:
        """去掉结构化标签和证据编号，避免影响答案-文档相关度计算。"""
        answer = (answer_text or "").strip()
        if not answer:
            return ""
        answer = re.sub(r"\[证据\d+\]", "", answer)
        answer = re.sub(r"(结论|依据|提示)\s*[:：]", "", answer)
        return answer.strip()

    def _answer_has_evidence_citations(self, answer_text: str) -> bool:
        """除“无法确认”类兜底文案外，答案必须显式带有证据编号。"""
        answer = (answer_text or "").strip()
        if not answer:
            return False
        if "无法确认" in answer:
            return True
        return bool(re.search(r"\[证据\d+\]", answer))

    def _answer_grounding_score(self, answer_text: str, rag_result: RAGRetrieveResult) -> float:
        """
        计算最终答案与检索文档的最大关联度。
        这里把“答案”当作查询，与每个检索块以及整体上下文做匹配；分数低则认为答案可能脱离证据。
        """
        answer = self._normalize_answer_for_eval(answer_text)
        if not answer or not rag_result.chunks:
            return 0.0
        candidates = []
        for c in rag_result.chunks:
            text = ((c.parent_content or "").strip() + "\n" + (c.content or "").strip()).strip()
            if text:
                candidates.append(text)
        whole_context = self._build_rag_context(rag_result).strip()
        if whole_context:
            candidates.append(whole_context)
        if not candidates:
            return 0.0
        return max(evaluate_retrieval(answer, text).normalized_score for text in candidates)

    def _generate_grounded_answer(
        self,
        question: str,
        rag_result: RAGRetrieveResult,
        trace: Optional[KnowledgeQueryTrace] = None,
    ) -> str:
        """
        基于检索结果生成最终答案；若答案与检索文档关联度低，则视为可能幻觉并重生成，最多 3 次。
        超过最大次数仍不达标时返回保守兜底文案。
        """
        context = self._build_rag_context(rag_result)
        if not context.strip():
            if trace is not None:
                trace.final_status = "rag_no_context"
                trace.fallback_reason = "empty_context"
            return "未找到相关知识，建议您换个问法或转人工客服。"

        settings = get_settings()
        min_score = float(getattr(settings, "rag_answer_grounding_min_score", 0.18))
        max_times = max(1, int(getattr(settings, "rag_answer_max_regenerate_times", 3)))
        selected_templates = select_prompt_templates(question)
        scenario_guidance = render_prompt_template_guidance(question)
        if trace is not None:
            trace.scenario_templates = [item.name for item in selected_templates]
        retry_note = ""
        last_answer = ""
        last_score = 0.0
        for attempt in range(max_times):
            try:
                resp = self._rag_chain.invoke(
                    {
                        "context": context,
                        "question": question,
                        "scenario_guidance": scenario_guidance,
                        "retry_note": retry_note,
                    }
                )
                answer_text = (resp.content or "").strip()
            except Exception:
                answer_text = ""
            if not answer_text:
                answer_text = "根据当前检索结果无法确认，建议您补充问题或转人工客服。"
            score = self._answer_grounding_score(answer_text, rag_result)
            has_citations = self._answer_has_evidence_citations(answer_text)
            last_answer = answer_text
            last_score = score
            if trace is not None:
                trace.grounding_score = score
                trace.has_evidence_citations = has_citations
                trace.regenerate_count = attempt
            if score >= min_score and has_citations:
                if trace is not None:
                    trace.final_status = "rag_regenerated" if attempt > 0 else "rag_grounded"
                return answer_text
            citation_note = "上一版答案缺少证据编号，请在每个关键结论后补充 [证据N]。" if not has_citations else ""
            retry_note = (
                f"上一版答案与参考内容关联度过低（score={score:.3f} < {min_score:.3f}），"
                "请仅保留能直接从参考内容中得到的事实，删除无法从参考内容明确支持的表述后重新回答。"
                + citation_note
            )
        if trace is not None:
            trace.final_status = "rag_fallback_unconfirmed"
            trace.fallback_reason = "low_grounding_after_regeneration"
        return (
            "根据当前检索结果无法确认，建议您补充更具体的问题或转人工客服。"
            if last_score < min_score
            else last_answer
        )

    def query(self, question: str) -> Tuple[str, Optional[str]]:
        """
        依次尝试 QA、Text2SQL、RAG，返回 (回复文案, 待确认 SQL 或 None)。
        当 Text2SQL 返回删除/修改类需人工确认时，第二项为待执行 SQL，由上层写入 state.pending_sql 并交 API 确认执行。
        """
        trace = self._reset_trace()
        # 1) 高频 QA 精准匹配
        answer = self.qa.find(question)
        if answer:
            trace.route = "qa"
            trace.final_status = "qa_hit"
            return (answer, None)

        # 2) 结构化数据 Text2SQL（可能返回待确认 SQL）
        result = self.text2sql.query(question)
        if result is not None:
            trace.route = "text2sql"
            if isinstance(result, Text2SQLConfirmRequired):
                trace.final_status = "text2sql_pending"
                trace.pending_sql = True
                return (result.message, result.sql)
            trace.final_status = "text2sql_answer"
            return (result, None)

        # 3) RAG：带评估与重检，并返回依据来源
        trace.route = "rag"
        rag_result = self.rag.retrieve_with_validation(
            question, top_k=10, use_rerank=True, rerank_top=5
        )
        trace.retrieve_attempt = rag_result.attempt
        trace.source_count = len(rag_result.chunks)
        trace.retrieved_doc_ids = list(dict.fromkeys([c.doc_id for c in rag_result.chunks if c.doc_id]))
        trace.retrieved_chunk_ids = list(dict.fromkeys([c.chunk_id for c in rag_result.chunks if c.chunk_id]))
        if rag_result.evals:
            trace.top_match_score = max(e.match_score for e in rag_result.evals if e is not None)
            trace.top_normalized_score = max(e.normalized_score for e in rag_result.evals if e is not None)
        if not rag_result.chunks:
            trace.final_status = "rag_no_hit"
            trace.fallback_reason = "no_retrieval_hit"
            return ("未找到相关知识，建议您换个问法或转人工客服。", None)

        answer_text = self._generate_grounded_answer(question, rag_result, trace=trace)

        # 答案 + 依据来源（依赖的切片全部展示）
        sources_block = _format_sources(rag_result)
        if sources_block:
            return (answer_text + "\n\n" + sources_block, None)
        return (answer_text, None)

    def _yield_text_chunked(self, text: str):
        """将整段文本按「首字 + 后续 64 字/块」yield，用于 QA/Text2SQL 结果的流式输出。"""
        if not text:
            return
        yield text[0:1]
        rest = text[1:]
        chunk_size = 64
        for i in range(0, len(rest), chunk_size):
            yield rest[i : i + chunk_size]

    def query_stream(self, question: str):
        """
        知识库流式回答：先出首字再逐 chunk。依次尝试 QA → Text2SQL → RAG，首个有结果即流式输出。
        Yields: 文本片段（str）。
        """
        trace = self._reset_trace()
        # 1) QA 精准匹配
        answer = self.qa.find(question)
        if answer:
            trace.route = "qa"
            trace.final_status = "qa_hit"
            for chunk in self._yield_text_chunked(answer):
                yield chunk
            return

        # 2) Text2SQL（可能为待确认 SQL，统一按文案流式输出）
        result = self.text2sql.query(question)
        if result is not None:
            trace.route = "text2sql"
            trace.final_status = "text2sql_pending" if isinstance(result, Text2SQLConfirmRequired) else "text2sql_answer"
            trace.pending_sql = isinstance(result, Text2SQLConfirmRequired)
            msg = result.message if isinstance(result, Text2SQLConfirmRequired) else result
            for chunk in self._yield_text_chunked(msg):
                yield chunk
            return

        # 3) RAG：检索后流式生成
        trace.route = "rag"
        rag_result = self.rag.retrieve_with_validation(
            question, top_k=10, use_rerank=True, rerank_top=5
        )
        trace.retrieve_attempt = rag_result.attempt
        trace.source_count = len(rag_result.chunks)
        trace.retrieved_doc_ids = list(dict.fromkeys([c.doc_id for c in rag_result.chunks if c.doc_id]))
        trace.retrieved_chunk_ids = list(dict.fromkeys([c.chunk_id for c in rag_result.chunks if c.chunk_id]))
        if rag_result.evals:
            trace.top_match_score = max(e.match_score for e in rag_result.evals if e is not None)
            trace.top_normalized_score = max(e.normalized_score for e in rag_result.evals if e is not None)
        if not rag_result.chunks:
            trace.final_status = "rag_no_hit"
            trace.fallback_reason = "no_retrieval_hit"
            for chunk in self._yield_text_chunked("未找到相关知识，建议您换个问法或转人工客服。"):
                yield chunk
            return

        # 为了在返回前做“答案-文档相关度校验 + 最多 3 次重生成”，这里先生成最终答案，再按块输出。
        answer_text = self._generate_grounded_answer(question, rag_result, trace=trace)
        for c in self._yield_text_chunked(answer_text):
            yield c

        sources_block = _format_sources(rag_result)
        if sources_block:
            for chunk in self._yield_text_chunked("\n\n" + sources_block):
                yield chunk

    async def aquery(self, question: str) -> Tuple[str, Optional[str]]:
        """异步：在线程池中执行 query，不阻塞事件循环。返回 (回复文案, 待确认 SQL 或 None)。"""
        return await asyncio.to_thread(self.query, question)

    async def aquery_stream(
        self, question: str, pending_sql_out: Optional[List[str]] = None
    ) -> AsyncIterator[str]:
        """
        异步流式回答：先出首字再逐 chunk；RAG 使用 LLM astream，其余在线程池执行后按段 yield。
        """
        trace = self._reset_trace()
        # 1) QA
        answer = await asyncio.to_thread(self.qa.find, question)
        if answer:
            trace.route = "qa"
            trace.final_status = "qa_hit"
            for chunk in self._yield_text_chunked(answer):
                yield chunk
            return

        # 2) Text2SQL（可能为待确认 SQL）
        result = await asyncio.to_thread(self.text2sql.query, question)
        if result is not None:
            trace.route = "text2sql"
            # 删除/修改类需人工确认：将 SQL 放入 pending_sql_out 供 API 写入 _pending_sql
            if isinstance(result, Text2SQLConfirmRequired):
                trace.final_status = "text2sql_pending"
                trace.pending_sql = True
                if pending_sql_out is not None:
                    pending_sql_out.append(result.sql)
                for chunk in self._yield_text_chunked(result.message):
                    yield chunk
            else:
                trace.final_status = "text2sql_answer"
                for chunk in self._yield_text_chunked(result):
                    yield chunk
            return

        # 3) RAG：检索在线程池，生成用 astream
        trace.route = "rag"
        rag_result = await asyncio.to_thread(
            self.rag.retrieve_with_validation,
            question, top_k=10, use_rerank=True, rerank_top=5,
        )
        trace.retrieve_attempt = rag_result.attempt
        trace.source_count = len(rag_result.chunks)
        trace.retrieved_doc_ids = list(dict.fromkeys([c.doc_id for c in rag_result.chunks if c.doc_id]))
        trace.retrieved_chunk_ids = list(dict.fromkeys([c.chunk_id for c in rag_result.chunks if c.chunk_id]))
        if rag_result.evals:
            trace.top_match_score = max(e.match_score for e in rag_result.evals if e is not None)
            trace.top_normalized_score = max(e.normalized_score for e in rag_result.evals if e is not None)
        if not rag_result.chunks:
            trace.final_status = "rag_no_hit"
            trace.fallback_reason = "no_retrieval_hit"
            for chunk in self._yield_text_chunked("未找到相关知识，建议您换个问法或转人工客服。"):
                yield chunk
            return

        answer_text = await asyncio.to_thread(self._generate_grounded_answer, question, rag_result, trace)
        for c in self._yield_text_chunked(answer_text):
            yield c

        sources_block = _format_sources(rag_result)
        if sources_block:
            for chunk in self._yield_text_chunked("\n\n" + sources_block):
                yield chunk
