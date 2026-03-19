from __future__ import annotations

import datetime as _dt
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "海关小智_面试深度技术剖析.pdf"


def _register_cjk_font() -> str:
    # Built-in CJK CID font in ReportLab (no external font files needed)
    font_name = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    except Exception:
        # If already registered or environment differs, just continue
        pass
    return font_name


def _p(text: str) -> str:
    # Keep Paragraph XML-safe enough for our content
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def build_pdf(out_path: Path) -> None:
    font = _register_cjk_font()
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "CJKBase",
        parent=styles["Normal"],
        fontName=font,
        fontSize=10.5,
        leading=15,
        spaceAfter=6,
    )
    h1 = ParagraphStyle(
        "CJKH1",
        parent=styles["Heading1"],
        fontName=font,
        fontSize=18,
        leading=24,
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "CJKH2",
        parent=styles["Heading2"],
        fontName=font,
        fontSize=13.5,
        leading=18,
        spaceBefore=6,
        spaceAfter=6,
    )
    mono = ParagraphStyle(
        "CJKMono",
        parent=base,
        fontName=font,
        backColor="#f4f4f4",
        leftIndent=6,
        rightIndent=6,
        spaceBefore=4,
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="海关小智：面试深度技术剖析",
        author="nanqi",
    )

    now = _dt.datetime.now().strftime("%Y-%m-%d")
    story: list = []

    story.append(Paragraph("海关小智智能问答（RAG + Text2SQL + Multi‑Agent）<br/>面试深度技术剖析（可背诵版）", h1))
    story.append(Paragraph(f"生成日期：{now}    版本：v1", base))
    story.append(Spacer(1, 10))
    story.append(Paragraph("本文的目标：把“我会用这些组件”讲成“我理解为什么有效、怎么实现、怎么权衡、如何评估”。内容全部基于本项目的真实实现与配置（FastAPI + LangGraph + Milvus + Redis + PostgreSQL + BGE）。", base))

    story.append(Paragraph("快速定位代码入口（让面试官相信你真的做过）", h2))
    bullets = [
        "智能体入口：api/main.py 的 POST /chat、POST /chat/stream，内部 get_graph() 后 graph.ainvoke(...)。",
        "图编排：src/graph/app.py，START → supervisor → (chat | knowledge | human | END)。",
        "路由逻辑：src/agents/supervisor.py（规则优先 + LLM 兜底，失败转 human）。",
        "知识库统一入口：src/kb/engine.py（QA → Text2SQL → RAG）。",
        "RAG 检索：src/kb/rag.py；分块：src/kb/chunking.py；配置：config/settings.py。",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(_p(x), base)) for x in bullets],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("一、RAG 优化：分块 / 检索 / 重排（面试追问逐条拆）", h1))

    story.append(Paragraph("Q1：为什么不是固定 1200/300，而是多尺寸分块？怎么证明？", h2))
    story.append(Paragraph("原理：分块尺寸决定“召回粒度”和“上下文完整性”的折中。太小：召回精准但缺背景，生成容易误解约束；太大：语义完整但召回噪音高、重排成本高。", base))
    story.append(Paragraph("实现：本项目将可验证的子块尺寸配置成列表，并支持多尺寸切片对比：settings.py 中 rag_chunk_sizes=[256,384,512,768]，并在 chunking.py 提供 chunk_text_multi_size(...)，便于离线对比不同 size 的效果。默认 chunk_text(...) 走“父子分段 + 句/段边界对齐 + 表格边界对齐”。", base))
    story.append(Paragraph("你可以补一句量化口径：我们用一个离线评估集（问题—标准答案/标准证据）衡量 Recall@K、MRR、以及“答案 grounding 分”。先选出在 Recall@K 与延迟之间最优的 size，再固定到线上。", base))
    story.append(Paragraph("关键配置（可背）：rag_chunk_sizes、rag_parent_overlap=150、rag_use_legacy_fixed_chunking=False。", mono))

    story.append(Paragraph("Q2：语义切割具体怎么实现？是模型切还是规则切？表格怎么处理？", h2))
    story.append(Paragraph("实现细节（规则切为主，保证稳定与成本）：", base))
    story.append(Paragraph("1) 句/段边界对齐：chunking.py 使用正则识别句末标点/双换行，切点在 pos±窗口内微调到最近句尾（_align_to_sentence_boundary）。这样避免在句子中间断开，又不会把块切得过碎。", base))
    story.append(Paragraph("2) 表格边界对齐：对 Markdown 风格表格（含“|”且列数足够）识别表格 span（_get_table_spans），若切点落在表内则扩展到表尾或回退到表头（_adjust_spans_for_tables），避免拆表。", base))
    story.append(Paragraph("3) 结构感知切片（实验备用）：chunk_text_structure_aware(...) 会识别 heading/table/paragraph，表格不拆、标题尽量和正文同块，段落内再做语义合并。", base))

    story.append(Paragraph("Q3：BM25 + 向量为什么要混合？权重/融合怎么定？", h2))
    story.append(Paragraph("原理：BM25 强在术语/编码/条款号等精确匹配；向量检索强在同义改写与语义召回。海关场景既有政策术语也有口语化问法，因此单路不稳。", base))
    story.append(Paragraph("实现：检索器在 rag.py 中两路召回（_bm25_search / _vector_search），默认启用 RRF 融合（settings.py: rag_use_rrf=True，rag_rrf_k=60）。RRF 的好处是对不同打分尺度不敏感，比固定比例更稳。若关闭 RRF，则按 rag_bm25_ratio=0.3、rag_vector_ratio=0.7 做比例合并。", base))
    story.append(Paragraph("你可以补一句“怎么定”：我们用离线评估集 sweep rag_rrf_k、nprobe、top_k，并看 Recall@K、Grounding 与 P95 延迟；线上再用用户反馈和坏例统计微调。", base))

    story.append(Paragraph("Q4：重排模型是什么？Cross‑Encoder 还是 Bi‑Encoder？为什么只对 Top‑K 精排？", h2))
    story.append(Paragraph("实现：本项目使用 BGE Reranker Large（settings.py: bge_reranker_model=BAAI/bge-reranker-large），属于 Cross‑Encoder 风格：query 与候选片段成对打分。它精度高但成本高，因此只对候选集合做 Top‑K 精排（rag.py 中 use_rerank=True 且 len(combined)>rerank_top 才会 compute_score，再按 rerank_score 排序后截断）。", base))
    story.append(Paragraph("为了可引用性与多样性：重排后会保留前 rag_rerank_anchor_count=3 条作为“必选证据”，其余名额用 MMR（Jaccard 近似）选多样性（rag_use_diversity_after_rerank=True, rag_diversity_mmr_lambda=0.8）。", base))

    story.append(PageBreak())
    story.append(Paragraph("二、Agent / LangGraph：决策、记忆、状态流转（追问点一口气讲透）", h1))

    story.append(Paragraph("Q5：你们的“自主决策”到底是什么？是 ReAct/CoT 还是 if‑else？", h2))
    story.append(Paragraph("本项目的“自主决策”主要体现在 Supervisor 路由：先规则（关键词/短语）快速判别，歧义时调用 LLM 输出固定路由标签（chat/knowledge）。它不是典型 ReAct（Thought/Action/Observation 循环），而是“确定性图 + 受控路由”的工程化做法，优先稳定性与可控性。", base))

    story.append(Paragraph("Q6：短期记忆/长期记忆怎么做？窗口多大？为什么要摘要？", h2))
    story.append(Paragraph("短期记忆：LangGraph 的 state.messages 作为对话消息列表，喂给模型时只取最近 llm_context_window_turns=10 轮（每轮 user+assistant）。", base))
    story.append(Paragraph("旧对话摘要：settings.py llm_context_summarize_old=True 时，会把窗口外旧消息压缩成一条【历史对话摘要】SystemMessage，再拼上最近 N 轮，既保留早期关键信息又控制 token。单条消息还会按 llm_context_max_chars_per_message_old=600 截断，避免某一条撑爆上下文。", base))
    story.append(Paragraph("长期记忆：可选 PostgreSQL（chat_history_postgresql_uri）持久化会话状态；进程重启可恢复。多 worker 时还可选 Redis checkpointer（chat_checkpointer_redis_url）共享短期状态。", base))

    story.append(Paragraph("Q7：LangGraph 的状态节点有哪些？条件边基于什么判断？", h2))
    story.append(Paragraph("状态：AgentState 至少包含 messages 与 next（NextAction=chat|knowledge|human|__end__）。Supervisor 写 next；子节点也可写 next=human 触发人工介入。", base))
    story.append(Paragraph("图：START→supervisor；supervisor 通过 conditional_edges(route_to_agent) 路由到 chat/knowledge/human；chat/knowledge 执行后若 next=human 则转 human，否则结束。这个结构让“分支/降级/人工介入”都可控且易维护。", base))

    story.append(PageBreak())
    story.append(Paragraph("三、幻觉抑制：Grounding 校验 + 重生成（如何回答到“底层机制”）", h1))

    story.append(Paragraph("Q8：你们的生成后校验怎么做？为什么有效？", h2))
    story.append(Paragraph("原理：幻觉常来自“检索证据不相关”或“生成偏离证据”。因此除了 prompt 约束，还需要一个客观的 grounding 信号来判断是否偏离。", base))
    story.append(Paragraph("实现：engine.py 在生成后会计算“答案与证据的最大相关度”（_answer_grounding_score）。做法是把答案文本做归一化（去掉[证据N]与结构化标签），再把“答案当查询”，与每个检索块文本以及整体 context 计算 normalized_score，取最大值作为 grounding_score。", base))
    story.append(Paragraph("若 grounding_score 低于 settings.rag_answer_grounding_min_score=0.3，则触发最多 rag_answer_max_regenerate_times=3 次重生成；仍不达标则保守拒答/提示补充信息。并且答案必须包含证据编号（_answer_has_evidence_citations），否则也会被视为不可信。", base))
    story.append(Paragraph("可背一句：我们不依赖“另一个大模型裁判”，而是用轻量可重复的检索相关度指标做 grounding gate，成本可控、行为可解释。", base))

    story.append(PageBreak())
    story.append(Paragraph("四、Text2SQL：安全护栏、schema 变化、错误恢复（追问点的“工程答案”）", h1))

    story.append(Paragraph("Q9：SQL 生成常见错在哪？你们怎么保证可执行与安全？", h2))
    story.append(Paragraph("实现要点：", base))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph(_p("SQL 提取与清洗：从代码块/行内代码/混合文本中提取 SQL，清理中文标点、零宽字符、尾部解释文本（text2sql.py:_sanitize_and_extract_sql）。"), base)),
                ListItem(Paragraph(_p("安全限制：自动执行路径仅允许 SELECT（_SQL_MUST_SELECT）；出现 DELETE/UPDATE/INSERT 等关键字视为危险（_SQL_DANGEROUS），走“生成但不执行”的人工确认流程。"), base)),
                ListItem(Paragraph(_p("时间范围解析：把“最近N天/去年/上月”等解析为具体日期区间注入 prompt，减少 where 条件缺失。"), base)),
                ListItem(Paragraph(_p("schema 变化：SchemaCache 定时刷新（settings.py: text2sql_schema_refresh_interval_seconds=3600），并支持人工 overrides（text2sql_schema_overrides_path）来补充表/列语义与关联。"), base)),
            ],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Paragraph("你可以补一句“错误恢复”：SQL 语法错误/执行失败会区分原因，触发最多 3 次带错误信息的重生成；写操作必须人工确认，避免误删误改。", base))

    story.append(PageBreak())
    story.append(Paragraph("五、异步与高并发：你要怎么讲才像做过生产", h1))
    story.append(Paragraph("Q10：高并发下你们怎么保证延迟与稳定？", h2))
    story.append(Paragraph("实现：FastAPI 路由全 async；LangGraph 图执行用 ainvoke；LLM 生成用 ainvoke/astream；同步阻塞（DB/Milvus/解析/部分 CPU）用 asyncio.to_thread 入线程池。线程池大小可配置 asyncio_thread_pool_workers。", base))
    story.append(Paragraph("一致性：同一会话 thread_id 加锁串行化（conversation_lock），避免并发写图状态与 pending_sql。", base))
    story.append(Paragraph("降级：依赖不可用（Redis/Milvus/LLM）有熔断与默认回退；非关键链路失败不阻塞主对话。", base))

    story.append(PageBreak())
    story.append(Paragraph("六、面试官最爱追问：你可以主动补的“替代方案与权衡”", h1))
    story.append(Paragraph("1) 为什么不用纯向量检索？→ 术语/条款号/编码类问题 BM25 更稳，混合能显著降低漏召回。", base))
    story.append(Paragraph("2) 为什么不做全量 cross‑encoder 重排？→ 成本与延迟不可控，因此只对 Top‑K 做精排并做多样性补充。", base))
    story.append(Paragraph("3) 为什么不用 ReAct 自由循环？→ 生产场景优先可控、可观测、可降级；图编排 + 受控路由更适合。", base))
    story.append(Paragraph("4) Grounding 为何不用 LLM Judge？→ 成本高且不稳定；用轻量检索相关度指标更可重复、易解释。", base))

    doc.build(story)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(OUT)
    print(f"Wrote: {OUT}")

