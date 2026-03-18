from __future__ import annotations

import datetime as _dt
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    PageBreak,
    ListFlowable,
    ListItem,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "海关小智_异步缓存降级与性能追踪.pdf"


def _register_cjk_font() -> str:
    name = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(name))
    except Exception:
        pass
    return name


def _p(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _bullet(items: list[str], style) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(_p(x), style)) for x in items],
        bulletType="bullet",
        leftIndent=14,
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
    h3 = ParagraphStyle(
        "CJKH3",
        parent=styles["Heading3"],
        fontName=font,
        fontSize=12,
        leading=16,
        spaceBefore=4,
        spaceAfter=4,
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
        title="海关小智：异步缓存降级与性能追踪",
        author="nanqi",
    )

    now = _dt.datetime.now().strftime("%Y-%m-%d")
    story: list = []
    story.append(Paragraph("海关小智智能问答<br/>异步 + 缓存 + 熔断/降级 + 性能追踪（面试背诵版）", h1))
    story.append(Paragraph(f"生成日期：{now}    版本：v2（新增白板时序图口述页）", base))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "目标：把“我做了异步/缓存/降级/可观测性”讲成可验证的工程设计：为什么要这样做、底层原理是什么、代码如何落地、如何评估效果与排障。",
            base,
        )
    )

    story.append(Paragraph("0. 快速定位（让面试官相信你真做过）", h2))
    story.append(
        _bullet(
            [
                "对话入口：api/main.py 的 POST /chat 与 POST /chat/stream（会话锁、缓存、图执行、观测落库都在这里串起来）。",
                "缓存实现：src/answer_cache.py（Redis async + 本地 LRU + single-flight 防击穿）。",
                "熔断与韧性：src/db_resilience.py（DB/Milvus 统一连接池、重试、熔断、critical/非关键降级）。",
                "LLM 多节点熔断与负载：src/llm.py（inflight/weight 选节点，连续失败熔断，半开探测）。",
                "共享状态：src/shared_state.py（多 worker pending_sql / parse_cache / interrupted 标记，Redis 或进程内降级）。",
                "观测与追踪：src/qa_monitoring.py（observation + rag_trace + feedback 三表）；api/main.py 写入。",
            ],
            base,
        )
    )
    story.append(PageBreak())

    # ---------- Whiteboard page ----------
    story.append(Paragraph("白板一页讲清：/chat 一次请求的全链路时序（含缓存/降级/观测）", h1))
    story.append(Paragraph("1) 你在白板上画什么（时序骨架）", h2))
    story.append(
        Paragraph(
            _p(
                "User → API(/chat or /chat/stream) → conversation_lock(thread_id)\n"
                "→ resume_if_interrupted → turn_limit_check\n"
                "→ answer_cache(get)\n"
                "  ├─ hit: update_graph_state + persist_history + save_observation(route=cache) → return\n"
                "  └─ miss: answer_lock(single-flight)\n"
                "        → answer_cache(get again)\n"
                "        → graph.ainvoke({messages}, thread_id)\n"
                "             ├─ timeout/cancelled: save_observation(route=timeout/cancelled) → return\n"
                "             ├─ __interrupt__: shared_state.add_interrupted + save_observation(route=human) → return\n"
                "             ├─ pending_sql: shared_state.set_pending_sql\n"
                "             └─ normal: extract AI reply\n"
                "        → answer_cache(set)\n"
                "        → persist_conversation_session + persist_chat_messages\n"
                "        → save_observation(route=chat/knowledge, trace=qa_trace)\n"
                "        → return"
            ),
            mono,
        )
    )

    story.append(Paragraph("2) 你怎么口述（逐句可背）", h2))
    story.append(
        _bullet(
            [
                "入口层先做 thread_id 会话锁：同一会话串行化，避免并发写坏 LangGraph 状态和 pending_sql。",
                "如果上一次会话在 human interrupt 中断，我会先 resume 再处理新消息，保证状态机一致。",
                "然后做轮数上限检查（15 轮），防止极长会话拖垮 token/延迟；触发就直接返回并记录观测。",
                "接着走问答缓存：先查 Redis（可选本地 LRU），命中直接回包，同时把 user/assistant 写入图状态与长期记忆，并记录 route=cache。",
                "未命中时进入 single-flight：同一问题只允许一个协程回源，其余等待，避免缓存击穿把 LLM/检索打爆。",
                "回源执行图：graph.ainvoke 有超时保护；如果返回 interrupt 说明转人工，我会把 thread_id 标记为 interrupted 存共享状态，下一次先 resume。",
                "如果知识库产生危险 SQL，会把 pending_sql 存共享状态，等待用户“确认执行”再真正执行，避免误删误改。",
                "正常返回后写缓存、写对话历史、写观测与 RAG trace（latency、route、grounding、retrieve_attempt、doc_id/chunk_id 等），形成可观测闭环。",
            ],
            base,
        )
    )

    story.append(Paragraph("3) 每一步的“目的”一句话总结（面试追问必杀）", h2))
    story.append(
        _bullet(
            [
                "会话锁：一致性；single-flight：削峰填谷；缓存：降成本降延迟。",
                "超时/熔断：保护下游，避免雪崩；降级：保证用户体验不 500。",
                "观测与 trace：能回答“慢在哪、差在哪、改动有没有用”。",
            ],
            base,
        )
    )

    story.append(PageBreak())

    # ---------- Part A: Async + Cache + Degrade ----------
    story.append(Paragraph("一、异步 + 高并发：底层原理与落地策略", h1))
    story.append(Paragraph("1.1 原理：为什么要“event loop 不阻塞”？", h2))
    story.append(
        Paragraph(
            "FastAPI 运行在异步事件循环上。任何同步阻塞（DB、文件、CPU 密集、网络阻塞）如果直接在 async 函数里执行，会阻塞整个事件循环，导致同进程其它请求延迟飙升甚至超时。",
            base,
        )
    )
    story.append(
        Paragraph(
            "因此本项目的硬约束是：LLM 等待、网络 IO 走 async；不可避免的同步段用 asyncio.to_thread() 扔到线程池；并用超时/限流/熔断保护下游。",
            base,
        )
    )

    story.append(Paragraph("1.2 落地：分层异步策略（你可以背的清单）", h2))
    story.append(
        _bullet(
            [
                "API 层：/chat、/chat/stream、/doc/upload 全 async。",
                "图执行：LangGraph 用 ainvoke/aget_state/aupdate_state。",
                "LLM：chain.ainvoke / chain.astream（等待模型不占线程）。",
                "同步段隔离：QA/Text2SQL/DB/校验等用 to_thread；线程池大小可配 asyncio_thread_pool_workers。",
                "限流：全局并发由 api_max_concurrent_requests 控制；MinerU 解析有信号量并发上限。",
                "会话一致性：conversation_lock(thread_id) 串行化同会话请求，避免并发写坏图状态与 pending_sql。",
            ],
            base,
        )
    )
    story.append(Paragraph("关键配置（背点）", h3))
    story.append(
        Paragraph(
            _p(
                "llm_context_window_turns=10（上下文窗口）\n"
                "agent_request_timeout_seconds=120（单次请求/LLM 超时）\n"
                "agent_llm_retry_times=2（LLM 重试）\n"
                "asyncio_thread_pool_workers（线程池）\n"
                "uvicorn_workers（多进程并发）\n"
                "conversation_lock_buckets=1024（会话锁分桶）"
            ),
            mono,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("二、缓存策略：命中路径、击穿/热 key、失效与一致性", h1))

    story.append(Paragraph("2.1 问答缓存（Answer Cache）：结果级缓存", h2))
    story.append(
        Paragraph(
            "定位：src/answer_cache.py + api/main.py。\n"
            "Key=kb:answer:v1:sha256(normalized_question)[:32]；Value=纯文本答案；TTL=86400s（24h）。",
            base,
        )
    )
    story.append(
        Paragraph(
            "底层原理：结果级缓存直接跳过图执行/检索/LLM，最有效降低延迟与成本；同时要处理缓存不可用、热 key、击穿三个生产问题。",
            base,
        )
    )
    story.append(Paragraph("你要能背出 3 个“生产级优化点”", h3))
    story.append(
        _bullet(
            [
                "断线重连：Redis 读写失败置空客户端，下次懒加载重连；读/写失败静默跳过，不阻断主流程。",
                "热 key：可选进程内 LRU（answer_cache_local_max_entries / ttl），命中本地则不打 Redis。",
                "防击穿（single-flight）：按 key 分桶锁（answer_cache_single_flight_buckets），同一问题未命中仅一个协程回源，其余等待后再查缓存。",
            ],
            base,
        )
    )

    story.append(Paragraph("2.2 RAG 检索缓存：中间结果缓存", h2))
    story.append(
        Paragraph(
            "定位：src/kb/rag.py（_merge_3_7 外层 LRU）。Key=sha256(query|total_k|use_rerank|rerank_top)。命中后跳过 Milvus/BM25/重排重复计算，但仍需要生成答案。",
            base,
        )
    )
    story.append(
        Paragraph(
            "面试常问差异：问答缓存是“问题→最终答案”，检索缓存是“query→候选证据”，一个是结果级、一个是中间级。中间级缓存可降低检索与重排成本，但不完全消除 LLM 成本。",
            base,
        )
    )

    story.append(Paragraph("2.3 一致性与失效：怎么回答才像做过生产", h2))
    story.append(
        Paragraph(
            "本项目默认是“最终一致”：按 TTL 自然失效，不做强一致。文档更新后短时间内可能命中旧缓存，属于可接受的业务折中（读多写少、允许短暂 stale）。若业务要求强一致：要么缩短 TTL、要么对 doc_id 相关 key 做主动失效（需要额外索引或反向映射）。",
            base,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("三、熔断与降级：如何保护下游、避免雪崩", h1))

    story.append(Paragraph("3.1 熔断 vs 重试：底层原理", h2))
    story.append(
        Paragraph(
            "重试解决“瞬时失败”；熔断解决“持续失败”。当下游持续失败/超时，高并发下重试会放大压力，造成线程/连接耗尽。熔断在 OPEN 期间快速失败或降级返回默认值，保护本服务与其它依赖。",
            base,
        )
    )

    story.append(Paragraph("3.2 DB / Milvus：统一韧性层（CircuitBreaker 状态机）", h2))
    story.append(
        Paragraph(
            "定位：src/db_resilience.py。\n"
            "状态机：CLOSED → 连续失败达到阈值 → OPEN（拒绝）→ 冷却期到 → HALF_OPEN（放行探测）→ 成功回 CLOSED / 失败回 OPEN。",
            base,
        )
    )
    story.append(Paragraph("关键点：critical=True/False 的差别（一定要讲出来）", h3))
    story.append(
        _bullet(
            [
                "critical=True：关键路径失败直接抛异常（例如 Text2SQL 执行必须连 DB）。",
                "critical=False：非关键路径熔断/失败时降级返回 None / [] / 0，不阻断主流程（例如对话历史落库、问答监控落库、RAG 向量检索等）。",
            ],
            base,
        )
    )

    story.append(Paragraph("3.3 LLM 多节点：加权负载 + 节点熔断", h2))
    story.append(
        Paragraph(
            "定位：src/llm.py。选择节点时按 inflight/weight 排序（低负载优先），同分用轮询打散；可重试错误（timeout/connection/429/5xx）累计失败，达到阈值后对节点熔断 open_seconds。所有节点都熔断时，放行“最早到期”的一个节点半开探测。",
            base,
        )
    )

    story.append(Paragraph("3.4 降级输出：用户体验如何兜住", h2))
    story.append(
        Paragraph(
            "对话链路里最重要的降级是：总控/LLM 失败不返回 500，而是走 human 节点或统一话术（agent_need_human_reply）；缓存不可用等非关键故障静默跳过；RAG 检索失败返回空证据并提示“无法确认”。",
            base,
        )
    )

    story.append(PageBreak())

    # ---------- Part B: Observability ----------
    story.append(Paragraph("四、智能体性能评估与追踪：指标口径、Trace 设计、如何闭环", h1))
    story.append(Paragraph("4.1 为什么要做“观测 + trace”？底层原理", h2))
    story.append(
        Paragraph(
            "Agent 系统的核心难点是：问题多样、链路复杂、线上波动大。没有可观测性就无法回答三个关键问题："
            "（1）慢在哪里？（2）差在哪里？（3）改动是否真的提升？因此必须把一次问答拆成可追踪的链路并结构化落库。",
            base,
        )
    )

    story.append(Paragraph("4.2 数据模型：Observation / RAG Trace / Feedback", h2))
    story.append(
        Paragraph(
            "定位：src/qa_monitoring.py。\n"
            "三张表：qa_observation（一次问答的宏观结果）+ qa_rag_trace（RAG 过程指标与证据）+ qa_feedback（用户反馈）。写入方在 api/main.py 的 _save_qa_observation_async。",
            base,
        )
    )

    story.append(Paragraph("核心字段（你要能背并解释“为什么”）", h3))
    table_data = [
        ["字段", "含义", "为什么要记录"],
        ["route", "本次走的链路：cache/chat/knowledge/human/timeout等", "做分桶分析：不同链路 P95/命中率/故障率不同"],
        ["latency_ms", "端到端耗时", "性能指标：P50/P95/P99；定位慢请求"],
        ["used_cache", "是否命中缓存", "评估缓存收益与容量/TTL是否合理"],
        ["fallback_reason", "降级原因（timeout/human_handoff/low_grounding等）", "把失败分类，形成改进 backlog"],
        ["llm_model/endpoint", "模型名与节点名", "对比不同节点延迟/故障；熔断前兆"],
        ["grounding_score", "答案与证据的相关度信号", "评估幻觉风险，作为重生成/拒答依据"],
        ["retrieve_attempt", "第几次检索通过（重检）", "评估检索质量与阈值设置是否过严/过松"],
        ["retrieved_doc_ids/chunk_ids", "命中的文档与切片", "复现坏例、做数据修复与索引修复"],
    ]
    tbl = Table(table_data, colWidths=[32 * mm, 52 * mm, 70 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef6")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c3d6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.3 指标体系：怎么讲得“有工程味”", h2))
    story.append(
        _bullet(
            [
                "性能：P50/P95/P99 latency（分 route、分模型节点、分是否命中缓存）。",
                "稳定性：success rate、timeout rate、human_handoff rate；熔断次数与恢复时间。",
                "质量：grounding_score 分布、has_evidence_citations 比例、bad case 列表与标签统计。",
                "业务：命中 QA/Text2SQL/RAG 的占比，高频问题覆盖度，人工介入占比。",
            ],
            base,
        )
    )

    story.append(Paragraph("4.4 闭环：从 trace 到迭代优化（你要能举例）", h2))
    story.append(
        Paragraph(
            "典型闭环：从 qa_observation 里筛选 fallback_reason=low_grounding 或 rating=bad 的案例 → "
            "用 retrieved_doc_ids/chunk_ids 复现检索证据 → "
            "判断是分块问题、召回问题还是 prompt 约束问题 → "
            "调整 rag_chunk_sizes/rrf_k/nprobe/grounding 阈值或补充 query rewrite 规则 → "
            "再看 grounding_score 与 bad-case rate 是否下降。",
            base,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("五、最常见面试追问（直接背答案）", h1))
    qa = [
        ("Q1：为什么 single-flight 能防击穿？",
         "同一 key 未命中时，如果不加锁，高并发会让所有请求同时回源（LLM/检索），造成下游雪崩。single-flight 保证同一 key 只有一个协程回源写缓存，其余等待后再读缓存，回源次数从 O(N) 降为 O(1)。我们用分桶锁避免全局锁竞争。"),
        ("Q2：熔断为什么一定要有 HALF_OPEN？",
         "OPEN 只会拒绝；HALF_OPEN 允许少量探测请求判断下游是否恢复，成功则自动恢复 CLOSED，失败则继续 OPEN。否则只能依赖人工恢复或一直不敢恢复。"),
        ("Q3：为什么对有些模块 critical=False？",
         "生产里要区分“核心链路”和“非核心链路”。比如问答监控落库失败不应该影响用户拿到答案，所以 critical=False 直接跳过；但 Text2SQL 真执行连不上库就没法完成动作，必须 critical=True 抛错或提示重试。"),
        ("Q4：多 worker 下状态怎么保证？",
         "LangGraph 内存 checkpointer（MemorySaver）不跨进程；因此我们把 pending_sql、interrupted 标记、解析缓存放进 shared_state Redis（shared_state_redis_url）以实现多 worker 共享；同一会话仍建议会话保持，并用 thread_id 会话锁避免并发写。"),
        ("Q5：怎么证明你做的优化真的有效？",
         "看三类指标：延迟（P95 降、首包更快）、成本（缓存命中率上升、LLM 调用次数下降）、质量（grounding_score 提升、bad-case rate 下降、human_handoff 下降）。所有指标都要按 route/模型节点分桶对比，避免平均数掩盖问题。"),
    ]
    for q, a in qa:
        story.append(Paragraph(q, h3))
        story.append(Paragraph(_p(a), base))

    doc.build(story)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(OUT)
    print(f"Wrote: {OUT}")

