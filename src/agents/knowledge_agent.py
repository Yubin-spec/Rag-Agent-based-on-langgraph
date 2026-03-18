# src/agents/knowledge_agent.py
"""
知识库 multi-agent 节点集合：将知识能力拆为 3 个独立节点（QA / Text2SQL / RAG）。
这样在 LangGraph 层面就是“多个 Agent/Node 协作”，而不是把路由藏在一个大节点里。
"""
from collections import OrderedDict
import json
from typing import Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from .state import AgentState


def _get_kb_engine():
    """延迟导入 KnowledgeEngine，避免循环依赖与未安装依赖时的启动报错。"""
    from src.kb.engine import KnowledgeEngine
    return KnowledgeEngine()


def _get_last_user_text(state: AgentState) -> str:
    """取最后一条用户消息，供 KnowledgeEngine 做 QA/Text2SQL/RAG 输入。"""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and hasattr(m, "content"):
            return (m.content or "").strip()
    return ""


class _RouteDecision(TypedDict, total=False):
    text2sql: bool
    confidence: float
    reason: str
    clarify_question: str


_router_cache: "OrderedDict[str, _RouteDecision]" = OrderedDict()


def _router_cache_get(key: str) -> Optional[_RouteDecision]:
    v = _router_cache.get(key)
    if v is None:
        return None
    _router_cache.move_to_end(key)
    return v


def _router_cache_put(key: str, value: _RouteDecision, max_entries: int) -> None:
    if max_entries <= 0:
        return
    _router_cache[key] = value
    _router_cache.move_to_end(key)
    while len(_router_cache) > max_entries:
        _router_cache.popitem(last=False)


def _rule_based_text2sql_candidate(question: str) -> Optional[bool]:
    """
    knowledge 内部路由用的轻量判别：像“数据查询/统计/筛选”才进入 Text2SQL 节点。
    目标是避免不必要的 Text2SQL LLM 调用，让大多数知识问答直接走 RAG。
    """
    from src.kb.intent_router import rule_based_text2sql_candidate

    return rule_based_text2sql_candidate(question)


_KB_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是知识库子链路的路由器。目标：在 QA 未命中后，决定下一步走 Text2SQL 还是 RAG。

判断规则（只需二选一）：
- text2sql：用户在问“数据查询/统计/排名/报表/筛选条件/时间范围/指标对比”，或涉及删除/修改数据（需生成待确认 SQL）。
- rag：用户在问“政策/流程/概念解释/材料要求/办事指南/规则解读/非结构化知识问答”。

请输出严格 JSON（不要 markdown，不要额外文字），格式：
{
  "route": "text2sql" | "rag",
  "confidence": 0.0~1.0,
  "reason": "一句话理由",
  "clarify_question": "当你不确定时给用户的一句澄清问题；如果很确定可留空字符串"
}
当你 confidence < 0.6 时，必须给出 clarify_question。""",
        ),
        ("human", "{question}"),
    ]
)


def _parse_router_json(text: str) -> Optional[_RouteDecision]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        # 尝试从杂讯中截取最外层 JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
        except Exception:
            return None
    route = str(obj.get("route", "")).strip().lower()
    if route not in ("text2sql", "rag"):
        return None
    confidence = obj.get("confidence", None)
    try:
        conf = float(confidence) if confidence is not None else 0.0
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reason = str(obj.get("reason", "") or "").strip()
    clarify = str(obj.get("clarify_question", "") or "").strip()
    return {
        "text2sql": route == "text2sql",
        "confidence": conf,
        "reason": reason,
        "clarify_question": clarify,
    }


async def _llm_route_text2sql_or_rag(question: str) -> _RouteDecision:
    from config import get_settings
    from src.llm import get_deepseek_llm

    settings = get_settings()
    cache_key = f"v2|{question.strip()}"
    cached = _router_cache_get(cache_key)
    if cached is not None:
        return cached

    llm = get_deepseek_llm(temperature=0)
    chain = _KB_ROUTER_PROMPT | llm
    resp = await chain.ainvoke({"question": question})
    out = (getattr(resp, "content", "") or "").strip()
    parsed = _parse_router_json(out)
    if not parsed:
        # 兼容兜底：仍允许模型只输出关键词
        lowered = out.lower()
        parsed = {
            "text2sql": ("text2sql" in lowered),
            "confidence": 0.0,
            "reason": "",
            "clarify_question": "",
        }
    _router_cache_put(
        cache_key,
        parsed,
        int(getattr(settings, "knowledge_router_cache_max_entries", 0) or 0),
    )
    return parsed


async def _decide_text2sql_candidate(question: str) -> Tuple[bool, float, str, str]:
    """
    混合路由：规则优先（高性能）+ 歧义时用 LLM（高准确）。
    返回 (text2sql_candidate, confidence, reason, clarify_question)。
    """
    from config import get_settings

    settings = get_settings()
    rule = _rule_based_text2sql_candidate(question)
    if rule is not None and getattr(settings, "knowledge_router_rules_short_circuit", True):
        return (bool(rule), 1.0, "rule_short_circuit", "")

    use_llm = bool(getattr(settings, "knowledge_router_use_llm_when_uncertain", True))
    if not use_llm:
        return (bool(rule) if rule is not None else False, 0.0, "llm_disabled", "")

    # 若 rule 已有结论但不短路：允许 LLM 覆盖；若 rule 为 None：LLM 补齐
    try:
        dec = await _llm_route_text2sql_or_rag(question)
        return (
            bool(dec.get("text2sql", False)),
            float(dec.get("confidence", 0.0) or 0.0),
            str(dec.get("reason", "") or ""),
            str(dec.get("clarify_question", "") or ""),
        )
    except Exception:
        # 路由失败时保守回退：优先遵循规则；规则未知则走 RAG（不触发 Text2SQL 误用）
        return (bool(rule) if rule is not None else False, 0.0, "router_exception_fallback", "")

def knowledge_agent_node(state: AgentState) -> dict:
    """
    兼容旧入口（同步）：仍按 QA → Text2SQL → RAG 顺序一口气跑完。
    新图会使用拆分后的三个节点：knowledge_qa / knowledge_text2sql / knowledge_rag。
    """
    engine = _get_kb_engine()
    last = _get_last_user_text(state)
    if not last:
        return {
            "messages": [AIMessage(content="请直接输入您要咨询的业务或数据问题。")],
            "next": "__end__",
        }
    answer, pending_sql = engine.query(last)
    out: dict = {
        "messages": [AIMessage(content=answer)],
        "next": "__end__",
        "route": "knowledge",
        "qa_trace": engine.get_last_trace(),
    }
    # 删除/修改类 SQL 只生成不执行时，由 API 层存入 _pending_sql，等用户确认后执行
    if pending_sql:
        out["pending_sql"] = pending_sql
    return out


def _is_retryable_error(e: Exception) -> bool:
    """是否可重试（超时、连接、5xx），与 chat_agent/supervisor 一致。"""
    msg = (getattr(e, "message", "") or str(e)).lower()
    return "timeout" in msg or "connection" in msg or "5" in str(getattr(e, "status_code", ""))


async def knowledge_qa_node_async(state: AgentState) -> dict:
    """
    QA 节点：命中则直接回复并结束；未命中则不产出消息，仅标记 qa_hit=False 交给下一步。
    """
    last = _get_last_user_text(state)
    if not last:
        return {"messages": [AIMessage(content="请直接输入您要咨询的业务或数据问题。")], "next": "__end__"}

    from config import get_settings
    settings = get_settings()
    max_retries = max(0, getattr(settings, "agent_llm_retry_times", 2))
    reply = getattr(settings, "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。")

    for attempt in range(max_retries + 1):
        try:
            engine = _get_kb_engine()
            trace = engine._reset_trace()
            # 先做 Text2SQL 意图识别：若明确是结构化数据查询/写操作，则直接短路跳过 QA 匹配
            # 目的：避免高频 QA 误命中以及不必要的匹配开销
            rule = _rule_based_text2sql_candidate(last)
            if rule is True:
                return {
                    "route": "knowledge_qa",
                    "qa_trace": engine.get_last_trace(),
                    "qa_hit": False,
                    "text2sql_candidate": True,
                    "kb_route_confidence": 1.0,
                    "kb_route_reason": "rule_short_circuit_pre_qa",
                }

            # 非明确 Text2SQL：再尝试高频 QA（快路径）
            answer = engine.qa.find(last)
            if answer:
                trace.route = "qa"
                trace.final_status = "qa_hit"
                return {
                    "messages": [AIMessage(content=answer)],
                    "next": "__end__",
                    "route": "knowledge_qa",
                    "qa_trace": engine.get_last_trace(),
                    "qa_hit": True,
                }

            # QA 未命中：进入 Text2SQL vs RAG 的混合路由（规则+LLM）
            text2sql_candidate, conf, reason, clarify_q = await _decide_text2sql_candidate(last)

            low_th = float(getattr(settings, "knowledge_router_low_confidence_threshold", 0.6) or 0.6)
            clarify_on_low = bool(getattr(settings, "knowledge_router_clarify_on_low_confidence", True))
            if clarify_on_low and conf > 0 and conf < low_th:
                # 低置信度：优先澄清，避免误判导致走错链路
                q = clarify_q or "你是想查“数据库里的统计/明细数据”（我可以帮你生成查询），还是想问“政策/流程/规则解释”（我可以基于知识库回答）？"
                return {
                    "messages": [AIMessage(content=q)],
                    "next": "__end__",
                    "route": "knowledge_router_clarify",
                    "qa_trace": engine.get_last_trace(),
                    "qa_hit": False,
                    "kb_clarify": True,
                    "kb_route_confidence": conf,
                    "kb_route_reason": reason,
                }

            return {
                "route": "knowledge_qa",
                "qa_trace": engine.get_last_trace(),
                "qa_hit": False,
                "text2sql_candidate": bool(text2sql_candidate),
                "kb_route_confidence": conf,
                "kb_route_reason": reason,
            }
        except Exception as e:
            if attempt < max_retries and _is_retryable_error(e):
                continue
            return {"next": "human", "human_message": reply}


async def knowledge_text2sql_node_async(state: AgentState) -> dict:
    """
    Text2SQL 节点：仅在确需 Text2SQL 时进入；若生成删除/修改类 SQL，则写入 pending_sql 等待确认。
    """
    last = _get_last_user_text(state)
    if not last:
        return {"messages": [AIMessage(content="请直接输入您要咨询的业务或数据问题。")], "next": "__end__"}

    from config import get_settings
    settings = get_settings()
    max_retries = max(0, getattr(settings, "agent_llm_retry_times", 2))
    reply = getattr(settings, "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。")

    for attempt in range(max_retries + 1):
        try:
            engine = _get_kb_engine()
            trace = engine._reset_trace()
            result = engine.text2sql.query(last)
            if result is not None:
                trace.route = "text2sql"
                # 删除/修改类 SQL 需人工确认
                from src.kb.text2sql import Text2SQLConfirmRequired

                if isinstance(result, Text2SQLConfirmRequired):
                    trace.final_status = "text2sql_pending"
                    trace.pending_sql = True
                    return {
                        "messages": [AIMessage(content=result.message)],
                        "next": "__end__",
                        "route": "knowledge_text2sql",
                        "qa_trace": engine.get_last_trace(),
                        "pending_sql": result.sql,
                        "text2sql_hit": True,
                    }
                trace.final_status = "text2sql_answer"
                return {
                    "messages": [AIMessage(content=result)],
                    "next": "__end__",
                    "route": "knowledge_text2sql",
                    "qa_trace": engine.get_last_trace(),
                    "text2sql_hit": True,
                }

            # 兜底：Text2SQL 无法产出时，继续走 RAG
            return {
                "route": "knowledge_text2sql",
                "qa_trace": engine.get_last_trace(),
                "text2sql_hit": False,
            }
        except Exception as e:
            if attempt < max_retries and _is_retryable_error(e):
                continue
            return {"next": "human", "human_message": reply}


async def knowledge_rag_node_async(state: AgentState) -> dict:
    """
    RAG 节点：只跑 RAG（含二次 RAG），输出带引用的答案。
    """
    last = _get_last_user_text(state)
    if not last:
        return {"messages": [AIMessage(content="请直接输入您要咨询的业务或数据问题。")], "next": "__end__"}

    from config import get_settings
    settings = get_settings()
    max_retries = max(0, getattr(settings, "agent_llm_retry_times", 2))
    reply = getattr(settings, "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。")

    for attempt in range(max_retries + 1):
        try:
            engine = _get_kb_engine()
            answer, _ = await engine.aquery_rag_only(last)
            return {
                "messages": [AIMessage(content=answer)],
                "next": "__end__",
                "route": "knowledge_rag",
                "qa_trace": engine.get_last_trace(),
            }
        except Exception as e:
            if attempt < max_retries and _is_retryable_error(e):
                continue
            return {"next": "human", "human_message": reply}


async def knowledge_agent_node_async(state: AgentState) -> dict:
    """
    知识库节点（异步）：不阻塞事件循环，内部使用 engine.aquery。
    当存在待确认 SQL（删除/修改）时，将 pending_sql 写入 state，供 API 确认执行。
    异常时返回需人工提示并结束本轮，可重试错误会重试。
    """
    last = _get_last_user_text(state)
    if not last:
        return {
            "messages": [AIMessage(content="请直接输入您要咨询的业务或数据问题。")],
            "next": "__end__",
        }
    from config import get_settings
    settings = get_settings()
    max_retries = max(0, getattr(settings, "agent_llm_retry_times", 2))
    reply = getattr(settings, "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。")
    for attempt in range(max_retries + 1):
        try:
            engine = _get_kb_engine()
            answer, pending_sql = await engine.aquery(last)
            out: dict = {
                "messages": [AIMessage(content=answer)],
                "next": "__end__",
                "route": "knowledge",
                "qa_trace": engine.get_last_trace(),
            }
            if pending_sql:
                out["pending_sql"] = pending_sql
            return out
        except Exception as e:
            if attempt < max_retries and _is_retryable_error(e):
                continue
            return {"next": "human", "human_message": reply}
