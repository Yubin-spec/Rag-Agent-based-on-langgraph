# src/agents/knowledge_agent.py
"""
知识库 Agent 节点：负责 QA 精准匹配、Text2SQL、RAG 三种机制。
内部使用 KnowledgeEngine（其 LLM 仅 DeepSeek），不调用 OpenAI。
"""
from langchain_core.messages import AIMessage, HumanMessage

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


def knowledge_agent_node(state: AgentState) -> dict:
    """
    知识库节点（同步）：从状态中取最后一条用户消息，经 QA → Text2SQL → RAG 顺序处理，
    返回生成的答案并结束本轮（next=__end__）。
    """
    engine = _get_kb_engine()
    last = _get_last_user_text(state)
    if not last:
        return {
            "messages": [AIMessage(content="请直接输入您要咨询的业务或数据问题。")],
            "next": "__end__",
        }
    answer, pending_sql = engine.query(last)
    out = {"messages": [AIMessage(content=answer)], "next": "__end__", "route": "knowledge", "qa_trace": engine.get_last_trace()}
    # 删除/修改类 SQL 只生成不执行时，由 API 层存入 _pending_sql，等用户确认后执行
    if pending_sql:
        out["pending_sql"] = pending_sql
    return out


def _is_retryable_error(e: Exception) -> bool:
    """是否可重试（超时、连接、5xx），与 chat_agent/supervisor 一致。"""
    msg = (getattr(e, "message", "") or str(e)).lower()
    return "timeout" in msg or "connection" in msg or "5" in str(getattr(e, "status_code", ""))


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
            out: dict = {"messages": [AIMessage(content=answer)], "next": "__end__", "route": "knowledge", "qa_trace": engine.get_last_trace()}
            if pending_sql:
                out["pending_sql"] = pending_sql
            return out
        except Exception as e:
            if attempt < max_retries and _is_retryable_error(e):
                continue
            return {"next": "human", "human_message": reply}
