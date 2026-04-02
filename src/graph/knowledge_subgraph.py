"""
knowledge 子图：LangGraph 上仅 **两个执行节点**（与实现划分一致）：

1) **knowledge_text2sql**：结构化查询，对应 `src/kb/text2sql.py`（NL2SQL、校验与执行）。
2) **knowledge_qa_rag**：非结构化知识路径——**Text2SQL 规则预判** → **高频 QA** → 未命中则 **混合路由**；
   若路由为 RAG 则在**本节点内**直接跑 RAG；若路由为 Text2SQL 则经边进入 `knowledge_text2sql`；
   Text2SQL 无结果时经边回到本节点并置 `kb_rag_only`，仅执行 RAG。

业务顺序与 `KnowledgeEngine` 对齐；仅把「原 knowledge_rag」合并进 `knowledge_qa_rag`，减少图上的节点数。
"""

from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.knowledge_agent import (
    knowledge_qa_rag_node_async,
    knowledge_text2sql_node_async,
)

_knowledge_graph = None


def _route_after_qa_rag(state: AgentState) -> str:
    # 错误时直接把 next=human 透出给外层图处理
    if state.get("next") == "human":
        return "__end__"
    # 低置信度澄清：直接结束子图，由调用方继续下一轮
    if state.get("kb_clarify") is True:
        return "__end__"
    # 高频 QA 命中则结束
    if state.get("qa_hit") is True:
        return "__end__"
    # 须先于 next==__end__：上一轮子图结束可能残留 next，不能挡住本次去 Text2SQL
    if state.get("text2sql_candidate") is True:
        return "knowledge_text2sql"
    # 本节点内已跑完 RAG 等并显式结束
    if state.get("next") == "__end__":
        return "__end__"
    return "__end__"


def _route_after_text2sql(state: AgentState) -> str:
    if state.get("next") == "human":
        return "__end__"
    # Text2SQL 产出答案/待确认 SQL 则结束，否则回到 QA+RAG 节点仅跑 RAG
    return state.get("text2sql_hit") is True and "__end__" or "knowledge_qa_rag"


def create_knowledge_graph():
    builder = StateGraph(AgentState)
    builder.add_node("knowledge_qa_rag", knowledge_qa_rag_node_async)
    builder.add_node("knowledge_text2sql", knowledge_text2sql_node_async)

    builder.add_edge(START, "knowledge_qa_rag")
    builder.add_conditional_edges(
        "knowledge_qa_rag",
        _route_after_qa_rag,
        {
            "knowledge_text2sql": "knowledge_text2sql",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "knowledge_text2sql",
        _route_after_text2sql,
        {"knowledge_qa_rag": "knowledge_qa_rag", "__end__": END},
    )
    return builder.compile()


def get_knowledge_graph():
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = create_knowledge_graph()
    return _knowledge_graph

