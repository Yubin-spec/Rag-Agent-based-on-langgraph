"""
knowledge 子图（知识 Agent 内部的多节点流水线）：

Supervisor 只路由到 knowledge（一个子智能体/子图），knowledge 内部再按：
QA → Text2SQL → RAG
执行，形成“外层 2-agent（chat/knowledge）+ 内层 knowledge multi-node”的结构。
"""

from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.knowledge_agent import (
    knowledge_qa_node_async,
    knowledge_text2sql_node_async,
    knowledge_rag_node_async,
)

_knowledge_graph = None


def _route_after_qa(state: AgentState) -> str:
    # 错误时直接把 next=human 透出给外层图处理
    if state.get("next") == "human":
        return "__end__"
    # 低置信度澄清：直接结束子图，由调用方继续下一轮
    if state.get("kb_clarify") is True:
        return "__end__"
    # QA 命中直接结束，否则进入 Text2SQL
    if state.get("qa_hit") is True:
        return "__end__"
    # 二级意图识别：像“数据查询/统计/写操作”才进入 Text2SQL，否则直接走 RAG
    return state.get("text2sql_candidate") is True and "knowledge_text2sql" or "knowledge_rag"


def _route_after_text2sql(state: AgentState) -> str:
    if state.get("next") == "human":
        return "__end__"
    # Text2SQL 产出答案/待确认 SQL 则结束，否则进入 RAG
    return state.get("text2sql_hit") is True and "__end__" or "knowledge_rag"


def create_knowledge_graph():
    builder = StateGraph(AgentState)
    builder.add_node("knowledge_qa", knowledge_qa_node_async)
    builder.add_node("knowledge_text2sql", knowledge_text2sql_node_async)
    builder.add_node("knowledge_rag", knowledge_rag_node_async)

    builder.add_edge(START, "knowledge_qa")
    builder.add_conditional_edges(
        "knowledge_qa",
        _route_after_qa,
        {
            "knowledge_text2sql": "knowledge_text2sql",
            "knowledge_rag": "knowledge_rag",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "knowledge_text2sql",
        _route_after_text2sql,
        {"knowledge_rag": "knowledge_rag", "__end__": END},
    )
    builder.add_edge("knowledge_rag", END)
    return builder.compile()


def get_knowledge_graph():
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = create_knowledge_graph()
    return _knowledge_graph

