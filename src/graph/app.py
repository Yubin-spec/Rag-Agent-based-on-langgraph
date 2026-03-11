# src/graph/app.py
"""
LangGraph 多智能体图：Supervisor 模式 + 人工介入（interrupt）。
流程：START → 总控(supervisor) → 条件边(chat | knowledge | human) → 对应子节点 → END。
chat、knowledge 异常时可路由到 human；human 节点内调用 interrupt(payload) 暂停，
调用方从 result["__interrupt__"] 取 payload 展示，恢复时用 Command(resume=...) 继续执行。
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AgentState, NextAction
from src.agents.supervisor import supervisor_node_async, route_to_agent
from src.agents.chat_agent import chat_agent_node_async
from src.agents.knowledge_agent import knowledge_agent_node_async
from src.agents.human_agent import human_handoff_node_async

# 全局图单例与检查点，避免重复编译（异步节点，支持 ainvoke 高性能并发）
_graph = None
_checkpointer = None


def _route_after_sub(state: AgentState) -> str:
    """
    chat / knowledge 子节点执行后的条件边：若子节点因异常设置了 next=human，则进入 human 节点；
    否则 next=__end__，直接结束本轮。
    """
    return state.get("next") == "human" and "human" or "__end__"


def create_graph(*, checkpointer=None):
    """
    构建并编译 LangGraph（异步节点版本）。
    - 节点：supervisor、chat、knowledge、human（人工介入）均为 async
    - 边：START→supervisor；supervisor→chat|knowledge|human|END；chat/knowledge→human|END；human→END
    """
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node_async)
    builder.add_node("chat", chat_agent_node_async)
    builder.add_node("knowledge", knowledge_agent_node_async)
    builder.add_node("human", human_handoff_node_async)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "chat": "chat",
            "knowledge": "knowledge",
            "human": "human",
            "__end__": END,
        },
    )
    builder.add_conditional_edges("chat", _route_after_sub, {"human": "human", "__end__": END})
    builder.add_conditional_edges("knowledge", _route_after_sub, {"human": "human", "__end__": END})
    builder.add_edge("human", END)

    memory = checkpointer or MemorySaver()
    compiled = builder.compile(checkpointer=memory)
    return compiled


def get_graph():
    """
    获取全局编译后的 LangGraph 单例，使用 MemorySaver 做 thread_id 维度的状态持久化。
    多轮对话依赖此检查点恢复 messages 等状态。
    """
    global _graph, _checkpointer
    if _graph is None:
        _checkpointer = MemorySaver()
        _graph = create_graph(checkpointer=_checkpointer)
    return _graph
