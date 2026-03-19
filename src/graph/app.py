# src/graph/app.py
"""
LangGraph 多智能体图：Supervisor 模式 + 人工介入（interrupt）。
流程：START → 总控(supervisor) → 条件边(chat | knowledge | human) → 对应子节点 → END。

其中 knowledge 为“知识库 Agent/子图”，内部拆分为 3 个独立节点（multi-node）：
QA → Text2SQL → RAG

chat、knowledge_* 异常时可路由到 human；human 节点内调用 interrupt(payload) 暂停，
调用方从 result["__interrupt__"] 取 payload 展示，恢复时用 Command(resume=...) 继续执行。
短期记忆：可选 checkpointer（chat_checkpointer_postgresql_uri / chat_checkpointer_redis_url）实现多 worker 共享状态，否则 MemorySaver（仅进程内）。
"""
import sys
from pathlib import Path

# 保证直接运行本文件时（如 python src/graph/app.py）项目根在 path 中，可 import src
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AgentState, NextAction
from src.agents.supervisor import supervisor_node_async, route_to_agent
from src.agents.chat_agent import chat_agent_node_async
from src.graph.knowledge_subgraph import get_knowledge_graph
from src.agents.human_agent import human_handoff_node_async

logger = logging.getLogger(__name__)

# 全局图单例与检查点，避免重复编译（异步节点，支持 ainvoke 高性能并发）
_graph = None
_checkpointer = None
_checkpointer_cm = None  # checkpointer 上下文（例如 PostgresSaver.from_conn_string 的 contextmanager）


def _make_checkpointer():
    """
    优先使用 PostgreSQL checkpointer（chat_checkpointer_postgresql_uri）。
    如未配置/不可用，再尝试 Redis checkpointer（chat_checkpointer_redis_url）。
    最终回退到进程内 MemorySaver。
    """
    from config import get_settings

    global _checkpointer_cm
    _checkpointer_cm = None

    settings = get_settings()

    pg_url = (getattr(settings, "chat_checkpointer_postgresql_uri", None) or "").strip()
    if pg_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[reportMissingImports]

            cm = PostgresSaver.from_conn_string(pg_url)
            # PostgresSaver.from_conn_string 是 contextmanager，需要显式 enter 才能拿到 saver
            saver = cm.__enter__() if hasattr(cm, "__enter__") else cm
            saver.setup()
            _checkpointer_cm = cm
            logger.info("短期记忆已使用 PostgreSQL checkpointer（支持跨进程 interrupt 后 resume）")
            return saver
        except Exception as e:
            logger.warning("PostgreSQL checkpointer 不可用，继续尝试 Redis/MemorySaver: %s", e)

    redis_url = (getattr(settings, "chat_checkpointer_redis_url", None) or "").strip()
    if not redis_url:
        return MemorySaver()

    try:
        from langgraph.checkpoint.redis import RedisSaver  # type: ignore[reportMissingImports]

        cm = RedisSaver.from_conn_string(redis_url)
        # RedisSaver.from_conn_string 可能也是 contextmanager；这里做兼容处理
        saver = cm.__enter__() if hasattr(cm, "__enter__") else cm
        saver.setup()
        _checkpointer_cm = cm if hasattr(cm, "__exit__") else None
        logger.info("短期记忆已使用 Redis checkpointer（多 worker 可共享状态）")
        return saver
    except Exception as e:
        logger.warning("Redis checkpointer 不可用，回退 MemorySaver: %s", e)
        return MemorySaver()


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
    # knowledge 作为一个“子智能体/子图”，内部再按 QA→Text2SQL→RAG 跑完
    builder.add_node("knowledge", get_knowledge_graph())
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
    获取全局编译后的 LangGraph 单例。按配置使用 PostgreSQL/Redis checkpointer 共享或 MemorySaver。
    多轮对话依赖此检查点恢复 messages 等状态。
    """
    global _graph, _checkpointer
    if _graph is None:
        _checkpointer = _make_checkpointer()
        _graph = create_graph(checkpointer=_checkpointer)
    return _graph


def close_checkpointer() -> None:
    """关闭 checkpointer 的上下文资源（例如 PostgresSaver 的数据库连接）。"""
    global _checkpointer_cm
    cm = _checkpointer_cm
    if cm is None:
        return
    try:
        if hasattr(cm, "__exit__"):
            cm.__exit__(None, None, None)
    except Exception as e:
        logger.warning("关闭 checkpointer 失败（忽略）：%s", e)
    finally:
        _checkpointer_cm = None
