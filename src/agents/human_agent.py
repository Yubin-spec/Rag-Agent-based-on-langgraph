# src/agents/human_agent.py
"""
人工介入节点：使用 LangGraph 的 interrupt() 实现 human-in-the-loop。
当总控/闲聊/知识库异常或需转人工时，路由到此节点；节点内调用 interrupt(payload) 暂停，
调用方从 result["__interrupt__"] 取 payload 展示给用户，恢复时用 Command(resume=...) 继续执行。
"""
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from config import get_settings
from .state import AgentState


async def human_handoff_node_async(state: AgentState) -> dict:
    """
    人工介入节点（异步）：先 interrupt 暂停并向外返回 human_message，恢复后再将回复写入 state 并结束。
    """
    msg = state.get("human_message") or getattr(
        get_settings(), "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。"
    )
    # 暂停执行，payload 会出现在调用方 result["__interrupt__"]，便于前端展示或对接工单
    interrupt({"human_message": msg})
    # resume 后继续：将转人工回复写入对话并结束
    return {
        "messages": [AIMessage(content=msg)],
        "next": "__end__",
    }
