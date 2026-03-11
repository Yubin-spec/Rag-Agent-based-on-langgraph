# src/agents/chat_agent.py
"""
闲聊 Agent：负责打招呼、引导、介绍能力。
使用的大模型仅限 DeepSeek（通过 src.llm.get_deepseek_llm），不调用 OpenAI。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from src.llm import get_deepseek_llm
from .state import AgentState
from .context_summary import summarize_old_messages_async

_settings = get_settings()
_llm = get_deepseek_llm(temperature=0.7)


def _messages_for_llm(state: AgentState) -> list:
    """取最近 llm_context_window_turns 轮消息喂给闲聊 LLM，与 supervisor 窗口逻辑一致。"""
    max_turns = _settings.llm_context_window_turns
    messages = state.get("messages") or []
    if len(messages) <= max_turns * 2:
        return messages
    return list(messages[-max_turns * 2 :])


async def _messages_for_llm_with_summary(state: AgentState) -> list:
    """若启用旧对话摘要，则返回【历史对话摘要】+ 最近 N 轮；否则仅最近 N 轮。"""
    max_turns = _settings.llm_context_window_turns
    messages = state.get("messages") or []
    if len(messages) <= max_turns * 2:
        return messages
    old_messages = list(messages[: -max_turns * 2])
    recent = list(messages[-max_turns * 2 :])
    summary = await summarize_old_messages_async(old_messages)
    if summary:
        return [SystemMessage(content="【历史对话摘要】\n" + summary)] + recent
    return recent

_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是海关12360智能客服的闲聊与引导助手。负责：
1. 友好闲聊、打招呼、寒暄。
2. 引导用户：若用户问“能做什么”“有什么功能”，简要介绍：可回答政策与业务问题（申报材料、AEO认证等）、可查数据（如口岸进口量、认证企业名单）、可讨论复杂政策（如跨境电商监管）。可以说“您可以直接问我具体业务或数据问题”。
请简洁、专业、友好。
回答要求：核心信息说完即结束，不要凑字数。简单问题简短答，复杂问题分阶段答（先结论再补充）。"""),
    ("placeholder", "{messages}"),
])


def chat_agent_node(state: AgentState) -> dict:
    """闲聊节点（同步）：根据当前消息生成回复，并返回 next=__end__ 结束本轮。"""
    chain = _CHAT_PROMPT | _llm
    response = chain.invoke({"messages": _messages_for_llm(state)})
    return {
        "messages": [AIMessage(content=response.content or "您好，请问有什么可以帮您？")],
        "next": "__end__",
        "route": "chat",
    }


def _is_retryable_error(e: Exception) -> bool:
    """判断是否为可重试错误（超时、连接、5xx），用于 LLM 调用重试。"""
    msg = (getattr(e, "message", "") or str(e)).lower()
    return "timeout" in msg or "connection" in msg or "5" in str(getattr(e, "status_code", ""))


async def chat_agent_node_async(state: AgentState) -> dict:
    """闲聊节点（异步）：不阻塞事件循环。异常时返回需人工提示并结束本轮，可重试错误会重试。"""
    chain = _CHAT_PROMPT | _llm
    max_retries = max(0, getattr(_settings, "agent_llm_retry_times", 2))
    for attempt in range(max_retries + 1):
        try:
            messages_to_send = await _messages_for_llm_with_summary(state)
            response = await chain.ainvoke({"messages": messages_to_send})
            return {
                "messages": [AIMessage(content=response.content or "您好，请问有什么可以帮您？")],
                "next": "__end__",
                "route": "chat",
            }
        except Exception as e:
            if attempt < max_retries and _is_retryable_error(e):
                continue
            reply = getattr(_settings, "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。")
            return {"next": "human", "human_message": reply}


def chat_agent_stream(state: AgentState):
    """
    闲聊流式生成（同步）：先出首字再逐 chunk。
    Yields: 文本片段（str），首 chunk 即首字/首 token。
    """
    chain = _CHAT_PROMPT | _llm
    for chunk in chain.stream({"messages": _messages_for_llm(state)}):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


async def chat_agent_stream_async(state: AgentState):
    """
    闲聊流式生成（异步）：先出首字再逐 chunk，用于 SSE，不阻塞事件循环。
    异常时 yield 需人工提示后结束。
    """
    chain = _CHAT_PROMPT | _llm
    try:
        messages_to_send = await _messages_for_llm_with_summary(state)
        async for chunk in chain.astream({"messages": messages_to_send}):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    except Exception:
        reply = getattr(_settings, "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。")
        yield reply
