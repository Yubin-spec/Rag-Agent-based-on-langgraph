# src/agents/context_summary.py
"""
将「窗口外」的旧对话压缩为简短摘要，与最近 N 轮一并喂给大模型，避免早期信息丢失。
仅在使用 DeepSeek 的 LLM 下做单次摘要调用，不调用 OpenAI。
"""
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from src.llm import get_deepseek_llm

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """请将以下对话压缩为一段简短摘要（2～5 句话），保留用户主要诉求、已讨论的关键信息与结论，供后续对话作上下文。只输出摘要正文，不要加「摘要：」等前缀。"""),
    ("human", "{dialog_text}"),
])


def _messages_to_dialog_text(messages: list) -> str:
    """把消息列表转成纯文本对话，便于摘要模型阅读。"""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"用户：{(m.content or '').strip()}")
        elif isinstance(m, AIMessage):
            lines.append(f"助手：{(m.content or '').strip()}")
    return "\n".join(lines) if lines else ""


async def summarize_old_messages_async(old_messages: list) -> str:
    """
    对窗口外的旧消息做一次 LLM 摘要。
    old_messages: 窗口之前的消息列表（HumanMessage / AIMessage）。
    返回摘要字符串；若为空或失败则返回空字符串（调用方可不带摘要继续）。
    """
    if not old_messages:
        return ""
    if not getattr(get_settings(), "llm_context_summarize_old", True):
        return ""
    dialog = _messages_to_dialog_text(old_messages)
    if not dialog.strip():
        return ""
    try:
        llm = get_deepseek_llm(temperature=0.2)
        chain = _SUMMARY_PROMPT | llm
        out = await chain.ainvoke({"dialog_text": dialog[:8000]})  # 限制输入长度
        return (out.content or "").strip() or ""
    except Exception:
        return ""
