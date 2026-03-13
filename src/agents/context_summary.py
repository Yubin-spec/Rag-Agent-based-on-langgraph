# src/agents/context_summary.py
"""
将「窗口外」的旧对话压缩为简短摘要，与最近 N 轮一并喂给大模型，避免早期信息丢失。
并对送入 LLM 的每条消息做字符上限截断，节省上下文与 token。
仅在使用 DeepSeek 的 LLM 下做单次摘要调用，不调用 OpenAI。
"""
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from src.llm import get_deepseek_llm

_TRUNCATE_SUFFIX = "…"


def truncate_messages_for_context(
    messages: list,
    max_chars_old: int = 0,
    max_chars_latest: int = 0,
) -> list:
    """
    对送入 LLM 的消息列表按条做字符截断，节省上下文。
    - 最后一条视为「当前轮用户输入」，适用 max_chars_latest（0 表示不截断）。
    - 其余每条适用 max_chars_old（0 表示不截断）。
    - SystemMessage 不截断。
    返回新列表，不修改原消息对象。
    """
    if not messages or (max_chars_old <= 0 and max_chars_latest <= 0):
        return list(messages)
    out = []
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            out.append(m)
            continue
        content = (getattr(m, "content", None) or "") or ""
        if isinstance(m, HumanMessage):
            is_latest = i == len(messages) - 1
            limit = max_chars_latest if is_latest else max_chars_old
        elif isinstance(m, AIMessage):
            is_latest = False
            limit = max_chars_old
        else:
            out.append(m)
            continue
        if limit <= 0 or len(content) <= limit:
            out.append(m)
            continue
        truncated = content[:limit].rstrip() + _TRUNCATE_SUFFIX
        out.append(type(m)(content=truncated))
    return out

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
    max_input = max(0, getattr(get_settings(), "llm_context_summary_input_max_chars", 8000))
    if max_input > 0 and len(dialog) > max_input:
        dialog = dialog[:max_input].rstrip() + "…"
    try:
        llm = get_deepseek_llm(temperature=0.2)
        chain = _SUMMARY_PROMPT | llm
        out = await chain.ainvoke({"dialog_text": dialog})
        return (out.content or "").strip() or ""
    except Exception:
        return ""
