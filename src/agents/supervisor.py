# src/agents/supervisor.py
"""
总控（Supervisor）节点：根据用户最后一句话路由到「闲聊」或「知识库」子智能体。
采用混合规则：规则预判（明显闲聊/明显知识库）→ 歧义时再调用 DeepSeek 推理。仅使用 DeepSeek，不调用 OpenAI。
"""
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from config import get_settings
from src.llm import get_deepseek_llm
from .state import AgentState, NextAction
from .context_summary import summarize_old_messages_async

_settings = get_settings()
# 仅使用 DeepSeek，禁止 OpenAI
_llm = get_deepseek_llm(temperature=0)

# ---------- 混合规则：规则命中则直接路由，否则交 LLM ----------
# 命中任一词即判为知识库（海关/政策/数据等业务相关）
KNOWLEDGE_KEYWORDS = (
    "海关", "申报", "AEO", "政策", "材料", "查询", "数据", "进出口", "认证", "口岸",
    "关税", "通关", "企业", "备案", "审批", "流程", "规定", "办法", "条例", "资质",
)
# 精确匹配则判为闲聊（打招呼/感谢/引导）
CHAT_PHRASES = frozenset([
    "你好", "您好", "嗨", "谢谢", "多谢", "感谢", "再见", "拜拜", "在吗", "有什么功能",
    "怎么用", "帮助", "干啥的", "你是谁", "你好呀", "您好呀", "谢谢啊", "在", "好", "嗯",
])


def _rule_based_intent(last: str) -> Optional[str]:
    """
    规则预判：明显闲聊或明显知识库则直接返回路由，歧义返回 None 交 LLM。
    """
    if not last or not last.strip():
        return "chat"
    text = last.strip()
    # 1) 含业务词 → knowledge
    for kw in KNOWLEDGE_KEYWORDS:
        if kw in text:
            return "knowledge"
    # 2) 纯闲聊短语（精确匹配）→ chat
    if text in CHAT_PHRASES:
        return "chat"
    # 3) 歧义，交给 LLM
    return None


# 总控仅需输出一个路由词，故 temperature=0
_SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是总控路由助手。根据用户最后一句话，决定交给哪个子智能体处理，只回复一个词。

可选路由：
- chat：闲聊、打招呼、无关业务、引导类问题（如“你好”“有什么功能”“怎么用”）
- knowledge：与海关/进出口/政策/数据/知识库相关的问题（如申报材料、AEO认证、数据查询、政策解读）

只回复: chat 或 knowledge"""),
    ("placeholder", "{messages}"),
])

def _get_last_user_text(state: AgentState) -> str:
    """从 state.messages 中取最后一条 HumanMessage 的 content，用于总控路由判断。"""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and hasattr(m, "content"):
            return (m.content or "").strip()
    return ""


def _messages_for_llm(state: AgentState) -> list:
    """取最近 llm_context_window_turns 轮消息（每轮 user+assistant 两条），节省 token；窗口内不压缩。"""
    from langchain_core.messages import HumanMessage, AIMessage
    max_turns = _settings.llm_context_window_turns
    messages = state.get("messages") or []
    if len(messages) <= max_turns * 2:
        return messages
    return list(messages[-max_turns * 2 :])


async def _messages_for_llm_with_summary(state: AgentState) -> list:
    """
    若启用 llm_context_summarize_old 且存在窗口外旧消息，则先对旧消息做摘要，
    再返回【历史对话摘要】+ 最近 N 轮，避免早期信息丢失。
    """
    from langchain_core.messages import HumanMessage, AIMessage
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


def supervisor_node(state: AgentState) -> dict:
    """
    总控节点（同步）：混合规则 + LLM。规则能判则直接路由，歧义时用 DeepSeek 判断。
    """
    last = _get_last_user_text(state)
    if not last:
        return {"next": "chat"}
    intent = _rule_based_intent(last)
    if intent is not None:
        return {"next": intent}

    chain = _SUPERVISOR_PROMPT | _llm
    response = chain.invoke({"messages": _messages_for_llm(state)})
    out = (response.content or "").strip().lower()
    if "knowledge" in out:
        next_action: NextAction = "knowledge"
    else:
        next_action = "chat"
    return {"next": next_action}


def _is_retryable_error(e: Exception) -> bool:
    """是否可重试（超时、连接错误等）。"""
    msg = (getattr(e, "message", "") or str(e)).lower()
    return "timeout" in msg or "connection" in msg or "5" in str(getattr(e, "status_code", ""))


async def supervisor_node_async(state: AgentState) -> dict:
    """
    总控节点（异步）：混合规则 + LLM。规则能判则直接路由不调模型，歧义时再调 DeepSeek。
    异常时返回需人工提示并结束本轮；可重试错误会重试若干次。
    """
    last = _get_last_user_text(state)
    if not last:
        return {"next": "chat"}
    intent = _rule_based_intent(last)
    if intent is not None:
        return {"next": intent}

    chain = _SUPERVISOR_PROMPT | _llm
    max_retries = max(0, getattr(_settings, "agent_llm_retry_times", 2))
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            messages_to_send = await _messages_for_llm_with_summary(state)
            response = await chain.ainvoke({"messages": messages_to_send})
            out = (response.content or "").strip().lower()
            if "knowledge" in out:
                next_action: NextAction = "knowledge"
            else:
                next_action = "chat"
            return {"next": next_action}
        except Exception as e:
            last_error = e
            if attempt < max_retries and _is_retryable_error(e):
                continue
            break
    # 重试耗尽后转人工，避免直接抛错导致 500
    reply = getattr(_settings, "agent_need_human_reply", "当前服务暂时异常，请稍后重试或转人工客服。")
    return {"next": "human", "human_message": reply}


def route_to_agent(state: AgentState) -> NextAction:
    """供 LangGraph conditional_edges 使用：根据 state.next 返回下一节点名。"""
    return state.get("next") or "chat"
