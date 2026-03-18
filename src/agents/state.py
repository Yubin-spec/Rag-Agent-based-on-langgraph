# src/agents/state.py
"""
多智能体图的状态定义。
用于 LangGraph：消息历史由 add_messages 做追加合并，next 由总控写入并驱动条件边。
pending_sql 用于删除/修改类 SQL 人工确认：知识库节点可写入，API 层取出后存到会话待确认表，确认后再执行。
"""
from typing import Annotated, Literal, NotRequired, Optional, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# 总控/子节点路由下一跳：chat（闲聊）、knowledge（知识库子图/子智能体）、human（人工介入）、__end__（结束）
NextAction = Literal["chat", "knowledge", "human", "__end__"]


class AgentState(TypedDict):
    """
    图状态：
    - messages: 对话消息列表，使用 add_messages 做增量合并
    - next: 总控/子节点写入的下一节点，供 conditional_edges 路由
    - pending_sql: 待人工确认执行的 SQL（仅当知识库返回删除/修改类 SQL 时写入，确认后由 API 执行并清空）
    - human_message: 转人工时由上游节点写入的提示文案，人工介入节点会将其作为回复发出
    - route: 实际执行的子链路（chat / knowledge / cache / text2sql_confirm 等），供 API 侧监控使用
    - qa_trace: 知识库链路的质量追踪信息，供问答效果监控与用户反馈分析使用
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next: NextAction
    pending_sql: NotRequired[Optional[str]]
    human_message: NotRequired[Optional[str]]
    route: NotRequired[Optional[str]]
    qa_trace: NotRequired[dict]
    # knowledge 子链路的中间标志（用于拆分节点间通信）
    qa_hit: NotRequired[bool]
    text2sql_hit: NotRequired[bool]
    text2sql_candidate: NotRequired[bool]
    kb_clarify: NotRequired[bool]
    kb_route_reason: NotRequired[str]
    kb_route_confidence: NotRequired[float]


def next_action_from_str(s: str) -> NextAction:
    """
    将 LLM 或外部传入的字符串规范为合法的 NextAction。
    非法或空时默认返回 "chat"，保证图不会走到未定义节点。
    """
    if s in ("chat", "knowledge", "human", "__end__"):
        return s
    return "chat"
