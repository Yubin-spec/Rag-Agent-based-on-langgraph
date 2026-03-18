"""
kb.intent_router

轻量意图识别（不调用 LLM）：
- 仅用于判断“是否应该优先走 Text2SQL”。
- 设计目标：极低成本 + 可解释 + 可在链路早期短路，避免不必要的 QA/RAG 计算。
"""

from __future__ import annotations


def rule_based_text2sql_candidate(question: str) -> bool | None:
    """
    返回：
    - True：强烈像数据查询/统计/写操作，应走 Text2SQL
    - False：明显不是（闲聊/政策解释等），不走 Text2SQL
    - None：不确定（交给更强路由器或后续链路）
    """
    q = (question or "").strip()
    if not q:
        return False
    # 明显闲聊
    if q in {"你好", "您好", "谢谢", "再见"}:
        return False
    # 写操作强信号：进入确认流
    if any(kw in q for kw in ("删除", "清空", "修改", "更新", "改成")):
        return True
    # 查询/统计强信号
    triggers = (
        "查询",
        "统计",
        "报表",
        "排名",
        "TOP",
        "top",
        "多少",
        "数量",
        "总数",
        "占比",
        "均值",
        "同比",
        "环比",
        "按",
        "分组",
    )
    time_markers = ("年", "月", "日", "季度", "Q1", "Q2", "Q3", "Q4", "202", "201")
    has_digit = any(ch.isdigit() for ch in q)
    if any(t in q for t in triggers):
        return True
    if has_digit and any(t in q for t in time_markers):
        return True
    return None

