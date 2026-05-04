# src/kb/kb_scope.py
"""
知识库分仓（总署 / 隶属海关 / 用户）领域模型与 Milvus 过滤表达式构建。

权限矩阵（服务端强制，不信赖客户端单独传来的 owner 字段）：
- national：全国人民可见的政策类知识；入库通常仅运维/同步任务，API 默认禁止普通上传。
- org：隶属海关业务知识；检索时仅当前请求所属 org_id 可命中 org 仓（expr 约束）。
- user：用户自有上传文档；检索时仅 owner_user_id == 当前登录用户 可命中 user 仓。

legacy：历史 chunk 无 kb_tier 或为空串时，视为「全国可检索」旧数据，避免升级后零结果。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config import get_settings


def _escape_milvus_str(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class RagSearchContext:
    """RAG 向量检索可见范围（由服务端根据登录态与会话构造，禁止前端直传 expr）。"""

    org_id: str = ""
    user_id: str = ""
    include_national: bool = True
    include_org: bool = True
    include_user: bool = True
    # True：检索全部隶属关 org 仓（不受 org_id 约束）；仅应由服务端在 verified 总署身份下置 True
    include_all_orgs: bool = False

    def cache_key_suffix(self) -> str:
        return (
            f"|o:{self.org_id}|u:{self.user_id}|N:{int(self.include_national)}"
            f"|O:{int(self.include_org)}|AO:{int(self.include_all_orgs)}|U:{int(self.include_user)}"
        )


def validate_upload_kb_fields(
    kb_tier: str,
    *,
    org_id: str = "",
    owner_user_id: str = "",
    request_user_id: Optional[str] = None,
    request_org_id: Optional[str] = None,
) -> tuple[str, str, str, str]:
    """
    校验并归一化入库字段。返回 (kb_tier, org_id, owner_user_id, doc_source_tag)。
    失败时抛 ValueError。
    """
    s = get_settings()
    t = (kb_tier or "user").strip().lower()
    if t not in ("national", "org", "user"):
        raise ValueError(f"invalid kb_tier: {kb_tier}")
    allowed = list(getattr(s, "doc_upload_allowed_kb_tiers", None) or ["user", "org"])
    if t not in allowed:
        raise ValueError(f"kb_tier '{t}' not allowed for this deployment")
    if t == "national" and not getattr(s, "doc_upload_allow_national_tier", False):
        raise ValueError("national tier upload is disabled; enable doc_upload_allow_national_tier")
    oid = (org_id or "").strip()
    oid_hdr = (request_org_id or "").strip()
    uid = (owner_user_id or "").strip()
    req_uid = (request_user_id or "").strip()
    if t == "org":
        if not oid:
            raise ValueError("org tier requires org_id")
        if oid_hdr and oid != oid_hdr:
            raise ValueError("org_id does not match X-Org-Id")
    if t == "user":
        if not req_uid:
            raise ValueError("user tier requires authenticated user_id")
        if uid and uid != req_uid:
            raise ValueError("owner_user_id must match current user")
        uid = req_uid
    if t == "national":
        oid = ""
        uid = ""
    doc_source = "api_upload"
    return t, oid, uid, doc_source


def build_rag_search_context_from_inputs(
    *,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
    include_national: Optional[bool] = None,
    include_org: Optional[bool] = None,
    include_user: Optional[bool] = None,
    include_all_orgs: Optional[bool] = None,
) -> RagSearchContext:
    s = get_settings()
    ia = False
    if include_all_orgs is not None:
        ia = bool(include_all_orgs)
    return RagSearchContext(
        org_id=(org_id or "").strip(),
        user_id=(user_id or "").strip(),
        include_national=bool(include_national) if include_national is not None else bool(getattr(s, "rag_search_include_national", True)),
        include_org=bool(include_org) if include_org is not None else bool(getattr(s, "rag_search_include_org", True)),
        include_user=bool(include_user) if include_user is not None else bool(getattr(s, "rag_search_include_user", True)),
        include_all_orgs=ia,
    )


def build_milvus_rag_expr(ctx: RagSearchContext) -> Optional[str]:
    """
    构造 Milvus search 的布尔表达式；None 表示不过滤（兼容未迁移 collection 或显式关闭）。
    """
    s = get_settings()
    if not getattr(s, "rag_kb_scope_filter_enabled", False):
        return None
    parts: list[str] = []
    if ctx.include_national:
        parts.append('(kb_tier == "national")')
        if getattr(s, "rag_kb_legacy_empty_tier_visible", True):
            parts.append('(kb_tier == "")')
    if ctx.include_org:
        if ctx.include_all_orgs:
            parts.append('(kb_tier == "org")')
        elif ctx.org_id:
            esc = _escape_milvus_str(ctx.org_id)
            parts.append(f'((kb_tier == "org") and (org_id == "{esc}"))')
    if ctx.include_user and ctx.user_id:
        esc_u = _escape_milvus_str(ctx.user_id)
        parts.append(f'((kb_tier == "user") and (owner_user_id == "{esc_u}"))')
    if not parts:
        return '(kb_tier == "national")'  # 保守：无任何可见仓时不应全库泄露
    return "(" + " or ".join(parts) + ")"


def rag_context_from_agent_state_slice(state: Any) -> RagSearchContext:
    """从 AgentState mapping 取可选 RAG 分仓字段（LangGraph state）。"""
    if not isinstance(state, dict):
        return build_rag_search_context_from_inputs()
    raw_ao = state.get("rag_include_all_orgs")
    return build_rag_search_context_from_inputs(
        org_id=state.get("rag_org_id"),
        user_id=state.get("rag_user_id"),
        include_national=state.get("rag_include_national"),
        include_org=state.get("rag_include_org"),
        include_user=state.get("rag_include_user"),
        include_all_orgs=bool(raw_ao) if raw_ao is not None else None,
    )
