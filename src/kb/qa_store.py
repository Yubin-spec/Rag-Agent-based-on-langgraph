# src/kb/qa_store.py —— 本文件在包内的路径标识，便于日志与错误栈定位

"""
高频 QA 匹配：
- local: 本地 JSON（exact/alias + 轻量语义）
- milvus: FAQ 独立 collection（exact/alias + 向量召回 + guardrail）
- hybrid: local 优先，未命中再走 milvus
"""

import json  # 解析 high_freq_qa.json 等 JSON 数据源
import logging  # 模块级日志（本文件暂少直接用 logger.debug，但保留扩展点）
from pathlib import Path  # 跨平台路径对象，便于 exists/open
from typing import Any, Optional  # 类型注解：任意结构与可选返回值

from config import get_settings  # 读取 pydantic Settings（后端、阈值、Milvus 名等）
from .retrieval_eval import evaluate_retrieval  # 无 LLM 的文本匹配/覆盖率/无关比例等综合打分
from .embedding_loader import get_bge_embedding  # 懒加载 BGE 向量模型（仅 Milvus 后端需要）
from src.db_resilience import get_milvus_collection, milvus_operation_with_retry  # Milvus 连接与失败重试封装

logger = logging.getLogger(__name__)  # 以模块名命名的日志器，子 logger 继承根配置


class QAStore:
    """
    从 JSON 文件加载高频问答对（如 12360 热线清洗数据），
    支持精确匹配与简单包含匹配，命中则直接返回答案。
    """

    def __init__(self, path: Optional[str] = None):
        settings = get_settings()  # 取全局单例配置
        self._path = path or settings.qa_data_path  # FAQ 数据源路径；可显式覆盖默认 data/high_freq_qa.json
        # 后端模式字符串规范化：去空白、小写，非法值后面会回退 local
        self._backend = str(getattr(settings, "qa_store_backend", "local") or "local").strip().lower()
        if self._backend not in ("local", "milvus", "hybrid"):  # 三重合法取值
            self._backend = "local"  # 未知配置时保守回退，避免运行时炸掉
        self._use_local = self._backend in ("local", "hybrid")  # 是否读本地 JSON 并建内存索引
        self._use_milvus = self._backend in ("milvus", "hybrid")  # 是否走 Milvus exact/语义
        self._qa: list[dict] = []  # 原始 QA 条列表（来自 JSON）
        self._exact_map: dict[str, dict[str, Any]] = {}  # 归一化问句/别名 -> 命中载荷（O(1) 精确）
        self._ngram_index: dict[str, list[int]] = {}  # ngram -> 候选文本下标列表（倒排预筛）
        self._cand_texts: list[dict[str, Any]] = []  # 每条「可向量/可精排」的候选条目（含 norm 文本）
        self._match_cache: dict[str, tuple[Optional[str], dict[str, Any]]] = {}  # 归一化 query -> (答案, meta)
        self._match_cache_order: list[str] = []  # 与上式配合实现简单 LRU 顺序
        self._embed = get_bge_embedding() if self._use_milvus else None  # Milvus 语义路径需要编码器；纯 local 不加载省内存
        if self._use_local:  # 仅 local/hybrid 需要读盘建索引
            self._load()  # 填充 self._qa
            self._build_index()  # 建 exact_map + ngram 倒排 + cand_texts

    def _milvus_collection(self):
        s = get_settings()  # 连接参数以当前配置为准
        return get_milvus_collection(
            s.milvus_uri,  # Milvus 服务地址
            s.qa_milvus_collection,  # FAQ 专用 collection 名（与 kb_chunks 隔离）
            create_if_missing=False,  # 运行期不自动建表，避免误创建空 schema
        )

    def _load(self) -> None:
        """从 qa_data_path 读取 JSON，支持 list 或 { \"qa_pairs\": [...] } 格式。"""
        p = Path(self._path)  # 将字符串路径转为 Path
        if not p.exists():  # 文件不存在则静默跳过（匹配时视为无本地库）
            return
        try:  # 解析失败不向外抛，避免整个应用起不来
            with open(p, "r", encoding="utf-8") as f:  # UTF-8 读 FAQ JSON
                data = json.load(f)  # 一次性反序列化
            if isinstance(data, list):  # 直接是数组格式
                self._qa = data  # 原样保存
            elif isinstance(data, dict) and "qa_pairs" in data:  # 带 qa_pairs 键的对象格式
                self._qa = data["qa_pairs"]  # 取出内层列表
            else:  # 其他结构不识别
                self._qa = []  # 置空
        except Exception:  # 含 JSONDecodeError 等
            self._qa = []  # 坏文件等价于无数据

    def _normalize(self, s: str) -> str:
        """去空格、小写，用于问句匹配。"""
        return "".join(s.split()).lower().strip()  # split 再 join 去掉任意空白；统一小写；去首尾空白（再保险）

    def _ngrams(self, norm: str, n: int) -> list[str]:
        if not norm:  # 空串无 ngram
            return []
        if n <= 1:  # 退化为字符级（极少用）
            return list(norm)  # 每个字单独成「gram」
        if len(norm) <= n:  # 长度不够切 n-gram
            return [norm]  # 整串作为一个 gram
        return [norm[i : i + n] for i in range(0, len(norm) - n + 1)]  # 滑动窗口切分

    def _iter_candidates(self):
        """统一抽取候选：question/answer/aliases。"""
        for item in self._qa or []:  # _qa 为 None 时等价空列表
            q = (item.get("question") or item.get("q") or "").strip()  # 支持 q 简写字段
            a = (item.get("answer") or item.get("a") or "").strip()  # 支持 a 简写字段
            aliases = item.get("aliases") or item.get("alias") or []  # 别名单字段或列表
            if isinstance(aliases, str):  # 单字符串别名
                aliases = [aliases]  # 包装成列表统一处理
            if not isinstance(aliases, list):  # 既不是 str 也不是 list
                aliases = []  # 丢弃异常类型
            aliases = [str(x).strip() for x in aliases if str(x).strip()]  # 转 str、去空
            if q and a:  # 问与答都必须非空才有意义
                yield {"question": q, "answer": a, "aliases": aliases, "raw": item}  # 统一结构 + 保留 raw 备查

    def _build_index(self) -> None:
        """
        构建：
        - exact/alias 映射（O(1)）
        - ngram 倒排（预筛候选，避免语义匹配全库扫描）
        """
        settings = get_settings()  # 读 ngram 大小等
        n = max(1, int(getattr(settings, "qa_semantic_ngram_size", 2) or 2))  # 至少为 1；中文常用 -bigram
        self._exact_map = {}  # 清空旧映射（若重复调用 _build_index）
        self._ngram_index = {}  # 清空倒排
        self._cand_texts = []  # 清空候选列表

        def add_exact(key: str, payload: dict[str, Any]):  # 闭包：写入 exact_map，先入为主不覆盖
            if key and key not in self._exact_map:  # 空 key 不写；已有键保留第一条
                self._exact_map[key] = payload  # 存 answer + match 元信息

        for cand in self._iter_candidates():  # 遍历每条 QA
            q_norm = self._normalize(cand["question"])  # 标准问法归一化键
            add_exact(
                q_norm,  # exact 键
                {
                    "answer": cand["answer"],  # 对应答案全文
                    "match_type": "exact",  # 命中类型标记
                    "matched_question": cand["question"],  # 原始标准问（展示/trace）
                    "matched_alias": "",  # exact 无 alias
                    "score": 1.0,  # 精确匹配满分
                    "query_coverage": 1.0,  # 覆盖率视为满
                },
            )
            for al in cand["aliases"]:  # 每条别名同样进 exact_map
                al_norm = self._normalize(al)  # 别名归一化键
                add_exact(
                    al_norm,  # alias 键
                    {
                        "answer": cand["answer"],  # 与标准问共享答案
                        "match_type": "alias",  # 类型为别名命中
                        "matched_question": cand["question"],  # 回指标准问
                        "matched_alias": al,  # 记录实际命中的别名字符串
                        "score": 1.0,  # 仍视为强精确
                        "query_coverage": 1.0,
                    },
                )
            # 预筛候选的“文本粒度”条目：标准问法+aliases 都作为独立候选文本
            texts = [(cand["question"], ""), *[(al, al) for al in cand["aliases"]]]  # (展示文本, alias_marker)
            for text, alias_marker in texts:  # 每个文本一条候选
                t_norm = self._normalize(text)  # 用于 ngram 与字符集
                if not t_norm:  # 全空白则跳过
                    continue
                idx = len(self._cand_texts)  # 当前将要 append 的下标
                self._cand_texts.append(
                    {
                        "answer": cand["answer"],  # 该候选对应的答案
                        "matched_question": cand["question"],  # 归属的标准问
                        "matched_alias": alias_marker,  # 若来自 alias 则非空
                        "text": text,  # 精排时给 evaluate_retrieval 的原文
                        "norm": t_norm,  # 归一化串，用于 ngram 与字符交集
                    }
                )
                for g in set(self._ngrams(t_norm, n)):  # 对去重后的每个 ngram
                    self._ngram_index.setdefault(g, []).append(idx)  # 倒排：gram -> 候选下标列表

    def _cache_get(self, q_norm: str) -> Optional[tuple[Optional[str], dict[str, Any]]]:
        val = self._match_cache.get(q_norm)  # 按归一化 query 查缓存
        if val is not None:  # 仅当真实命中缓存时才更新 LRU（避免 None 占位污染顺序）
            # 命中即刷新顺序，保持真实 LRU 行为
            try:
                self._match_cache_order.remove(q_norm)  # 从队列头侧摘掉旧位置
            except ValueError:  # 防御：顺序表与 dict 偶发不同步
                pass
            self._match_cache_order.append(q_norm)  # 挪到队尾表示最近使用
        return val  # 可能为 (None, meta) 表示曾缓存过「无命中」

    def _cache_put(self, q_norm: str, value: tuple[Optional[str], dict[str, Any]]) -> None:
        settings = get_settings()
        cap = max(0, int(getattr(settings, "qa_match_cache_max_entries", 0) or 0))  # 0 表示关闭缓存写入
        if cap <= 0 or not q_norm:  # 无容量或空键不写
            return
        if q_norm in self._match_cache:  # 已存在则只更新值、不动 LRU 顺序（简化实现）
            self._match_cache[q_norm] = value  # 覆盖
            return
        self._match_cache[q_norm] = value  # 新键插入
        self._match_cache_order.append(q_norm)  # 记录插入顺序
        if len(self._match_cache_order) > cap:  # 超出容量淘汰最久未使用
            old = self._match_cache_order.pop(0)  # 队头为最旧
            self._match_cache.pop(old, None)  # 安全删除，无键则忽略

    @staticmethod
    def _escape_expr(s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace('"', '\\"')  # Milvus 表达式里反斜杠与双引号需转义

    def _milvus_exact(self, q_norm: str) -> Optional[tuple[Optional[str], dict[str, Any]]]:
        s = get_settings()
        expr = f'question_norm == "{self._escape_expr(q_norm)}"'  # 标量过滤：归一化问句等值

        def _do_query(coll):
            rows = coll.query(
                expr=expr,  # 过滤条件
                output_fields=["answer", "matched_question", "matched_alias", "match_type"],  # 需要回传的字段
                limit=1,  # exact 最多一条
            )
            if not rows:  # Milvus 无命中
                return None
            row = rows[0]  # 取首条
            return (
                row.get("answer") or "",  # 答案字符串；缺失时用空串
                {
                    "match_type": row.get("match_type") or "exact",  # 兜底为 exact
                    "matched_question": row.get("matched_question") or "",
                    "matched_alias": row.get("matched_alias") or "",
                    "score": 1.0,  # 与本地 exact 一致
                    "query_coverage": 1.0,
                },
            )

        return milvus_operation_with_retry(
            s.milvus_uri,  # 用于日志/重连
            s.qa_milvus_collection,  # collection 名
            _do_query,  # 实际查询 lambda
            retries=1,  # 少量重试抗抖动
            critical=False,  # 失败不当作致命错误
            default=None,  # 失败返回 None，上层继续 local/语义
        )

    def _milvus_semantic(self, question: str, q_norm: str) -> Optional[tuple[Optional[str], dict[str, Any]]]:
        settings = get_settings()
        if self._embed is None:  # 未初始化编码器（不应发生若 _use_milvus）
            return None
        qv = self._embed.encode([question]).tolist()  # BGE 批量编码；这里 batch=1
        if not qv:  # 编码异常或空
            return None

        limit = max(1, int(getattr(settings, "qa_milvus_semantic_top_k", 80) or 80))  # ANN 召回条数上限
        nprobe = max(1, int(getattr(settings, "qa_milvus_nprobe", 64) or 64))  # 探测聚类数，越大越准越慢
        search_params = {"metric_type": "IP", "params": {"nprobe": nprobe}}  # 内积与归一化向量配合常用 cosine 等价

        def _do_search(coll):
            results = coll.search(
                data=qv,  # 查询向量列表
                anns_field="embedding",  # Milvus 里向量字段名（与入库 schema 一致）
                param=search_params,  # 检索参数（拼写为 param 非 params， pymilvus API）
                limit=limit,  # topK
                output_fields=["candidate_text", "answer", "matched_question", "matched_alias"],  # 回传标量
            )
            out = []  # 扁平化 hits
            for hits in results:  # 每个查询向量一组 hits（此处仅 1 个 query）
                for h in hits:  # 单条命中
                    out.append(
                        {
                            "candidate_text": h.entity.get("candidate_text") or "",  # 用于精排的问法文本
                            "answer": h.entity.get("answer") or "",
                            "matched_question": h.entity.get("matched_question") or "",
                            "matched_alias": h.entity.get("matched_alias") or "",
                            "vector_score": float(h.score),  # 原始向量相似度（后续门控主要用 evaluate_retrieval）
                        }
                    )
            return out

        cands = milvus_operation_with_retry(
            settings.milvus_uri,
            settings.qa_milvus_collection,
            _do_search,
            retries=1,
            critical=False,
            default=[],  # 失败当作无候选
        )
        if not cands:  # 空的候选列表
            return None

        min_score = float(getattr(settings, "qa_semantic_min_score", 0.72))  # Milvus 分支默认与 settings 一致或可单独覆盖读代码内默认值
        min_cov = float(getattr(settings, "qa_semantic_min_query_coverage", 0.6))  # 问句词在候选文本中的覆盖率下限
        top_k = max(1, int(getattr(settings, "qa_semantic_top_k", 20) or 20))  # 精排后参与 margin 的头部条数
        min_margin = float(getattr(settings, "qa_semantic_min_margin", 0.08) or 0.0)  # top1-top2 分差；0 关闭
        n = max(1, int(getattr(settings, "qa_semantic_ngram_size", 2) or 2))  # ngram 重叠统计窗口
        max_irrel = float(getattr(settings, "qa_semantic_max_irrelevant_ratio", 0.6) or 0.6)  # 无关文本占比上限
        min_overlap_cnt = int(getattr(settings, "qa_semantic_min_ngram_overlap_count", 4) or 4)  # 字符 ngram 交集个数下限
        q_grams_set = set(self._ngrams(q_norm, n))  # 用户问句的 ngram 集合，用于与候选算 overlap

        scored: list[dict[str, Any]] = []  # 每条候选的最终门控特征
        for cand in cands:  # 遍历向量召回
            text = cand["candidate_text"]  # 与入库时写入的一致
            if not text:  # 无文本无法精排
                continue
            ev = evaluate_retrieval(question, text)  # 计算 match_score/coverage/irrel/normalized_score
            cand_grams_set = set(self._ngrams(self._normalize(text), n))  # 候选文本 ngram 集合
            overlap_cnt = len(q_grams_set & cand_grams_set)  # 交集中 ngram 个数（越大越像同主题）
            scored.append(
                {
                    "answer": cand["answer"],  # 若通过门控则返此答案
                    "matched_question": cand["matched_question"],
                    "matched_alias": cand["matched_alias"],
                    "score": float(ev.normalized_score),  # 主排序分（已含 irrelevant 惩罚）
                    "query_coverage": float(ev.query_coverage),
                    "irrelevant_ratio": float(ev.irrelevant_ratio),
                    "ngram_overlap_cnt": int(overlap_cnt),
                }
            )
        if not scored:  # 全部被跳过
            return None
        scored.sort(key=lambda x: x["score"], reverse=True)  # 按综合分降序
        top = scored[:top_k]  # 只保留前 top_k 算 margin
        best = top[0]  # 第一名
        second = top[1] if len(top) > 1 else None  # 第二名可能不存在
        margin = float(best["score"]) - float(second["score"]) if second else float("inf")  # 无第二名则视为极大 margin
        if (
            float(best["score"]) >= min_score  # 综合分达标
            and float(best["query_coverage"]) >= min_cov  # 问句被候选覆盖够多
            and float(best.get("irrelevant_ratio", 1.0)) <= max_irrel  # 候选里「无关」比例不能太高
            and int(best.get("ngram_overlap_cnt", 0)) >= min_overlap_cnt  # ngram 硬重叠防向量飘移
            and (min_margin <= 0 or margin >= min_margin)  # 关闭 margin 或分差够大
        ):
            return (
                best["answer"],
                {
                    "match_type": "semantic",  # 表示非 exact 的语义门控命中
                    "matched_question": best["matched_question"],
                    "matched_alias": best["matched_alias"],
                    "score": float(best["score"]),
                    "query_coverage": float(best["query_coverage"]),
                    "top2_score": float(second["score"]) if second else 0.0,  # 方便 trace 与调参
                    "margin": float(margin) if margin != float("inf") else 0.0,  # inf 时记 0 避免 JSON 难序列化
                    "recall_top_k": int(top_k),
                    "ngram_size": int(n),
                    "irrelevant_ratio": float(best.get("irrelevant_ratio", 0.0)),
                    "ngram_overlap_cnt": int(best.get("ngram_overlap_cnt", 0)),
                },
            )
        return None  # 未过门控则视为未命中，上层可走 RAG/其他

    def match(self, question: str) -> tuple[Optional[str], dict[str, Any]]:
        """
        返回 (answer, meta)。
        meta: {match_type, matched_question, matched_alias, score, query_coverage}
        match_type: exact | alias | semantic | contains | none
        """
        settings = get_settings()  # 每轮读取最新配置（支持热更 pydantic 场景有限）
        q = (question or "").strip()  # 原始问句去首尾空白
        q_norm = self._normalize(q)  # 标准化键
        if not q_norm:  # 空问句无法匹配
            return None, {"match_type": "none"}  # 无答案，类型 none

        # 1) exact / alias（强精确）
        cached = self._cache_get(q_norm)  # 先走缓存避免重复 IO/向量
        if cached is not None:  # 含「曾缓存未命中」的元组
            return cached  # 直接返回

        if self._use_local:  # local/hybrid：查内存 exact_map
            direct = self._exact_map.get(q_norm)  # O(1)
            if direct:  # 命中标准问或别名
                out = (direct["answer"], dict(direct))  # 拷贝 dict 避免外部修改内部结构
                self._cache_put(q_norm, out)  # 写入缓存
                return out
        if self._use_milvus:  # milvus/hybrid：标量 exact
            milvus_exact = self._milvus_exact(q_norm)  # 远端 question_norm 匹配
            if milvus_exact and milvus_exact[0]:  # 答案非空才算命中
                self._cache_put(q_norm, milvus_exact)
                return milvus_exact

        # 2) semantic（轻量相似度，兜底召回；阈值控制 precision）
        if bool(getattr(settings, "qa_enable_semantic_match", True)):  # 总开关
            min_query_chars = int(getattr(settings, "qa_semantic_min_query_chars", 0) or 0)  # 过短问句只做 exact
            if min_query_chars > 0 and len(q_norm) < min_query_chars:  # 短 query 高歧义
                out = (None, {"match_type": "none"})  # 明确不语义
                self._cache_put(q_norm, out)  # 缓存否定结果，防反复算
                return out
            min_score = float(getattr(settings, "qa_semantic_min_score", 0.72))  # local 精排默认阈值
            min_cov = float(getattr(settings, "qa_semantic_min_query_coverage", 0.6))
            require_overlap = bool(getattr(settings, "qa_semantic_require_any_overlap", True))  # 是否先要求字符交集
            top_k = max(1, int(getattr(settings, "qa_semantic_top_k", 30) or 30))  # local 默认比 milvus 分支 top 略大
            min_margin = float(getattr(settings, "qa_semantic_min_margin", 0.05) or 0.0)
            n = max(1, int(getattr(settings, "qa_semantic_ngram_size", 2) or 2))
            pre_n = max(1, int(getattr(settings, "qa_semantic_prefilter_top_n", 80) or 80))  # 进入精排的最大候选数
            max_irrel = float(getattr(settings, "qa_semantic_max_irrelevant_ratio", 0.75) or 0.75)  # local 默认略宽
            min_overlap_cnt = int(getattr(settings, "qa_semantic_min_ngram_overlap_count", 3) or 3)

            if self._use_local:  # 先尝试 JSON+内存语义
                # 2.1 倒排预筛：用 ngram 重叠统计快速缩小候选集合（避免全库扫描）
                q_grams = self._ngrams(q_norm, n)  # 用户问 ngram 序列
                cand_count: dict[int, int] = {}  # 候选下标 -> 重叠计数（越多越相关）
                for g in set(q_grams):  # 对每个唯一 ngram
                    for idx in self._ngram_index.get(g, []):  # 倒排链上的候选
                        cand_count[idx] = cand_count.get(idx, 0) + 1  # 累计命中次数
                idxs = []  # 精排下标列表
                if cand_count:  # 至少有一个候选被击中
                    pre_idxs = sorted(cand_count.items(), key=lambda kv: kv[1], reverse=True)[:pre_n]  # 按重叠数排序截断
                    idxs = [i for i, _ in pre_idxs]  # 只保留下标

                if idxs:  # 有预筛结果才精排
                    # 2.2 精排：只对少量候选计算 evaluate_retrieval 分数
                    scored: list[dict[str, Any]] = []
                    q_chars = set(q_norm) if require_overlap else set()  # 字符级快速过滤
                    q_grams_set = set(self._ngrams(q_norm, n))  # 与 milvus 分支一致的 overlap
                    for idx in idxs:  # 仅对 pre_n 内候选算分
                        cand = self._cand_texts[idx]  # 取条目
                        if require_overlap and not (q_chars & set(cand["norm"])):  # 无任何同字符则丢弃
                            continue
                        ev = evaluate_retrieval(q, cand["text"])  # 用原始展示文本算分（保留空格等）
                        cand_grams_set = set(self._ngrams(cand["norm"], n))  # 候选 norm 的 ngram 集合
                        overlap_cnt = len(q_grams_set & cand_grams_set)  # 与用户问的 ngram 求交
                        scored.append(
                            {
                                "answer": cand["answer"],
                                "matched_question": cand["matched_question"],
                                "matched_alias": cand["matched_alias"],
                                "score": float(ev.normalized_score),
                                "irrelevant_ratio": float(ev.irrelevant_ratio),
                                "query_coverage": float(ev.query_coverage),
                                "ngram_overlap_cnt": int(overlap_cnt),
                            }
                        )
                    if scored:  # 精排后非空
                        scored.sort(key=lambda x: x["score"], reverse=True)
                        top = scored[:top_k]
                        best = top[0]
                        second = top[1] if len(top) > 1 else None
                        margin = float(best["score"]) - float(second["score"]) if second else float("inf")
                        if (
                            float(best["score"]) >= min_score
                            and float(best["query_coverage"]) >= min_cov
                            and float(best.get("irrelevant_ratio", 1.0)) <= max_irrel
                            and int(best.get("ngram_overlap_cnt", 0)) >= min_overlap_cnt
                            and (min_margin <= 0 or margin >= min_margin)
                        ):
                            out = (
                                best["answer"],
                                {
                                    "match_type": "semantic",
                                    "matched_question": best["matched_question"],
                                    "matched_alias": best["matched_alias"],
                                    "score": float(best["score"]),
                                    "query_coverage": float(best["query_coverage"]),
                                    "top2_score": float(second["score"]) if second else 0.0,
                                    "margin": float(margin) if margin != float("inf") else 0.0,
                                    "recall_top_k": int(top_k),
                                    "prefilter_top_n": int(pre_n),
                                    "ngram_size": int(n),
                                    "irrelevant_ratio": float(best.get("irrelevant_ratio", 0.0)),
                                    "ngram_overlap_cnt": int(best.get("ngram_overlap_cnt", 0)),
                                },
                            )
                            self._cache_put(q_norm, out)
                            return out
            if self._use_milvus:  # local 未命中或仅 milvus：走向量召回 + 门控
                milvus_sem = self._milvus_semantic(question=q, q_norm=q_norm)  # 传原始 q 给 embedding 与 eval
                if milvus_sem and milvus_sem[0]:  # 答案非空
                    self._cache_put(q_norm, milvus_sem)
                    return milvus_sem

        # 3) contains（历史兼容：风险较高，默认可开，建议逐步关）
        if bool(getattr(settings, "qa_enable_legacy_contains_match", True)):  # Settings 通常 False；缺省参数仅为防御
            for cand in self._iter_candidates():  # 全量扫描（慢但仅兜底）
                q_cand_norm = self._normalize(cand["question"])  # 标准问归一化
                if q_norm in q_cand_norm or q_cand_norm in q_norm:  # 子串互含：易误判
                    out = (
                        cand["answer"],
                        {
                            "match_type": "contains",
                            "matched_question": cand["question"],
                            "matched_alias": "",
                            "score": 0.0,
                            "query_coverage": 0.0,
                        },
                    )
                    self._cache_put(q_norm, out)
                    return out
        out = (None, {"match_type": "none"})  # 全阶段未命中
        self._cache_put(q_norm, out)  # 缓存未命中，减少重复计算
        return out

    def find(self, question: str) -> Optional[str]:
        """兼容旧接口：仅返回答案。"""
        ans, _ = self.match(question)  # 忽略 meta
        return ans  # 可能为 None
