"""
高频 QA 阈值标定（offline sweep）。

目标：
- 给“语义兜底匹配（QA semantic）”找到可复现的阈值：
  - qa_semantic_min_score
  - qa_semantic_min_query_coverage
- 输出在不同阈值下的 precision/recall/FPR，以及按业务成本函数选点的推荐值。

数据：
1) 高频 QA 库：Settings.qa_data_path（默认 data/high_freq_qa.json）
   - 支持字段：question/answer/aliases
2) 标注集：--labeled <path>
   JSONL，每行一个样本：
   {"id":"1","query":"...","gold_question":"标准问题文本","label":1}
   - label=1 表示应该命中到 gold_question（或其 aliases）
   - label=0 表示不应该命中到 gold_question（常用于“相似但不同意图”的对抗样本）

输出：
- 控制台表格（可选 --out_csv 导出）
"""

from __future__ import annotations

import sys
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# 确保从项目根导入本地模块，避免与 site-packages 中的同名包冲突
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_settings  # noqa: E402
from src.kb.retrieval_eval import evaluate_retrieval  # noqa: E402


def _normalize(s: str) -> str:
    return "".join((s or "").split()).lower().strip()


def _load_qa_candidates(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "qa_pairs" in data:
        data = data["qa_pairs"]
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or item.get("q") or "").strip()
        a = (item.get("answer") or item.get("a") or "").strip()
        aliases = item.get("aliases") or item.get("alias") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []
        aliases = [str(x).strip() for x in aliases if str(x).strip()]
        if q and a:
            out.append({"question": q, "answer": a, "aliases": aliases})
    return out


def _iter_texts_for_candidate(cand: dict[str, Any]) -> Iterable[str]:
    yield cand["question"]
    for al in cand.get("aliases") or []:
        if al:
            yield al


@dataclass
class BestMatch:
    matched_question: str
    matched_text: str
    score: float
    query_coverage: float


def _best_match(query: str, qa: list[dict[str, Any]]) -> BestMatch | None:
    q = (query or "").strip()
    if not q:
        return None
    best: BestMatch | None = None
    # 轻过滤：完全没字符交集的直接跳过，避免分数噪音（与线上配置一致）
    q_norm = _normalize(q)
    q_chars = set(q_norm)
    for cand in qa:
        for text in _iter_texts_for_candidate(cand):
            t_norm = _normalize(text)
            if not t_norm or not (q_chars & set(t_norm)):
                continue
            ev = evaluate_retrieval(q, text)
            score = float(ev.normalized_score)
            cov = float(ev.query_coverage)
            if best is None or score > best.score:
                best = BestMatch(
                    matched_question=str(cand["question"]),
                    matched_text=str(text),
                    score=score,
                    query_coverage=cov,
                )
    return best


def _load_labeled_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _is_hit(best: BestMatch | None, *, min_score: float, min_cov: float) -> bool:
    return bool(best and best.score >= min_score and best.query_coverage >= min_cov)


def _compute_metrics(
    samples: list[dict[str, Any]],
    qa: list[dict[str, Any]],
    *,
    min_score: float,
    min_cov: float,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    # 这里的“命中”定义：达到阈值 + 且最佳匹配落到 gold_question（或其别名文本）
    # label=1: 应该命中 gold_question；label=0: 不应该命中 gold_question（即使相似也不行）
    # 注意：这是“是否命中到指定 gold_question”的判定，不等价于“命中任意 QA”。
    # 用于阈值校准时更严：防止错命中相似问法。
    # 若你想评估“命中任意 QA 是否合理”，可扩展为多标签或将 gold_question 置空并另写判定。

    # 建 gold_question -> (question/aliases) 映射
    gold_texts: dict[str, set[str]] = {}
    for cand in qa:
        q = str(cand["question"])
        s = {str(q)}
        for al in cand.get("aliases") or []:
            s.add(str(al))
        gold_texts[q] = s

    for s in samples:
        query = str(s.get("query") or "")
        gold = str(s.get("gold_question") or "")
        label = int(s.get("label") or 0)

        best = _best_match(query, qa)
        hit_any = _is_hit(best, min_score=min_score, min_cov=min_cov)
        hit_gold = False
        if hit_any and best is not None and gold:
            allowed = gold_texts.get(gold) or {gold}
            hit_gold = best.matched_text in allowed or best.matched_question == gold

        pred = 1 if hit_gold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "min_score": round(float(min_score), 4),
        "min_cov": round(float(min_cov), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, help="标注集 JSONL 路径")
    ap.add_argument("--qa_path", default="", help="覆盖 Settings.qa_data_path")
    ap.add_argument("--score_grid", default="0.60,0.62,0.64,0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84")
    ap.add_argument("--cov_grid", default="0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--target_precision", type=float, default=0.98, help="希望达到的 precision（用于选推荐阈值）")
    ap.add_argument("--cost_fp", type=float, default=10.0, help="误命中（错用高频 QA）的成本系数")
    ap.add_argument("--cost_fn", type=float, default=1.0, help="漏命中（走 RAG/澄清）的成本系数")
    ap.add_argument("--out_csv", default="", help="可选：导出 CSV")
    args = ap.parse_args()

    settings = get_settings()
    qa_path = args.qa_path.strip() or settings.qa_data_path
    qa = _load_qa_candidates(qa_path)
    if not qa:
        raise SystemExit(f"QA 库为空或不存在: {qa_path}")
    samples = _load_labeled_jsonl(args.labeled)
    if not samples:
        raise SystemExit(f"标注集为空: {args.labeled}")

    score_grid = [float(x) for x in args.score_grid.split(",") if x.strip()]
    cov_grid = [float(x) for x in args.cov_grid.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    for s in score_grid:
        for c in cov_grid:
            m = _compute_metrics(samples, qa, min_score=s, min_cov=c)
            # 简单成本函数：错答成本远高于漏召回
            m["cost"] = round(args.cost_fp * m["fp"] + args.cost_fn * m["fn"], 3)
            rows.append(m)

    # 先按“满足 target_precision 的最低成本”选点；否则选最低成本
    eligible = [r for r in rows if float(r["precision"]) >= float(args.target_precision)]
    pick_pool = eligible or rows
    pick_pool.sort(key=lambda r: (r["cost"], -r["precision"], -r["recall"], r["fpr"]))
    best = pick_pool[0]

    # 输出 topN，便于面试时说“我们扫阈值曲线选点”
    rows_sorted = sorted(rows, key=lambda r: (r["cost"], -r["precision"], -r["recall"]))[:15]
    print("\n== QA 阈值 sweep（top 15 by cost） ==")
    for r in rows_sorted:
        flag = "  <== RECOMMEND" if (r["min_score"], r["min_cov"]) == (best["min_score"], best["min_cov"]) else ""
        print(
            f"score>={r['min_score']:.2f} cov>={r['min_cov']:.2f} | "
            f"P={r['precision']:.3f} R={r['recall']:.3f} FPR={r['fpr']:.3f} | "
            f"TP={r['tp']} FP={r['fp']} FN={r['fn']} cost={r['cost']}{flag}"
        )

    print("\n== 推荐阈值 ==")
    print(
        f"qa_semantic_min_score = {best['min_score']}\n"
        f"qa_semantic_min_query_coverage = {best['min_cov']}\n"
        f"（在标注集上 precision={best['precision']}, recall={best['recall']}, fp={best['fp']}, fn={best['fn']}）"
    )

    if args.out_csv.strip():
        outp = Path(args.out_csv.strip())
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "min_score",
                    "min_cov",
                    "precision",
                    "recall",
                    "fpr",
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "cost",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        print(f"\n已导出: {outp}")


if __name__ == "__main__":
    main()

