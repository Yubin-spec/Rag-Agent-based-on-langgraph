"""
将本地 high_freq_qa.json 同步到 FAQ 专属 Milvus collection。

用法：
  python scripts/sync_faq_to_milvus.py
  python scripts/sync_faq_to_milvus.py --path data/high_freq_qa.json --replace
"""

import argparse
import hashlib
import json
from pathlib import Path

from config import get_settings
from src.db_resilience import get_milvus_collection, milvus_operation_with_retry
from src.kb.embedding_loader import get_bge_embedding


def _normalize(s: str) -> str:
    return "".join((s or "").split()).lower().strip()


def _build_collection_schema(collection_name: str):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

    dim = get_settings().milvus_dim
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="question_norm", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="candidate_text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="matched_question", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="matched_alias", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="match_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="faq chunks (exact/alias + semantic)")
    coll = Collection(name=collection_name, schema=schema)
    coll.create_index(
        field_name="embedding",
        index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 256}},
    )
    coll.load()
    return coll


def _load_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("qa_pairs"), list):
        return data["qa_pairs"]
    return []


def _build_rows(items: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in items:
        q = (item.get("question") or item.get("q") or "").strip()
        a = (item.get("answer") or item.get("a") or "").strip()
        aliases = item.get("aliases") or item.get("alias") or []
        if isinstance(aliases, str):
            aliases = [aliases]
        if not isinstance(aliases, list):
            aliases = []
        aliases = [str(x).strip() for x in aliases if str(x).strip()]
        if not q or not a:
            continue

        # 标准问法条目
        q_norm = _normalize(q)
        rows.append(
            {
                "id": hashlib.sha1(f"exact::{q_norm}".encode("utf-8")).hexdigest(),
                "question_norm": q_norm,
                "candidate_text": q,
                "matched_question": q,
                "matched_alias": "",
                "answer": a,
                "match_type": "exact",
            }
        )

        # alias 条目
        for al in aliases:
            al_norm = _normalize(al)
            if not al_norm:
                continue
            rows.append(
                {
                    "id": hashlib.sha1(f"alias::{al_norm}::{q_norm}".encode("utf-8")).hexdigest(),
                    "question_norm": al_norm,
                    "candidate_text": al,
                    "matched_question": q,
                    "matched_alias": al,
                    "answer": a,
                    "match_type": "alias",
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="", help="qa json path, default from settings.qa_data_path")
    parser.add_argument("--replace", action="store_true", help="clear old FAQ rows before insert")
    args = parser.parse_args()

    s = get_settings()
    path = Path(args.path or s.qa_data_path)
    items = _load_items(path)
    rows = _build_rows(items)
    if not rows:
        print(f"[WARN] no FAQ rows loaded from: {path}")
        return

    embed = get_bge_embedding()
    if embed is None:
        print("[ERROR] embedding model unavailable")
        return

    coll = get_milvus_collection(
        s.milvus_uri,
        s.qa_milvus_collection,
        create_if_missing=True,
        schema_builder=_build_collection_schema,
    )
    if coll is None:
        print("[ERROR] milvus collection unavailable")
        return

    if args.replace:
        def _do_delete(c):
            c.delete('id != ""')
            c.flush()
            return True
        milvus_operation_with_retry(
            s.milvus_uri,
            s.qa_milvus_collection,
            _do_delete,
            retries=1,
            critical=False,
            default=False,
        )

    texts = [r["candidate_text"] for r in rows]
    vecs = embed.encode(texts).tolist()
    entities = [
        [r["id"] for r in rows],
        [r["question_norm"] for r in rows],
        [r["candidate_text"] for r in rows],
        [r["matched_question"] for r in rows],
        [r["matched_alias"] for r in rows],
        [r["answer"] for r in rows],
        [r["match_type"] for r in rows],
        vecs,
    ]

    def _do_insert(c):
        c.insert(entities)
        c.flush()
        return len(rows)

    count = milvus_operation_with_retry(
        s.milvus_uri,
        s.qa_milvus_collection,
        _do_insert,
        retries=2,
        critical=False,
        default=0,
    )
    print(f"[OK] synced {count} FAQ rows to {s.qa_milvus_collection}")


if __name__ == "__main__":
    main()
