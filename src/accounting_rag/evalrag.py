"""Harnais d'évaluation retrieval : recall@k / MRR sur citations gold."""
import json
from collections import defaultdict
from pathlib import Path


def load_benchmark(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def match(record_id: str, citation: str) -> bool:
    if record_id == citation:
        return True
    if not record_id.startswith(citation):
        return False
    nxt = record_id[len(citation):][:1]
    return nxt in {"@", "#"} or record_id[len(citation):].startswith("-c")


def evaluate(searcher, questions: list[dict], mode: str, k: int = 10) -> dict:
    recalls5, recalls10, mrrs = [], [], []
    par_cat: dict[str, list[float]] = defaultdict(list)
    for q in questions:
        hits = searcher.search(q["question"], k=k, mode=mode)
        ids = [h["record_id"] for h in hits]
        covered10 = sum(any(match(i, c) for i in ids[:10]) for c in q["citations"])
        covered5 = sum(any(match(i, c) for i in ids[:5]) for c in q["citations"])
        recalls10.append(covered10 / len(q["citations"]))
        recalls5.append(covered5 / len(q["citations"]))
        rank = next((r + 1 for r, i in enumerate(ids)
                     if any(match(i, c) for c in q["citations"])), None)
        mrrs.append(1.0 / rank if rank else 0.0)
        par_cat[q["categorie"]].append(covered10 / len(q["citations"]))
    return {
        "recall@5": round(sum(recalls5) / len(recalls5), 3),
        "recall@10": round(sum(recalls10) / len(recalls10), 3),
        "mrr": round(sum(mrrs) / len(mrrs), 3),
        "par_categorie": {c: {"recall@10": round(sum(v) / len(v), 3), "n": len(v)}
                          for c, v in sorted(par_cat.items())},
        "n": len(questions),
    }
