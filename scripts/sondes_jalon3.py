"""Persiste les deux mesures ad hoc que le rapport du jalon 3 publiait sans JSON.

La loi du projet est que tout chiffre publié soit recalculable depuis un JSON versionné.
Deux mesures y échappaient (relevé par la revue finale de branche) :

  1. le tableau CPU vs GPU qui justifie la correction d'un facteur ~70 sur les latences de
     reranking publiées au jalon 2.5 — c'est le chiffre le plus lourd de conséquences du
     jalon, il ne peut pas rester un contrôle jetable ;
  2. l'anatomie de q023 (contributions RRF du gold contre les trois premiers du top-10,
     composition mono-canal / bi-canal du pool), qui fonde la piste retenue pour la suite.

Écrit `docs/mesures/jalon3/sondes.json`.

Coût : ~3 min (une question rerankée sur CPU coûte ~150 s à elle seule).

Usage : uv run python scripts/sondes_jalon3.py
"""
import json
import time
from pathlib import Path

from accounting_rag.embed import Embedder
from accounting_rag.evalrag import load_benchmark, match
from accounting_rag.rerank import Reranker
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT = ROOT / "docs/mesures/jalon3/sondes.json"
RRF_K = 60  # constante de la fusion, cf. Searcher._rrf


def sonde_device(embedder, question: str) -> dict:
    """Latence d'une question rerankée (bge, 25 candidats), GPU puis CPU forcé."""
    out = {}
    for device in ("cuda", "cpu"):
        r = Reranker(model_name="BAAI/bge-reranker-v2-m3")
        r.model.model.to(device)
        s = Searcher(DB, embedder=embedder, reranker=r, pool=50, n_rerank=25)
        t0 = time.perf_counter()
        s.search(question, mode="hybrid+rerank", k=10)
        out[device] = round(time.perf_counter() - t0, 2)
    out["facteur"] = round(out["cpu"] / out["cuda"], 1)
    return out


def sonde_q023(embedder) -> dict:
    """Anatomie de l'éviction RRF : contributions par canal, composition du pool."""
    q = next(x for x in load_benchmark(ROOT / "benchmark/dev.jsonl") if x["id"] == "q023")
    s = Searcher(DB, embedder=embedder)
    bm, de = s._bm25(q["question"], s.pool), s._dense(q["question"], s.pool)
    rbm = {r: i for i, r in enumerate(sorted(bm, key=bm.get, reverse=True), 1)}
    rde = {r: i for i, r in enumerate(sorted(de, key=de.get, reverse=True), 1)}
    fus = s._rrf([bm, de])
    classe = sorted(fus, key=fus.get, reverse=True)
    gold = next(r for r in rbm if any(match(r, c) for c in q["citations"]))

    def detail(rid: str) -> dict:
        b, d = rbm.get(rid), rde.get(rid)
        return {"record": rid.split("@")[0], "rang_bm25": b, "rang_dense": d,
                "contribution_bm25": round(1 / (RRF_K + b), 5) if b else 0.0,
                "contribution_dense": round(1 / (RRF_K + d), 5) if d else 0.0,
                "score_rrf": round(fus[rid], 5),
                "rang_apres_fusion": classe.index(rid) + 1}

    mono = sum(1 for r in fus if (r in rbm) != (r in rde))
    return {
        "question": q["id"], "citations_attendues": q["citations"],
        "convention_rangs": "1-indexée (comme scripts/diagnostic_rangs.py)",
        "rrf_k": RRF_K,
        "gold": detail(gold),
        "trois_premiers_apres_fusion": [detail(r) for r in classe[:3]],
        "pool_fusionne": {"total": len(fus), "un_seul_canal": mono,
                          "les_deux_canaux": len(fus) - mono},
        "n_rerank_de_la_config_livree": 25,
        "gold_a_portee_du_reranker": classe.index(gold) + 1 <= 25,
    }


def main() -> None:
    embedder = Embedder()
    questions = load_benchmark(ROOT / "benchmark/dev.jsonl")
    out = {
        "latence_par_device": {
            "config": "hybrid+rerank, BAAI/bge-reranker-v2-m3, n_rerank=25, pool=50",
            "question": questions[1]["id"], "n_questions": 1,
            "secondes_par_question": sonde_device(embedder, questions[1]["question"]),
            "note": "Justifie la correction des latences publiées au jalon 2.5 "
                    "(129,5 s/question y était une latence CPU).",
        },
        "anatomie_q023": sonde_q023(embedder),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[sondes_jalon3] écrit {OUT}")


if __name__ == "__main__":
    main()
