"""Script de clôture (T6, jalon 2.5) — campagne dev finale + référence test gelée.

Régénère, dans UN SEUL processus python (embedder + reranker partagés, alignement
des ids `par_question` garanti comme en T3/T4), les dicts `par_question` bruts
comparant la baseline hybrid (paramètres neutres) à la configuration finale
`hybrid+rerank` (BAAI/bge-reranker-v2-m3, adopté en Ablation B / T4) — cf.
docs/eval-jalon25.md, section « Clôture — dev final et référence test gelée (T6) ».

Ce script :
  1. Contrôle de fraîcheur : `hybrid` seul sur `dev` doit redonner recall@10=0,672
     (référence T2/T3/T4). Abandonne immédiatement sinon (index ou code changés
     depuis la mesure de référence : toute réutilisation serait invalide).
  2. Campagne dev complète (61 questions) : A=hybrid baseline neutre, B=hybrid+rerank
     -> bootstrap global + par catégorie, persistance des dicts `par_question` bruts.
  3. Campagne test complète (29 questions, UNE SEULE fois, jamais re-exécutée en
     développement) : mêmes deux runs, mêmes mesures.

Écrit `docs/mesures/jalon25/cloture_dev.json` et `docs/mesures/jalon25/cloture_test.json`
(écrase les fichiers déjà versionnés — ne relancer que pour vérifier une reproduction,
pas pour un nouvel ajustement : `test` est gelé, cf. benchmark/README.md).

Coût : le run `hybrid+rerank` (bge-reranker-v2-m3, CPU) prend ≈120-130 s/question,
soit environ ~2h de calcul au total pour dev+test (90 questions). Le run `hybrid`
seul (contrôle de fraîcheur + baseline) est rapide (~15-20 s pour les deux splits).

Usage (depuis la racine du dépôt, avec uv installé) :
    uv run python scripts/mesure_cloture.py
"""
import sys
import time
import json
from pathlib import Path
from collections import defaultdict

from accounting_rag.evalrag import load_benchmark, evaluate, paired_bootstrap
from accounting_rag.search import Searcher
from accounting_rag.embed import Embedder
from accounting_rag.rerank import Reranker

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon25"

REF_RECALL10_DEV_HYBRID = 0.672


def par_categorie_bootstrap(par_question_a, par_question_b, questions):
    """Bootstrap apparié restreint à chaque catégorie (sous-ensemble des ids)."""
    cat_of = {q["id"]: q["categorie"] for q in questions}
    by_cat = defaultdict(list)
    for qid in par_question_a:
        by_cat[cat_of[qid]].append(qid)
    out = {}
    for cat, ids in sorted(by_cat.items()):
        a_sub = {i: par_question_a[i] for i in ids}
        b_sub = {i: par_question_b[i] for i in ids}
        out[cat] = {"n": len(ids), **paired_bootstrap(a_sub, b_sub)}
    return out


def run_split(split_name, embedder, reranker):
    bench_path = ROOT / f"benchmark/{split_name}.jsonl"
    questions = load_benchmark(bench_path)
    print(f"[cloture] {split_name} : {len(questions)} questions chargées depuis {bench_path}", flush=True)

    s_hybrid = Searcher(DB, embedder=embedder)
    print(f"[cloture] {split_name} : run A (hybrid baseline neutre)...", flush=True)
    t0 = time.perf_counter()
    a = evaluate(s_hybrid, questions, mode="hybrid", k=10)
    dt_a = time.perf_counter() - t0
    print(f"[cloture] {split_name} : A terminé en {dt_a:.1f}s "
          f"({dt_a/len(questions)*1000:.0f} ms/question) -> "
          f"recall@10={a['recall@10']}, mrr={a['mrr']}", flush=True)

    s_rerank = Searcher(DB, embedder=embedder, reranker=reranker)
    print(f"[cloture] {split_name} : run B (hybrid+rerank, config finale)...", flush=True)
    t0 = time.perf_counter()
    b = evaluate(s_rerank, questions, mode="hybrid+rerank", k=10)
    dt_b = time.perf_counter() - t0
    print(f"[cloture] {split_name} : B terminé en {dt_b:.1f}s "
          f"({dt_b/len(questions)*1000:.0f} ms/question) -> "
          f"recall@10={b['recall@10']}, mrr={b['mrr']}", flush=True)

    comp_global = paired_bootstrap(a["par_question"], b["par_question"])
    comp_cat = par_categorie_bootstrap(a["par_question"], b["par_question"], questions)
    print(f"[cloture] {split_name} : bootstrap global A vs B : {comp_global}", flush=True)
    print(f"[cloture] {split_name} : bootstrap par catégorie : "
          f"{json.dumps(comp_cat, indent=2, ensure_ascii=False)}", flush=True)

    out = {
        "split": split_name,
        "n": len(questions),
        "a": {"recall@5": a["recall@5"], "recall@10": a["recall@10"], "mrr": a["mrr"],
              "par_categorie": a["par_categorie"], "par_question": a["par_question"]},
        "b": {"recall@5": b["recall@5"], "recall@10": b["recall@10"], "mrr": b["mrr"],
              "par_categorie": b["par_categorie"], "par_question": b["par_question"]},
        "latence_a_s": dt_a, "latence_a_ms_par_q": dt_a / len(questions) * 1000,
        "latence_b_s": dt_b, "latence_b_ms_par_q": dt_b / len(questions) * 1000,
        "bootstrap_global": comp_global,
        "bootstrap_par_categorie": comp_cat,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"cloture_{split_name}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cloture] {split_name} : résultat écrit dans {out_path}", flush=True)
    return out


def main():
    print("[cloture] chargement embedder (e5-small, partagé)...", flush=True)
    t0 = time.perf_counter()
    embedder = Embedder()
    print(f"[cloture] embedder chargé en {time.perf_counter()-t0:.1f}s", flush=True)

    # --- Contrôle de fraîcheur (avant tout engagement sur le run coûteux) ---
    print("[cloture] contrôle de fraîcheur : hybrid seul sur dev...", flush=True)
    dev_questions = load_benchmark(ROOT / "benchmark/dev.jsonl")
    s_control = Searcher(DB, embedder=embedder)
    t0 = time.perf_counter()
    control = evaluate(s_control, dev_questions, mode="hybrid", k=10)
    dt_control = time.perf_counter() - t0
    print(f"[cloture] contrôle terminé en {dt_control:.1f}s -> recall@10={control['recall@10']} "
          f"(référence T2/T3/T4 = {REF_RECALL10_DEV_HYBRID})", flush=True)
    if control["recall@10"] != REF_RECALL10_DEV_HYBRID:
        print(f"[cloture] ÉCHEC DU CONTRÔLE : recall@10={control['recall@10']} != "
              f"{REF_RECALL10_DEV_HYBRID} -- l'index ou le code a changé depuis T2/T3/T4. "
              f"ABANDON.", flush=True)
        sys.exit(1)
    print("[cloture] contrôle OK -- index/code identiques à la mesure de référence T2/T3/T4, "
          "on procède à la campagne complète (dev + test, config finale).", flush=True)

    print("[cloture] chargement reranker (défaut de code, BAAI/bge-reranker-v2-m3)...", flush=True)
    t0 = time.perf_counter()
    reranker = Reranker()
    print(f"[cloture] reranker chargé en {time.perf_counter()-t0:.1f}s", flush=True)

    dev_out = run_split("dev", embedder, reranker)
    test_out = run_split("test", embedder, reranker)

    print("\n[cloture] === RÉSUMÉ FINAL ===", flush=True)
    for out in (dev_out, test_out):
        print(f"{out['split']} (n={out['n']}) : "
              f"baseline recall@10={out['a']['recall@10']} mrr={out['a']['mrr']} | "
              f"finale recall@10={out['b']['recall@10']} mrr={out['b']['mrr']} | "
              f"delta={out['bootstrap_global']['delta']} p={out['bootstrap_global']['p_amelioration']}",
              flush=True)
    print("[cloture] terminé.", flush=True)


if __name__ == "__main__":
    main()
