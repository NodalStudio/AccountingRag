"""Script de clôture du jalon 3 (T4) — campagne dev finale + référence test gelée.

Régénère, dans UN SEUL processus python (embedder, reranker et rewriter partagés, donc
alignement garanti des ids `par_question`), les trois configurations qui résument
l'histoire du projet :

  A — `hybrid` neutre .................... baseline du jalon 2.5 (recall@10 dev = 0,672)
  B — `hybrid+rerank`, bge, n_rerank=25 ... config ADOPTÉE au jalon 2.5 (dev = 0,738)
  C — réécriture `etend` + B ............. config ADOPTÉE au jalon 3 (dev = 0,877)

Deux comparaisons par bootstrap apparié (10 000 tirages, seed 42) :
  - C vs A : l'effet cumulé de tout ce qui a été adopté depuis le premier benchmark ;
  - C vs B : l'apport propre de ce jalon, par-dessus le reranking déjà adopté.

Contrôles de fraîcheur AVANT tout engagement sur le split gelé : A sur dev doit redonner
0,672 et B sur dev doit redonner 0,738. Un écart signifie que l'index ou le code a changé
depuis les mesures publiées — toute comparaison serait invalide, le script abandonne.

**Le split `test` (29 questions) est GELÉ** : il n'a servi qu'une fois au jalon 2.5 et une
fois ici, jamais pour choisir un paramètre (cf. benchmark/README.md). Ne relancer ce
script que pour vérifier une reproduction, jamais après un nouvel ajustement.

Coût : les réécritures de `dev` sont déjà dans le cache committé ; celles de `test` sont
appelées ici pour la première fois (29 appels à l'API Claude, quelques centimes), puis
mises en cache à leur tour. Les runs B et C coûtent ~2 s/question sur GPU (~150 s/question
sur CPU, cf. docs/eval-jalon3.md).

Usage (depuis la racine du dépôt) :
    uv run python scripts/cloture_jalon3.py
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from accounting_rag.embed import Embedder
from accounting_rag.evalrag import evaluate, load_benchmark, paired_bootstrap
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon3"
CACHE_REECRITURES = OUT_DIR / "reecritures.json"

REF_DEV_A = 0.672  # hybrid neutre, référence jalon 2.5
REF_DEV_B = 0.738  # hybrid+rerank bge n_rerank=25, config adoptée jalon 2.5


def _par_categorie_bootstrap(pq_a: dict, pq_b: dict, questions: list[dict]) -> dict:
    cat_of = {q["id"]: q["categorie"] for q in questions}
    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid in pq_a:
        by_cat[cat_of[qid]].append(qid)
    return {
        cat: {"n": len(ids), **paired_bootstrap({i: pq_a[i] for i in ids},
                                                {i: pq_b[i] for i in ids})}
        for cat, ids in sorted(by_cat.items())
    }


def _pire_perte_categorie(pq_ref: dict, pq_cfg: dict, questions: list[dict]) -> tuple[str, float]:
    cat_of = {q["id"]: q["categorie"] for q in questions}
    ref, cfg = defaultdict(list), defaultdict(list)
    for qid, v in pq_ref.items():
        ref[cat_of[qid]].append(v)
    for qid, v in pq_cfg.items():
        cfg[cat_of[qid]].append(v)
    pire_cat, pire = "", 0.0
    for cat in ref:
        d = sum(cfg[cat]) / len(cfg[cat]) - sum(ref[cat]) / len(ref[cat])
        if d < pire:
            pire, pire_cat = d, cat
    return pire_cat, pire


def _mesurer(label: str, searcher: Searcher, mode: str, questions: list[dict]) -> dict:
    t0 = time.perf_counter()
    r = evaluate(searcher, questions, mode=mode, k=10)
    dt = time.perf_counter() - t0
    r["latence_s_par_question"] = dt / len(questions)
    print(f"[cloture3] {label} : recall@5={r['recall@5']} recall@10={r['recall@10']} "
          f"mrr={r['mrr']} ({dt:.1f}s, {r['latence_s_par_question']:.2f} s/question)", flush=True)
    return r


def _comparer(ref: dict, cfg: dict, questions: list[dict]) -> dict:
    boot = paired_bootstrap(ref["par_question"], cfg["par_question"])
    cat, delta = _pire_perte_categorie(ref["par_question"], cfg["par_question"], questions)
    return {**boot,
            "pire_perte_categorie": {"categorie": cat, "delta": round(delta, 4)},
            "adopte": boot["p_amelioration"] >= 0.95 and delta >= -0.05,
            "par_categorie": _par_categorie_bootstrap(ref["par_question"],
                                                       cfg["par_question"], questions)}


def _hors_atteinte(b: dict, c: dict) -> dict:
    """Questions que la réécriture répare, et celles qui résistent aux deux configs.

    Répond à la question posée par le plan du jalon : combien de questions restaient
    hors d'atteinte SANS réécriture par LLM ?
    """
    reparees = sorted(q for q in b["par_question"]
                      if b["par_question"][q] < c["par_question"][q])
    cassees = sorted(q for q in b["par_question"]
                     if b["par_question"][q] > c["par_question"][q])
    resistantes = sorted(q for q in b["par_question"]
                         if b["par_question"][q] == 0.0 and c["par_question"][q] == 0.0)
    return {
        "reparees_par_la_reecriture": reparees,
        "n_reparees": len(reparees),
        "degradees_par_la_reecriture": cassees,
        "n_degradees": len(cassees),
        "resistantes_aux_deux_configs": resistantes,
        "n_resistantes": len(resistantes),
    }


def run_split(split: str, embedder, reranker, rewriter) -> dict:
    questions = load_benchmark(ROOT / f"benchmark/{split}.jsonl")
    print(f"\n[cloture3] === split {split} ({len(questions)} questions) ===", flush=True)

    a = _mesurer(f"{split} A — hybrid neutre",
                 Searcher(DB, embedder=embedder), "hybrid", questions)
    b = _mesurer(f"{split} B — hybrid+rerank (config jalon 2.5)",
                 Searcher(DB, embedder=embedder, reranker=reranker), "hybrid+rerank", questions)
    c = _mesurer(f"{split} C — réécriture etend + rerank (config jalon 3)",
                 Searcher(DB, embedder=embedder, reranker=reranker,
                          rewriter=rewriter, mode_reecriture="etend"),
                 "hybrid+rerank", questions)

    out = {
        "split": split, "n": len(questions),
        "configs": {
            "A_hybrid_neutre": {k: a[k] for k in
                                ("recall@5", "recall@10", "mrr", "par_categorie",
                                 "par_question", "latence_s_par_question")},
            "B_rerank_jalon25": {k: b[k] for k in
                                 ("recall@5", "recall@10", "mrr", "par_categorie",
                                  "par_question", "latence_s_par_question")},
            "C_reecriture_rerank_jalon3": {k: c[k] for k in
                                           ("recall@5", "recall@10", "mrr", "par_categorie",
                                            "par_question", "latence_s_par_question")},
        },
        "bootstrap_C_vs_A": _comparer(a, c, questions),
        "bootstrap_C_vs_B": _comparer(b, c, questions),
        "questions_hors_atteinte_sans_reecriture": _hors_atteinte(b, c),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"cloture_{split}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cloture3] {split} : écrit {path}", flush=True)
    return out


def main() -> None:
    print("[cloture3] chargement de l'embedder partagé...", flush=True)
    embedder = Embedder()
    dev_questions = load_benchmark(ROOT / "benchmark/dev.jsonl")

    # --- Contrôle de fraîcheur 1 : la baseline neutre du jalon 2.5 ---
    ctrl_a = evaluate(Searcher(DB, embedder=embedder), dev_questions, mode="hybrid", k=10)
    print(f"[cloture3] contrôle A : recall@10 dev = {ctrl_a['recall@10']} "
          f"(attendu {REF_DEV_A})", flush=True)
    if ctrl_a["recall@10"] != REF_DEV_A:
        sys.exit(f"STATUS: BLOCKED — contrôle de fraîcheur A échoué "
                 f"({ctrl_a['recall@10']} != {REF_DEV_A}) : index ou code modifiés depuis "
                 f"les mesures publiées, toute comparaison serait invalide.")

    print("[cloture3] chargement du reranker (défaut de code, bge-reranker-v2-m3)...", flush=True)
    reranker = Reranker()

    # --- Contrôle de fraîcheur 2 : la config adoptée au jalon 2.5 ---
    ctrl_b = evaluate(Searcher(DB, embedder=embedder, reranker=reranker),
                      dev_questions, mode="hybrid+rerank", k=10)
    print(f"[cloture3] contrôle B : recall@10 dev = {ctrl_b['recall@10']} "
          f"(attendu {REF_DEV_B})", flush=True)
    if ctrl_b["recall@10"] != REF_DEV_B:
        sys.exit(f"STATUS: BLOCKED — contrôle de fraîcheur B échoué "
                 f"({ctrl_b['recall@10']} != {REF_DEV_B}).")
    print("[cloture3] contrôles OK — index et code identiques aux mesures publiées.", flush=True)

    rewriter = Rewriter(cache_path=CACHE_REECRITURES)
    print(f"[cloture3] cache de réécritures : {len(rewriter._cache)} entrée(s). Les "
          f"questions du split gelé n'y sont pas encore : elles seront appelées une fois "
          f"puis mises en cache.", flush=True)

    dev = run_split("dev", embedder, reranker, rewriter)
    test = run_split("test", embedder, reranker, rewriter)

    print("\n[cloture3] === RÉFÉRENCE GELÉE DU JALON 3 ===")
    print("| split | n | A hybrid | B rerank (j2.5) | C réécriture+rerank (j3) | "
          "delta C−B | p(C>B) | delta C−A | p(C>A) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for o in (dev, test):
        cc, cb, ca = (o["configs"]["C_reecriture_rerank_jalon3"],
                      o["configs"]["B_rerank_jalon25"], o["configs"]["A_hybrid_neutre"])
        print(f"| {o['split']} | {o['n']} | {ca['recall@10']} | {cb['recall@10']} | "
              f"**{cc['recall@10']}** | {o['bootstrap_C_vs_B']['delta']} | "
              f"{o['bootstrap_C_vs_B']['p_amelioration']} | "
              f"{o['bootstrap_C_vs_A']['delta']} | "
              f"{o['bootstrap_C_vs_A']['p_amelioration']} |")
    h = dev["questions_hors_atteinte_sans_reecriture"]
    print(f"\ndev : {h['n_reparees']} question(s) réparée(s) par la réécriture "
          f"({', '.join(h['reparees_par_la_reecriture'])}), {h['n_degradees']} dégradée(s) "
          f"({', '.join(h['degradees_par_la_reecriture']) or 'aucune'}), "
          f"{h['n_resistantes']} résistante(s) aux deux configs "
          f"({', '.join(h['resistantes_aux_deux_configs'])}).")
    print(f"coût API de cette clôture : {rewriter.appels} appel(s) à {rewriter.modele}, "
          f"{rewriter.tokens_entree} tokens d'entrée, {rewriter.tokens_sortie} de sortie.")


if __name__ == "__main__":
    main()
