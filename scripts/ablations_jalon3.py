"""Squelette commun de mesure des ablations du jalon 3, par bootstrap apparié.

Charge le benchmark (`benchmark/<split>.jsonl`), instancie **un seul** `Embedder`
partagé entre toutes les configurations d'une ablation (économie de ~50 s par config
— l'essentiel du coût d'une campagne dev tient dans ce chargement unique), exécute
`evaluate()` pour chaque configuration, compare chacune à la configuration de
référence par `paired_bootstrap()`, imprime un tableau markdown et écrit le JSON
complet (agrégats **et** `par_question` bruts de CHAQUE configuration, pas seulement
la référence — tout chiffre publié dans docs/eval-jalon3.md doit être recalculable
sans re-runner ce script) dans `docs/mesures/jalon3/<ablation>_<split>.json`.

Ablations supportées par ce jalon (`--ablation D|E|F|cumul`) :
  - D (T1) : `df_max ∈ {None, 0.10, 0.05, 0.02}`, mode `hybrid`. Référence = `df_max=None`
    (baseline jalon 2.5, recall@10 attendu = 0,672 sur dev — contrôle de non-régression
    effectué en tête de `run_ablation()` avant tout bootstrap : échoue avec
    `STATUS: BLOCKED` si la config neutre ne redonne pas exactement 0,672).
  - E (T2, cette tâche) : `pool ∈ {50, 100, 200, 400}` (mode `hybrid`) + `dedup_termes=True`
    à pool constant (50). Référence = `pool=50, dedup_termes=False` (même baseline
    neutre, recall@10 attendu = 0,672 — même contrôle de non-régression que D). Mesure
    en plus la **couverture du pool** (part des questions dont au moins une citation
    gold est présente dans l'union bm25 ∪ dense AVANT fusion, à chaque largeur) — le
    vrai livrable de cette tâche, qui conditionne l'ablation F (reranking). Une
    configuration combinée (meilleure largeur + dedup) n'est mesurée QUE si l'une des
    deux mesures séparées est adoptée (`run_ablation_E`, pas le chemin générique
    `run_ablation` utilisé par D).
  - F, cumul : introduits par les tâches suivantes du jalon 3 (paramètre `n_rerank` pas
    encore sur `Searcher` à ce stade) — non implémentées ici, choix reconnus par
    l'argparse mais qui lèvent `NotImplementedError` avec un renvoi explicite vers la
    tâche qui les introduit.

Usage :
    uv run python scripts/ablations_jalon3.py --ablation D --split dev
    uv run python scripts/ablations_jalon3.py --ablation E --split dev
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from accounting_rag.embed import Embedder
from accounting_rag.evalrag import evaluate, load_benchmark, match, paired_bootstrap
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon3"

REF_RECALL10_DEV_HYBRID_NEUTRE = 0.672  # baseline jalon 2.5, cf. docs/eval-jalon25.md

POOL_WIDTHS_E = (50, 100, 200, 400)  # ablation E : largeurs de pool mesurées


def _configs_D() -> tuple[str, list[tuple[str, dict]]]:
    """df_max ∈ {None, 0.10, 0.05, 0.02}, mode hybrid. Référence = df_max=None (index 0)."""
    configs = [
        ("df_max=None (neutre, jalon 2.5)", {"df_max": None}),
        ("df_max=0.10", {"df_max": 0.10}),
        ("df_max=0.05", {"df_max": 0.05}),
        ("df_max=0.02", {"df_max": 0.02}),
    ]
    return "hybrid", configs


def _configs_non_implementees(nom: str) -> tuple[str, list[tuple[str, dict]]]:
    raise NotImplementedError(
        f"--ablation {nom} n'est pas implémentée par cette tâche (T2, jalon 3) : le "
        f"paramètre qu'elle mesure (n_rerank) n'est pas encore introduit sur Searcher. "
        f"Voir docs/superpowers/plans/2026-08-16-jalon3-recouvrement-nul.md, "
        f"Task 3 (ablation F, n_rerank), Task 4 (cumul)."
    )


_CONFIGS_PAR_ABLATION = {
    "D": _configs_D,
    "F": lambda: _configs_non_implementees("F"),
    "cumul": lambda: _configs_non_implementees("cumul"),
}


def _par_categorie_bootstrap(par_question_a: dict, par_question_b: dict,
                              questions: list[dict]) -> dict:
    cat_of = {q["id"]: q["categorie"] for q in questions}
    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid in par_question_a:
        by_cat[cat_of[qid]].append(qid)
    out = {}
    for cat, ids in sorted(by_cat.items()):
        a_sub = {i: par_question_a[i] for i in ids}
        b_sub = {i: par_question_b[i] for i in ids}
        out[cat] = {"n": len(ids), **paired_bootstrap(a_sub, b_sub)}
    return out


def _pire_perte_categorie(par_question_ref: dict, par_question_cfg: dict,
                           questions: list[dict]) -> tuple[str, float]:
    """Pire delta (recall@10 moyen par catégorie, cfg - ref) — négatif = régression."""
    cat_of = {q["id"]: q["categorie"] for q in questions}
    par_cat_ref: dict[str, list[float]] = defaultdict(list)
    par_cat_cfg: dict[str, list[float]] = defaultdict(list)
    for qid, v in par_question_ref.items():
        par_cat_ref[cat_of[qid]].append(v)
    for qid, v in par_question_cfg.items():
        par_cat_cfg[cat_of[qid]].append(v)
    pire_cat, pire_delta = "", 0.0
    for cat in par_cat_ref:
        moy_ref = sum(par_cat_ref[cat]) / len(par_cat_ref[cat])
        moy_cfg = sum(par_cat_cfg[cat]) / len(par_cat_cfg[cat])
        delta = moy_cfg - moy_ref
        if delta < pire_delta:
            pire_delta, pire_cat = delta, cat
    return pire_cat, pire_delta


def _couverture_pool(searcher: Searcher, questions: list[dict]) -> dict:
    """Couverture du pool AVANT fusion, à la largeur `searcher.pool` : part des
    questions dont au moins une citation gold figure dans l'union bm25 ∪ dense
    récupérée par les deux canaux (avant construction du RRF), globalement et par
    catégorie.

    C'est la métrique demandée par le brief T2 (§ Step 6) : le vrai livrable de cette
    tâche, car elle conditionne l'ablation F (reranking) — un gold absent du pool ne
    peut jamais être remonté par aucun reranker, quelle que soit sa qualité. Distincte
    du recall@10 mesuré par `evaluate()` : celui-ci porte sur le résultat APRÈS fusion
    RRF et troncature à k=10, alors que le pool est bien plus large (jusqu'à `pool`
    candidats par canal, donc jusqu'à `2*pool` ids distincts avant fusion).
    """
    par_cat: dict[str, list[bool]] = defaultdict(list)
    for q in questions:
        bm25_ids = set(searcher._bm25(q["question"], searcher.pool))
        dense_ids = set(searcher._dense(q["question"], searcher.pool))
        pool_ids = bm25_ids | dense_ids
        couvert = any(match(rid, c) for rid in pool_ids for c in q["citations"])
        par_cat[q["categorie"]].append(couvert)
    toutes = [v for vs in par_cat.values() for v in vs]
    return {
        "pool": searcher.pool,
        "couverture": round(sum(toutes) / len(toutes), 3),
        "n": len(toutes),
        "par_categorie": {c: {"couverture": round(sum(v) / len(v), 3), "n": len(v)}
                           for c, v in sorted(par_cat.items())},
    }


def run_ablation(nom: str, split: str) -> dict:
    if nom not in _CONFIGS_PAR_ABLATION:
        raise ValueError(f"ablation inconnue : {nom}")
    mode, configs = _CONFIGS_PAR_ABLATION[nom]()

    questions = load_benchmark(ROOT / f"benchmark/{split}.jsonl")
    print(f"[ablations_jalon3] ablation {nom}, split {split} ({len(questions)} questions), "
          f"mode {mode} — chargement de l'embedder partagé (~50 s)...", flush=True)
    embedder = Embedder()

    runs = []
    for label, params in configs:
        s = Searcher(DB, embedder=embedder, **params)
        t0 = time.perf_counter()
        r = evaluate(s, questions, mode=mode, k=10)
        dt = time.perf_counter() - t0
        r["latence_s_par_question"] = dt / len(questions)
        print(f"[ablations_jalon3] {label} : recall@5={r['recall@5']} recall@10={r['recall@10']} "
              f"mrr={r['mrr']} ({dt:.1f}s, {r['latence_s_par_question']*1000:.0f} ms/question)",
              flush=True)
        runs.append((label, params, r))

    ref_label, ref_params, ref = runs[0]
    if nom == "D" and split == "dev":
        if ref["recall@10"] != REF_RECALL10_DEV_HYBRID_NEUTRE:
            print(f"[ablations_jalon3] CONTRÔLE DE NON-RÉGRESSION ÉCHOUÉ : la config neutre "
                  f"({ref_label}) donne recall@10={ref['recall@10']} != "
                  f"{REF_RECALL10_DEV_HYBRID_NEUTRE} (référence jalon 2.5). ARRÊT.", flush=True)
            raise SystemExit(
                f"STATUS: BLOCKED — recall@10 neutre = {ref['recall@10']}, attendu "
                f"{REF_RECALL10_DEV_HYBRID_NEUTRE}"
            )
        print(f"[ablations_jalon3] contrôle de non-régression OK : recall@10 neutre = "
              f"{ref['recall@10']} == {REF_RECALL10_DEV_HYBRID_NEUTRE}.", flush=True)

    comparaisons = []
    for label, params, r in runs[1:]:
        boot = paired_bootstrap(ref["par_question"], r["par_question"])
        pire_cat, pire_delta = _pire_perte_categorie(ref["par_question"], r["par_question"], questions)
        adopte = boot["p_amelioration"] >= 0.95 and pire_delta >= -0.05
        comparaisons.append({
            "label": label, "params": params, **boot,
            "pire_perte_categorie": {"categorie": pire_cat, "delta": round(pire_delta, 4)},
            "adopte": adopte,
        })

    print(f"\n| config | recall@5 | recall@10 | MRR | latence/question |")
    print("|---|---|---|---|---|")
    for label, _, r in runs:
        print(f"| {label} | {r['recall@5']} | {r['recall@10']} | {r['mrr']} | "
              f"{r['latence_s_par_question']:.3f} s |")
        for cat, v in r["par_categorie"].items():
            print(f"|   ↳ {cat} | | {v['recall@10']} | | (n={v['n']}) |")

    print(f"\n| comparaison (vs {ref_label}) | delta | IC95 | p_amelioration | "
          f"pire perte catégorie | adopté ? |")
    print("|---|---|---|---|---|---|")
    for c in comparaisons:
        print(f"| {c['label']} | {c['delta']} | {c['ic95']} | {c['p_amelioration']} | "
              f"{c['pire_perte_categorie']['delta']} ({c['pire_perte_categorie']['categorie']}) | "
              f"{'oui' if c['adopte'] else 'non'} |")

    out = {
        "ablation": nom, "split": split, "mode": mode, "n": len(questions),
        "configs": [
            {"label": label, "params": params,
             "recall@5": r["recall@5"], "recall@10": r["recall@10"], "mrr": r["mrr"],
             "par_categorie": r["par_categorie"], "par_question": r["par_question"],
             "latence_s_par_question": r["latence_s_par_question"]}
            for label, params, r in runs
        ],
        "comparaisons_vs_reference": comparaisons,
        "bootstrap_par_categorie_vs_reference": {
            c["label"]: _par_categorie_bootstrap(ref["par_question"],
                                                  next(r for l, p, r in runs if l == c["label"])["par_question"],
                                                  questions)
            for c in comparaisons
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{nom}_{split}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ablations_jalon3] écrit {out_path}")
    return out


def run_ablation_E(split: str) -> dict:
    """Ablation E (T2) : largeur du pool de candidats (`pool`) + dédup des termes du
    MATCH (`dedup_termes`) comme configuration mesurée à part entière. Chemin dédié
    (pas `run_ablation()` générique) car cette ablation a deux particularités que le
    squelette générique ne couvre pas : (1) la métrique de couverture du pool, calculée
    directement via `_bm25`/`_dense` sur les Searcher des configs pool=50/100/200/400 ;
    (2) une configuration combinée (meilleure largeur + dedup) mesurée SEULEMENT si
    l'une des deux mesures séparées est adoptée (brief T2, step 5.3).
    """
    nom, mode = "E", "hybrid"
    questions = load_benchmark(ROOT / f"benchmark/{split}.jsonl")
    print(f"[ablations_jalon3] ablation E, split {split} ({len(questions)} questions), "
          f"mode {mode} — chargement de l'embedder partagé (~50 s)...", flush=True)
    embedder = Embedder()

    configs = [
        (f"pool={POOL_WIDTHS_E[0]} (neutre, jalon 2.5)",
         {"pool": POOL_WIDTHS_E[0], "dedup_termes": False}),
        *[(f"pool={w}", {"pool": w, "dedup_termes": False}) for w in POOL_WIDTHS_E[1:]],
        ("dedup_termes=True (pool=50)", {"pool": POOL_WIDTHS_E[0], "dedup_termes": True}),
    ]

    runs: list[tuple[str, dict, dict]] = []
    searchers: dict[str, Searcher] = {}
    for label, params in configs:
        s = Searcher(DB, embedder=embedder, **params)
        searchers[label] = s
        t0 = time.perf_counter()
        r = evaluate(s, questions, mode=mode, k=10)
        dt = time.perf_counter() - t0
        r["latence_s_par_question"] = dt / len(questions)
        print(f"[ablations_jalon3] {label} : recall@5={r['recall@5']} recall@10={r['recall@10']} "
              f"mrr={r['mrr']} ({dt:.1f}s, {r['latence_s_par_question']*1000:.0f} ms/question)",
              flush=True)
        runs.append((label, params, r))

    ref_label, ref_params, ref = runs[0]
    if split == "dev" and ref["recall@10"] != REF_RECALL10_DEV_HYBRID_NEUTRE:
        print(f"[ablations_jalon3] CONTRÔLE DE NON-RÉGRESSION ÉCHOUÉ : la config neutre "
              f"({ref_label}) donne recall@10={ref['recall@10']} != "
              f"{REF_RECALL10_DEV_HYBRID_NEUTRE} (référence jalon 2.5). ARRÊT.", flush=True)
        raise SystemExit(
            f"STATUS: BLOCKED — recall@10 neutre = {ref['recall@10']}, attendu "
            f"{REF_RECALL10_DEV_HYBRID_NEUTRE}"
        )
    print(f"[ablations_jalon3] contrôle de non-régression OK : recall@10 neutre = "
          f"{ref['recall@10']} == {REF_RECALL10_DEV_HYBRID_NEUTRE}.", flush=True)

    # Couverture du pool : un calcul par largeur (les 4 configs dedup_termes=False),
    # sur les MÊMES Searcher que ceux utilisés pour evaluate() ci-dessus (embedder
    # partagé, aucune reconstruction).
    couverture_pool: dict[str, dict] = {}
    for label, params in configs:
        if params.get("dedup_termes") is False:
            t0 = time.perf_counter()
            couverture_pool[label] = _couverture_pool(searchers[label], questions)
            print(f"[ablations_jalon3] couverture du pool, {label} : "
                  f"{couverture_pool[label]['couverture']} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    comparaisons = []
    for label, params, r in runs[1:]:
        boot = paired_bootstrap(ref["par_question"], r["par_question"])
        pire_cat, pire_delta = _pire_perte_categorie(ref["par_question"], r["par_question"], questions)
        adopte = boot["p_amelioration"] >= 0.95 and pire_delta >= -0.05
        comparaisons.append({
            "label": label, "params": params, **boot,
            "pire_perte_categorie": {"categorie": pire_cat, "delta": round(pire_delta, 4)},
            "adopte": adopte,
        })

    # Meilleure largeur de pool (recall@10, parmi 100/200/400, hors dedup) et sa
    # comparaison ; comparaison de dedup_termes=True (pool=50). Config combinée
    # (meilleure largeur + dedup) mesurée SEULEMENT si l'une des deux est adoptée
    # séparément (brief T2, step 5.3) — sinon, documenter l'absence de mesure est une
    # réserve honnête, pas un oubli.
    labels_pool_seul = [l for l, p in configs[1:4]]  # pool=100, pool=200, pool=400
    label_meilleur_pool = max(
        labels_pool_seul, key=lambda l: next(r for ll, pp, r in runs if ll == l)["recall@10"]
    )
    comp_meilleur_pool = next(c for c in comparaisons if c["label"] == label_meilleur_pool)
    comp_dedup = next(c for c in comparaisons if c["label"] == "dedup_termes=True (pool=50)")
    pool_adopte, dedup_adopte = comp_meilleur_pool["adopte"], comp_dedup["adopte"]

    label_combo = None
    if pool_adopte or dedup_adopte:
        largeur = next(p["pool"] for l, p in configs if l == label_meilleur_pool)
        label_combo = f"pool={largeur}+dedup_termes=True"
        params_combo = {"pool": largeur, "dedup_termes": True}
        print(f"[ablations_jalon3] {label_meilleur_pool} adopté={pool_adopte}, "
              f"dedup adopté={dedup_adopte} -> mesure de la config combinée {label_combo}.",
              flush=True)
        s_combo = Searcher(DB, embedder=embedder, **params_combo)
        t0 = time.perf_counter()
        r_combo = evaluate(s_combo, questions, mode=mode, k=10)
        dt = time.perf_counter() - t0
        r_combo["latence_s_par_question"] = dt / len(questions)
        runs.append((label_combo, params_combo, r_combo))
        boot = paired_bootstrap(ref["par_question"], r_combo["par_question"])
        pire_cat, pire_delta = _pire_perte_categorie(ref["par_question"], r_combo["par_question"], questions)
        comparaisons.append({
            "label": label_combo, "params": params_combo, **boot,
            "pire_perte_categorie": {"categorie": pire_cat, "delta": round(pire_delta, 4)},
            "adopte": boot["p_amelioration"] >= 0.95 and pire_delta >= -0.05,
        })
    else:
        print(f"[ablations_jalon3] ni {label_meilleur_pool} ni dedup_termes=True ne sont "
              f"adoptés séparément -> config combinée NON mesurée (règle du brief T2, "
              f"step 5.3) — réserve à documenter, pas un oubli.", flush=True)

    print(f"\n| config | recall@5 | recall@10 | MRR | latence/question |")
    print("|---|---|---|---|---|")
    for label, _, r in runs:
        print(f"| {label} | {r['recall@5']} | {r['recall@10']} | {r['mrr']} | "
              f"{r['latence_s_par_question']:.3f} s |")
        for cat, v in r["par_categorie"].items():
            print(f"|   ↳ {cat} | | {v['recall@10']} | | (n={v['n']}) |")

    print(f"\n| comparaison (vs {ref_label}) | delta | IC95 | p_amelioration | "
          f"pire perte catégorie | adopté ? |")
    print("|---|---|---|---|---|---|")
    for c in comparaisons:
        print(f"| {c['label']} | {c['delta']} | {c['ic95']} | {c['p_amelioration']} | "
              f"{c['pire_perte_categorie']['delta']} ({c['pire_perte_categorie']['categorie']}) | "
              f"{'oui' if c['adopte'] else 'non'} |")

    print(f"\n| couverture du pool | globale | {' | '.join(sorted({cat for cp in couverture_pool.values() for cat in cp['par_categorie']}))} |")
    cats = sorted({cat for cp in couverture_pool.values() for cat in cp["par_categorie"]})
    print("|---|---|" + "---|" * len(cats))
    for label, cp in couverture_pool.items():
        par_cat_str = " | ".join(f"{cp['par_categorie'][c]['couverture']}" for c in cats)
        print(f"| {label} | {cp['couverture']} | {par_cat_str} |")

    out = {
        "ablation": nom, "split": split, "mode": mode, "n": len(questions),
        "configs": [
            {"label": label, "params": params,
             "recall@5": r["recall@5"], "recall@10": r["recall@10"], "mrr": r["mrr"],
             "par_categorie": r["par_categorie"], "par_question": r["par_question"],
             "latence_s_par_question": r["latence_s_par_question"]}
            for label, params, r in runs
        ],
        "comparaisons_vs_reference": comparaisons,
        "bootstrap_par_categorie_vs_reference": {
            c["label"]: _par_categorie_bootstrap(ref["par_question"],
                                                  next(r for l, p, r in runs if l == c["label"])["par_question"],
                                                  questions)
            for c in comparaisons
        },
        "couverture_pool": couverture_pool,
        "combo": {
            "mesuree": label_combo is not None,
            "label": label_combo,
            "raison": (f"{label_meilleur_pool} ou dedup_termes=True adopté séparément"
                       if label_combo is not None else
                       f"ni {label_meilleur_pool} ni dedup_termes=True n'est adopté séparément "
                       f"(règle du brief T2, step 5.3)"),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{nom}_{split}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ablations_jalon3] écrit {out_path}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation", required=True, choices=["D", "E", "F", "cumul"])
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    args = p.parse_args()
    if args.ablation == "E":
        run_ablation_E(args.split)
    else:
        run_ablation(args.ablation, args.split)


if __name__ == "__main__":
    main()
