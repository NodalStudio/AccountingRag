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
  - D (T1, cette tâche) : `df_max ∈ {None, 0.10, 0.05, 0.02}`, mode `hybrid`. Référence
    = `df_max=None` (baseline jalon 2.5, recall@10 attendu = 0,672 sur dev — contrôle
    de non-régression effectué en tête de `run_ablation()` avant tout bootstrap :
    échoue avec `STATUS: BLOCKED` si la config neutre ne redonne pas exactement 0,672).
  - E, F, cumul : introduits par les tâches suivantes du jalon 3 (paramètres `pool`,
    `n_rerank` pas encore sur `Searcher` à ce stade) — non implémentées ici, choix
    reconnus par l'argparse mais qui lèvent `NotImplementedError` avec un renvoi
    explicite vers la tâche qui les introduit.

Usage :
    uv run python scripts/ablations_jalon3.py --ablation D --split dev
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from accounting_rag.embed import Embedder
from accounting_rag.evalrag import evaluate, load_benchmark, paired_bootstrap
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon3"

REF_RECALL10_DEV_HYBRID_NEUTRE = 0.672  # baseline jalon 2.5, cf. docs/eval-jalon25.md


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
        f"--ablation {nom} n'est pas implémentée par cette tâche (T1, jalon 3) : les "
        f"paramètres qu'elle mesure (pool, n_rerank) ne sont pas encore introduits sur "
        f"Searcher. Voir docs/superpowers/plans/2026-08-16-jalon3-recouvrement-nul.md, "
        f"Task 2 (ablation E, pool), Task 3 (ablation F, n_rerank), Task 4 (cumul)."
    )


_CONFIGS_PAR_ABLATION = {
    "D": _configs_D,
    "E": lambda: _configs_non_implementees("E"),
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation", required=True, choices=["D", "E", "F", "cumul"])
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    args = p.parse_args()
    run_ablation(args.ablation, args.split)


if __name__ == "__main__":
    main()
