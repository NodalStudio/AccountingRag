"""Correctif du jalon 3 : la règle de fusion récompense le consensus, pas l'excellence.

Le défaut est nommé, mesuré et persisté au jalon 3 (docs/eval-jalon3.md, § « Anatomie de
q023 » ; docs/mesures/jalon3/sondes.json, champ `anatomie_q023`) : le meilleur candidat
lexical du corpus (rang 2 sur 1 653), absent du canal dense, sort du top-10 fusionné
derrière un candidat 5ᵉ et 6ᵉ, parce que la somme RRF additionne deux contributions
médiocres au lieu de récompenser une contribution excellente. Le jalon 3 l'a laissé
ouvert au motif que le reranker rattrape le cas — le gold sortant au rang 11, donc dans
la fenêtre `n_rerank=25`.

Ce script mesure deux leviers, un par grille, jamais combinés :

  - `poids_consensus` (neutre 1,0) : `score = max + poids_consensus × (somme − max)`.
    À 1,0 c'est la somme RRF historique ; à 0 seule l'excellence dans un canal compte.
  - `rrf_k` (neutre 60) : l'escompte de rang, jusqu'ici figé.

`rrf_k` est mesuré bien qu'un calcul le disqualifie d'avance — il faudrait `rrf_k ≤ 1`
pour renverser q023, cf. `tests/test_fusion.py`. C'est précisément pour cela qu'il est
mesuré : la loi 6 du dépôt interdit de publier un mécanisme avant d'avoir exécuté le
contrôle qui le tranche, et trois mécanismes plausibles du jalon 3 étaient faux.

**Le livrable central n'est pas le recall@10, c'est la marge avant éviction** — le rang
du gold dans la fusion, avant reranking. Le recall d'aujourd'hui dit que le reranker
rattrape ; la marge dit s'il rattrapera encore quand le corpus grandira d'un ordre de
grandeur au jalon suivant. C'est l'analogue de la couverture du pool, vrai livrable de
l'ablation E du jalon 3, dont le recall seul ne disait rien.

Deux contextes, tenus constants à l'intérieur de chaque grille (loi 2, une variable à
la fois) :

  - `hybrid`  : fusion nue, sans réécriture ni reranking — le mécanisme y est visible ;
  - `livree`  : la configuration livrée au jalon 3 (réécriture `etend` + `hybrid+rerank`,
                `bge-reranker-v2-m3`, `n_rerank=25`, `pool=50`) — celle qui décide de
                l'adoption, parce que c'est celle que quelqu'un exécute.

Critère d'adoption, fixé avant toute mesure et identique à celui du jalon 3 (loi 3) :
`p_amelioration ≥ 0,95` sur recall@10 (bootstrap apparié, `n_boot=10000`, `seed=42`)
ET aucune catégorie ne perdant plus de 0,05. Aucun réglage n'est adopté sur une autre
métrique après coup.

Coût : **aucun appel API**. Les réécritures sont lues en LECTURE SEULE dans l'ancrage
versionné du jalon 4, qui couvre les 93 questions de dev ; `Rewriter(ecrire_cache=False)`
lève plutôt que d'appeler l'API si une question y manque (loi 10).

Usage :
  uv run python scripts/ablations_fusion.py --controle-seul      # gratuit, ~1 min
  uv run python scripts/ablations_fusion.py --split dev
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from accounting_rag.embed import Embedder
from accounting_rag.evalrag import evaluate, load_benchmark, match, paired_bootstrap
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon3-fix"
# Ancrages en LECTURE SEULE (loi 10) : ce script ne réécrit jamais docs/mesures/**
# hors de son propre répertoire de sortie.
PERIMETRE_JALON3 = ROOT / "docs/mesures/jalon3/cloture_dev.json"
CACHE_REECRITURES = ROOT / "docs/mesures/jalon4/reecritures_dev.json"

# Grilles FIXÉES AVANT MESURE. Géométriques, sur deux décades, choisies sans regarder
# le résultat : le point de bascule arithmétique de q023 (~0,05, cf. tests) tombe DANS
# la grille par conséquence des rangs mesurés, pas parce que la grille a été taillée
# autour de lui.
GRILLES = {
    "consensus": ("poids_consensus", [1.0, 0.5, 0.25, 0.10, 0.025]),
    "escompte": ("rrf_k", [60, 20, 5, 1]),
}
NEUTRE = {"poids_consensus": 1.0, "rrf_k": 60}

SEUIL_ADOPTION = 0.95
GARDE_CATEGORIE = -0.05


def _searcher(contexte: str, embedder, reranker, rewriter, **leviers) -> Searcher:
    if contexte == "hybrid":
        return Searcher(DB, embedder=embedder, **leviers)
    if contexte == "livree":
        return Searcher(DB, embedder=embedder, reranker=reranker, rewriter=rewriter,
                        mode_reecriture="etend", n_rerank=25, pool=50, **leviers)
    raise ValueError(f"contexte inconnu : {contexte!r}")


MODE = {"hybrid": "hybrid", "livree": "hybrid+rerank"}


def controle_fraicheur(embedder, reranker, rewriter) -> dict:
    """Les deux configurations publiées au jalon 3 doivent redonner leur chiffre exact.

    Le périmètre n'est PAS une constante : il est lu dans le JSON qui porte le chiffre
    publié. Un contrôle qui figerait « 61 questions » mesurerait la taille du benchmark
    au lieu du périmètre audité, et se mettrait à bloquer dès que dev grandit — ce qui
    est arrivé au jalon 4, où dev est passé de 61 à 93 questions.
    """
    publie = json.loads(PERIMETRE_JALON3.read_text(encoding="utf-8"))["configs"]
    toutes = load_benchmark(ROOT / "benchmark/dev.jsonl")
    attendus = {
        "hybrid": ("A_hybrid_neutre", publie["A_hybrid_neutre"]["recall@10"]),
        "livree": ("C_reecriture_rerank_jalon3", publie["C_reecriture_rerank_jalon3"]["recall@10"]),
    }
    out = {}
    for contexte, (cle, attendu) in attendus.items():
        ids = set(publie[cle]["par_question"])
        questions = [q for q in toutes if q["id"] in ids]
        if len(questions) != len(ids):
            manquants = sorted(ids - {q["id"] for q in questions})
            print(f"STATUS: BLOCKED — {len(manquants)} question(s) du périmètre publié "
                  f"absente(s) de benchmark/dev.jsonl : {manquants[:5]}", flush=True)
            sys.exit(1)
        s = _searcher(contexte, embedder, reranker, rewriter)
        obtenu = evaluate(s, questions, mode=MODE[contexte])["recall@10"]
        etat = "OK" if obtenu == attendu else "ÉCART"
        print(f"[fraîcheur] {contexte:7s} sur {len(questions)} questions : "
              f"{obtenu} (publié {attendu}) — {etat}", flush=True)
        out[contexte] = {"n": len(questions), "attendu": attendu, "obtenu": obtenu,
                         "conforme": obtenu == attendu}
        if obtenu != attendu:
            print("STATUS: BLOCKED — le retrieval a bougé depuis les chiffres publiés ; "
                  "aucune ablation ne serait interprétable.", flush=True)
            sys.exit(1)
    return out


def marge_avant_eviction(s: Searcher, questions: list[dict], mode: str) -> dict:
    """Rang du gold DANS LA FUSION, avant reranking — le livrable central.

    Le recall@10 de la configuration livrée dit que le reranker rattrape les golds que
    la fusion évince. Il ne dit pas de combien. Cette mesure-ci le dit : tant que le
    gold évincé reste dans les `n_rerank` premiers de la fusion, le rattrapage tient ;
    au-delà, il ne tient plus, et rien ne garantit qu'il tiendra sur un corpus plus
    grand. Une question dont le gold est ROUTÉ (référence d'article explicite) n'est
    pas exposée au défaut et est comptée à part, jamais comme un succès de la fusion.
    """
    rangs: dict[str, int | None] = {}
    routees: list[str] = []
    for q in questions:
        routes, classes, _ = s.avant_rerank(q["question"], mode=mode)
        if any(match(r["record_id"], c) for r in routes for c in q["citations"]):
            routees.append(q["id"])
            continue
        rang = next((i + 1 for i, rid in enumerate(classes)
                     if any(match(rid, c) for c in q["citations"])), None)
        rangs[q["id"]] = rang
    trouves = [r for r in rangs.values() if r is not None]
    n = len(rangs)
    return {
        "n_exposees": n,
        "n_routees": len(routees),
        "questions_routees": sorted(routees),
        "n_gold_absent_de_la_fusion": n - len(trouves),
        # Médiane des rangs TROUVÉS seulement — donc une statistique partielle, qui
        # flatte d'autant plus qu'il y a de golds absents. Elle ne se lit qu'à côté de
        # `n_gold_absent_de_la_fusion`, jamais seule.
        "rang_median_des_golds_trouves": statistics.median(trouves) if trouves else None,
        # Les deux seuils qui décident quelque chose : 10 est le top-k rendu, 25 est la
        # fenêtre du reranker de la configuration livrée. Un gold ABSENT de la fusion
        # compte au-delà de tous les seuils — il est hors de portée du reranker au même
        # titre qu'un gold au rang 300, et l'omettre ferait sous-estimer précisément le
        # risque que cette mesure existe pour rendre visible.
        "part_au_dela_de_10": round(sum(r is None or r > 10 for r in rangs.values()) / n, 4) if n else None,
        "part_au_dela_de_25": round(sum(r is None or r > 25 for r in rangs.values()) / n, 4) if n else None,
        "rangs": rangs,
    }


def _pire_perte_categorie(ref: dict, cfg: dict, questions: list[dict]) -> tuple[str, float]:
    cat_of = {q["id"]: q["categorie"] for q in questions}
    a: dict[str, list[float]] = defaultdict(list)
    b: dict[str, list[float]] = defaultdict(list)
    for qid, v in ref.items():
        a[cat_of[qid]].append(v)
    for qid, v in cfg.items():
        b[cat_of[qid]].append(v)
    pire_cat, pire = "", 0.0
    for cat in a:
        delta = sum(b[cat]) / len(b[cat]) - sum(a[cat]) / len(a[cat])
        if delta < pire:
            pire, pire_cat = delta, cat
    return pire_cat, round(pire, 4)


def mesurer(contexte: str, leviers: dict, questions: list[dict],
            embedder, reranker, rewriter) -> dict:
    s = _searcher(contexte, embedder, reranker, rewriter, **leviers)
    t0 = time.time()
    res = evaluate(s, questions, mode=MODE[contexte])
    duree = time.time() - t0
    res["latence_s_par_question"] = duree / len(questions)
    res["marge"] = marge_avant_eviction(s, questions, mode=MODE[contexte])
    res["leviers"] = dict(leviers)
    return res


def run(split: str, contextes: list[str], grilles: list[str]) -> dict:
    questions = load_benchmark(ROOT / f"benchmark/{split}.jsonl")
    embedder = Embedder()
    reranker = Reranker()
    # LECTURE SEULE : lève sur une question absente plutôt que d'appeler l'API payante
    # et de réécrire un ancrage versionné (loi 10).
    rewriter = Rewriter(cache_path=CACHE_REECRITURES, ecrire_cache=False)

    sortie = {
        "split": split,
        "n": len(questions),
        "grilles": {k: {"levier": v[0], "valeurs": v[1]} for k, v in GRILLES.items()},
        "neutre": NEUTRE,
        "critere_adoption": {"p_amelioration_min": SEUIL_ADOPTION,
                             "garde_categorie": GARDE_CATEGORIE,
                             "metrique": "recall@10", "n_boot": 10000, "seed": 42},
        "fraicheur": controle_fraicheur(embedder, reranker, rewriter),
        "contextes": {},
    }

    for contexte in contextes:
        print(f"\n=== contexte {contexte} ({MODE[contexte]}) ===", flush=True)
        ref = mesurer(contexte, dict(NEUTRE), questions, embedder, reranker, rewriter)
        print(f"  référence (neutre) : recall@10={ref['recall@10']} "
              f"marge médiane={ref['marge']['rang_median_des_golds_trouves']}", flush=True)
        configs = {"reference": ref}
        for grille in grilles:
            levier, valeurs = GRILLES[grille]
            for valeur in valeurs:
                if valeur == NEUTRE[levier]:
                    continue  # la référence, déjà mesurée — jamais deux fois
                leviers = dict(NEUTRE, **{levier: valeur})
                res = mesurer(contexte, leviers, questions, embedder, reranker, rewriter)
                boot = paired_bootstrap(ref["par_question"], res["par_question"])
                cat, perte = _pire_perte_categorie(ref["par_question"],
                                                   res["par_question"], questions)
                res["bootstrap_vs_reference"] = boot
                res["pire_categorie"] = {"categorie": cat, "delta": perte}
                res["adopte"] = (boot["p_amelioration"] >= SEUIL_ADOPTION
                                 and perte >= GARDE_CATEGORIE)
                configs[f"{levier}={valeur}"] = res
                print(f"  {levier}={valeur:<6} recall@10={res['recall@10']:<6} "
                      f"delta={boot['delta']:+.4f} p={boot['p_amelioration']:.4f} "
                      f"pire_cat={perte:+.4f} marge_med={res['marge']['rang_median_des_golds_trouves']} "
                      f"au-delà de 25={res['marge']['part_au_dela_de_25']}", flush=True)
        sortie["contextes"][contexte] = configs

    adoptes = [f"{c}/{n}" for c, cfgs in sortie["contextes"].items()
               for n, r in cfgs.items() if r.get("adopte")]
    sortie["configurations_adoptees"] = adoptes
    return sortie


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--contexte", default="tous", choices=["hybrid", "livree", "tous"])
    p.add_argument("--grille", default="toutes", choices=["consensus", "escompte", "toutes"])
    p.add_argument("--controle-seul", action="store_true",
                   help="n'exécute que le contrôle de fraîcheur, puis sort")
    args = p.parse_args()

    if args.controle_seul:
        embedder, reranker = Embedder(), Reranker()
        rewriter = Rewriter(cache_path=CACHE_REECRITURES, ecrire_cache=False)
        controle_fraicheur(embedder, reranker, rewriter)
        print("contrôle de fraîcheur seul : OK")
        return

    contextes = ["hybrid", "livree"] if args.contexte == "tous" else [args.contexte]
    grilles = ["consensus", "escompte"] if args.grille == "toutes" else [args.grille]
    sortie = run(args.split, contextes, grilles)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chemin = OUT_DIR / f"fusion_{args.split}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=2, sort_keys=True),
                      encoding="utf-8")
    print(f"\n[ablations_fusion] écrit {chemin.relative_to(ROOT)}")
    print(f"configurations franchissant le critère d'adoption : "
          f"{sortie['configurations_adoptees'] or 'AUCUNE'}")


if __name__ == "__main__":
    main()
