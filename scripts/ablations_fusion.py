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
from pathlib import Path

from accounting_rag.ablation import (  # noqa: F401 — ré-exports pour les tests
    CACHE_REECRITURES, GARDE_CATEGORIE, MODE, PERIMETRE_JALON3, ROOT,
    SEUIL_ADOPTION, _affiche, controle_fraicheur, evaluate, marge_avant_eviction,
    mesurer, pire_perte_categorie, run_grilles, searcher_du_contexte)
from accounting_rag.embed import Embedder
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter

OUT_DIR = ROOT / "docs/mesures/jalon3-fix"

# Grilles FIXÉES AVANT MESURE. Géométriques, sur deux décades, choisies sans regarder
# le résultat : le point de bascule arithmétique de q023 (~0,05, cf. tests) tombe DANS
# la grille par conséquence des rangs mesurés, pas parce que la grille a été taillée
# autour de lui.
GRILLES = {
    "consensus": ("poids_consensus", [1.0, 0.5, 0.25, 0.10, 0.025]),
    "escompte": ("rrf_k", [60, 20, 5, 1]),
}
NEUTRE = {"poids_consensus": 1.0, "rrf_k": 60}

# Alias historique : les tests de ce script l'appellent sous son ancien nom privé.
_pire_perte_categorie = pire_perte_categorie
_searcher = searcher_du_contexte


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--contexte", default="tous", choices=["hybrid", "livree", "tous"])
    p.add_argument("--grille", default="toutes", choices=["consensus", "escompte", "toutes"])
    p.add_argument("--sortie", type=Path, default=None,
                   help="répertoire de sortie ; obligatoire pour une exécution PARTIELLE")
    p.add_argument("--controle-seul", action="store_true",
                   help="n'exécute que le contrôle de fraîcheur, puis sort")
    args = p.parse_args()

    if args.controle_seul:
        embedder, reranker = Embedder(), Reranker()
        rewriter = Rewriter(cache_path=CACHE_REECRITURES, ecrire_cache=False)
        controle_fraicheur(embedder, reranker, rewriter)
        print("contrôle de fraîcheur seul : OK")
        return

    partielle = args.contexte != "tous" or args.grille != "toutes"
    if partielle and args.sortie is None:
        # Sans cette garde, `--contexte hybrid` écraserait l'ancrage publié par un
        # fichier amputé de la moitié de ses mesures — et un artefact partiel se lit
        # exactement comme un artefact complet. C'est la loi 10 prise au sérieux : le
        # répertoire de sortie n'est en écriture QUE pour l'exécution qui le remplit.
        raise SystemExit(
            "STATUS: BLOCKED — exécution partielle (--contexte/--grille) sans --sortie. "
            f"Elle écrirait un artefact incomplet par-dessus {_affiche(OUT_DIR)}. "
            "Passez --sortie vers un répertoire de travail.")

    contextes = ["hybrid", "livree"] if args.contexte == "tous" else [args.contexte]
    grilles = ["consensus", "escompte"] if args.grille == "toutes" else [args.grille]
    sortie = run_grilles(args.split, contextes, grilles, GRILLES, NEUTRE)

    out_dir = args.sortie or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    chemin = out_dir / f"fusion_{args.split}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=2, sort_keys=True),
                      encoding="utf-8")
    print(f"\n[ablations_fusion] écrit {_affiche(chemin)}")
    print(f"configurations franchissant le critère d'adoption : "
          f"{sortie['configurations_adoptees'] or 'AUCUNE'}")


if __name__ == "__main__":
    main()
