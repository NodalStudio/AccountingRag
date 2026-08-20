"""Second correctif du jalon 3 : la réécriture casse trois questions qu'elle devrait aider.

Le jalon 3 a adopté la réécriture de requête sur un gain net mesuré — douze questions
réparées, trois cassées — et a laissé les trois en dette, avec une piste : « conditionner
la réécriture à un signal de recouvrement faible plutôt que l'appliquer systématiquement ».

**Cette piste ne correspond pas au mécanisme**, et c'est la première chose que ce
correctif a établie. q080 est la question la plus familière du lot — « j'ai avalé une
boîte » — donc n'importe quelle porte fondée sur le recouvrement déclencherait la
réécriture précisément là où elle casse.

Les trois questions cassées ont trois causes différentes, que la phrase « la réécriture
dégrade trois questions » masquait (rangs mesurés sur `data/corpus.db`, pool=400) :

  - **q080** : rang bm25 7 -> 84 avec la réécriture. Le gold sort du pool de 50, donc
    hors de portée de tout reranker. Le canal dense est hors de cause : il ne trouve
    JAMAIS ce gold, réécriture ou non.
  - **q025** : rang bm25 1 -> 13 ; son gold arrive au rang 22 de la fusion, donc DANS la
    fenêtre du reranker, qui le voit et ne le retient pas.
  - **q008** : rang bm25 1 partout, dense 1 -> 2. Sa dégradation n'est pas lexicale ;
    c'est le cas d'éviction par la fusion, que le premier correctif du jalon 3 réparait
    dans le contexte livré. Les deux dettes se recoupent sur cette question.

Le levier mesuré ici est donc lexical : `poids_question`, le nombre de répétitions des
tokens de la question originale devant la réécriture dans la requête envoyée à BM25. Il
exploite une propriété déjà mesurée au jalon 3 (`bm25()` de FTS5 pondère la multiplicité
des termes) et ne touche pas le canal dense — sans quoi il déplacerait aussi l'embedding
et mesurerait deux variables sous un seul nom.

Sonde préalable, rang bm25 du gold selon le nombre de répétitions (0 = réécriture seule) :

    question   n=0    n=1*   n=2    n=3    n=5    n=8   question seule
    q080       280     84     26     12      5      6        7
    q025        14     13     10      6      3      2        1
    q021         7      4      4      7     18     51      154
    q060         3     11     19     42    159   None     None
    q070        10     10     11     13     31     46      136
    (* n=1 = le mode `etend` actuel)

L'arbitrage est réel et monotone : ce qui répare q080 détruit q060. C'est ce que la
grille tranche.

**Prédiction enregistrée avant mesure.** Le premier correctif a montré que le reranker
absorbe les changements de fusion. Je m'attends à ce qu'il n'absorbe PAS celui-ci, pour
une raison structurelle : la fusion réordonne l'intérieur du pool, ce levier change la
COMPOSITION du pool — un candidat absent du pool est hors de portée de tout reranker.
Elle est écrite ici pour être démentie : au premier correctif, j'avais annoncé `rrf_k`
« disqualifié d'avance » en généralisant à une grille un calcul portant sur une question,
et la mesure m'a corrigé.

Deux contextes, tenus constants dans la grille (loi 2) :

  - `reecriture` : réécriture `etend` SANS reranking — le contexte de l'ablation G du
    jalon 3, où le mécanisme est visible. Le levier serait inerte dans `hybrid` nu, faute
    de réécriture à pondérer.
  - `livree`     : la configuration livrée au jalon 3 — celle qui décide de l'adoption.

Critère d'adoption inchangé et fixé avant mesure : `p_amelioration >= 0,95` sur recall@10
(bootstrap apparié, `n_boot=10000`, `seed=42`) ET aucune catégorie perdant plus de 0,05.

Coût : aucun appel API. Les réécritures sont lues en LECTURE SEULE dans l'ancrage
versionné du jalon 4.

Usage :
  uv run python scripts/ablations_reecriture.py --controle-seul     # gratuit
  uv run python scripts/ablations_reecriture.py --split dev
"""
import argparse
import json
from pathlib import Path

from accounting_rag.ablation import (  # noqa: F401 — ré-exports pour les tests
    CACHE_REECRITURES, GARDE_CATEGORIE, MODE, PUBLIE, ROOT, SEUIL_ADOPTION,
    _affiche, controle_fraicheur, marge_avant_eviction, run_grilles)
from accounting_rag.embed import Embedder
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter

OUT_DIR = ROOT / "docs/mesures/jalon3-fix"

# Grille FIXÉE AVANT MESURE, et lisible dans la sonde du docstring : elle couvre le
# domaine où le rang de q080 redevient inférieur au pool (n>=2) ET celui où les questions
# que la réécriture répare commencent à se dégrader (n>=5). Elle n'a pas été taillée
# autour d'un optimum : la sonde ne mesure que des rangs lexicaux, pas du recall.
GRILLES = {"poids_question": ("poids_question", [1, 2, 3, 5, 8])}
NEUTRE = {"poids_question": 1}

CONTEXTES = ["reecriture", "livree"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--contexte", default="tous", choices=CONTEXTES + ["tous"])
    p.add_argument("--sortie", type=Path, default=None,
                   help="répertoire de sortie ; obligatoire pour une exécution PARTIELLE")
    p.add_argument("--controle-seul", action="store_true")
    args = p.parse_args()

    if args.controle_seul:
        embedder, reranker = Embedder(), Reranker()
        rewriter = Rewriter(cache_path=CACHE_REECRITURES, ecrire_cache=False)
        controle_fraicheur(embedder, reranker, rewriter, contextes=CONTEXTES)
        print("contrôle de fraîcheur seul : OK")
        return

    partielle = args.contexte != "tous"
    if partielle and args.sortie is None:
        # Un artefact amputé se lit exactement comme un artefact complet.
        raise SystemExit(
            "STATUS: BLOCKED — exécution partielle (--contexte) sans --sortie. Elle "
            f"écrirait un artefact incomplet par-dessus {_affiche(OUT_DIR)}.")

    contextes = CONTEXTES if args.contexte == "tous" else [args.contexte]
    sortie = run_grilles(args.split, contextes, ["poids_question"], GRILLES, NEUTRE)

    out_dir = args.sortie or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    chemin = out_dir / f"reecriture_{args.split}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=2, sort_keys=True),
                      encoding="utf-8")
    print(f"\n[ablations_reecriture] écrit {_affiche(chemin)}")
    print(f"configurations franchissant le critère d'adoption : "
          f"{sortie['configurations_adoptees'] or 'AUCUNE'}")


if __name__ == "__main__":
    main()
