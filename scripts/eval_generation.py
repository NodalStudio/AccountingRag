"""Campagne de génération : mesure ce que le système RÉPOND, pas ce qu'il retrouve.

Trois jalons ont amélioré le retrieval sans jamais mesurer la réponse. Ce script produit
la première mesure de génération du projet : les taux de citation de la brique
`citations.py` (aucun LLM, aucun jugement) sur la configuration de retrieval livrée au
jalon 3, sans y toucher.

Le générateur ne reçoit que la question et les passages — jamais les citations attendues
du benchmark (loi 9 du dépôt). Le contrôle de fraîcheur en tête de script refuse de
dépenser un centime d'API si le retrieval n'est plus celui qui a été publié.

Caches :
  - réécritures : `docs/mesures/jalon3/reecritures.json` en LECTURE SEULE ;
  - réponses : `docs/mesures/jalon4/reponses_{split}.json`, écrit par ce script et par
    lui seul, puis versionné comme ancrage de reproductibilité.

Coût : un appel API par question (61 sur dev), gratuit à toute ré-exécution.

Usage :
  uv run python scripts/eval_generation.py --split dev --controle-seul   # gratuit
  uv run python scripts/eval_generation.py --split dev
"""
import argparse
import json
import sys
from pathlib import Path

from accounting_rag.citations import metriques
from accounting_rag.embed import Embedder
from accounting_rag.evalrag import evaluate, load_benchmark
from accounting_rag.generate import Generator
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon4"
CACHE_REECRITURES = ROOT / "docs/mesures/jalon3/reecritures.json"

# Baseline publiée du retrieval neutre sur dev (jalon 3, docs/eval-jalon3.md).
# Le contrôle de fraîcheur compare à cette valeur AVANT tout appel payant.
RECALL_NEUTRE_DEV = 0.672

CONFIG = {
    "retrieval": "hybrid+rerank",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "n_rerank": 25,
    "pool": 50,
    "mode_reecriture": "etend",
    "k": 10,
}


def controle_fraicheur(embedder) -> float:
    """`Searcher()` neutre sur dev doit rendre exactement la valeur publiée.

    Gratuit (aucun appel API). S'il échoue, quelque chose a bougé dans l'index ou dans
    le retrieval, et la campagne mesurerait un système différent de celui décrit.
    """
    questions = load_benchmark(ROOT / "benchmark/dev.jsonl")
    r = evaluate(Searcher(DB, embedder=embedder), questions, mode="hybrid", k=10)
    obtenu = r["recall@10"]
    if obtenu != RECALL_NEUTRE_DEV:
        print(f"STATUS: BLOCKED — contrôle de fraîcheur en échec : recall@10 neutre sur "
              f"dev = {obtenu}, attendu {RECALL_NEUTRE_DEV}. Aucun appel API n'a été "
              f"fait. L'index ou le retrieval a changé : ne pas mesurer avant de savoir "
              f"quoi.", file=sys.stderr)
        raise SystemExit(1)
    print(f"[eval_generation] contrôle OK : recall@10 = {obtenu}", flush=True)
    return obtenu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--controle-seul", action="store_true",
                    help="n'exécute que le contrôle de fraîcheur, gratuit")
    args = ap.parse_args()

    embedder = Embedder()
    recall_neutre = controle_fraicheur(embedder)
    if args.controle_seul:
        return

    questions = load_benchmark(ROOT / f"benchmark/{args.split}.jsonl")
    cache_reponses = OUT_DIR / f"reponses_{args.split}.json"
    searcher = Searcher(DB, embedder=embedder, reranker=Reranker(),
                        rewriter=Rewriter(cache_path=CACHE_REECRITURES,
                                          ecrire_cache=False),
                        mode_reecriture=CONFIG["mode_reecriture"])
    generateur = Generator(cache_path=cache_reponses)

    reponses: dict[str, dict] = {}
    for i, q in enumerate(questions, 1):
        passages = searcher.search(q["question"], mode=CONFIG["retrieval"],
                                   k=CONFIG["k"])
        # Le générateur ne voit QUE la question et les passages : ni les citations
        # gold du benchmark, ni ses notes, ni sa catégorie.
        reponses[q["id"]] = generateur.repondre(q["question"], passages)
        print(f"[eval_generation] {i}/{len(questions)} {q['id']} "
              f"abstention={reponses[q['id']]['abstention']} "
              f"citations={len(reponses[q['id']]['citations'])}", flush=True)

    m = metriques(reponses, DB)
    resultat = {
        "split": args.split,
        "n": len(questions),
        "config": CONFIG,
        "controle_fraicheur": {"recall@10_neutre_dev": recall_neutre,
                               "attendu": RECALL_NEUTRE_DEV},
        "modele_generation": generateur.modele,
        "metriques": m,
        "cout": {"appels_api": generateur.appels,
                 "tokens_entree": generateur.tokens_entree,
                 "tokens_sortie": generateur.tokens_sortie},
        "reponses_cache": str(cache_reponses.relative_to(ROOT)),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sortie = OUT_DIR / f"generation_{args.split}.json"
    sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    print(f"\n[eval_generation] === {args.split}, n={len(questions)} ===")
    for cle in ("n_citations", "taux_citations_inexistantes",
                "taux_citations_non_portantes", "taux_reponses_sans_citation",
                "taux_citations_version_omise", "taux_abstention",
                "taux_correspondance_brute"):
        print(f"  {cle:32s} {m[cle]}")
    print(f"  citations_par_reponse            {m['citations_par_reponse']}")
    print(f"[eval_generation] écrit {sortie.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
