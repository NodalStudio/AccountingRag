"""CLI d'évaluation : uv run python scripts/run_eval.py --mode all --split dev"""
import argparse
from pathlib import Path
from accounting_rag.evalrag import load_benchmark, evaluate
from accounting_rag.search import Searcher

p = argparse.ArgumentParser()
p.add_argument("--mode", default="all",
               choices=["bm25", "dense", "hybrid", "hybrid+graph", "hybrid+rerank", "all"],
               help="hybrid+rerank : mode lourd (~2 min/question CPU), invoquer explicitement "
                    "(absent de --mode all).")
p.add_argument("--split", default="dev", choices=["dev", "test"])
p.add_argument("--k", type=int, default=10)
# Ablation A (T3, jalon 2.5) : pondération par champ. Défauts = valeurs neutres (1.0) —
# ni poids_chemin=2.0 ni boost_commentaire=0.7 n'ont été adoptés (p_amelioration << 0,95
# sur le bootstrap apparié, cf. docs/eval-jalon25.md, section « Ablation A »).
p.add_argument("--poids-chemin", type=float, default=1.0)
p.add_argument("--boost-commentaire", type=float, default=1.0)
# df_max, pool (T1/T2, jalon 3) : REJETÉS par bootstrap (cf. docs/eval-jalon3.md,
# ablations D et E) — défauts = valeurs neutres (comportement jalon 2.5 inchangé).
# Exposés ici pour permettre une mesure ad hoc sans modifier le code.
p.add_argument("--df-max", type=float, default=None)
p.add_argument("--pool", type=int, default=50)
# n_rerank (T3, jalon 3, ablation F) : nombre de candidats soumis au reranker en mode
# hybrid+rerank. Défaut = valeur adoptée après mesure, cf. docs/eval-jalon3.md, § Ablation F.
p.add_argument("--n-rerank", type=int, default=25)
args = p.parse_args()

questions = load_benchmark(Path(f"benchmark/{args.split}.jsonl"))
searcher = Searcher(Path("data/corpus.db"),
                     poids_chemin=args.poids_chemin,
                     boost_commentaire=args.boost_commentaire,
                     df_max=args.df_max,
                     pool=args.pool,
                     n_rerank=args.n_rerank)
# Ablation B (T4, jalon 2.5) : hybrid+rerank ADOPTÉ (bootstrap sur le meilleur des deux
# rerankers mesurés, BAAI/bge-reranker-v2-m3 : p_amelioration=0,952, aucune catégorie
# perdant du recall@10 — cf. docs/eval-jalon25.md, section « Ablation B »). MAIS ce mode
# coûte ~117s/question sur cette machine (reranker lourd, CPU) : il est volontairement
# EXCLU de --mode all (qui doit rester une campagne rapide, ~1 min) — révision suite
# revue T4 (fix round 1). Invoquer --mode hybrid+rerank explicitement pour l'exercer ;
# aucun chargement du reranker n'a lieu tant que ce mode n'est pas sélectionné (propriété
# lazy). Le modèle plus léger initialement pressenti
# (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, ~10s/question) a été mesuré et REJETÉ
# (p_amelioration=0,858) ; les deux restent sélectionnables via ACCRAG_RERANKER.
modes = ["bm25", "dense", "hybrid", "hybrid+graph"] if args.mode == "all" else [args.mode]

print("| mode | recall@5 | recall@10 | MRR | n |")
print("|---|---|---|---|---|")
for mode in modes:
    m = evaluate(searcher, questions, mode=mode, k=args.k)
    print(f"| {mode} | {m['recall@5']} | {m['recall@10']} | {m['mrr']} | {m['n']} |")
    for cat, v in m["par_categorie"].items():
        print(f"|   ↳ {cat} | | {v['recall@10']} | | {v['n']} |")
