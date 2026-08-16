"""CLI d'évaluation : uv run python scripts/run_eval.py --mode all --split dev"""
import argparse
from pathlib import Path
from accounting_rag.evalrag import load_benchmark, evaluate
from accounting_rag.search import Searcher

p = argparse.ArgumentParser()
p.add_argument("--mode", default="all",
               choices=["bm25", "dense", "hybrid", "hybrid+graph", "hybrid+rerank", "all"])
p.add_argument("--split", default="dev", choices=["dev", "test"])
p.add_argument("--k", type=int, default=10)
# Ablation A (T3, jalon 2.5) : pondération par champ. Défauts = valeurs neutres (1.0) —
# ni poids_chemin=2.0 ni boost_commentaire=0.7 n'ont été adoptés (p_amelioration << 0,95
# sur le bootstrap apparié, cf. docs/eval-jalon25.md, section « Ablation A »).
p.add_argument("--poids-chemin", type=float, default=1.0)
p.add_argument("--boost-commentaire", type=float, default=1.0)
args = p.parse_args()

questions = load_benchmark(Path(f"benchmark/{args.split}.jsonl"))
searcher = Searcher(Path("data/corpus.db"),
                     poids_chemin=args.poids_chemin,
                     boost_commentaire=args.boost_commentaire)
# Ablation B (T4, jalon 2.5) : hybrid+rerank ADOPTÉ (bootstrap sur le meilleur des deux
# rerankers mesurés, BAAI/bge-reranker-v2-m3 : p_amelioration=0,952, aucune catégorie
# perdant du recall@10 — cf. docs/eval-jalon25.md, section « Ablation B »). Attention :
# ce mode coûte ~117s/question sur cette machine (reranker lourd, CPU) — une campagne
# --mode all sur les 61 questions du split dev passe de ~1 min à ~2h. Le modèle plus
# léger initialement pressenti (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, ~10s/question)
# a été mesuré et REJETÉ (p_amelioration=0,858) ; il reste disponible via ACCRAG_RERANKER.
modes = (["bm25", "dense", "hybrid", "hybrid+graph", "hybrid+rerank"] if args.mode == "all"
         else [args.mode])

print("| mode | recall@5 | recall@10 | MRR | n |")
print("|---|---|---|---|---|")
for mode in modes:
    m = evaluate(searcher, questions, mode=mode, k=args.k)
    print(f"| {mode} | {m['recall@5']} | {m['recall@10']} | {m['mrr']} | {m['n']} |")
    for cat, v in m["par_categorie"].items():
        print(f"|   ↳ {cat} | | {v['recall@10']} | | {v['n']} |")
