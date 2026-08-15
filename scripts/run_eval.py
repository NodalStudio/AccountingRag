"""CLI d'évaluation : uv run python scripts/run_eval.py --mode all --split dev"""
import argparse
from pathlib import Path
from accounting_rag.evalrag import load_benchmark, evaluate
from accounting_rag.search import Searcher

p = argparse.ArgumentParser()
p.add_argument("--mode", default="all", choices=["bm25", "dense", "hybrid", "hybrid+graph", "all"])
p.add_argument("--split", default="dev", choices=["dev", "test"])
p.add_argument("--k", type=int, default=10)
args = p.parse_args()

questions = load_benchmark(Path(f"benchmark/{args.split}.jsonl"))
searcher = Searcher(Path("data/corpus.db"))
modes = ["bm25", "dense", "hybrid", "hybrid+graph"] if args.mode == "all" else [args.mode]

print(f"| mode | recall@5 | recall@10 | MRR | n |")
print(f"|---|---|---|---|---|")
for mode in modes:
    m = evaluate(searcher, questions, mode=mode, k=args.k)
    print(f"| {mode} | {m['recall@5']} | {m['recall@10']} | {m['mrr']} | {m['n']} |")
    for cat, v in m["par_categorie"].items():
        print(f"|   ↳ {cat} | | {v['recall@10']} | | {v['n']} |")
