"""Imprime les passages remontés pour chaque question d'abstention, pour lecture.

Aucune métrique : le recall n'a aucun sens sur un split dont les citations sont vides par
construction, et `evalrag.evaluate` y lèverait une ZeroDivisionError. Ce que ce script
sert à vérifier est qualitatif et ne s'automatise pas — qu'aucun des passages remontés ne
réponde réellement à la question. Une question d'abstention à laquelle le corpus répond
est un piège invalide et doit être retirée du split.

La liste des passages est écrite dans `docs/mesures/jalon4/passages_abstention.json` pour
que la vérification soit rejouable et que le choix de retirer ou garder une question soit
traçable.

Cache de réécriture : `data/reecritures-cache.json` (gitignoré). L'ancrage versionné du
jalon 3 n'est PAS ouvert ici : ces questions n'y figurent pas et il ne doit pas grossir
hors d'une campagne.

Usage : uv run python scripts/inspecter_abstention.py [--n-passages 10]
"""
import argparse
import json
from pathlib import Path

from accounting_rag.embed import Embedder
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT = ROOT / "docs/mesures/jalon4/passages_abstention.json"
CACHE_REECRITURES = ROOT / "data/reecritures-cache.json"


def charger() -> list[dict]:
    lignes = (ROOT / "benchmark/abstention.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lignes if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-passages", type=int, default=10)
    args = ap.parse_args()

    questions = charger()
    searcher = Searcher(DB, embedder=Embedder(), reranker=Reranker(),
                        rewriter=Rewriter(cache_path=CACHE_REECRITURES),
                        mode_reecriture="etend")

    sortie = []
    for q in questions:
        passages = searcher.search(q["question"], mode="hybrid+rerank",
                                   k=args.n_passages)
        print("=" * 100)
        print(f"{q['id']} [{q['raison']}] {q['question']}")
        vus = []
        for r in passages:
            extrait = " ".join(r["texte"].split())[:240]
            print(f"  {r['record_id']:30s} {extrait}")
            vus.append({"record_id": r["record_id"], "article": r["article"],
                        "chemin": r["chemin"], "extrait": extrait})
        sortie.append({"id": q["id"], "raison": q["raison"],
                       "question": q["question"], "passages": vus})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_questions": len(questions),
                               "n_passages": args.n_passages,
                               "questions": sortie},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[inspecter_abstention] écrit {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
