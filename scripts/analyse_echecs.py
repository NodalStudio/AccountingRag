"""Analyse des échecs recall@10 < 1 sur un split donné, pour un mode donné.

Réutilise `Searcher` (retrieval) et `evalrag` (harnais de mesure) — aucune logique
dupliquée. Pour chaque question dont recall@10 < 1 (au moins une citation gold
absente du top-10 obtenu), affiche :
  - la question, sa catégorie, ses citations gold, le recall@10 obtenu ;
  - le top-10 retourné par `Searcher.search()`, avec une coche sur les hits gold ;
  - le texte des records gold (utile pour lire l'écart de vocabulaire) ;
  - le diff des tokens normalisés : tokens de la question absents du texte gold
    normalisé, et réciproquement — matériau brut pour proposer des entrées
    SYNONYMES ciblées (ruling J2-5 : un terme courant -> son équivalent PCG
    EXACT, jamais un rapprochement de deux concepts distincts).

Sortie markdown sur stdout (rediriger vers un fichier pour la consigner).

Usage :
    uv run python scripts/analyse_echecs.py --split dev --mode hybrid > docs/echecs-dev-jalon25.md
"""
import argparse
from datetime import date
from pathlib import Path

from accounting_rag.evalrag import evaluate, load_benchmark, match
from accounting_rag.normalize import normalize
from accounting_rag.search import Searcher

# Mots fonctionnels français à exclure du diff pour ne garder que les tokens
# porteurs de sens (candidats à un rapprochement de vocabulaire). Calculés via
# normalize() elle-même : le stemming/pliage s'applique identiquement, pas de
# liste de stems à maintenir séparément de la chaîne de normalisation.
_STOPWORDS_SOURCE = [
    "de", "le", "la", "les", "un", "une", "des", "et", "que", "qui", "dans", "sur",
    "pour", "est", "ce", "se", "son", "sa", "ses", "je", "du", "en", "au", "aux",
    "à", "ne", "pas", "plus", "on", "il", "elle", "ils", "elles", "avec", "par",
    "ou", "si", "être", "avoir", "quel", "quelle", "comment", "cette", "ces",
    "mon", "ma", "mes", "notre", "nos", "leur", "leurs", "y", "ai", "j'ai",
    "suis", "sont", "a", "ont", "peut", "peux", "dois", "doit", "faire",
]
STOPWORDS = {tok for w in _STOPWORDS_SOURCE for tok in normalize(w).split()}

_DIFF_MAX = 40  # tronque les diffs trop longs (texte gold complet) pour rester lisible


def _tokens(text: str) -> set[str]:
    return {t for t in normalize(text).split() if t not in STOPWORDS}


def _fmt_diff(tokens: set[str]) -> str:
    if not tokens:
        return "*(vide)*"
    ordered = sorted(tokens)
    truncated = len(ordered) > _DIFF_MAX
    shown = ordered[:_DIFF_MAX]
    out = ", ".join(f"`{t}`" for t in shown)
    if truncated:
        out += f" *(+{len(ordered) - _DIFF_MAX} tronqués)*"
    return out


def _gold_records(con, citations: list[str]) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Pour chaque citation gold, les records (id, article, texte) qui la matchent."""
    all_ids = [r[0] for r in con.execute("SELECT id FROM records").fetchall()]
    out = []
    for c in citations:
        matched_ids = [i for i in all_ids if match(i, c)]
        rows = []
        for rid in matched_ids:
            row = con.execute(
                "SELECT article, texte FROM records WHERE id = ?", (rid,)
            ).fetchone()
            rows.append((rid, row[0], row[1]))
        out.append((c, rows))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    # Ruling J25-7 : la « meilleure config » pour cette tâche est hybrid (pas
    # hybrid+rerank — le canal synonymes n'agit que sur la jambe bm25, et hybrid
    # tourne en ~1 min contre ~2h pour hybrid+rerank).
    p.add_argument("--mode", default="hybrid",
                    choices=["bm25", "dense", "hybrid", "hybrid+graph", "hybrid+rerank"])
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    questions = load_benchmark(Path(f"benchmark/{args.split}.jsonl"))
    searcher = Searcher(Path("data/corpus.db"))
    resultat = evaluate(searcher, questions, mode=args.mode, k=args.k)

    echecs = [q for q in questions if resultat["par_question"][q["id"]] < 1.0]

    print(f"# Analyse des échecs — split `{args.split}`, mode `{args.mode}` ({date.today().isoformat()})")
    print()
    print(f"Recall@10 global : {resultat['recall@10']} ; MRR : {resultat['mrr']} ; n={resultat['n']}.")
    print()
    print(f"Questions en échec (recall@10 < 1) : {len(echecs)}/{len(questions)}.")
    print()
    for cat, v in resultat["par_categorie"].items():
        print(f"- `{cat}` : recall@10={v['recall@10']} (n={v['n']})")
    print()
    print("---")
    print()

    for q in echecs:
        r10 = resultat["par_question"][q["id"]]
        hits = searcher.search(q["question"], k=args.k, mode=args.mode)
        gold = _gold_records(searcher.con, q["citations"])
        gold_ids_matched = {rid for _, rows in gold for rid, _, _ in rows}

        print(f"## {q['id']} ({q['categorie']}) — recall@10={r10:.3f}")
        print()
        print(f"**Question** : {q['question']}")
        print()
        print(f"**Citations gold** : {q['citations']}")
        print()
        print(f"**Top-{args.k} obtenu ({args.mode})** :")
        print()
        for i, h in enumerate(hits, 1):
            marque = " **← gold**" if h["record_id"] in gold_ids_matched else ""
            print(f"{i}. `{h['record_id']}` (source={h['source']}, score={h['score']}){marque}")
        print()
        print("**Records gold** :")
        print()
        for citation, rows in gold:
            if not rows:
                print(f"- `{citation}` : *(aucun record ne matche cette citation dans `records` — anomalie de benchmark)*")
                continue
            for rid, article, texte in rows:
                extrait = texte[:300].replace("\n", " ")
                print(f"- `{rid}` (article {article}) : {extrait}{'…' if len(texte) > 300 else ''}")
        print()

        q_tokens = _tokens(q["question"])
        g_tokens = set()
        for _, rows in gold:
            for _, _, texte in rows:
                g_tokens |= _tokens(texte)

        print(f"**Tokens question absents du(des) record(s) gold** : {_fmt_diff(q_tokens - g_tokens)}")
        print()
        print(f"**Tokens gold absents de la question** : {_fmt_diff(g_tokens - q_tokens)}")
        print()
        print("---")
        print()


if __name__ == "__main__":
    main()
