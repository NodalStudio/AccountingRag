"""Diagnostic outillé : rang du record gold par canal (lexical filtré ou non, dense).

Outil versionné remplaçant les sondes jetables du diagnostic fondateur du jalon 3
(`sonde_discriminance.py`, `sonde_dense.py`, non versionnées) : pour chaque question
demandée, calcule et affiche le rang (1-indexé, records dédupliqués) du premier record
gold dans TROIS classements complets (sans troncature `LIMIT`, contrairement à
`Searcher.search()` en production qui plafonne `_bm25()`/`_dense()` à 50 candidats) :

  1. lexical NON filtré  : `df_max=None`, comportement jalon 2.5 (OR de tous les
     tokens normalisés de la requête).
  2. lexical FILTRÉ      : `df_max=<seuil>` (T1, jalon 3) — les tokens trop fréquents
     (mots fonctionnels) sont écartés du MATCH avant classement.
  3. dense                : classement bm25 remplacé par la distance vectorielle
     (`Searcher._dense`, non tronquée — `limit=n_chunks`).

But : vérifier, sans reconstruire l'index, si le filtrage lexical (ou le canal dense)
repêche un record gold qui se classe dernier en lexical non filtré faute de token
discriminant commun avec la question — c'est le diagnostic qui motive `df_max`
(cf. `Searcher._termes_match`, docs/eval-jalon3.md).

Coût : le canal dense charge le modèle d'embeddings (`Embedder`, e5-small) au premier
appel, ~50 s sur cette machine (CPU). Le canal lexical est quasi instantané (SQLite
FTS5). Le chargement n'a lieu qu'une fois pour toutes les questions demandées (`Embedder`
partagé). Chaque canal exécute une requête SQL SANS LIMIT sur toute la base (`ORDER BY`
sans troncature) : coût proportionnel au nombre de chunks du corpus, pas au nombre de
questions.

Usage :
    uv run python scripts/diagnostic_rangs.py --questions q021,q026,q060,q023 \
        [--split dev] [--df-max 0.02]

Écrit `docs/mesures/jalon3/diagnostic_rangs.json` (une entrée par question demandée) et
imprime un tableau markdown sur stdout.
"""
import argparse
import json
from pathlib import Path

from accounting_rag.evalrag import load_benchmark, match
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon3"


def _rang_gold_lexical(ranked: list[str], gold: str) -> tuple[int | None, int]:
    """Rang du premier id de `ranked` qui matche `gold`, convention `sonde_discriminance.py`.

    Cette sonde s'arrêtait dès que le gold était trouvé (`return rang, len(vus)` DANS
    la boucle) : à cet instant précis, le nombre de records distincts scannés (`vus`)
    est TOUJOURS égal au rang lui-même (ils sont incrémentés ensemble à chaque record
    distinct). D'où les paires "154/154", "46/46", "1430/1430" du diagnostic fondateur —
    le second nombre n'est PAS la taille du classement complet, c'est le rang lui-même
    répété. Convention conservée à l'identique pour reproduire ces chiffres bit à bit
    (contrôle de l'outil, cf. docs/eval-jalon3.md). Si le gold est absent, la boucle va
    jusqu'au bout : le second élément est alors la taille réelle de `ranked`.
    """
    for rang, rid in enumerate(ranked, start=1):
        if match(rid, gold):
            return rang, rang
    return None, len(ranked)


def _rang_gold_dense(ranked: list[str], gold: str) -> tuple[int | None, int]:
    """Rang du premier id de `ranked` qui matche `gold`, convention `sonde_dense.py`.

    Cette sonde ne s'arrêtait PAS à la première correspondance (elle continuait à
    parcourir tous les rows pour compter `records_distincts`) : le second élément
    retourné est donc la taille réelle du classement complet dédupliqué (ex. 1660,
    le nombre total de records du corpus), pas le rang. Convention distincte de
    `_rang_gold_lexical` ci-dessus, conservée à l'identique par canal.
    """
    rang = None
    for i, rid in enumerate(ranked, start=1):
        if rang is None and match(rid, gold):
            rang = i
    return rang, len(ranked)


def _classement_lexical(searcher: Searcher, query: str, df_max: float | None) -> list[str]:
    """Classement lexical complet (sans troncature `LIMIT`), à `df_max` donné.

    Bascule temporairement `searcher.df_max` (attribut d'instance lu par
    `_termes_match`) plutôt que d'instancier un second `Searcher` — évite un second
    chargement de connexion pour une bascule de paramètre pur.
    """
    ancien = searcher.df_max
    searcher.df_max = df_max
    try:
        scores = searcher._bm25(query, limit=searcher.n_chunks)
    finally:
        searcher.df_max = ancien
    return sorted(scores, key=scores.get, reverse=True)


def _classement_dense(searcher: Searcher, query: str) -> list[str]:
    """Classement dense complet (sans troncature `LIMIT` — `limit=n_chunks`)."""
    scores = searcher._dense(query, limit=searcher.n_chunks)
    return sorted(scores, key=scores.get, reverse=True)


def diagnostiquer(searcher: Searcher, questions: dict[str, dict], qids: list[str],
                   df_max: float) -> dict[str, dict]:
    resultats = {}
    for qid in qids:
        q = questions[qid]
        query, gold = q["question"], q["citations"][0]

        classement_brut = _classement_lexical(searcher, query, df_max=None)
        rang_brut, total_brut = _rang_gold_lexical(classement_brut, gold)

        classement_filtre = _classement_lexical(searcher, query, df_max=df_max)
        rang_filtre, total_filtre = _rang_gold_lexical(classement_filtre, gold)

        classement_dense = _classement_dense(searcher, query)
        rang_dense, total_dense = _rang_gold_dense(classement_dense, gold)

        resultats[qid] = {
            "gold": gold,
            "categorie": q["categorie"],
            "lexical_non_filtre": {"rang": rang_brut, "candidats": total_brut},
            "lexical_filtre": {"df_max": df_max, "rang": rang_filtre, "candidats": total_filtre},
            "dense": {"rang": rang_dense, "candidats": total_dense},
        }
    return resultats


def _fmt(entree: dict) -> str:
    rang, total = entree["rang"], entree["candidats"]
    return f"{rang}/{total}" if rang is not None else f"ABSENT/{total}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", required=True,
                   help="ids séparés par des virgules, ex. q021,q026,q060,q023")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--df-max", type=float, default=0.02,
                   help="seuil de fréquence documentaire pour le classement lexical filtré "
                        "(défaut 0.02, celui du diagnostic fondateur)")
    args = p.parse_args()

    qids = args.questions.split(",")
    questions = {q["id"]: q for q in load_benchmark(ROOT / f"benchmark/{args.split}.jsonl")}
    manquants = [qid for qid in qids if qid not in questions]
    if manquants:
        raise SystemExit(f"questions absentes de {args.split}.jsonl : {manquants}")

    print(f"[diagnostic_rangs] chargement de l'embedder (~50 s)...", flush=True)
    searcher = Searcher(DB)
    print(f"[diagnostic_rangs] corpus : {searcher.n_chunks} chunks", flush=True)

    resultats = diagnostiquer(searcher, questions, qids, df_max=args.df_max)

    print(f"\n| question | catégorie | gold | lexical (non filtré) | "
          f"lexical (df≤{args.df_max*100:g}%) | dense |")
    print("|---|---|---|---|---|---|")
    for qid, r in resultats.items():
        print(f"| {qid} | {r['categorie']} | {r['gold']} | "
              f"{_fmt(r['lexical_non_filtre'])} | {_fmt(r['lexical_filtre'])} | "
              f"{_fmt(r['dense'])} |")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "diagnostic_rangs.json"
    payload = {"split": args.split, "n_chunks": searcher.n_chunks, "df_max": args.df_max,
               "resultats": resultats}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[diagnostic_rangs] écrit {out_path}")


if __name__ == "__main__":
    main()
