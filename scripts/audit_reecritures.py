"""Contrôle d'intégrité de l'ablation G : une réécriture par LLM ne doit jamais
introduire un numéro d'article qui n'était pas déjà dans la question.

Motivation (ruling J3-8, jalon 3) : le rewriter ne reçoit que le texte de la question,
jamais les citations attendues — c'est verrouillé par un test structurel. Mais un modèle
qui CONNAÎT le Plan comptable général pourrait citer de lui-même le bon numéro d'article,
ce qui ferait entrer la réponse dans la requête et invaliderait la mesure : le gain
mesuré ne mesurerait plus le retrieval mais la mémoire du modèle.

Le canal de fuite est la CORRESPONDANCE LEXICALE ET DENSE sur le token numérique lui-même
(« 214-13 » dans la requête matche le texte de l'article 214-13 dans `chunks_norm`), PAS
le routeur de référence exacte : `Searcher.search()` appelle `_route()` sur la question
ORIGINALE, jamais sur la réécriture, donc un numéro inventé ne peut pas déclencher le
routage. Formulation corrigée après la revue finale de branche, qui a relevé que le modèle
de menace tel qu'énoncé était impossible par construction.

Ce script audite TOUTES les réécritures du cache committé — les deux splits, dev ET le
split gelé — et classe chaque numéro d'article trouvé en trois catégories :
  - DÉJÀ DANS LA QUESTION : simple recopie, sans effet (le routeur lit de toute façon la
    question ORIGINALE, cf. `Searcher.search`) ;
  - INVENTÉ, HORS GOLD : le modèle cite un article de sa mémoire, mais pas celui attendu —
    bruit, pas fuite ;
  - INVENTÉ ET ÉGAL AU GOLD : FUITE. La mesure de l'ablation G serait invalidée.

Le script ÉCHOUE aussi (code 1) si une entrée du cache ne correspond à aucune question des
deux splits : une réécriture non traçable est une anomalie, pas une ligne à sauter en
silence. Défaut corrigé après la revue finale de branche du jalon 3 : la première version
ne chargeait que `dev.jsonl` et sautait les 29 réécritures du split gelé — précisément
celles qui produisent le chiffre mis en avant — tout en affichant `len(cache)` comme
nombre de réécritures auditées.

Usage : uv run python scripts/audit_reecritures.py
Sortie : tableau Markdown + code de retour 1 si une fuite ou une entrée orpheline est
détectée.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from accounting_rag.evalrag import load_benchmark  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "docs/mesures/jalon3/reecritures.json"

# Numéro d'article du PCG : 3 ou 4 chiffres, puis un ou plusieurs segments -N
# (211-1, 214-13, 1222-74...). Les références au code de commerce (L 123-16) sont
# capturées par la même regex sous leur forme numérique (123-16) : c'est voulu, elles
# comptent comme des références et doivent être tracées.
_ARTICLE = re.compile(r"\b(\d{3,4}(?:-\d+)+)\b")


# Verdicts possibles, exposés comme constantes pour que les tests s'y accrochent sans
# recopier les libellés.
RECOPIE = "déjà dans la question"
INVENTE = "inventé, hors gold"
FUITE = "**FUITE — inventé et égal au gold**"


def numeros_golds(citations: list[str]) -> set[str]:
    """Numéros d'article attendus, extraits des citations gold (`pcg-214-13@...` -> `214-13`)."""
    return {c.split("pcg-", 1)[-1].split("@")[0] for c in citations}


def classer(question: str, reecriture: str, citations: list[str]) -> list[tuple[str, str]]:
    """Classe chaque numéro d'article de `reecriture` en RECOPIE / INVENTE / FUITE.

    Fonction pure, testée par `tests/test_audit_reecritures.py` — dont un test injecte
    une fuite synthétique pour vérifier que ce contrôle sait ÉCHOUER. Un contrôle
    d'intégrité que personne n'a vu échouer ne prouve rien (leçon de la tâche 2 de ce
    jalon : un contrôle de reproduction avait validé le bug qu'il devait attraper).
    """
    dans_question = set(_ARTICLE.findall(question))
    golds = numeros_golds(citations)
    out = []
    for num in sorted(set(_ARTICLE.findall(reecriture))):
        if num in dans_question:
            out.append((num, RECOPIE))
        elif num in golds:
            out.append((num, FUITE))
        else:
            out.append((num, INVENTE))
    return out


def main() -> int:
    # Les DEUX splits : une fuite dans le split gelé invaliderait le chiffre de clôture
    # aussi sûrement qu'une fuite dans dev.
    par_texte = {q["question"]: q
                 for split in ("dev", "test")
                 for q in load_benchmark(ROOT / f"benchmark/{split}.jsonl")}
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    lignes, fuites, recopies, inventes, orphelines = [], [], 0, 0, []
    for texte, reecriture in sorted(cache.items()):
        q = par_texte.get(texte)
        if q is None:
            orphelines.append(texte[:60])
            continue
        for num, verdict in classer(texte, reecriture, q["citations"]):
            if verdict == RECOPIE:
                recopies += 1
            elif verdict == FUITE:
                fuites.append((q["id"], num))
            else:
                inventes += 1
            lignes.append(f"| {q['id']} | `{num}` | {verdict} |")

    print(f"| question | numéro trouvé dans la réécriture | verdict |")
    print("|---|---|---|")
    for l in lignes:
        print(l)
    auditees = len(cache) - len(orphelines)
    print(f"\n{auditees} réécriture(s) auditée(s) sur {len(cache)} entrée(s) de cache "
          f"({len(par_texte)} questions dans les deux splits) ; {len(lignes)} numéro(s) "
          f"d'article trouvé(s) : {recopies} recopié(s) depuis la question, {inventes} "
          f"inventé(s) hors gold, {len(fuites)} fuite(s).")
    if orphelines:
        print(f"\nÉCHEC : {len(orphelines)} entrée(s) de cache ne correspondent à aucune "
              f"question des splits — réécriture non traçable : "
              f"{', '.join(repr(o) for o in orphelines)}")
        return 1
    if fuites:
        print("\nCONTRÔLE D'INTÉGRITÉ ÉCHOUÉ — la mesure de l'ablation G est invalidée "
              f"pour : {', '.join(f'{qid} ({num})' for qid, num in fuites)}")
        return 1
    print("\nCONTRÔLE D'INTÉGRITÉ OK : aucune réécriture n'introduit le numéro d'article "
          "attendu. Le gain mesuré ne peut donc pas venir d'une correspondance lexicale "
          "sur le numéro de l'article gold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
