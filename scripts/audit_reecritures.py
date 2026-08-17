"""Contrôle d'intégrité de l'ablation G : une réécriture par LLM ne doit jamais
introduire un numéro d'article qui n'était pas déjà dans la question.

Motivation (ruling J3-8, jalon 3) : le rewriter ne reçoit que le texte de la question,
jamais les citations attendues — c'est verrouillé par un test structurel. Mais un modèle
qui CONNAÎT le Plan comptable général pourrait citer de lui-même le bon numéro d'article,
ce qui ferait entrer la réponse dans la requête par une autre porte et invaliderait la
mesure : le routeur regex retrouverait alors le gold par référence exacte, et le gain
mesuré ne mesurerait plus le retrieval mais la mémoire du modèle.

Ce script audite les 61 réécritures du cache committé et classe chaque numéro d'article
trouvé en trois catégories :
  - DÉJÀ DANS LA QUESTION : simple recopie, sans effet (le routeur lit de toute façon la
    question ORIGINALE, cf. `Searcher.search`) ;
  - INVENTÉ, HORS GOLD : le modèle cite un article de sa mémoire, mais pas celui attendu —
    bruit, pas fuite ;
  - INVENTÉ ET ÉGAL AU GOLD : FUITE. La mesure de l'ablation G serait invalidée.

Usage : uv run python scripts/audit_reecritures.py
Sortie : tableau Markdown + code de retour 1 si au moins une fuite est détectée.
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
    questions = {q["id"]: q for q in load_benchmark(ROOT / "benchmark/dev.jsonl")}
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    par_texte = {q["question"]: q for q in questions.values()}

    lignes, fuites, recopies, inventes = [], [], 0, 0
    for texte, reecriture in sorted(cache.items()):
        q = par_texte.get(texte)
        if q is None:
            continue  # réécriture d'un autre split : hors périmètre de cet audit
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
    print(f"\n{len(cache)} réécritures auditées ; {len(lignes)} numéro(s) d'article "
          f"trouvé(s) : {recopies} recopié(s) depuis la question, {inventes} inventé(s) "
          f"hors gold, {len(fuites)} fuite(s).")
    if fuites:
        print("\nCONTRÔLE D'INTÉGRITÉ ÉCHOUÉ — la mesure de l'ablation G est invalidée "
              f"pour : {', '.join(f'{qid} ({num})' for qid, num in fuites)}")
        return 1
    print("\nCONTRÔLE D'INTÉGRITÉ OK : aucune réécriture n'introduit le numéro d'article "
          "attendu. Le gain mesuré ne peut pas venir du routeur de référence exacte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
