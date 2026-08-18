"""Vérification programmatique des citations : aucun LLM, aucun réseau.

C'est la brique la moins truquable du projet, et elle passe volontairement AVANT
le LLM-juge : le juge mesure la qualité d'une réponse, ceci mesure son honnêteté.
La faute la plus grave qu'un RAG comptable puisse commettre est de citer un
article qui n'existe pas, et elle se détecte en SQL.

Trois verdicts possibles par citation, plus un refus de principe :
  - "ok"                  : le record existe et contient l'extrait
  - "record_inexistant"   : citation hallucinée
  - "extrait_absent"      : le record existe mais ne porte pas l'affirmation
  - "extrait_trop_court"  : l'extrait ne prouve rien, on refuse de le valider
"""
import re
import sqlite3
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

# Un extrait plus court que cela matcherait du texte par accident : le valider
# donnerait un faux « ok » et rendrait tout le contrôle inutile.
EXTRAIT_MINIMUM = 30

_ESPACES = re.compile(r"\s+")


def normaliser_pour_comparaison(texte: str) -> str:
    """Replie espaces, casse et forme Unicode — RIEN d'autre.

    Volontairement plus strict que `normalize.normalize()` : on ne déplie pas les
    élisions, on ne stemme pas, on ne replie pas les accents. Un extrait doit être
    verbatim ; seules les différences de mise en page (retours à la ligne, espaces
    doubles du PDF) sont tolérées.
    """
    return _ESPACES.sub(" ", unicodedata.normalize("NFC", texte)).strip().casefold()


def verifier_citation(con: sqlite3.Connection, record_id: str, extrait: str) -> str:
    if len(extrait.strip()) < EXTRAIT_MINIMUM:
        return "extrait_trop_court"
    row = con.execute("SELECT texte FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return "record_inexistant"
    if normaliser_pour_comparaison(extrait) in normaliser_pour_comparaison(row[0]):
        return "ok"
    return "extrait_absent"


def correspond_brut(con: sqlite3.Connection, record_id: str, extrait: str) -> bool:
    """L'extrait figure-t-il dans le texte SANS aucune normalisation ?

    Mesuré à côté du verdict pour rendre visible ce que la normalisation tolère. La sonde
    du jalon (docs/mesures/jalon4/sonde_verbatim.json) donne 77/77 sans normalisation :
    tant que ce taux égale le taux « ok », la tolérance ne fait aucun travail.
    """
    row = con.execute("SELECT texte FROM records WHERE id = ?", (record_id,)).fetchone()
    return row is not None and bool(extrait) and extrait in row[0]


def metriques(reponses: dict[str, dict], db_path: str | Path) -> dict:
    """Agrège les trois taux sur un dict {question_id: sortie de Generator.repondre}."""
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    verdicts: Counter[str] = Counter()
    par_question: dict[str, str] = {}
    details: dict[str, list[dict]] = {}
    n_sans_citation = n_abstentions = n_non_abstentions = n_brut = 0
    nb_citations: list[int] = []

    for qid, r in sorted(reponses.items()):
        if r.get("abstention"):
            n_abstentions += 1
            par_question[qid] = "abstention"
            continue
        n_non_abstentions += 1
        cits = r.get("citations") or []
        if not cits:
            n_sans_citation += 1
            nb_citations.append(0)
            par_question[qid] = "sans_citation"
            continue
        nb_citations.append(len(cits))
        vus = []
        for c in cits:
            v = verifier_citation(con, c["record_id"], c["extrait"])
            brut = correspond_brut(con, c["record_id"], c["extrait"])
            verdicts[v] += 1
            n_brut += int(brut)
            vus.append({"record_id": c["record_id"], "verdict": v,
                        "correspond_brut": brut})
        details[qid] = vus
        # Le verdict de la question est le pire de ses citations : une seule
        # citation hallucinée suffit à rendre la réponse malhonnête.
        for pire in ("record_inexistant", "extrait_absent", "extrait_trop_court", "ok"):
            if any(v["verdict"] == pire for v in vus):
                par_question[qid] = pire
                break
    con.close()

    total_cit = sum(verdicts.values())
    def taux(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    return {
        "n": len(reponses),
        "n_citations": total_cit,
        "verdicts": dict(verdicts),
        "taux_citations_inexistantes": taux(verdicts["record_inexistant"], total_cit),
        "taux_citations_non_portantes": taux(
            verdicts["extrait_absent"] + verdicts["extrait_trop_court"], total_cit),
        "taux_reponses_sans_citation": taux(n_sans_citation, n_non_abstentions),
        "taux_abstention": taux(n_abstentions, len(reponses)),
        # Publié À CÔTÉ du verdict, pas à sa place : l'écart entre ce taux et le taux
        # « ok » est exactement ce que la normalisation tolère.
        "taux_correspondance_brute": taux(n_brut, total_cit),
        "citations_par_reponse": ({
            "min": min(nb_citations), "max": max(nb_citations),
            "median": statistics.median(nb_citations),
            "moyenne": round(statistics.fmean(nb_citations), 2),
        } if nb_citations else None),
        "par_question": par_question,
        "details": details,
    }
