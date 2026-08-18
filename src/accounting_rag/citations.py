"""Vérification programmatique des citations : aucun LLM, aucun réseau.

C'est la brique la moins truquable du projet, et elle passe volontairement AVANT
le LLM-juge : le juge mesure la qualité d'une réponse, ceci mesure son honnêteté.
La faute la plus grave qu'un RAG comptable puisse commettre est de citer un
article qui n'existe pas, et elle se détecte en SQL.

Verdicts possibles par citation :
  - "ok"                  : le record existe et contient l'extrait
  - "record_inexistant"   : citation hallucinée — l'article n'existe pas
  - "version_omise"       : le bon article, mais l'identifiant a perdu son suffixe de
                            version (`pcg-1121-1` au lieu de `pcg-1121-1@2026-01-01`)
  - "version_ambigue"     : identifiant sans version, et l'article en a plusieurs
  - "extrait_absent"      : le record existe mais ne porte pas l'affirmation
  - "extrait_trop_court"  : l'extrait ne prouve rien, on refuse de le valider

Pourquoi `version_omise` existe séparément, et c'est la leçon la plus chère de la
première campagne : elle a d'abord affiché 15,64 % de « citations inexistantes ». Aucune
n'était inventée. Les 69 concernées portaient le bon article avec un extrait verbatim, et
seul le `@2026-01-01` manquait. Confondre « article inventé » et « identifiant abrégé »
gonflait de quinze points la métrique la plus grave du projet. Les deux restent des
défauts — 39 articles du corpus portent plusieurs versions, donc omettre la version est
réellement une perte de traçabilité — mais ce sont deux défauts différents, et un rapport
qui les additionne ne mesure plus rien.
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
# Apostrophes typographiques ramenées à l'ASCII. Ajouté APRÈS la première campagne, sur
# preuve : l'unique citation « non portante » des 422 mesurées sur dev divergeait du texte
# du corpus d'un seul caractère — `l'actif` contre `l’actif` — dans un extrait qui écrivait
# par ailleurs correctement `l’écart d’acquisition`. Une apostrophe droite ne fait pas dire
# autre chose à un article, donc classer cela « non portant » est une erreur de catégorie :
# la métrique existe pour attraper une citation qui prête à un texte un propos qu'il ne
# tient pas. Le repli ne cache rien, `taux_correspondance_brute` publiant en parallèle le
# taux de correspondance SANS aucune normalisation.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "＇": "'"})


def normaliser_pour_comparaison(texte: str) -> str:
    """Replie espaces, casse, forme Unicode et apostrophes — RIEN d'autre.

    Volontairement plus strict que `normalize.normalize()` : on ne déplie pas les
    élisions, on ne stemme pas, on ne replie pas les accents. Un extrait doit être
    verbatim ; seules les différences de mise en page (retours à la ligne, espaces
    doubles du PDF) et de style d'apostrophe sont tolérées.
    """
    plie = unicodedata.normalize("NFC", texte).translate(_APOSTROPHES)
    return _ESPACES.sub(" ", plie).strip().casefold()


def _porte(extrait: str, texte: str) -> bool:
    return normaliser_pour_comparaison(extrait) in normaliser_pour_comparaison(texte)


def _sans_version(con: sqlite3.Connection, record_id: str) -> list[tuple[str, str]]:
    """Records dont l'identifiant est `record_id` suivi d'un suffixe de version.

    Un identifiant versionné faux (`pcg-999-99@2026-01-01`) ne peut pas être rattrapé par
    ce chemin sans garde explicite : le motif deviendrait `pcg-999-99@2026-01-01@%`, et
    aucun identifiant du corpus ne porte deux `@`. Un garde `if "@" in record_id` a été
    écrit puis retiré parce qu'aucune mutation ne pouvait le faire échouer — une ligne
    qu'aucun test ne peut mettre en défaut ne protège rien et se contente de rassurer.
    """
    return con.execute("SELECT id, texte FROM records WHERE id LIKE ?",
                       (record_id + "@%",)).fetchall()


def verifier_citation(con: sqlite3.Connection, record_id: str, extrait: str) -> str:
    if len(extrait.strip()) < EXTRAIT_MINIMUM:
        return "extrait_trop_court"
    row = con.execute("SELECT texte FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is not None:
        return "ok" if _porte(extrait, row[0]) else "extrait_absent"

    versions = _sans_version(con, record_id)
    if not versions:
        return "record_inexistant"
    if len(versions) > 1:
        # On ne choisit pas à la place du modèle : la citation ne dit pas laquelle des
        # versions elle invoque, donc elle n'est pas traçable.
        return "version_ambigue"
    return "version_omise" if _porte(extrait, versions[0][1]) else "extrait_absent"


def correspond_brut(con: sqlite3.Connection, record_id: str, extrait: str) -> bool:
    """L'extrait figure-t-il dans le texte SANS aucune normalisation ?

    Mesuré à côté du verdict pour rendre visible ce que la normalisation tolère. La sonde
    du jalon (docs/mesures/jalon4/sonde_verbatim.json) donne 77/77 sans normalisation :
    tant que ce taux égale le taux « ok », la tolérance ne fait aucun travail.

    Applique le MÊME refus de principe que `verifier_citation` sur les extraits trop
    courts. Sans cela, un extrait de dix caractères présent verbatim serait compté ici et
    refusé là, et l'écart entre les deux taux ne mesurerait plus la normalisation mais un
    mélange des deux causes — ce qui viderait ce compteur de sa raison d'être.
    """
    if len(extrait.strip()) < EXTRAIT_MINIMUM:
        return False
    row = con.execute("SELECT texte FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is not None:
        return extrait in row[0]
    # Résout le suffixe de version comme `verifier_citation`, pour la même raison : sans
    # cela l'écart entre les deux taux mesurerait l'omission de version et non la
    # normalisation, et ce compteur ne dirait plus ce qu'il est là pour dire.
    versions = _sans_version(con, record_id)
    return len(versions) == 1 and extrait in versions[0][1]


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
        # citation hallucinée suffit à rendre la réponse malhonnête. L'ordre descend de la
        # faute la plus grave (article inventé) à la plus bénigne (version omise, qui coûte
        # de la traçabilité mais ne prête aucun propos à un texte qui ne le tient pas).
        for pire in ("record_inexistant", "extrait_absent", "extrait_trop_court",
                     "version_ambigue", "version_omise", "ok"):
            if any(v["verdict"] == pire for v in vus):
                par_question[qid] = pire
                break
    con.close()

    total_cit = sum(verdicts.values())

    def taux(n: int, d: int) -> float | None:
        """`None`, jamais 0,0, quand le dénominateur est nul.

        Sur un split où le système s'abstient partout, il n'y a aucune citation à juger :
        publier « taux de citations inexistantes : 0,0 » se lirait comme un sans-faute
        alors que le taux n'est pas défini. C'est exactement le genre de chiffre juste et
        trompeur que ce dépôt refuse.
        """
        return round(n / d, 4) if d else None

    return {
        "n": len(reponses),
        "n_citations": total_cit,
        "verdicts": dict(verdicts),
        "taux_citations_inexistantes": taux(verdicts["record_inexistant"], total_cit),
        "taux_citations_non_portantes": taux(
            verdicts["extrait_absent"] + verdicts["extrait_trop_court"], total_cit),
        # Publié séparément, jamais additionné aux citations inexistantes : le bon article
        # cité sans son suffixe de version n'est pas un article inventé.
        "taux_citations_version_omise": taux(
            verdicts["version_omise"] + verdicts["version_ambigue"], total_cit),
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
