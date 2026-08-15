import re
from collections import Counter
from .model import Record
from .parse import Anomalie

_ROMANS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(s: str) -> int:
    """Convertit un nombre romain en entier."""
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMANS[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def check(records: list[Record]) -> list[Anomalie]:
    """Vérifications d'intégrité post-assemblage du corpus."""
    out: list[Anomalie] = []

    # (a) Doublons d'identifiants
    ids = Counter(r.id for r in records)
    for rid, n in ids.items():
        if n > 1:
            out.append(Anomalie(0, rid, f"identifiant en double ({n}×)", "identifiant en double"))

    # (b) Articles connus pour le contrôle des renvois internes
    known_articles = {r.article for r in records if r.article}

    for r in records:
        # (c) Texte vide
        if not r.texte.strip():
            out.append(Anomalie(r.page_debut, r.id, "texte vide", "texte vide"))

        # (a) Cohérence numérotation ↔ chemin (Ruling 21: TITRE, pas LIVRE)
        # Le 1er chiffre de l'article doit correspondre au numéro du TITRE du chemin
        if r.article and len(r.article.split("-")[0]) == 3:
            # Extraire le numéro du TITRE du chemin (peut être romain ou arabe)
            titre_match = re.search(r"Titre\s+([IVXLC0-9]+)", r.chemin, re.I)
            if titre_match:
                titre_str = titre_match.group(1)
                try:
                    if titre_str.isdigit():
                        titre_num = int(titre_str)
                    else:
                        titre_num = _roman_to_int(titre_str)
                except KeyError:
                    titre_num = None

                if titre_num is not None and int(r.article[0]) != titre_num:
                    out.append(
                        Anomalie(
                            r.page_debut,
                            r.id,
                            f"article {r.article} sous {titre_match.group(0)} : incohérence de Titre",
                            "incohérence de Titre"
                        )
                    )

        # (d) Renvois internes pointant vers un article inexistant (dangling)
        for rv in r.renvois:
            if rv.famille == "interne":
                num = rv.cible.removeprefix("pcg-")
                if num not in known_articles:
                    out.append(
                        Anomalie(
                            r.page_debut, r.id, f"renvoi interne sans cible : {num}",
                            "renvoi interne sans cible"
                        )
                    )

    return out


def report(records: list[Record], anomalies: list[Anomalie]) -> str:
    """Génère un rapport markdown des anomalies, groupé par catégorie stable."""
    n_reg = sum(1 for r in records if r.type == "reglementaire")
    n_com = sum(1 for r in records if r.type == "commentaire_ANC")
    lines = [
        "# Rapport de build du corpus",
        "",
        f"- **{len(records)} enregistrement(s)** : {n_reg} réglementaires, {n_com} commentaires ANC",
        f"- **{len(anomalies)} anomalie(s)**",
        "",
    ]
    by_categorie = Counter(a.categorie for a in anomalies)
    for categorie, n in by_categorie.most_common():
        lines.append(f"## {categorie} ({n})")
        for a in anomalies:
            if a.categorie == categorie:
                lines.append(f"- p.{a.page} — `{a.ligne[:100]}` — {a.raison}")
        lines.append("")
    return "\n".join(lines)
