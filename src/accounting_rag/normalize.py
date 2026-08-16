"""Chaîne d'analyse lexicale domaine — appliquée à l'identique au build et à la requête."""
import re
import unicodedata
import snowballstemmer

_stemmer = snowballstemmer.stemmer("french")

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_ELISION = re.compile(
    r"\b(?:jusqu|lorsqu|puisqu|quoiqu|presqu|aujourd|qu|[ldjmnstc])'", re.I
)
_REF_LETTREE = re.compile(r"\b([lrd])\.?\s*(\d{1,4}(?:-\d+)+)\b", re.I)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9à-ÿ-]+")
_HAS_DIGIT = re.compile(r"\d")

# Synonymes métier : clé (forme trouvée) -> valeur (forme canonique), en minuscules.
SYNONYMES: dict[str, str] = {
    "fonds de commerce": "fonds commercial",
    "leasing": "credit-bail",
    "location avec option d'achat": "credit-bail",
    "ifc": "indemnites de fin de carriere",
    "indemnite de depart a la retraite": "indemnites de fin de carriere",
    "actif incorporel": "immobilisation incorporelle",
    "actif corporel": "immobilisation corporelle",
    "stock-options": "options de souscription ou d'achat d'actions",
    "goodwill": "fonds commercial",
}
# T5 (jalon 2.5) — lot piloté par l'analyse des échecs dev (docs/echecs-dev-jalon25.md) :
# "aides"->"subventions" et "la/une boite"->"l'/une entite" ont été mesurés par bootstrap
# apparié (paired_bootstrap) sur benchmark/dev.jsonl, mode hybrid, et REJETÉS globalement
# (p_amelioration=0.000, delta=0.0 — les nouveaux tokens n'entrent jamais dans la fenêtre
# top-50 de _bm25() pour les questions ciblées). Voir docs/eval-jalon25.md, section
# « Ablation C », pour le détail (lot proposé, entrées écartées par prudence, mesure).


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    t = text.lower().translate(_APOSTROPHES)
    t = _ELISION.sub("", t)
    t = _REF_LETTREE.sub(lambda m: f"{m.group(1)}{m.group(2)}", t)
    # Vue repliée utilisée uniquement pour détecter les synonymes de façon insensible
    # aux accents (les clés de SYNONYMES sont écrites sans accent). _fold conserve la
    # longueur caractère pour caractère sur le français (une lettre accentuée -> une
    # lettre de base), donc les positions trouvées dans `matching` sont valides dans `t`.
    matching = _fold(t)
    for src, dst in SYNONYMES.items():
        idx = 0
        while True:
            pos = matching.find(src, idx)
            if pos == -1:
                break
            end = pos + len(src)
            t = t[:pos] + dst + t[end:]
            matching = matching[:pos] + dst + matching[end:]
            idx = pos + len(dst)
    # Tokenise sur le texte ACCENTUÉ : le stemmer Snowball français attend des flexions
    # accentuées (ex. "généré"/"génération"/"générer" -> "géner") ; le pliage n'intervient
    # qu'en dernier, sur chaque stem, pour produire des tokens ASCII comparables.
    tokens = [tok for tok in _TOKEN_SPLIT.split(t) if tok and tok != "-"]
    out = []
    for tok in tokens:
        if _HAS_DIGIT.search(tok):
            out.append(tok.strip("-"))
        else:
            out.append(_fold(_stemmer.stemWord(tok)))
    return " ".join(o for o in out if o)
