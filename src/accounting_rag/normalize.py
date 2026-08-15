"""Chaîne d'analyse lexicale domaine — appliquée à l'identique au build et à la requête."""
import re
import unicodedata
import snowballstemmer

_stemmer = snowballstemmer.stemmer("french")

_APOSTROPHES = str.maketrans({"'": "'", "'": "'", "ʼ": "'"})
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
    "stock-options": "options de souscription d'actions",
    "goodwill": "fonds commercial",
    "amortissement degressif": "amortissement derogatoire",
}


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    t = text.lower().translate(_APOSTROPHES)
    t = _ELISION.sub("", t)
    t = _REF_LETTREE.sub(lambda m: f"{m.group(1)}{m.group(2)}", t)
    folded = _fold(t)
    for src, dst in SYNONYMES.items():
        folded = folded.replace(_fold(src), _fold(dst))
    tokens = [tok for tok in _TOKEN_SPLIT.split(folded) if tok and tok != "-"]
    out = []
    for tok in tokens:
        if _HAS_DIGIT.search(tok):
            out.append(tok.strip("-"))
        else:
            out.append(_fold(_stemmer.stemWord(tok)))
    return " ".join(o for o in out if o)
