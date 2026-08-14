import re
from .model import Line, Kind

ART_RE = re.compile(r"^Art\.\s*\d")
SECTION_RE = re.compile(r"^(Livre|Titre|Chapitre|Section|Sous-section)\b", re.I)
_BULLETS = {"•", "-", "–", "*"}


def classify(line: Line) -> Kind:
    if line.font.startswith("Symbol") or line.text in _BULLETS:
        return Kind.PUCE
    if line.size <= 9.1:                       # en-têtes, pieds, folios
        return Kind.BRUIT
    if line.size >= 10.4:                      # titres de haut niveau (10.6, 12.0, 11.0, 20.0) — gras ou non
        return Kind.SECTION_HEADER
    if 9.8 <= line.size <= 10.3:               # strate réglementaire (10.0)
        if line.bold:
            if ART_RE.match(line.text):
                return Kind.ARTICLE_HEADER
            if SECTION_RE.match(line.text):
                return Kind.SECTION_HEADER
            return Kind.REGLEMENTAIRE          # gras d'emphase dans le corps
        return Kind.REGLEMENTAIRE
    if 9.3 <= line.size <= 9.7:                # strate commentaire (9.5)
        return Kind.COMMENTAIRE_TITRE if line.bold else Kind.COMMENTAIRE
    return Kind.INCONNU
