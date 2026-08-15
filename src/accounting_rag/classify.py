import re
from .model import Line, Kind

# Ruling 23: les amendements ANC récents (cryptoactifs 619-x, adaptations
# sectorielles 111-x/121-x/.../401-x) rédigent l'en-tête en forme LONGUE
# « Article NNN-N » au lieu de la forme abrégée « Art. NNN-N » du texte de
# 2014. Étendu pour couvrir les deux formes — sans risque de faux positif :
# « Article » doit être immédiatement suivi (après espace(s)) d'un CHIFFRE,
# ce qui exclut « Article premier » (lettre) ; les phrases de corps du type
# « ...l'article 628 dispose... » ne sont jamais concernées puisque ART_RE
# n'est testé que sur des lignes en gras de la bande réglementaire qui
# COMMENCENT la ligne (ancre ^), jamais sur du texte à l'intérieur d'une
# phrase.
ART_RE = re.compile(r"^Art(?:\.\s*|icle\s+)\d")
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
