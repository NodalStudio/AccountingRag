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
    # Correctif positionnel : le folio dupliqué en HAUT de page (taille 10,0,
    # Tahoma non gras) tombe dans la bande réglementaire (9,8-10,3) et produit
    # un faux record dont le texte est un simple numéro de page (ex.
    # pcg-420-8@2026-01-01#2 = « 186 »). Une ligne dont le texte est
    # UNIQUEMENT des chiffres, ÉGAL à son propre numéro de page, ET proche
    # d'une marge (x < 60 OU y < 60) est du bruit.
    #
    # Le deuxième bras (x < 60) couvre les pages pivotées à 90° (annexes en
    # tableau large, ex. p.250-252/311/321-323/377/384) : PyMuPDF y renvoie
    # des coordonnées déjà corrigées de la rotation, donc CE MÊME folio
    # dupliqué s'y retrouve à x≈39-58 (marge de tête dans le référentiel du
    # tableau pivoté) et non plus y<60. Vérifié empiriquement sur les 9
    # occurrences réelles (toutes x<60, y=272.7 EXACTEMENT, valeur = numéro de
    # page, croissant de page en page) — signature d'un artefact de mise en
    # page, pas d'une coïncidence.
    #
    # Cas distingué et PRÉSERVÉ (ne doit jamais devenir BRUIT) : p.518, ligne
    # « 518 » à x=106.2/y=253.8, non gras — un compte réel du plan de comptes
    # (« 518 – Intérêts courus ») qui vaut par coïncidence le numéro de page,
    # sur une page NON pivotée, dans la colonne des codes de compte (x≈106,
    # jamais < 60). x et y restent tous deux ≥ 60 pour ce cas → il survit.
    stripped = line.text.strip()
    if stripped.isdigit() and stripped == str(line.page) and (line.x < 60 or line.y < 60):
        return Kind.BRUIT
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
