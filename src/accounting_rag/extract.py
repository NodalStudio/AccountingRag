import re
from pathlib import Path
import pymupdf
from .model import Line


def _merge_spans(spans: list[dict], dir_vec: tuple[float, float] = (1.0, 0.0)) -> tuple[str, dict]:
    """Fusionne les spans d'une ligne ; retourne (texte, span_dominant).

    Fix Round 3:
    1. Garde tous les spans (même blancs) pour préserver l'espacement justifié
    2. Spans blancs → séparateurs forcés (« » ), SI leur largeur propre le justifie
    3. Supprime spans invisibles (size < 2.0 pt)
    4. Puce « o » Courier → normalisée comme puce

    Fix Round 4:
    - Un span blanc n'agit comme séparateur que si sa largeur propre "le
      long du texte" dépasse 0.15 * taille dominante (mandat initial). Pour
      une ligne horizontale (dir≈(1,0)), cette largeur est bbox x1-x0. Pour
      une ligne pivotée (dir non horizontale, ex: texte tourné à 90°,
      p.461), le bbox x1-x0 mesure l'épaisseur perpendiculaire (constante,
      ~taille de police, pour TOUS les spans de la ligne) et non l'étendue
      le long du texte : c'est bbox y1-y0 qui la mesure sur ces lignes.
      Ce filtre reste défensif (il écarte les spans blancs de largeur
      réellement nulle/négative) mais NE SUFFIT PAS seul à résoudre p.461 :
      vérifié caractère par caractère (get_text("rawdict")) sur le PDF réel,
      le glyphe blanc entre « d' » et « affaires » sur cette ligne pivotée a
      une avance non nulle (≈0.76 × taille — même 2 à 3× plus large qu'un
      espace normal ≈0.28-0.31 × taille observé sur p.34/p.184/p.410/p.420),
      donc PAS "quasi nulle" comme l'hypothèse initiale le supposait. La
      largeur seule ne discrimine donc pas ce cas.
    - Garde-fou complémentaire, vérifié comme la cause réelle et suffisante
      du bug : un span blanc n'insère jamais de séparateur immédiatement
      après une apostrophe d'élision française (’ ' ‘) — un mot élidé
      (« d' », « l' », « qu' »…) n'est jamais suivi d'une espace visible en
      français, quelle que soit la géométrie du span blanc qui suit.
    - Les espaces multiples résultants (ex: un span blanc suivi d'un span
      qui porte déjà un espace de tête) sont normalisés en un seul espace.
    """
    dx, dy = dir_vec
    line_is_horizontal = abs(dx) >= abs(dy)
    # Filtre les spans invisibles (artefacts Word)
    visible_spans = [s for s in spans if s["size"] >= 2.0]
    if not visible_spans:
        return "", spans[0] if spans else {"size": 10.0, "font": "Tahoma"}

    # Span dominant sur spans NON BLANCS seulement
    non_blank_spans = [s for s in visible_spans if s["text"].strip()]
    dominant = max(non_blank_spans, key=lambda s: len(s["text"])) if non_blank_spans else visible_spans[0]

    # Détecte puce « o » Courier en premier span non blanc
    _BULLETS = {"•", "-", "–", "*"}
    first_non_blank = next((s for s in visible_spans if s["text"].strip()), None)
    is_puce_o = (
        first_non_blank
        and first_non_blank["text"].strip() == "o"
        and first_non_blank["font"].startswith("Courier")
        and len(non_blank_spans) > 1
    )

    _ELISIONS = ("'", "’", "‘")

    # Fusion avec gestion des spans blancs
    text = ""
    for s in visible_spans:
        s_text = s["text"]

        # Span blanc → séparateur, seulement si sa largeur propre "le long
        # du texte" (axe déterminé par la direction d'écriture de la ligne)
        # dépasse le seuil mandaté, ET que le texte accumulé ne se termine
        # pas par une apostrophe d'élision (voir docstring)
        if not s_text.strip():
            if line_is_horizontal:
                width = s["bbox"][2] - s["bbox"][0]
            else:
                width = s["bbox"][3] - s["bbox"][1]
            ends_with_elision = bool(text) and text[-1] in _ELISIONS
            if width > 0.15 * dominant["size"] and not ends_with_elision:
                if text and not text[-1].isspace():
                    text += " "
            continue

        # Premier span non blanc
        if not text:
            text = s_text
            continue

        # Décide de l'espacement avant ce span
        sep = ""
        if text and not text[-1].isspace() and s_text[:1] and not s_text[0].isspace():
            # Seuil géométrique: écart > 0.2 * taille dominante → espace
            prev_bbox = visible_spans[visible_spans.index(s) - 1]["bbox"]
            gap = s["bbox"][0] - prev_bbox[2]
            if gap > 0.2 * dominant["size"]:
                sep = " "

        text += sep + s_text

    text = re.sub(r"\s{2,}", " ", text).strip()

    # Normalise puce « o » ou autres glyphes de puce en début
    if is_puce_o or (first_non_blank and (first_non_blank["font"].startswith("Symbol") or first_non_blank["text"].strip() in _BULLETS) and len(non_blank_spans) > 1):
        first_text = first_non_blank["text"].strip()
        rest = text[len(first_text):].lstrip()
        text = "- " + rest if rest else "-"

    return text, dominant


def extract_lines(pdf_path: Path, pages: range | None = None) -> list[Line]:
    doc = pymupdf.open(pdf_path)
    page_nums = pages if pages is not None else range(doc.page_count)
    out: list[Line] = []
    for pno in page_nums:
        d = doc[pno].get_text("dict")
        for block in d["blocks"]:
            if block["type"] != 0:
                continue
            for raw_line in block["lines"]:
                # NE FILTRE PLUS les spans blancs — garde tous les spans pour préserver l'espacement
                spans = raw_line["spans"]
                # Saute seulement si TOUS les spans sont blancs
                if not any(s["text"].strip() for s in spans):
                    continue
                # dir=(0,-1) etc. pour du texte pivoté (ex: en-têtes de
                # tableau tournés à 90°, cf. p.461) : détermine l'axe le
                # long duquel mesurer la largeur des spans blancs
                dir_vec = raw_line.get("dir", (1.0, 0.0))
                text, dom = _merge_spans(spans, dir_vec=dir_vec)
                # Pour x, y, utilise le premier span NON blanc
                first_non_blank = next((s for s in spans if s["text"].strip()), spans[0])
                out.append(Line(
                    text=text,
                    size=round(dom["size"], 1),
                    bold="Bold" in dom["font"],
                    font=dom["font"],
                    x=round(first_non_blank["bbox"][0], 1),
                    y=round(first_non_blank["bbox"][1], 1),
                    page=pno + 1,
                ))
    return out
