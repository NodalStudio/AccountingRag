from pathlib import Path
import pymupdf
from .model import Line


def _merge_spans(spans: list[dict]) -> tuple[str, dict]:
    """Fusionne les spans d'une ligne ; retourne (texte, span_dominant).

    Fix Round 3:
    1. Garde tous les spans (même blancs) pour préserver l'espacement justifié
    2. Spans blancs → séparateurs forcés (« » )
    3. Supprime spans invisibles (size < 2.0 pt)
    4. Puce « o » Courier → normalisée comme puce
    """
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

    # Fusion avec gestion des spans blancs
    text = ""
    for s in visible_spans:
        s_text = s["text"]

        # Span blanc → séparateur forcé
        if not s_text.strip():
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

    text = text.strip()

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
                text, dom = _merge_spans(spans)
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
