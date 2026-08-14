from pathlib import Path
import pymupdf
from .model import Line


def _merge_spans(spans: list[dict]) -> tuple[str, dict]:
    """Fusionne les spans d'une ligne ; retourne (texte, span_dominant).

    Critère GÉOMÉTRIQUE: écart horizontal réel entre spans détermine les séparations.
    - Petites capitales, exposants, indices: géométriquement contigus → collés
    - Blanc typographique (puce→texte, mot→repère): écart mesurable → espace
    """
    dominant = max(spans, key=lambda s: len(s["text"]))

    # Détecte si le premier span est une puce
    _BULLETS = {"•", "-", "–", "*"}
    first_span = spans[0]
    first_text = first_span["text"].strip()
    is_first_bullet = (
        first_span["font"].startswith("Symbol") or first_text in _BULLETS
    ) and len(spans) > 1

    # Fusion basée sur écart géométrique
    text = spans[0]["text"]
    for prev, cur in zip(spans, spans[1:]):
        gap = cur["bbox"][0] - prev["bbox"][2]  # écart horizontal

        # Décide si on ajoute un espace
        sep = ""
        if text and not text[-1].isspace() and cur["text"][:1] and not cur["text"][0].isspace():
            # Seuil géométrique: écart > 0.2 * taille dominante → espace
            if gap > 0.2 * dominant["size"]:
                sep = " "

        text += sep + cur["text"]

    text = text.strip()

    # Normalise puce en première position: « - » + espace + reste
    if is_first_bullet:
        # Enlève le premier span (la puce) et le remplace par « - »
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
                spans = [s for s in raw_line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text, dom = _merge_spans(spans)
                out.append(Line(
                    text=text,
                    size=round(dom["size"], 1),
                    bold="Bold" in dom["font"],
                    font=dom["font"],
                    x=round(spans[0]["bbox"][0], 1),
                    y=round(spans[0]["bbox"][1], 1),
                    page=pno + 1,
                ))
    return out
