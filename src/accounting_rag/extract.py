from pathlib import Path
import pymupdf
from .model import Line


def _merge_spans(spans: list[dict]) -> tuple[str, dict]:
    """Fusionne les spans d'une ligne ; retourne (texte, span_dominant).

    Règles:
    - Exposants vrais (size < 0.8×dominant ET précédent finit par chiffre): recollés sans espace
    - Autres spans: espaces préservés entre spans
    - Puce en première position: normalisée en « - » + espace + reste
    """
    import re

    dominant = max(spans, key=lambda s: len(s["text"]))

    # Détecte si le premier span est une puce
    _BULLETS = {"•", "-", "–", "*"}
    first_span = spans[0]
    first_text = first_span["text"].strip()
    is_first_bullet = (
        first_span["font"].startswith("Symbol") or first_text in _BULLETS
    ) and len(spans) > 1

    parts: list[str] = []
    for i, s in enumerate(spans):
        t = s["text"]

        # exposant vrais: nettement plus petit ET le span précédent finit par un chiffre (1er, 2ème, etc.)
        is_superscript = (
            s["size"] < 0.8 * dominant["size"]
            and parts
            and re.search(r"\d\s*$", parts[-1])  # précédent finit par chiffre (possiblement suivi d'espace)
        )

        if is_superscript:
            parts[-1] = parts[-1].rstrip() + t.strip()
        else:
            # Préserver les espaces entre spans (sauf recollage d'exposant)
            if parts and not parts[-1].endswith(" ") and not t.startswith(" "):
                parts[-1] += " "
            parts.append(t)

    text = "".join(parts).strip()

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
