from pathlib import Path
import pymupdf
from .model import Line


def _merge_spans(spans: list[dict]) -> tuple[str, dict]:
    """Fusionne les spans d'une ligne ; retourne (texte, span_dominant)."""
    dominant = max(spans, key=lambda s: len(s["text"]))
    parts: list[str] = []
    for s in spans:
        t = s["text"]
        # exposant : nettement plus petit que le span dominant -> collé sans espace
        if s["size"] < 0.8 * dominant["size"] and parts:
            parts[-1] = parts[-1].rstrip() + t.strip()
        else:
            parts.append(t)
    return "".join(parts).strip(), dominant


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
