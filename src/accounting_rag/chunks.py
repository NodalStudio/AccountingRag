"""Fenêtrage des records longs pour l'embedding (small-to-big : on retourne le record)."""


def _windows(words: list[str], max_words: int, overlap: int) -> list[str]:
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, max(len(words) - overlap, 1), step)]


def make_chunks(record_id: str, texte: str, max_words: int = 220, overlap: int = 40):
    paras = [p.strip() for p in texte.split("\n") if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for p in paras:
        n = len(p.split())
        if n > max_words:
            if current:
                pieces.append("\n".join(current))
                current, count = [], 0
            pieces.extend(_windows(p.split(), max_words, overlap))
            continue
        if count + n > max_words and current:
            pieces.append("\n".join(current))
            current, count = [], 0
        current.append(p)
        count += n
    if current:
        pieces.append("\n".join(current))
    if not pieces:
        pieces = [texte]
    return [(f"{record_id}::{i}", i, piece) for i, piece in enumerate(pieces)]
