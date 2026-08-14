import re
from .model import Renvoi

_INTERNE = re.compile(r"\barticles?\s+((?:\d{3,4}-\d+(?:-\d+)?)(?:\s*(?:,|et)\s*\d{3,4}-\d+(?:-\d+)?)*)", re.I)
_NUM = re.compile(r"\d{3,4}-\d+(?:-\d+)?")
_EXTERNE = re.compile(r"\barticle\s+(?P<art>[LRD])\.?\s*(?P<num>[\d\-\.]+)\s+du\s+(?P<code>[Cc]ode\s+[a-zé\s]+?)(?=[,;.)]|$)")
_CRC = re.compile(r"règlement\s+n°\s*(\d{2,4}-\d{2})\s+du\s+CRC", re.I)
_AVIS = re.compile(r"Avis\s+(?P<org>CNC|CU)\s+n°\s*(?P<num>\d{4}-[\w]+)", re.I)

_CODE_SLUGS = {
    "code monétaire et financier": "comofi",
    "code de commerce": "code-de-commerce",
    "code général des impôts": "cgi",
}


def _slug_code(code: str) -> str:
    key = " ".join(code.lower().split())
    return _CODE_SLUGS.get(key, key.replace(" ", "-"))


def extract_renvois(texte: str) -> list[Renvoi]:
    out: list[Renvoi] = []
    for m in _INTERNE.finditer(texte):
        for num in _NUM.findall(m.group(1)):
            out.append(Renvoi(f"pcg-{num}", "interne"))
    for m in _EXTERNE.finditer(texte):
        num = m.group("num").rstrip(".").replace(".", "-")
        out.append(Renvoi(f"legi-{m.group('art')}{num}-{_slug_code(m.group('code'))}", "externe_legal"))
    for m in _CRC.finditer(texte):
        out.append(Renvoi(f"crc-{m.group(1)}", "historique"))
    for m in _AVIS.finditer(texte):
        out.append(Renvoi(f"avis-{m.group('org').lower()}-{m.group('num')}", "historique"))
    # dédoublonnage en préservant l'ordre
    seen, uniq = set(), []
    for r in out:
        if (r.cible, r.famille) not in seen:
            seen.add((r.cible, r.famille))
            uniq.append(r)
    return uniq
