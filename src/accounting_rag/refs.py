import re
from .model import Renvoi

# Ruling 19 (F3): un renvoi interne \d{3,4}-\d+ ne doit pas matcher si la
# référence est en réalité un article de code externe — reconnu par la
# présence de « du code »/« du même code » à ≤ 30 caractères après la liste
# complète de numéros (ex. « article 1844-5 du code civil »).
_INTERNE = re.compile(
    r"\barticles?\s+((?:\d{3,4}-\d+(?:-\d+)?)(?:\s*(?:,|et|à)\s*\d{3,4}-\d+(?:-\d+)?)*)"
    r"(?!.{0,30}?\bdu\s+(?:même\s+)?code\b)",
    re.I,
)
_NUM = re.compile(r"\d{3,4}-\d+(?:-\d+)?")

# F1: Whitelist de codes légaux connus (au lieu de capture unbounded)
_CODE_PATTERNS = [
    "code monétaire et financier",
    "code de commerce",
    "code général des impôts",
    "code civil",
    "code de la sécurité sociale",
    "code de la construction et de l'habitation",
    "code des assurances",
    "code de l'urbanisme",
    "code du travail",
    "code du sport",
]
_CODE_ALTS = "|".join(re.escape(c) for c in _CODE_PATTERNS)

# F2: Support articles pluriel avec liste de références
_EXTERNE = re.compile(
    r"\barticles?\s+(?P<list>(?:[LRD]\.\s*[\d\-\.]+(?:\s+(?:et|à|,)\s*[LRD]\.\s*[\d\-\.]+)*)?)\s+du\s+(?P<code>" + _CODE_ALTS + ")",
    re.I
)
_EXTERNE_REFS = re.compile(r"([LRD])\.?\s*([\d\-\.]+)")

# Ruling 19 (F3): alternative externe SANS préfixe [LRD] — numéros au format
# interne (\d{3,4}-\d+) mais suivis de « du code <connu> » (ex. « article
# 1844-5 du code civil » → legi-1844-5-code-civil, pas pcg-1844-5).
_EXTERNE_NOPREFIX = re.compile(
    r"\barticles?\s+(?P<list>(?:\d{3,4}-\d+(?:-\d+)?)(?:\s*(?:et|à|,)\s*\d{3,4}-\d+(?:-\d+)?)*)\s+du\s+(?P<code>"
    + _CODE_ALTS + r")",
    re.I,
)

# F6: CRC pattern avec format "04-01" (alternative ordre inverse "CRC n°")
_CRC = re.compile(r"règlement\s+(?:n°\s*(\d{2,4}-\d{2})\s+du\s+CRC|CRC\s+n°\s*(\d{2,4}-\d{2}))", re.I)
_AVIS = re.compile(r"Avis\s+(?P<org>CNC|CU)\s+n°\s*(?P<num>\d{4}-[\w]+)", re.I)

_CODE_SLUGS = {
    "code monétaire et financier": "comofi",
    "code de commerce": "code-de-commerce",
    "code général des impôts": "cgi",
    "code civil": "code-civil",
    "code de la sécurité sociale": "css",
    "code de la construction et de l'habitation": "cch",
    "code des assurances": "code-des-assurances",
    "code de l'urbanisme": "code-de-l-urbanisme",
    "code du travail": "code-du-travail",
    "code du sport": "code-du-sport",
}


def _slug_code(code: str) -> str:
    key = " ".join(code.lower().split())
    return _CODE_SLUGS.get(key, key.replace(" ", "-"))


def extract_renvois(texte: str) -> list[Renvoi]:
    out: list[Renvoi] = []
    last_code_slug = None  # F5: Track last code seen for "du même code"

    # Références internes (PCG)
    for m in _INTERNE.finditer(texte):
        for num in _NUM.findall(m.group(1)):
            out.append(Renvoi(f"pcg-{num}", "interne"))

    # Ruling 19 (F3): références "article <num> du code <connu>" sans
    # préfixe [LRD] — écartées de _INTERNE par le lookahead ci-dessus,
    # rattachées ici comme renvois externes (numéro brut, sans lettre).
    for m in _EXTERNE_NOPREFIX.finditer(texte):
        code_name = m.group("code").lower()
        code_slug = _slug_code(code_name)
        last_code_slug = code_slug
        for num in _NUM.findall(m.group("list")):
            out.append(Renvoi(f"legi-{num}-{code_slug}", "externe_legal"))

    # Références externes (légales) - F2: Handle plural with lists
    for m in _EXTERNE.finditer(texte):
        code_name = m.group("code").lower()
        code_slug = _slug_code(code_name)
        last_code_slug = code_slug

        # Extract all [LRD]. num pairs from the list
        refs_str = m.group("list")
        if refs_str:
            for ref_m in _EXTERNE_REFS.finditer(refs_str):
                art = ref_m.group(1).upper()
                num = ref_m.group(2).rstrip(".").replace(".", "-")
                out.append(Renvoi(f"legi-{art}{num}-{code_slug}", "externe_legal"))

    # F5: Handle "du même code" references
    same_code_pattern = re.compile(r"\barticles?\s+(?P<list>(?:[LRD]\.\s*[\d\-\.]+(?:\s+(?:et|à|,)\s*[LRD]\.\s*[\d\-\.]+)*)?)\s+du\s+même\s+code", re.I)
    for m in same_code_pattern.finditer(texte):
        if last_code_slug:
            refs_str = m.group("list")
            if refs_str:
                for ref_m in _EXTERNE_REFS.finditer(refs_str):
                    art = ref_m.group(1).upper()
                    num = ref_m.group(2).rstrip(".").replace(".", "-")
                    out.append(Renvoi(f"legi-{art}{num}-{last_code_slug}", "externe_legal"))

    # Références historiques (CRC) - F6: Support both orders
    for m in _CRC.finditer(texte):
        # Group 1 is "règlement n°" order, Group 2 is "CRC n°" order
        crc_num = m.group(1) or m.group(2)
        if crc_num:
            out.append(Renvoi(f"crc-{crc_num}", "historique"))

    # Références historiques (Avis)
    for m in _AVIS.finditer(texte):
        out.append(Renvoi(f"avis-{m.group('org').lower()}-{m.group('num')}", "historique"))

    # Dédoublonnage en préservant l'ordre
    seen, uniq = set(), []
    for r in out:
        if (r.cible, r.famille) not in seen:
            seen.add((r.cible, r.famille))
            uniq.append(r)
    return uniq
