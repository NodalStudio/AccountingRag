import re
from dataclasses import dataclass
from pathlib import Path
from .model import Line, Kind, Record
from .extract import extract_lines
from .classify import classify
from .clean import join_lines
from .refs import extract_renvois

_ART_NUM = re.compile(r"^Art\.\s*(\d{3,4}-\d+(?:-\d+)?)")
_LEVELS = ["livre", "titre", "chapitre", "section", "sous-section"]
_CITATION = re.compile(r"(Avis\s+(?:CNC|CU)\s+n°\s*[\w\-]+[^–]*|règlement\s+n°\s*[\d\-]+\s+du\s+CRC[^–]*)", re.I)
_TOC_DOTS = re.compile(r"\.{4,}")  # entrées de sommaire (points de conduite) — Ruling 7


@dataclass(frozen=True)
class Anomalie:
    page: int
    ligne: str
    raison: str


class _Builder:
    def __init__(self, edition: str):
        self.edition = edition
        self.records: list[Record] = []
        self.anomalies: list[Anomalie] = []
        self.path: dict[str, str] = {}          # niveau -> libellé
        self.cur_article: str | None = None
        self.cur_kind: str | None = None        # "reglementaire" | "commentaire"
        self.buf: list[str] = []
        self.buf_start: int = 0
        self.buf_end: int = 0
        self.com_count: int = 0
        self.com_title: str = ""
        self.seen_ids: set[str] = set()

    def chemin(self) -> str:
        return " > ".join(self.path[lv] for lv in _LEVELS if lv in self.path)

    def flush(self):
        if not self.buf or self.cur_article is None:
            self.buf = []
            return
        texte = join_lines(self.buf)
        if self.cur_kind == "reglementaire":
            rid = f"pcg-{self.cur_article}@{self.edition}"
            rtype, citation = "reglementaire", None
        else:
            self.com_count += 1
            rid = f"pcg-{self.cur_article}-c{self.com_count}@{self.edition}"
            rtype = "commentaire_ANC"
            m = _CITATION.search(self.com_title)
            citation = m.group(1).strip() if m else self.com_title[:200] or None
            texte = (self.com_title + "\n" + texte).strip() if self.com_title else texte
        if rid in self.seen_ids:
            # Collision d'identifiant — le plus souvent un article d'annexe
            # rédigé hors format « Art. NNN-N » (ex. « Article 1er »), qui ne
            # matche jamais _ART_NUM et laisse cur_article figé sur le dernier
            # article régulièrement numéroté. On garantit l'unicité par un
            # suffixe plutôt que d'écraser silencieusement un record existant.
            suffix = 2
            while f"{rid}#{suffix}" in self.seen_ids:
                suffix += 1
            self.anomalies.append(Anomalie(
                self.buf_start, rid,
                "collision d'identifiant — article probablement hors format (Article 1er ?)",
            ))
            rid = f"{rid}#{suffix}"
        self.seen_ids.add(rid)
        self.records.append(Record(
            id=rid, article=self.cur_article, chemin=self.chemin(), texte=texte,
            type=rtype, nature="comptable", opposable=False,
            valide_du=self.edition, valide_au=None, source_citation=citation,
            page_debut=self.buf_start, page_fin=self.buf_end,
            renvois=extract_renvois((self.com_title or "") + " " + texte),
        ))
        self.buf = []
        # Ruling 2: réinitialise le titre de commentaire après émission — sinon
        # les titres de commentaires successifs fusionnent.
        if self.cur_kind == "commentaire":
            self.com_title = ""

    def feed(self, line: Line, kind: Kind):
        if kind == Kind.BRUIT:
            return
        if kind == Kind.SECTION_HEADER:
            # Ruling 7: entrées de sommaire (pages ~5-12) — points de conduite
            # « .... » — ne doivent pas polluer le chemin hiérarchique.
            if _TOC_DOTS.search(line.text):
                return
            self.flush()
            lowered = line.text.lower()
            for lv in _LEVELS:
                if lowered.startswith(lv):
                    self.path[lv] = line.text
                    idx = _LEVELS.index(lv)
                    for deeper in _LEVELS[idx + 1:]:
                        self.path.pop(deeper, None)
                    return
            self.anomalies.append(Anomalie(line.page, line.text, "section sans niveau reconnu"))
            return
        if kind == Kind.ARTICLE_HEADER:
            self.flush()
            m = _ART_NUM.match(line.text)
            if not m:
                self.anomalies.append(Anomalie(line.page, line.text, "en-tête d'article illisible"))
                return
            self.cur_article = m.group(1)
            self.cur_kind = "reglementaire"
            self.com_count = 0
            self.buf_start = self.buf_end = line.page
            # texte éventuel sur la même ligne que « Art. N »
            reste = line.text[m.end():].strip()
            if reste:
                self.buf.append(reste)
            return
        if kind == Kind.COMMENTAIRE_TITRE:
            if self.cur_kind != "commentaire" or self.buf:
                self.flush()
            if self.cur_article is None:
                self.anomalies.append(Anomalie(line.page, line.text, "commentaire orphelin (aucun article ouvert)"))
                return
            if self.cur_kind == "commentaire" and not self.buf:
                self.com_title = (self.com_title + " " + line.text).strip()  # titre multi-lignes
            else:
                self.cur_kind = "commentaire"
                self.com_title = line.text
                self.buf_start = self.buf_end = line.page
            return
        if kind in (Kind.REGLEMENTAIRE, Kind.COMMENTAIRE, Kind.PUCE):
            expected = "reglementaire" if kind == Kind.REGLEMENTAIRE else "commentaire"
            if kind == Kind.PUCE:
                # Ruling 4 (T3-T6): une puce peut arriver isolée comme Kind.PUCE
                # valant exactement "-" ; on la bufferise pour la fusionner à la
                # ligne suivante (REGLEMENTAIRE ou COMMENTAIRE).
                self.buf.append("- ")
                self.buf_end = line.page
                return
            if self.cur_article is None:
                self.anomalies.append(Anomalie(line.page, line.text, "texte avant tout article (préambule ?)"))
                return
            if self.cur_kind != expected:
                self.flush()
                self.cur_kind = expected
                if expected == "commentaire":
                    self.com_title = ""
                self.buf_start = line.page
            if self.buf and self.buf[-1] == "- ":
                self.buf[-1] = "- " + line.text
            else:
                self.buf.append(line.text)
            self.buf_end = line.page
            return
        self.anomalies.append(Anomalie(line.page, line.text, f"ligne inclassable ({line.size}/{line.font})"))


def parse(pdf_path: Path, edition: str = "2026-01-01") -> tuple[list[Record], list[Anomalie]]:
    b = _Builder(edition)
    for line in extract_lines(pdf_path):
        b.feed(line, classify(line))
    b.flush()
    return b.records, b.anomalies
