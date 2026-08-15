import re
from dataclasses import dataclass
from pathlib import Path
from .model import Line, Kind, Record
from .extract import extract_lines
from .classify import classify
from .clean import join_lines
from .refs import extract_renvois

# Ruling 23: même extension que ART_RE (classify.py) — les en-têtes en forme
# longue « Article NNN-N » (amendements ANC récents : cryptoactifs 619-x,
# adaptations sectorielles 111-x/121-x/.../401-x) doivent être capturés au
# même titre que la forme abrégée « Art. NNN-N ».
_ART_NUM = re.compile(r"^Art(?:\.\s*|icle\s+)(\d{3,4}-\d+(?:-\d+)?)")
_LEVELS = ["livre", "titre", "chapitre", "section", "sous-section"]
_CITATION = re.compile(r"(Avis\s+(?:CNC|CU)\s+n°\s*[\w\-]+[^–]*|règlement\s+n°\s*[\d\-]+\s+du\s+CRC[^–]*)", re.I)
_TOC_DOTS = re.compile(r"\.{4,}")  # entrées de sommaire (points de conduite) — Ruling 7


@dataclass(frozen=True)
class Anomalie:
    page: int
    ligne: str
    raison: str
    categorie: str  # catégorie stable pour groupage (ex: "renvoi interne sans cible", "incohérence de Titre")


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
        self.na_count: int = 0          # Ruling 18: compteur global des records ancrés à une section (sans article)
        self.last_level: str | None = None          # F2: dernier niveau de path posé par un SECTION_HEADER
        self.last_was_section_header: bool = False  # Ruling 20: la ligne PRÉCÉDENTE (toute nature, BRUIT compris) était-elle un SECTION_HEADER ?
        self.last_page: int | None = None           # Ruling 20: page de cette ligne précédente

    def chemin(self) -> str:
        return " > ".join(self.path[lv] for lv in _LEVELS if lv in self.path)

    def _check_orphan_suspect(self, line: Line):
        # Ruling 18: le contenu hors article n'est plus une anomalie-poubelle
        # (il devient un record pcg-na-<n>) — sauf cas suspect : aucun chemin
        # de section ET hors des 14 premières pages (avant-propos/sommaire).
        if self.cur_article is None and not self.chemin() and line.page > 14:
            self.anomalies.append(Anomalie(
                line.page, line.text, "texte hors article et hors section (cas suspect)",
                "cas suspect"
            ))

    def flush(self):
        if not self.buf:
            self.buf = []
            return
        texte = join_lines(self.buf)
        if self.cur_kind == "reglementaire":
            if self.cur_article is not None:
                rid = f"pcg-{self.cur_article}@{self.edition}"
            else:
                self.na_count += 1
                rid = f"pcg-na-{self.na_count}@{self.edition}"
            rtype, citation = "reglementaire", None
        else:
            if self.cur_article is not None:
                self.com_count += 1
                rid = f"pcg-{self.cur_article}-c{self.com_count}@{self.edition}"
            else:
                self.na_count += 1
                rid = f"pcg-na-{self.na_count}@{self.edition}"
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
                "collision d'identifiant"
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
        # Ruling 20: l'adjacence stricte requise pour la concaténation d'un
        # titre multi-lignes se calcule sur CHAQUE ligne du flux, BRUIT
        # compris — une ligne de bruit intercalée (ex. p.101-102 : 12 lignes
        # de tableau classées BRUIT entre « I Coûts internes » et
        # « II Coûts externes », deux titres DISTINCTS) invalide la
        # continuation. On capture l'état "ligne précédente" AVANT de le
        # mettre à jour pour la ligne courante.
        prev_was_section_header = self.last_was_section_header and self.last_page == line.page
        self.last_was_section_header = (kind == Kind.SECTION_HEADER)
        self.last_page = line.page

        if kind == Kind.BRUIT:
            return
        if kind == Kind.SECTION_HEADER:
            # Ruling 7: entrées de sommaire (pages ~5-12) — points de conduite
            # « .... » — ne doivent pas polluer le chemin hiérarchique.
            # Ruling 20 (correctif angle mort) : une ligne de sommaire ignorée
            # ne doit PAS se faire passer pour un vrai titre — elle rompt
            # l'adjacence (comme le BRUIT) mais ne l'établit JAMAIS. Sans ce
            # correctif, le passage au sommet de feed() l'aurait déjà classée
            # last_was_section_header = True juste avant cette garde.
            if _TOC_DOTS.search(line.text):
                self.last_was_section_header = False
                return
            lowered = line.text.lower()
            starts_with_level = any(lowered.startswith(lv) for lv in _LEVELS)
            # F2/Ruling 20: un SECTION_HEADER sans mot-clé de niveau,
            # IMMÉDIATEMENT successeur d'un autre SECTION_HEADER (aucune
            # ligne intercalée d'aucune sorte, BRUIT compris) ET sur la même
            # page, est la SUITE du titre précédent — concatène-le au dernier
            # niveau posé au lieu de l'anomalie. Sinon (adjacence non
            # satisfaite), comportement d'avant F2 : anomalie.
            if not starts_with_level and prev_was_section_header and self.last_level is not None:
                self.path[self.last_level] = self.path[self.last_level] + " " + line.text
                return
            self.flush()
            # Ruling 17: une SECTION_HEADER ferme le périmètre de l'article
            # courant — aucun article ne se poursuit après un titre de
            # section intercalé (structure du Recueil vérifiée).
            self.cur_article = None
            self.cur_kind = None
            self.com_title = ""
            self.com_count = 0
            for lv in _LEVELS:
                if lowered.startswith(lv):
                    self.path[lv] = line.text
                    idx = _LEVELS.index(lv)
                    for deeper in _LEVELS[idx + 1:]:
                        self.path.pop(deeper, None)
                    self.last_level = lv
                    return
            self.anomalies.append(Anomalie(line.page, line.text, "section sans niveau reconnu", "section sans niveau reconnu"))
            return
        if kind == Kind.ARTICLE_HEADER:
            self.flush()
            m = _ART_NUM.match(line.text)
            if not m:
                self.anomalies.append(Anomalie(line.page, line.text, "en-tête d'article illisible", "en-tête d'article illisible"))
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
            # F1: décide AVANT de flusher si ce titre est un nouveau titre
            # (je viens de flusher un commentaire déjà pourvu d'un corps, ou
            # je change de strate) ou la suite d'un titre multi-lignes (le
            # titre précédent n'a pas encore de corps ET je ne flush pas).
            # flush() vide self.buf, donc ce calcul doit précéder l'appel.
            just_flushed = self.cur_kind != "commentaire" or bool(self.buf)
            if just_flushed:
                self.flush()
            # Ruling 18: un titre de commentaire sans article ouvert n'est
            # plus rejeté en anomalie — il devient le titre d'un record
            # ancré à la section (pcg-na-<n>), ex. annexes d'exemples.
            self._check_orphan_suspect(line)
            if just_flushed:
                # Nouveau titre : réinitialise le titre ET les pages du
                # buffer sur la ligne courante — sinon le nouveau commentaire
                # hérite des pages de l'ancien (F1).
                self.cur_kind = "commentaire"
                self.com_title = line.text
                self.buf_start = self.buf_end = line.page
            else:
                self.com_title = (self.com_title + " " + line.text).strip()  # titre multi-lignes
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
            # Ruling 18: du texte réglementaire/commentaire sans article
            # ouvert n'est plus rejeté en anomalie — il devient un record
            # ancré à la section courante (pcg-na-<n>).
            self._check_orphan_suspect(line)
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
        self.anomalies.append(Anomalie(line.page, line.text, f"ligne inclassable ({line.size}/{line.font})", "ligne inclassable"))


def parse(pdf_path: Path, edition: str = "2026-01-01") -> tuple[list[Record], list[Anomalie]]:
    b = _Builder(edition)
    for line in extract_lines(pdf_path):
        b.feed(line, classify(line))
    b.flush()
    return b.records, b.anomalies
