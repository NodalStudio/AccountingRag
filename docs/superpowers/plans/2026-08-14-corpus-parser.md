# Corpus v1 — Parseur du Recueil ANC → SQLite : Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformer le Recueil des normes comptables françaises 2026 (PDF, 662 p.) en un dataset SQLite structuré (articles réglementaires + commentaires ANC + renvois), avec rapport d'anomalies, via un parseur déterministe guidé par la typographie.

**Architecture:** Pipeline en étages purs et testables : extraction PyMuPDF (lignes typées) → classification typographique → nettoyage → assemblage hiérarchique (machine à états) → extraction des renvois → vérifications d'intégrité → écriture SQLite. Aucun LLM dans le pipeline ; les anomalies sont rapportées, pas devinées.

**Tech Stack:** Python ≥3.12 via `uv`, PyMuPDF (`pymupdf`), sqlite3 (stdlib), pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-accountingrag-design.md` (sections 2, 3, 7)

## Global Constraints

- Gestionnaire : `uv` (pas de pip global). Lancer les tests : `uv run pytest`.
- Le PDF source n'est PAS commité (gitignoré) ; il est téléchargé par script depuis anc.gouv.fr.
- Texte en UTF-8, accents préservés partout.
- Pipeline 100 % déterministe et rejouable ; toute ligne inclassable devient une anomalie rapportée, jamais une supposition.
- Signatures typographiques de référence (sondage empirique du 2026-08-14, voir JOURNAL.md) : réglementaire = taille 10,0 ; commentaire = 9,5 ; en-tête d'article = gras 10,0 « Art. N » ; titre de commentaire = gras 9,5 ; sections = gras ≥10,5 ou gras 10,0 à mot-clé ; bruit (en-tête/pied) = ≤9,0.
- Étapes marquées **[mécanique]** : exécutables par un sous-agent économique (Haiku/Sonnet).
- À la fin de chaque tâche : commit. À la fin du plan : mise à jour de `JOURNAL.md`.

---

### Task 1: Scaffolding du projet + téléchargement des données

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/accounting_rag/__init__.py`, `scripts/download_data.py`, `tests/conftest.py`

**Interfaces:**
- Produces: fixture pytest `recueil_path` (Path vers `data/raw/recueil-pcg-2026.pdf`, skip si absent) ; constante `RECUEIL_URL` et fonction `download(dest: Path) -> Path` dans `scripts/download_data.py`.

- [ ] **Step 1: Créer la structure**

```toml
# pyproject.toml
[project]
name = "accounting-rag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pymupdf>=1.24"]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/accounting_rag"]
```

```gitignore
# .gitignore
data/raw/
*.db
__pycache__/
.pytest_cache/
.venv/
```

`src/accounting_rag/__init__.py` : vide.

- [ ] **Step 2: Script de téléchargement**

```python
# scripts/download_data.py
"""Télécharge le Recueil des normes comptables françaises depuis anc.gouv.fr."""
import urllib.request
from pathlib import Path

RECUEIL_URL = (
    "https://www.anc.gouv.fr/files/anc/files/"
    "1_Normes_fran%C3%A7aises/recueil/2026/Recueil-PCG-Janvier-2026.pdf"
)
DEST = Path("data/raw/recueil-pcg-2026.pdf")


def download(dest: Path = DEST) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(RECUEIL_URL, dest)
    return dest


if __name__ == "__main__":
    p = download()
    print(f"OK: {p} ({p.stat().st_size / 1e6:.1f} Mo)")
```

- [ ] **Step 3: Fixture partagée**

```python
# tests/conftest.py
from pathlib import Path
import pytest

RECUEIL = Path("data/raw/recueil-pcg-2026.pdf")


@pytest.fixture(scope="session")
def recueil_path() -> Path:
    if not RECUEIL.exists():
        pytest.skip("PDF absent — lancer scripts/download_data.py")
    return RECUEIL
```

- [ ] **Step 4: Télécharger et vérifier** — Run: `uv run python scripts/download_data.py` puis `uv run python -c "import pymupdf; d=pymupdf.open('data/raw/recueil-pcg-2026.pdf'); print(d.page_count)"`. Expected: `662`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src scripts tests
git commit -m "chore: scaffolding uv + téléchargement du Recueil 2026"
```

---

### Task 2: Modèle de données (`model.py`)

**Files:**
- Create: `src/accounting_rag/model.py`, `tests/test_model.py`

**Interfaces:**
- Produces:
  - `Line(text: str, size: float, bold: bool, font: str, x: float, y: float, page: int)` (frozen dataclass)
  - `Kind` (Enum: `REGLEMENTAIRE, COMMENTAIRE, ARTICLE_HEADER, COMMENTAIRE_TITRE, SECTION_HEADER, PUCE, BRUIT, INCONNU`)
  - `Renvoi(cible: str, famille: str)` — famille ∈ `{"interne", "externe_legal", "historique"}`
  - `Record(id, article, chemin, texte, type, nature, opposable, valide_du, valide_au, source_citation, page_debut, page_fin, renvois: list[Renvoi])` avec `Record.to_dict()`.

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_model.py
from accounting_rag.model import Record, Renvoi


def test_record_to_dict():
    r = Record(
        id="pcg-212-5@2026-01-01", article="212-5",
        chemin="Livre II > Titre I > Chapitre II", texte="Le titulaire…",
        type="reglementaire", nature="comptable", opposable=False,
        valide_du="2026-01-01", valide_au=None, source_citation=None,
        page_debut=40, page_fin=40,
        renvois=[Renvoi("legi-L313-7-comofi", "externe_legal")],
    )
    d = r.to_dict()
    assert d["id"] == "pcg-212-5@2026-01-01"
    assert d["renvois"] == [{"cible": "legi-L313-7-comofi", "famille": "externe_legal"}]
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_model.py -v`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/model.py
from dataclasses import dataclass, field, asdict
from enum import Enum, auto


@dataclass(frozen=True)
class Line:
    text: str
    size: float
    bold: bool
    font: str
    x: float
    y: float
    page: int  # 1-indexé


class Kind(Enum):
    REGLEMENTAIRE = auto()
    COMMENTAIRE = auto()
    ARTICLE_HEADER = auto()
    COMMENTAIRE_TITRE = auto()
    SECTION_HEADER = auto()
    PUCE = auto()
    BRUIT = auto()
    INCONNU = auto()


@dataclass(frozen=True)
class Renvoi:
    cible: str
    famille: str  # interne | externe_legal | historique


@dataclass
class Record:
    id: str
    article: str | None
    chemin: str
    texte: str
    type: str          # reglementaire | commentaire_ANC
    nature: str        # comptable
    opposable: bool
    valide_du: str
    valide_au: str | None
    source_citation: str | None
    page_debut: int
    page_fin: int
    renvois: list[Renvoi] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["renvois"] = [asdict(r) for r in self.renvois]
        return d
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_model.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: modèle de données (Line, Kind, Record, Renvoi)"`

---

### Task 3: Extraction des lignes typées (`extract.py`)

**Files:**
- Create: `src/accounting_rag/extract.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `Line` (Task 2).
- Produces: `extract_lines(pdf_path: Path, pages: range | None = None) -> list[Line]` — une `Line` par ligne visuelle, spans fusionnés (exposants recollés sans espace : « 1 » + « er » → « 1er »), `size`/`bold`/`font` = ceux du span dominant (plus grand nombre de caractères), `bold` vrai si « Bold » dans le nom de police du span dominant.

- [ ] **Step 1: Test qui échoue (sur le vrai PDF, page 40)**

```python
# tests/test_extract.py
from accounting_rag.extract import extract_lines


def test_page40_contains_article_header(recueil_path):
    lines = extract_lines(recueil_path, pages=range(39, 40))  # page 40 (0-indexée 39)
    headers = [l for l in lines if l.text.startswith("Art. 212-5")]
    assert len(headers) == 1
    h = headers[0]
    assert h.bold is True
    assert abs(h.size - 10.0) < 0.1
    assert h.page == 40


def test_superscript_merged(recueil_path):
    lines = extract_lines(recueil_path, pages=range(39, 40))
    footer = [l for l in lines if "1er janvier 2026" in l.text.replace(" ", " ")]
    assert footer, "l'exposant 'er' doit être recollé à '1'"
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_extract.py -v`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/extract.py
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
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_extract.py -v`. Expected: PASS. Si le test d'exposant échoue : imprimer les lignes de pied de page réelles et ajuster le seuil 0.8 — ne PAS supprimer le test.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: extraction PyMuPDF en lignes typées"`

---

### Task 4: Classification typographique (`classify.py`)

**Files:**
- Create: `src/accounting_rag/classify.py`, `tests/test_classify.py`

**Interfaces:**
- Consumes: `Line`, `Kind` (Task 2).
- Produces: `classify(line: Line) -> Kind` (fonction pure) ; regex exportées `ART_RE`, `SECTION_RE`.

- [ ] **Step 1: Tests qui échouent (synthétiques + réels)**

```python
# tests/test_classify.py
from accounting_rag.model import Line, Kind
from accounting_rag.classify import classify
from accounting_rag.extract import extract_lines


def L(text, size, bold=False, font="Tahoma", x=99.0):
    return Line(text=text, size=size, bold=bold, font=font, x=x, y=300.0, page=40)


def test_signatures_synthetiques():
    assert classify(L("Art. 212-5", 10.0, bold=True)) == Kind.ARTICLE_HEADER
    assert classify(L("Le titulaire d'un contrat…", 10.0)) == Kind.REGLEMENTAIRE
    assert classify(L("Les immobilisations exploitées…", 9.5)) == Kind.COMMENTAIRE
    assert classify(L("Exclusion des contrats – Avis CU n° 2006-C", 9.5, bold=True)) == Kind.COMMENTAIRE_TITRE
    assert classify(L("Sous-section 2 – Dispositions particulières", 10.0, bold=True)) == Kind.SECTION_HEADER
    assert classify(L("Chapitre IV – Immobilisations", 10.6, bold=True)) == Kind.SECTION_HEADER
    assert classify(L("RECUEIL DES NORMES COMPTABLES FRANÇAISES", 8.5)) == Kind.BRUIT
    assert classify(L("•", 10.0, font="Symbol")) == Kind.PUCE
    assert classify(L("-", 9.5, font="Calibri")) == Kind.PUCE


def test_page40_reelle_sans_inconnu(recueil_path):
    lines = extract_lines(recueil_path, pages=range(39, 41))
    kinds = [classify(l) for l in lines]
    assert Kind.INCONNU not in kinds, [l.text for l, k in zip(lines, kinds) if k == Kind.INCONNU]
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_classify.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/classify.py
import re
from .model import Line, Kind

ART_RE = re.compile(r"^Art\.\s*\d")
SECTION_RE = re.compile(r"^(Livre|Titre|Chapitre|Section|Sous-section)\b", re.I)
_BULLETS = {"•", "-", "–", "*"}


def classify(line: Line) -> Kind:
    if line.font.startswith("Symbol") or line.text in _BULLETS:
        return Kind.PUCE
    if line.size <= 9.1:                       # en-têtes, pieds, folios
        return Kind.BRUIT
    if line.size >= 10.4 and line.bold:        # titres de haut niveau (10.6, 12.0)
        return Kind.SECTION_HEADER
    if 9.8 <= line.size <= 10.3:               # strate réglementaire (10.0)
        if line.bold:
            if ART_RE.match(line.text):
                return Kind.ARTICLE_HEADER
            if SECTION_RE.match(line.text):
                return Kind.SECTION_HEADER
            return Kind.REGLEMENTAIRE          # gras d'emphase dans le corps
        return Kind.REGLEMENTAIRE
    if 9.3 <= line.size <= 9.7:                # strate commentaire (9.5)
        return Kind.COMMENTAIRE_TITRE if line.bold else Kind.COMMENTAIRE
    return Kind.INCONNU
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_classify.py -v`. Expected: PASS.

- [ ] **Step 5 [mécanique]: Recensement plein document** — Run: `uv run python -c "
from accounting_rag.extract import extract_lines
from accounting_rag.classify import classify
from collections import Counter
from pathlib import Path
c = Counter(classify(l).name for l in extract_lines(Path('data/raw/recueil-pcg-2026.pdf')))
print(c)"`. Expected: `INCONNU` < 1 % des lignes. Noter le décompte exact dans le message de commit.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: classification typographique (INCONNU=<n>/<total> lignes)"`

---

### Task 5: Nettoyage et assemblage des paragraphes (`clean.py`)

**Files:**
- Create: `src/accounting_rag/clean.py`, `tests/test_clean.py`

**Interfaces:**
- Consumes: rien (fonctions pures sur `list[str]`).
- Produces: `join_lines(lines: list[str]) -> str` — recolle les césures (`immobili-` + `sations` → `immobilisations`), joint les lignes en paragraphes (une ligne se terminant par `.`, `;`, `:` ou une ligne suivante commençant par une puce `- ` ouvre un nouveau paragraphe `\n`), normalise les espaces insécables.

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_clean.py
from accounting_rag.clean import join_lines


def test_cesure_recollee():
    assert join_lines(["les immobili-", "sations existantes."]) == "les immobilisations existantes."


def test_paragraphes():
    txt = join_lines(["Première phrase.", "Deuxième phrase", "qui continue."])
    assert txt == "Première phrase.\nDeuxième phrase qui continue."


def test_puce_conservee():
    txt = join_lines(["traitement suivant :", "- les frais de constitution ;", "- les frais de fusion."])
    assert txt == "traitement suivant :\n- les frais de constitution ;\n- les frais de fusion."


def test_espace_insecable():
    assert join_lines(["n° 2006-C"]) == "n° 2006-C"
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_clean.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/clean.py
_TERMINATORS = (".", ";", ":", "!", "?")


def join_lines(lines: list[str]) -> str:
    paras: list[str] = []
    current = ""
    for raw in lines:
        line = raw.replace(" ", " ").strip()
        if not line:
            continue
        starts_bullet = line.startswith(("- ", "• "))
        if current and (starts_bullet or current.endswith(_TERMINATORS)):
            paras.append(current)
            current = line
        elif current.endswith("-") and not current.endswith(" -"):
            current = current[:-1] + line          # césure
        elif current:
            current = current + " " + line
        else:
            current = line
    if current:
        paras.append(current)
    return "\n".join(paras)
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_clean.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: nettoyage (césures, paragraphes, insécables)"`

---

### Task 6: Extraction des renvois (`refs.py`)

**Files:**
- Create: `src/accounting_rag/refs.py`, `tests/test_refs.py`

**Interfaces:**
- Consumes: `Renvoi` (Task 2).
- Produces: `extract_renvois(texte: str) -> list[Renvoi]`. Cibles normalisées : interne `pcg-<num>` ; externe `legi-<art>-<code>` (code ∈ slug : `comofi`, `code-de-commerce`, `cgi`…) ; historique = citation verbatim compactée (`crc-2004-06`, `avis-cnc-2004-15`, `avis-cu-2006-C`).

- [ ] **Step 1: Tests qui échouent (phrases réelles du sondage)**

```python
# tests/test_refs.py
from accounting_rag.refs import extract_renvois


def _cibles(txt):
    return {(r.cible, r.famille) for r in extract_renvois(txt)}


def test_interne_pluriel():
    c = _cibles("définis aux articles 212-1 et 212-2.")
    assert ("pcg-212-1", "interne") in c and ("pcg-212-2", "interne") in c


def test_interne_cf():
    assert ("pcg-1214-48", "interne") in _cibles("Cf. article 1214-48")


def test_externe_comofi():
    c = _cibles("à l'article L. 313-7 du Code monétaire et financier")
    assert ("legi-L313-7-comofi", "externe_legal") in c


def test_historique_crc_et_avis():
    txt = ("du règlement n° 2004-06 du CRC ; Avis CNC n° 2004-15 du 23 juin 2004 ; "
           "Avis CU n° 2006-C du 4 octobre 2006")
    c = _cibles(txt)
    assert ("crc-2004-06", "historique") in c
    assert ("avis-cnc-2004-15", "historique") in c
    assert ("avis-cu-2006-C", "historique") in c
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_refs.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/refs.py
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
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_refs.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: extraction des renvois (interne/externe/historique)"`

---

### Task 7: Assemblage hiérarchique (`parse.py`)

**Files:**
- Create: `src/accounting_rag/parse.py`, `tests/test_parse.py`

**Interfaces:**
- Consumes: `extract_lines` (T3), `classify` (T4), `join_lines` (T5), `extract_renvois` (T6), `Record` (T2).
- Produces: `parse(pdf_path: Path, edition: str = "2026-01-01") -> tuple[list[Record], list[Anomalie]]` où `Anomalie(page: int, ligne: str, raison: str)` (dataclass exportée par `parse.py`). Règles d'identifiants : article → `pcg-<num>@<edition>` ; n-ième commentaire d'un article → `pcg-<num>-c<n>@<edition>`. `chemin` = chemin des sections courantes joint par ` > `. `source_citation` d'un commentaire = première citation `Avis…`/`règlement…` trouvée dans son titre, sinon le titre entier tronqué à 200 caractères.

- [ ] **Step 1: Test qui échoue (pages 40-41 réelles)**

```python
# tests/test_parse.py
from accounting_rag.parse import parse


def test_pages_40_41(recueil_path):
    records, anomalies = parse(recueil_path)
    by_id = {r.id: r for r in records}

    art = by_id["pcg-212-5@2026-01-01"]
    assert art.type == "reglementaire"
    assert art.texte.startswith("Le titulaire d'un contrat de crédit-bail")
    assert art.page_debut == 40
    assert "Sous-section" not in art.texte           # les titres ne fuient pas dans le texte

    # le commentaire qui suit 212-5 lui est rattaché, avec sa provenance
    c = by_id.get("pcg-212-5-c1@2026-01-01")
    assert c is not None and c.type == "commentaire_ANC"
    assert c.opposable is False
    assert "avis-cu-2006-C" in [r.cible for r in c.renvois]

    # la série d'articles de la sous-section 2 est présente
    for num in ("212-6", "212-7", "212-8", "212-9", "212-10", "212-11"):
        assert f"pcg-{num}@2026-01-01" in by_id

    # le chemin porte la sous-section pour 212-6
    assert "Sous-section 2" in by_id["pcg-212-6@2026-01-01"].chemin


def test_articles_reglementaires_opposables_non(recueil_path):
    records, _ = parse(recueil_path)
    assert all(r.opposable is False for r in records)  # rien d'opposable dans l'ANC (≠ BOFiP)
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_parse.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/parse.py
import re
from dataclasses import dataclass
from pathlib import Path
from .model import Line, Kind, Record
from .extract import extract_lines
from .classify import classify, ART_RE
from .clean import join_lines
from .refs import extract_renvois

_ART_NUM = re.compile(r"^Art\.\s*(\d{3,4}-\d+(?:-\d+)?)")
_LEVELS = ["livre", "titre", "chapitre", "section", "sous-section"]
_CITATION = re.compile(r"(Avis\s+(?:CNC|CU)\s+n°\s*[\w\-]+[^–]*|règlement\s+n°\s*[\d\-]+\s+du\s+CRC[^–]*)", re.I)


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
        self.records.append(Record(
            id=rid, article=self.cur_article, chemin=self.chemin(), texte=texte,
            type=rtype, nature="comptable", opposable=False,
            valide_du=self.edition, valide_au=None, source_citation=citation,
            page_debut=self.buf_start, page_fin=self.buf_end,
            renvois=extract_renvois((self.com_title or "") + " " + texte),
        ))
        self.buf = []

    def feed(self, line: Line, kind: Kind):
        if kind == Kind.BRUIT:
            return
        if kind == Kind.SECTION_HEADER:
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
                self.buf.append("- ")
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
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_parse.py -v`. Expected: PASS. En cas d'écart sur les pages réelles (ex. titre de commentaire multi-lignes mal recollé), imprimer les records des pages 40-41 et corriger la machine à états — ne pas affaiblir les assertions.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: assemblage hiérarchique articles/commentaires"`

---

### Task 8: Intégrité et rapport d'anomalies (`integrity.py`)

**Files:**
- Create: `src/accounting_rag/integrity.py`, `tests/test_integrity.py`

**Interfaces:**
- Consumes: `Record`, `Anomalie` (T7).
- Produces: `check(records) -> list[Anomalie]` (vérifications post-assemblage) et `report(records, anomalies) -> str` (markdown). Vérifications : (a) cohérence numérotation↔chemin pour les articles à 3 chiffres — 1er chiffre = n° de Livre extrait du chemin (romain→arabe) ; (b) doublons d'identifiants ; (c) articles au texte vide ; (d) renvois internes pointant vers un article inexistant (dangling — signalés, pas bloquants).

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_integrity.py
from accounting_rag.integrity import check, report, _roman_to_int
from accounting_rag.model import Record, Renvoi


def _rec(rid, article, chemin="Livre II > Titre I", texte="x", renvois=()):
    return Record(id=rid, article=article, chemin=chemin, texte=texte,
                  type="reglementaire", nature="comptable", opposable=False,
                  valide_du="2026-01-01", valide_au=None, source_citation=None,
                  page_debut=1, page_fin=1, renvois=list(renvois))


def test_roman():
    assert _roman_to_int("II") == 2 and _roman_to_int("IX") == 9


def test_incoherence_livre_detectee():
    bad = _rec("pcg-312-1@2026-01-01", "312-1", chemin="Livre II > Titre I")
    assert any("Livre" in a.raison for a in check([bad]))


def test_dangling_renvoi():
    r = _rec("pcg-212-1@2026-01-01", "212-1",
             renvois=[Renvoi("pcg-999-99", "interne")])
    assert any("999-99" in a.raison for a in check([r]))


def test_report_contient_compteurs():
    r = _rec("pcg-212-1@2026-01-01", "212-1")
    md = report([r], [])
    assert "1 enregistrement" in md
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_integrity.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/integrity.py
import re
from collections import Counter
from .model import Record
from .parse import Anomalie

_LIVRE = re.compile(r"Livre\s+([IVXLC]+)", re.I)
_ROMANS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(s: str) -> int:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMANS[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def check(records: list[Record]) -> list[Anomalie]:
    out: list[Anomalie] = []
    ids = Counter(r.id for r in records)
    for rid, n in ids.items():
        if n > 1:
            out.append(Anomalie(0, rid, f"identifiant en double ({n}×)"))
    known_articles = {r.article for r in records if r.article}
    for r in records:
        if not r.texte.strip():
            out.append(Anomalie(r.page_debut, r.id, "texte vide"))
        if r.article and len(r.article.split("-")[0]) == 3:
            m = _LIVRE.search(r.chemin)
            if m and int(r.article[0]) != _roman_to_int(m.group(1)):
                out.append(Anomalie(r.page_debut, r.id,
                                    f"article {r.article} sous {m.group(0)} : incohérence de Livre"))
        for rv in r.renvois:
            if rv.famille == "interne":
                num = rv.cible.removeprefix("pcg-")
                if num not in known_articles:
                    out.append(Anomalie(r.page_debut, r.id, f"renvoi interne sans cible : {num}"))
    return out


def report(records: list[Record], anomalies: list[Anomalie]) -> str:
    n_reg = sum(1 for r in records if r.type == "reglementaire")
    n_com = sum(1 for r in records if r.type == "commentaire_ANC")
    lines = [
        "# Rapport de build du corpus", "",
        f"- **{len(records)} enregistrement(s)** : {n_reg} réglementaires, {n_com} commentaires ANC",
        f"- **{len(anomalies)} anomalie(s)**", "",
    ]
    by_reason = Counter(a.raison.split(":")[0] for a in anomalies)
    for reason, n in by_reason.most_common():
        lines.append(f"## {reason} ({n})")
        for a in anomalies:
            if a.raison.split(":")[0] == reason:
                lines.append(f"- p.{a.page} — `{a.ligne[:100]}` — {a.raison}")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_integrity.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: vérifications d'intégrité et rapport d'anomalies"`

---

### Task 9: Écriture SQLite (`db.py`)

**Files:**
- Create: `src/accounting_rag/db.py`, `tests/test_db.py`

**Interfaces:**
- Consumes: `Record` (T2).
- Produces: `write_db(records: list[Record], dest: Path) -> None` — tables `records` (colonnes = champs du Record hors renvois), `renvois(source_id, cible, famille)`, et FTS5 `records_fts(texte, chemin)` synchronisée par content-table. Écrase `dest` si existant.

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_db.py
import sqlite3
from accounting_rag.db import write_db
from tests.test_integrity import _rec  # réutilise la fabrique
from accounting_rag.model import Renvoi


def test_roundtrip(tmp_path):
    dest = tmp_path / "corpus.db"
    rec = _rec("pcg-212-5@2026-01-01", "212-5", texte="Le titulaire d'un contrat de crédit-bail…",
               renvois=[Renvoi("legi-L313-7-comofi", "externe_legal")])
    write_db([rec], dest)
    con = sqlite3.connect(dest)
    assert con.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    assert con.execute("SELECT famille FROM renvois").fetchone()[0] == "externe_legal"
    hits = con.execute("SELECT rowid FROM records_fts WHERE records_fts MATCH 'crédit'").fetchall()
    assert len(hits) == 1
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_db.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/db.py
import sqlite3
from pathlib import Path
from .model import Record

_SCHEMA = """
CREATE TABLE records(
  id TEXT PRIMARY KEY, article TEXT, chemin TEXT, texte TEXT,
  type TEXT NOT NULL, nature TEXT NOT NULL, opposable INTEGER NOT NULL,
  valide_du TEXT, valide_au TEXT, source_citation TEXT,
  page_debut INTEGER, page_fin INTEGER
);
CREATE TABLE renvois(
  source_id TEXT NOT NULL REFERENCES records(id),
  cible TEXT NOT NULL, famille TEXT NOT NULL
);
CREATE INDEX idx_renvois_source ON renvois(source_id);
CREATE VIRTUAL TABLE records_fts USING fts5(
  texte, chemin, content='records', content_rowid='rowid'
);
"""


def write_db(records: list[Record], dest: Path) -> None:
    dest.unlink(missing_ok=True)
    con = sqlite3.connect(dest)
    con.executescript(_SCHEMA)
    for r in records:
        con.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.id, r.article, r.chemin, r.texte, r.type, r.nature,
             int(r.opposable), r.valide_du, r.valide_au, r.source_citation,
             r.page_debut, r.page_fin),
        )
        rowid = con.execute("SELECT rowid FROM records WHERE id=?", (r.id,)).fetchone()[0]
        con.execute("INSERT INTO records_fts(rowid, texte, chemin) VALUES (?,?,?)",
                    (rowid, r.texte, r.chemin))
        con.executemany("INSERT INTO renvois VALUES (?,?,?)",
                        [(r.id, rv.cible, rv.famille) for rv in r.renvois])
    con.commit()
    con.close()
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_db.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: écriture SQLite (records + renvois + FTS5)"`

---

### Task 10: Pipeline complet (`scripts/build_corpus.py`)

**Files:**
- Create: `scripts/build_corpus.py`
- Modify: aucun

**Interfaces:**
- Consumes: `parse` (T7), `check`/`report` (T8), `write_db` (T9).
- Produces: `data/corpus.db` (gitignoré) + `docs/rapport-build.md` (commité).

- [ ] **Step 1: Écrire le script**

```python
# scripts/build_corpus.py
"""Pipeline complet : PDF -> corpus.db + rapport d'anomalies."""
from pathlib import Path
from accounting_rag.parse import parse
from accounting_rag.integrity import check, report
from accounting_rag.db import write_db

PDF = Path("data/raw/recueil-pcg-2026.pdf")
DB = Path("data/corpus.db")
RAPPORT = Path("docs/rapport-build.md")

records, anomalies = parse(PDF)
anomalies += check(records)
write_db(records, DB)
RAPPORT.write_text(report(records, anomalies), encoding="utf-8")
print(f"{len(records)} enregistrements -> {DB}")
print(f"{len(anomalies)} anomalies -> {RAPPORT}")
```

- [ ] **Step 2 [mécanique]: Lancer et relever les compteurs** — Run: `uv run python scripts/build_corpus.py`. Expected: plusieurs centaines d'articles réglementaires ET plusieurs centaines de commentaires ; taux d'anomalies < 2 % des enregistrements. Relever les chiffres exacts.

- [ ] **Step 3: Contrôles de plausibilité en SQL** — Run:
`sqlite3 data/corpus.db "SELECT type, COUNT(*) FROM records GROUP BY type; SELECT COUNT(*) FROM renvois; SELECT texte FROM records WHERE id='pcg-212-5@2026-01-01';"`
Expected: le texte de 212-5 commence par « Le titulaire d'un contrat de crédit-bail ».

- [ ] **Step 4: Commit (rapport inclus)** — `git add scripts/build_corpus.py docs/rapport-build.md && git commit -m "feat: pipeline complet — <n> records, <m> anomalies"`

---

### Task 11 [mécanique]: Validation par échantillonnage (sous-agents)

**Files:**
- Create: `docs/validation-echantillon.md`

**Interfaces:**
- Consumes: `data/corpus.db`, le PDF source.

- [ ] **Step 1: Tirer un échantillon reproductible** — 12 pages : `uv run python -c "import random; random.seed(42); print(sorted(random.sample(range(20, 660), 12)))"`.

- [ ] **Step 2: Dispatcher la vérification à des sous-agents économiques** — Pour chaque page de l'échantillon, un sous-agent (modèle économique) reçoit : (a) le texte brut de la page (`pymupdf` get_text), (b) les enregistrements de la base couvrant cette page (`SELECT id, type, texte FROM records WHERE page_debut <= P AND page_fin >= P`). Consigne au sous-agent : « Vérifie que chaque phrase du texte brut (hors en-têtes/pieds) figure dans un enregistrement, que la frontière réglementaire/commentaire est correcte, que les numéros d'articles correspondent. Rapporte chaque écart en une ligne : page, id, nature de l'écart. Rapporte "OK" sinon. »

- [ ] **Step 3: Consolider** — Écrire `docs/validation-echantillon.md` : pages vérifiées, écarts trouvés, corrections apportées (le cas échéant, corriger le parseur et relancer Task 10 avant de conclure).

- [ ] **Step 4: Commit** — `git add docs/validation-echantillon.md && git commit -m "test: validation par échantillonnage (12 pages, sous-agents)"`

---

### Task 12: README, licence et publication GitHub

**Files:**
- Create: `README.md`, `LICENSE` (MIT), `DATA_LICENSE.md`

- [ ] **Step 1: README** — Contenu minimal : objectif du projet (agent comptable français open source ; étoile polaire DSCG UE4), avertissement (« synthèse par LLM, pas une doctrine — ne remplace pas un expert-comptable »), quickstart (`uv run python scripts/download_data.py && uv run python scripts/build_corpus.py`), description du schéma SQLite, licences (code MIT / dataset Licence Ouverte 2.0 / sources : ANC), lien vers `JOURNAL.md`.

- [ ] **Step 2: Licences** — `LICENSE` = texte MIT standard au nom de l'auteur ; `DATA_LICENSE.md` = mention Licence Ouverte 2.0 (Etalab) pour le contenu extrait de l'ANC, avec attribution « Autorité des normes comptables ».

- [ ] **Step 3: Mise à jour du journal** — Ajouter à `JOURNAL.md` une section « Jalon 1 livré » : compteurs du build, taux d'anomalies, écarts trouvés en validation, surprises rencontrées.

- [ ] **Step 4: Publication** — **⚠️ Demander à l'utilisateur le nom du dépôt et la visibilité avant d'exécuter** : `gh repo create <nom> --public --source=. --push`.

- [ ] **Step 5: Commit final** — `git add -A && git commit -m "docs: README + licences" && git push`

---

## Self-Review

- **Couverture spec (sections 2, 3, 7)** : schéma → T2/T9 ; parseur typographique + double source + rapport d'anomalies → T3-T8 ; publication GitHub/licences → T12 ; LLM aux marges → T11 (validation), aucun LLM dans le pipeline ✓. Hors périmètre assumé de ce plan (plans suivants) : chaîne d'analyse lexicale, retrieval, benchmark, BOFiP/NEP/IFRS.
- **Placeholders** : chaque étape porte son code ou sa commande exacte ✓.
- **Cohérence des types** : `Line/Kind/Record/Renvoi` définis en T2, consommés à l'identique en T3-T9 ; `Anomalie` définie en T7, réutilisée en T8 ✓.
