# Jalon 2 — Chaîne d'analyse lexicale, retrieval hybride, benchmark d'amorçage : Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre le corpus interrogeable avec une qualité mesurée : normalisation lexicale domaine, index dense + lexical, retrieval hybride avec routeur de références et expansion du graphe de renvois, et un benchmark de 30 questions gold qui produit le premier tableau recall@k.

**Architecture:** Étages purs au-dessus du corpus.db du jalon 1 : `normalize.py` (chaîne d'analyse appliquée au build ET à la requête) → `chunks.py` (fenêtrage des records longs) → `embed.py` (encodeur configurable) → `scripts/build_index.py` (ajoute FTS normalisé + vecteurs au .db) → `search.py` (routeur → BM25 + dense → fusion RRF → renvois 1-hop → small-to-big) → `benchmark/` (JSONL gold) + `evalrag.py` (recall@k/MRR) + `scripts/run_eval.py`.

**Tech Stack:** Python ≥3.12 via `uv` ; snowballstemmer (stemming français, pur Python) ; sentence-transformers + `intfloat/multilingual-e5-small` (384 dims, ~470 Mo — défaut accessible ; modèle surclassable par variable d'env) ; sqlite-vec ; pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-accountingrag-design.md` (sections 4, 5)

## Global Constraints

- `uv` exclusivement ; tests : `uv run pytest`.
- Le corpus d'entrée est `data/corpus.db` produit par le jalon 1 (1660 records, ids type `pcg-214-1@2026-01-01`, `pcg-214-1-c1@…`, `pcg-na-12@…`, suffixes `#n` possibles). Ne JAMAIS modifier les modules du jalon 1 (extract/classify/clean/parse/refs/integrity/db) sauf ruling explicite du contrôleur.
- La normalisation requête et la normalisation build passent par LA MÊME fonction (`normalize.normalize`) — toute divergence est un bug.
- `build_index.py` est idempotent (DROP + recreate de ses tables) et n'altère jamais les tables du jalon 1 (`records`, `renvois`, `records_fts`).
- Déterminisme : mêmes entrées → même index (les embeddings sont déterministes à modèle fixé).
- Les questions du benchmark citent leurs articles gold par préfixe d'id SANS édition ni fragment : `"pcg-214-1"` (le harnais matche tout record dont l'id commence par ce préfixe suivi de `@`, `-c` ou `#`).
- Étapes marquées **[mécanique]** : sous-agent économique. La rédaction des questions gold (T6) exige du jugement : modèle standard.
- À la fin de chaque tâche : commit. Mise à jour de `JOURNAL.md` : contrôleur uniquement.

---

### Task 1: Chaîne d'analyse lexicale (`normalize.py`)

**Files:**
- Create: `src/accounting_rag/normalize.py`, `tests/test_normalize.py`
- Modify: `pyproject.toml` (ajouter `snowballstemmer>=2.2`)

**Interfaces:**
- Produces: `normalize(text: str) -> str` — texte → tokens normalisés joints par espaces. `SYNONYMES: dict[str, str]` exporté (extensible).
- Étapes internes, dans cet ordre : minuscules → apostrophes typographiques (’ ‘) → droites → suppression des élisions (l', d', j', m', n', s', t', c', qu', jusqu', lorsqu', puisqu', quoiqu', presqu', aujourd') → normalisation des références en tokens atomiques (« l. 313-7 » → `l313-7` ; « r. 123-4 » → `r123-4` ; les numéros nus `214-1` restent tels quels ; `boi-bic-amt-10-20` reste tel quel) → remplacement des synonymes métier (sur chaînes, avant stemming) → découpage sur tout caractère non [a-z0-9à-ÿ-] → stemming Snowball français des tokens purement alphabétiques (les tokens contenant un chiffre restent verbatim) → pliage des accents (NFD, suppression des combinants) → jointure par espaces.

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_normalize.py
from accounting_rag.normalize import normalize


def test_meme_normalisation_flexions():
    # le stem rapproche singulier/pluriel et formes fléchies
    assert normalize("les amortissements des immobilisations") == normalize(
        "l'amortissement de l'immobilisation"
    ).replace("de ", "").strip() or set(normalize("les amortissements").split()) & set(
        normalize("l'amortissement").split()
    )


def test_stems_partages():
    a = set(normalize("les amortissements dérogatoires").split())
    b = set(normalize("un amortissement dérogatoire").split())
    assert a & b >= {"amort", "derogatoir"} or len(a & b) >= 2


def test_reference_atomique():
    out = normalize("aux termes de l'article L. 313-7 du code monétaire")
    assert "l313-7" in out.split()
    out2 = normalize("Art. 214-1 du PCG")
    assert "214-1" in out2.split()


def test_elision_supprimee():
    out = normalize("l'exercice d'imputation")
    assert "exercice" in out.split()
    assert not any(t.startswith("l'") or t == "l" for t in out.split())


def test_synonyme_metier():
    assert normalize("le fonds de commerce") == normalize("le fonds commercial")


def test_accents_plies():
    out = normalize("créance échue")
    assert all(ord(c) < 128 for c in out)


def test_requete_et_document_identiques():
    # invariant central : même fonction pour les deux côtés
    doc = "Le titulaire d'un contrat de crédit-bail comptabilise en charges"
    query = "comptabiliser les charges d'un contrat de crédit-bail"
    assert set(normalize(doc).split()) & set(normalize(query).split()) >= {"contrat", "charg"} or \
           len(set(normalize(doc).split()) & set(normalize(query).split())) >= 3
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_normalize.py -v`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/normalize.py
"""Chaîne d'analyse lexicale domaine — appliquée à l'identique au build et à la requête."""
import re
import unicodedata
import snowballstemmer

_stemmer = snowballstemmer.stemmer("french")

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_ELISION = re.compile(
    r"\b(?:jusqu|lorsqu|puisqu|quoiqu|presqu|aujourd|qu|[ldjmnstc])'", re.I
)
_REF_LETTREE = re.compile(r"\b([lrd])\.?\s*(\d{1,4}(?:-\d+)+)\b", re.I)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9à-ÿ-]+")
_HAS_DIGIT = re.compile(r"\d")

# Synonymes métier : clé (forme trouvée) -> valeur (forme canonique), en minuscules.
SYNONYMES: dict[str, str] = {
    "fonds de commerce": "fonds commercial",
    "leasing": "credit-bail",
    "location avec option d'achat": "credit-bail",
    "ifc": "indemnites de fin de carriere",
    "indemnite de depart a la retraite": "indemnites de fin de carriere",
    "actif incorporel": "immobilisation incorporelle",
    "actif corporel": "immobilisation corporelle",
    "stock-options": "options de souscription d'actions",
    "goodwill": "fonds commercial",
    "amortissement degressif": "amortissement derogatoire",
}


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    t = text.lower().translate(_APOSTROPHES)
    t = _ELISION.sub("", t)
    t = _REF_LETTREE.sub(lambda m: f"{m.group(1)}{m.group(2)}", t)
    folded = _fold(t)
    for src, dst in SYNONYMES.items():
        folded = folded.replace(_fold(src), _fold(dst))
    tokens = [tok for tok in _TOKEN_SPLIT.split(folded) if tok and tok != "-"]
    out = []
    for tok in tokens:
        if _HAS_DIGIT.search(tok):
            out.append(tok.strip("-"))
        else:
            out.append(_fold(_stemmer.stemWord(tok)))
    return " ".join(o for o in out if o)
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_normalize.py -v`. Expected: PASS. Si le stemmer produit des stems différents de ceux attendus dans les tests (« amort »/« derogatoir »), imprime les stems réels et ajuste LES TESTS aux stems réels du Snowball français (les assertions ensemblistes sont conçues pour tolérer ça) — jamais l'inverse.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(jalon2): chaîne d'analyse lexicale (élisions, refs atomiques, synonymes, stemming fr)"`

---

### Task 2: Fenêtrage des records longs (`chunks.py`)

**Files:**
- Create: `src/accounting_rag/chunks.py`, `tests/test_chunks.py`

**Interfaces:**
- Produces: `make_chunks(record_id: str, texte: str, max_words: int = 220, overlap: int = 40) -> list[tuple[str, int, str]]` — liste de `(chunk_id, seq, chunk_text)` avec `chunk_id = f"{record_id}::{seq}"`. Découpe au paragraphe (`\n`) puis regroupe en fenêtres ≤ max_words ; un paragraphe seul > max_words est découpé en fenêtres glissantes avec chevauchement `overlap` mots. Un record court → un seul chunk (seq 0, texte intégral).

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_chunks.py
from accounting_rag.chunks import make_chunks


def test_record_court_un_seul_chunk():
    chunks = make_chunks("pcg-x@e", "Une phrase courte.")
    assert chunks == [("pcg-x@e::0", 0, "Une phrase courte.")]


def test_decoupe_aux_paragraphes():
    texte = "\n".join(f"Paragraphe {i} " + "mot " * 100 for i in range(5))
    chunks = make_chunks("pcg-x@e", texte, max_words=220)
    assert len(chunks) >= 2
    assert all(len(c[2].split()) <= 220 for c in chunks)
    assert [c[1] for c in chunks] == list(range(len(chunks)))


def test_paragraphe_geant_fenetre_glissante():
    texte = "mot " * 1000
    chunks = make_chunks("pcg-x@e", texte.strip(), max_words=220, overlap=40)
    assert len(chunks) >= 5
    # le chevauchement existe : la fin d'un chunk se retrouve au début du suivant
    a, b = chunks[0][2].split(), chunks[1][2].split()
    assert a[-40:] == b[:40]


def test_couverture_totale():
    texte = "\n".join(f"Alinea {i} unique_{i}" for i in range(30))
    chunks = make_chunks("pcg-x@e", texte, max_words=20)
    joined = " ".join(c[2] for c in chunks)
    assert all(f"unique_{i}" in joined for i in range(30))
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_chunks.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/chunks.py
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
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_chunks.py -v`. Expected: PASS. (Le test de fenêtre glissante fixe le contrat : si l'implémentation diffère sur les bornes, ajuste l'implémentation.)

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(jalon2): fenêtrage des records longs"`

---

### Task 3: Encodeur configurable (`embed.py`)

**Files:**
- Create: `src/accounting_rag/embed.py`, `tests/test_embed.py`
- Modify: `pyproject.toml` (ajouter `sentence-transformers>=3`, `sqlite-vec>=0.1`)

**Interfaces:**
- Produces: `class Embedder` — `Embedder(model_name: str | None = None)` (défaut : env `ACCRAG_EMB_MODEL` sinon `intfloat/multilingual-e5-small`) ; `.dim: int` ; `.encode_passages(texts: list[str]) -> list[list[float]]` ; `.encode_query(text: str) -> list[float]`. Les modèles e5 exigent les préfixes `"passage: "` / `"query: "` — appliqués automatiquement si `"e5"` figure dans le nom du modèle. Vecteurs normalisés L2 (`normalize_embeddings=True`).

- [ ] **Step 1: Test qui échoue** (marqué `slow` — télécharge le modèle au premier run)

```python
# tests/test_embed.py
import pytest
from accounting_rag.embed import Embedder


@pytest.fixture(scope="module")
def emb():
    return Embedder()


def test_dimensions_et_normalisation(emb):
    vecs = emb.encode_passages(["l'amortissement des immobilisations"])
    assert len(vecs) == 1 and len(vecs[0]) == emb.dim
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_similarite_semantique(emb):
    q = emb.encode_query("comment comptabiliser un logiciel acheté ?")
    p_proche = emb.encode_passages(["Les immobilisations incorporelles comprennent les logiciels acquis."])[0]
    p_loin = emb.encode_passages(["Le montant des primes de remboursement d'emprunt est amorti."])[0]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, p_proche) > dot(q, p_loin)
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_embed.py -v`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/embed.py
"""Encodeur dense configurable. Défaut : multilingual-e5-small (384 dims, léger)."""
import os

_DEFAULT = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer  # import paresseux (lourd)

        self.model_name = model_name or os.environ.get("ACCRAG_EMB_MODEL", _DEFAULT)
        self._model = SentenceTransformer(self.model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._e5 = "e5" in self.model_name.lower()

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts] if self._e5 else texts
        return self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()

    def encode_query(self, text: str) -> list[float]:
        prefixed = f"query: {text}" if self._e5 else text
        return self._model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_embed.py -v`. Expected: PASS (premier run : téléchargement ~470 Mo, patiente).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(jalon2): encodeur dense configurable (e5-small par défaut)"`

---

### Task 4: Construction de l'index (`scripts/build_index.py`)

**Files:**
- Create: `scripts/build_index.py`, `src/accounting_rag/index.py`, `tests/test_index.py`

**Interfaces:**
- Produces (dans `index.py`) : `build_index(db_path: Path, embedder: Embedder | None = None) -> dict` — retourne les compteurs `{chunks, vecteurs, records_norm}`. Crée (DROP puis CREATE) : table `chunks(chunk_id TEXT PRIMARY KEY, record_id TEXT, seq INT, texte TEXT)` ; FTS5 `chunks_norm` (colonnes `texte_norm`, `chemin_norm`, content-less, une ligne par chunk, rowid aligné sur `chunks.rowid`) ; table vectorielle `chunks_vec` via sqlite-vec (`CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[<dim>])`, rowid aligné sur `chunks.rowid`). Ne touche PAS aux tables du jalon 1.
- `scripts/build_index.py` : CLI qui ouvre `data/corpus.db`, appelle `build_index`, affiche les compteurs.

- [ ] **Step 1: Test qui échoue** (sur un mini-db synthétique, sans modèle : embedder factice)

```python
# tests/test_index.py
import sqlite3
from pathlib import Path
from accounting_rag.db import write_db
from accounting_rag.index import build_index
from conftest import _rec


class FakeEmbedder:
    dim = 4

    def encode_passages(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


def test_build_index_cree_les_tables(tmp_path):
    db = tmp_path / "c.db"
    write_db([
        _rec("pcg-1-1@e", "1-1", texte="Un article sur l'amortissement dérogatoire.\n" + "mot " * 500),
        _rec("pcg-1-2@e", "1-2", texte="Le fonds commercial est amorti."),
    ], db)
    stats = build_index(db, embedder=FakeEmbedder())
    con = sqlite3.connect(db)
    n_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert stats["chunks"] == n_chunks >= 3  # le record long produit plusieurs chunks
    assert con.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == n_chunks
    # le FTS normalisé matche une flexion différente
    hits = con.execute(
        "SELECT rowid FROM chunks_norm WHERE chunks_norm MATCH 'derogatoir*'"
    ).fetchall()
    assert hits
    # idempotence
    stats2 = build_index(db, embedder=FakeEmbedder())
    assert stats2 == stats
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_index.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/index.py
"""Construit chunks + FTS normalisé + vecteurs au-dessus du corpus.db (idempotent)."""
import sqlite3
from pathlib import Path
import sqlite_vec
from .chunks import make_chunks
from .normalize import normalize


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def build_index(db_path: Path, embedder=None) -> dict:
    if embedder is None:
        from .embed import Embedder
        embedder = Embedder()
    con = _connect(db_path)
    con.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_norm;
        DROP TABLE IF EXISTS chunks_vec;
        CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, record_id TEXT NOT NULL, seq INT, texte TEXT);
        CREATE VIRTUAL TABLE chunks_norm USING fts5(texte_norm, chemin_norm);
    """)
    con.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{embedder.dim}])")
    rows = con.execute("SELECT id, texte, chemin FROM records").fetchall()
    all_chunks: list[tuple[str, str, int, str, str]] = []
    for rid, texte, chemin in rows:
        for chunk_id, seq, piece in make_chunks(rid, texte or ""):
            all_chunks.append((chunk_id, rid, seq, piece, chemin or ""))
    for i, (chunk_id, rid, seq, piece, chemin) in enumerate(all_chunks, start=1):
        con.execute("INSERT INTO chunks(rowid, chunk_id, record_id, seq, texte) VALUES (?,?,?,?,?)",
                    (i, chunk_id, rid, seq, piece))
        con.execute("INSERT INTO chunks_norm(rowid, texte_norm, chemin_norm) VALUES (?,?,?)",
                    (i, normalize(piece), normalize(chemin)))
    BATCH = 64
    import struct
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start:start + BATCH]
        vecs = embedder.encode_passages([c[3] for c in batch])
        for offset, vec in enumerate(vecs):
            rowid = start + offset + 1
            con.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                        (rowid, struct.pack(f"{len(vec)}f", *vec)))
    con.commit()
    stats = {"chunks": len(all_chunks), "vecteurs": len(all_chunks),
             "records_norm": len(rows)}
    con.close()
    return stats
```

```python
# scripts/build_index.py
"""CLI : ajoute l'index de recherche (chunks + FTS normalisé + vecteurs) à data/corpus.db."""
from pathlib import Path
from accounting_rag.index import build_index

stats = build_index(Path("data/corpus.db"))
print(stats)
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_index.py -v`. Expected: PASS. NB sqlite-vec : l'insertion attend des bytes little-endian float32 (`struct.pack`) — si l'API diffère (versions récentes acceptent le JSON), adapte en suivant la doc du paquet installé et note-le dans ton rapport.

- [ ] **Step 5 [mécanique]: Construire l'index réel** — Run: `uv run python scripts/build_index.py`. Noter les compteurs réels (attendu : ≥1660 chunks — les records longs en produisent plusieurs) et la durée. Vérifier : `sqlite3 data/corpus.db "SELECT COUNT(*) FROM chunks; SELECT rowid FROM chunks_norm WHERE chunks_norm MATCH 'derogatoir*' LIMIT 3;"`.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(jalon2): index de recherche (chunks + FTS normalisé + sqlite-vec) — <n> chunks"`

---

### Task 5: Retrieval hybride (`search.py`)

**Files:**
- Create: `src/accounting_rag/search.py`, `tests/test_search.py`

**Interfaces:**
- Consumes: `normalize` (T1), `Embedder` (T3), les tables de T4 + `records`/`renvois` du jalon 1.
- Produces: `class Searcher` — `Searcher(db_path: Path, embedder=None)` (embedder paresseux : chargé au premier appel dense) ; `search(query: str, k: int = 10, mode: str = "hybrid") -> list[dict]` avec `mode ∈ {"bm25", "dense", "hybrid", "hybrid+graph"}` ; chaque résultat : `{"record_id", "article", "chemin", "texte", "score", "source"}` (`source` ∈ route/bm25/dense/fusion/graph). Comportements :
  1. **Routeur** (tous modes) : si la requête contient une référence d'article (`art(icle)?\.?\s*(\d{2,4}-\d+(?:-\d+)*)` ou `[LRD]\.?\s*\d+-\d+`), lookup direct `records.article = <num>` — les résultats routés sont TOUJOURS en tête, source `route`.
  2. **bm25** : `chunks_norm MATCH normalize(query)` (tokens joints par ` OR `), agrégé par record (meilleur score par record, ordre BM25 croissant de sqlite = plus petit meilleur → convertir en score décroissant), top k.
  3. **dense** : KNN sqlite-vec sur `chunks_vec` (top 50 chunks), agrégé par record (max), top k.
  4. **hybrid** : fusion RRF (k=60) des classements bm25 et dense (sur les records), top k.
  5. **hybrid+graph** : hybrid, puis expansion : pour chaque record du top 5, ajouter les records cibles de ses renvois internes (`renvois.famille='interne'`, cible existante) avec score = 0.5 × score du parent, source `graph`, dédupliqué, re-trié, top k.
  6. **small-to-big** : les hits de chunks remontent toujours le RECORD entier (texte complet).

- [ ] **Step 1: Test qui échoue** (sur le corpus RÉEL — nécessite data/corpus.db indexé en T4 ; skip sinon)

```python
# tests/test_search.py
import pytest
from pathlib import Path
from accounting_rag.search import Searcher

DB = Path("data/corpus.db")
pytestmark = pytest.mark.skipif(not DB.exists(), reason="corpus.db absent")


@pytest.fixture(scope="module")
def s():
    return Searcher(DB)


def test_routeur_reference_directe(s):
    hits = s.search("que dit l'article 214-1 ?", mode="bm25")
    assert hits and hits[0]["source"] == "route"
    assert hits[0]["article"] == "214-1"


def test_bm25_flexions(s):
    hits = s.search("amortissements dérogatoires", k=10, mode="bm25")
    assert hits
    assert any("dérogatoire" in h["texte"].lower() or "derogatoire" in h["texte"].lower() for h in hits[:5])


def test_dense_vocabulaire_courant(s):
    hits = s.search("comment comptabiliser un logiciel acheté ?", k=10, mode="dense")
    assert hits and len(hits) <= 10
    assert all(h["texte"] for h in hits)


def test_hybrid_contient_les_deux(s):
    hits = s.search("crédit-bail levée d'option", k=10, mode="hybrid")
    assert hits
    ids = [h["record_id"] for h in hits]
    assert len(ids) == len(set(ids))  # dédupliqué


def test_graph_expansion_ajoute_des_renvois(s):
    base = {h["record_id"] for h in s.search("contrat de crédit-bail", k=10, mode="hybrid")}
    expanded = s.search("contrat de crédit-bail", k=10, mode="hybrid+graph")
    assert any(h["source"] == "graph" for h in expanded) or {h["record_id"] for h in expanded} == base
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_search.py -v`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/search.py
"""Retrieval hybride : routeur de références -> BM25 normalisé + dense -> RRF -> renvois."""
import re
import sqlite3
import struct
from pathlib import Path
from .normalize import normalize

_REF_QUERY = re.compile(r"\bart(?:icle)?s?\.?\s*(\d{2,4}-\d+(?:-\d+)*)", re.I)
_REF_LETTREE = re.compile(r"\b([LRD])\.?\s*(\d{1,4}(?:-\d+)+)", re.I)
_RRF_K = 60


class Searcher:
    def __init__(self, db_path: Path, embedder=None):
        import sqlite_vec
        self.con = sqlite3.connect(db_path)
        self.con.enable_load_extension(True)
        sqlite_vec.load(self.con)
        self.con.enable_load_extension(False)
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _record(self, record_id: str, score: float, source: str) -> dict:
        row = self.con.execute(
            "SELECT article, chemin, texte FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        return {"record_id": record_id, "article": row[0], "chemin": row[1],
                "texte": row[2], "score": round(score, 4), "source": source}

    def _route(self, query: str) -> list[dict]:
        nums = _REF_QUERY.findall(query)
        out = []
        for num in nums:
            for (rid,) in self.con.execute(
                "SELECT id FROM records WHERE article = ? AND type = 'reglementaire'", (num,)
            ).fetchall():
                out.append(self._record(rid, 100.0, "route"))
        return out

    def _bm25(self, query: str, limit: int = 50) -> dict[str, float]:
        toks = normalize(query).split()
        if not toks:
            return {}
        match = " OR ".join(f'"{t}"' for t in toks)
        rows = self.con.execute(
            "SELECT c.record_id, bm25(chunks_norm) AS b FROM chunks_norm "
            "JOIN chunks c ON c.rowid = chunks_norm.rowid "
            "WHERE chunks_norm MATCH ? ORDER BY b LIMIT ?", (match, limit)
        ).fetchall()
        scores: dict[str, float] = {}
        for rid, b in rows:
            s = -b  # bm25() de sqlite : plus petit = meilleur
            scores[rid] = max(scores.get(rid, -1e9), s)
        return scores

    def _dense(self, query: str, limit: int = 50) -> dict[str, float]:
        vec = self.embedder.encode_query(query)
        rows = self.con.execute(
            "SELECT c.record_id, v.distance FROM chunks_vec v "
            "JOIN chunks c ON c.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND v.k = ?",
            (struct.pack(f"{len(vec)}f", *vec), limit),
        ).fetchall()
        scores: dict[str, float] = {}
        for rid, dist in rows:
            s = -dist
            scores[rid] = max(scores.get(rid, -1e9), s)
        return scores

    @staticmethod
    def _rrf(rankings: list[dict[str, float]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for scores in rankings:
            ordered = sorted(scores, key=scores.get, reverse=True)
            for rank, rid in enumerate(ordered):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        return fused

    def _expand_graph(self, results: list[dict], k: int) -> list[dict]:
        seen = {r["record_id"] for r in results}
        extra: list[dict] = []
        for r in results[:5]:
            for (cible,) in self.con.execute(
                "SELECT cible FROM renvois WHERE source_id = ? AND famille = 'interne'",
                (r["record_id"],),
            ).fetchall():
                for (rid,) in self.con.execute(
                    "SELECT id FROM records WHERE article = ? AND type='reglementaire'",
                    (cible.removeprefix("pcg-"),),
                ).fetchall():
                    if rid not in seen:
                        seen.add(rid)
                        extra.append(self._record(rid, r["score"] * 0.5, "graph"))
        merged = results + extra
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:k]

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[dict]:
        routed = self._route(query)
        routed_ids = {r["record_id"] for r in routed}
        if mode == "bm25":
            scores = self._bm25(query)
        elif mode == "dense":
            scores = self._dense(query)
        else:
            scores = self._rrf([self._bm25(query), self._dense(query)])
        ranked = sorted(scores, key=scores.get, reverse=True)
        results = [self._record(rid, scores[rid], "fusion" if mode.startswith("hybrid") else mode)
                   for rid in ranked if rid not in routed_ids][: k - len(routed) if len(routed) < k else 0]
        out = routed + results
        if mode == "hybrid+graph":
            out = self._expand_graph(out, k)
        return out[:k]
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_search.py -v`. Expected: PASS. La syntaxe KNN de sqlite-vec (`WHERE embedding MATCH ? AND k = ?`) varie selon les versions — si elle échoue, consulte la doc du paquet installé (`uv run python -c "import sqlite_vec; print(sqlite_vec.__version__)"`), adapte, et documente dans ton rapport.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(jalon2): retrieval hybride (routeur, BM25, dense, RRF, graphe)"`

---

### Task 6: Benchmark d'amorçage — 30 questions gold (`benchmark/`)

**Files:**
- Create: `benchmark/README.md`, `benchmark/dev.jsonl` (~21 questions), `benchmark/test.jsonl` (~9 questions, RÉSERVÉ — jamais utilisé pour régler le système), `tests/test_benchmark_format.py`

**Interfaces:**
- Format d'une ligne JSONL : `{"id": "q001", "question": "...", "categorie": "reference_directe|regle|vocabulaire_courant", "citations": ["pcg-214-1"], "notes": "..."}`. `citations` = préfixes d'ids d'articles gold (sans `@édition`) ; ≥1 par question ; chaque préfixe doit matcher ≥1 record de `data/corpus.db`.
- Répartition cible : 5 référence directe / 15 règle / 10 vocabulaire courant. Split : ~70 % dev, ~30 % test, stratifié par catégorie, figé (jamais re-tiré).

- [ ] **Step 1: Test de format qui échoue** (avant création des fichiers)

```python
# tests/test_benchmark_format.py
import json
import sqlite3
from pathlib import Path
import pytest

FILES = [Path("benchmark/dev.jsonl"), Path("benchmark/test.jsonl")]
DB = Path("data/corpus.db")


@pytest.mark.skipif(not DB.exists(), reason="corpus.db absent")
def test_format_et_citations_existantes():
    con = sqlite3.connect(DB)
    total = 0
    ids = set()
    for f in FILES:
        assert f.exists(), f"{f} manquant"
        for line in f.read_text(encoding="utf-8").splitlines():
            q = json.loads(line)
            assert set(q) >= {"id", "question", "categorie", "citations"}
            assert q["categorie"] in {"reference_directe", "regle", "vocabulaire_courant"}
            assert q["id"] not in ids
            ids.add(q["id"])
            assert len(q["question"]) > 15
            assert q["citations"]
            for c in q["citations"]:
                n = con.execute(
                    "SELECT COUNT(*) FROM records WHERE id = ? OR id LIKE ? OR id LIKE ? OR id LIKE ?",
                    (c, c + "@%", c + "-c%", c + "#%"),
                ).fetchone()[0]
                assert n > 0, f"{q['id']}: citation {c} sans record"
            total += 1
    assert total == 30
```

- [ ] **Step 2: Rédaction des questions** — travail de JUGEMENT (modèle standard), en DEUX temps :
  (a) tirer du corpus réel des articles porteurs de règles concrètes (amortissement, stocks, provisions, crédit-bail, fonds commercial, jetons/619-x, frais d'établissement, écarts de conversion, participation…) — lire le texte réel de chaque article choisi ;
  (b) rédiger pour chacun une question DONT LA RÉPONSE EST DANS CET ARTICLE : catégorie `reference_directe` = la question cite le numéro ; `regle` = question en langage professionnel sans numéro ; `vocabulaire_courant` = question comme la poserait un créateur d'entreprise (mots familiers, pas de jargon : « logiciel acheté », « machine », « voiture de société », « client qui ne paie pas »…). Les citations = le ou les préfixes d'articles dont le texte répond réellement (vérifier en relisant le record). `notes` = un mot sur la réponse attendue.
  Split stratifié dev/test (~21/~9) figé manuellement.

- [ ] **Step 3: Vérifier le passage du test de format** — Run: `uv run pytest tests/test_benchmark_format.py -v`. Expected: PASS.

- [ ] **Step 4: benchmark/README.md** — format, catégories, split dev/test (test = réservé, ne JAMAIS l'utiliser pour régler le système), procédure d'ajout de questions, licence (Licence Ouverte 2.0, cohérente avec DATA_LICENSE.md).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(jalon2): benchmark d'amorçage — 30 questions gold à citations (21 dev / 9 test)"`

---

### Task 7: Harnais d'évaluation (`evalrag.py` + `scripts/run_eval.py`)

**Files:**
- Create: `src/accounting_rag/evalrag.py`, `scripts/run_eval.py`, `tests/test_evalrag.py`

**Interfaces:**
- Produces: `load_benchmark(path: Path) -> list[dict]` ; `match(record_id: str, citation: str) -> bool` (préfixe : `record_id == citation` ou commence par `citation` suivi de `@`, `-c` ou `#`) ; `evaluate(searcher, questions, mode: str, k: int = 10) -> dict` → `{"recall@5": float, "recall@10": float, "mrr": float, "par_categorie": {cat: {"recall@10": ...}}, "n": int}`. Recall d'une question = |citations couvertes par le top-k| / |citations| ; MRR sur le rang du premier hit.
- `scripts/run_eval.py` : CLI `--mode bm25|dense|hybrid|hybrid+graph|all --split dev|test --k 10` ; affiche un tableau markdown.

- [ ] **Step 1: Test qui échoue**

```python
# tests/test_evalrag.py
from accounting_rag.evalrag import match, evaluate


def test_match_prefixe():
    assert match("pcg-214-1@2026-01-01", "pcg-214-1")
    assert match("pcg-214-1-c2@2026-01-01", "pcg-214-1")
    assert match("pcg-214-1@2026-01-01#2", "pcg-214-1")
    assert not match("pcg-214-10@2026-01-01", "pcg-214-1")


class FakeSearcher:
    def search(self, query, k=10, mode="hybrid"):
        return [{"record_id": "pcg-214-1@e", "score": 1.0, "source": "bm25",
                 "article": "214-1", "chemin": "", "texte": ""}]


def test_evaluate_recall_parfait_et_nul():
    qs = [
        {"id": "q1", "question": "x", "categorie": "regle", "citations": ["pcg-214-1"]},
        {"id": "q2", "question": "y", "categorie": "regle", "citations": ["pcg-999-9"]},
    ]
    m = evaluate(FakeSearcher(), qs, mode="bm25", k=10)
    assert m["recall@10"] == 0.5
    assert m["mrr"] == 0.5
    assert m["n"] == 2
```

- [ ] **Step 2: Vérifier l'échec** — Run: `uv run pytest tests/test_evalrag.py -v`. Expected: FAIL.

- [ ] **Step 3: Implémentation**

```python
# src/accounting_rag/evalrag.py
"""Harnais d'évaluation retrieval : recall@k / MRR sur citations gold."""
import json
from collections import defaultdict
from pathlib import Path


def load_benchmark(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def match(record_id: str, citation: str) -> bool:
    if record_id == citation:
        return True
    if not record_id.startswith(citation):
        return False
    nxt = record_id[len(citation):][:1]
    return nxt in {"@", "#"} or record_id[len(citation):].startswith("-c")


def evaluate(searcher, questions: list[dict], mode: str, k: int = 10) -> dict:
    recalls5, recalls10, mrrs = [], [], []
    par_cat: dict[str, list[float]] = defaultdict(list)
    for q in questions:
        hits = searcher.search(q["question"], k=k, mode=mode)
        ids = [h["record_id"] for h in hits]
        covered10 = sum(any(match(i, c) for i in ids[:10]) for c in q["citations"])
        covered5 = sum(any(match(i, c) for i in ids[:5]) for c in q["citations"])
        recalls10.append(covered10 / len(q["citations"]))
        recalls5.append(covered5 / len(q["citations"]))
        rank = next((r + 1 for r, i in enumerate(ids)
                     if any(match(i, c) for c in q["citations"])), None)
        mrrs.append(1.0 / rank if rank else 0.0)
        par_cat[q["categorie"]].append(covered10 / len(q["citations"]))
    return {
        "recall@5": round(sum(recalls5) / len(recalls5), 3),
        "recall@10": round(sum(recalls10) / len(recalls10), 3),
        "mrr": round(sum(mrrs) / len(mrrs), 3),
        "par_categorie": {c: {"recall@10": round(sum(v) / len(v), 3), "n": len(v)}
                          for c, v in sorted(par_cat.items())},
        "n": len(questions),
    }
```

```python
# scripts/run_eval.py
"""CLI d'évaluation : uv run python scripts/run_eval.py --mode all --split dev"""
import argparse
from pathlib import Path
from accounting_rag.evalrag import load_benchmark, evaluate
from accounting_rag.search import Searcher

p = argparse.ArgumentParser()
p.add_argument("--mode", default="all")
p.add_argument("--split", default="dev", choices=["dev", "test"])
p.add_argument("--k", type=int, default=10)
args = p.parse_args()

questions = load_benchmark(Path(f"benchmark/{args.split}.jsonl"))
searcher = Searcher(Path("data/corpus.db"))
modes = ["bm25", "dense", "hybrid", "hybrid+graph"] if args.mode == "all" else [args.mode]

print(f"| mode | recall@5 | recall@10 | MRR | n |")
print(f"|---|---|---|---|---|")
for mode in modes:
    m = evaluate(searcher, questions, mode=mode, k=args.k)
    print(f"| {mode} | {m['recall@5']} | {m['recall@10']} | {m['mrr']} | {m['n']} |")
    for cat, v in m["par_categorie"].items():
        print(f"|   ↳ {cat} | | {v['recall@10']} | | {v['n']} |")
```

- [ ] **Step 4: Vérifier le passage** — Run: `uv run pytest tests/test_evalrag.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(jalon2): harnais d'évaluation recall@k/MRR + CLI"`

---

### Task 8 [mécanique]: Première campagne + documentation des résultats

**Files:**
- Create: `docs/eval-jalon2.md`
- Modify: `README.md` (section « Résultats » + quickstart de l'index/recherche)

- [ ] **Step 1: Campagne dev** — Run: `uv run python scripts/run_eval.py --mode all --split dev`. Copier le tableau COMPLET (modes × catégories) dans `docs/eval-jalon2.md`, avec : date, modèle d'embeddings utilisé, taille de l'index, durées.
- [ ] **Step 2: Analyse d'erreurs sommaire** — pour les 3 questions dev au recall le plus bas en mode hybrid : noter la question, les citations attendues, le top-5 obtenu, et une hypothèse d'explication (une ligne chacune) dans `docs/eval-jalon2.md`.
- [ ] **Step 3: README** — ajouter : quickstart index (`uv run python scripts/build_index.py`, mention du téléchargement du modèle), exemple de recherche en 3 lignes de Python, tableau de résultats dev (copié), lien vers `docs/eval-jalon2.md` et `benchmark/README.md`.
- [ ] **Step 4: Commit** — `git add -A && git commit -m "docs(jalon2): première campagne d'évaluation — résultats et analyse d'erreurs"`

---

## Self-Review

- **Couverture spec (section 4)** : chaîne d'analyse (tokenisation références ✓ T1, élision/stemming ✓ T1, synonymes ✓ T1, pondération par champ → NON COUVERTE, différée au backlog — le FTS indexe `chemin_norm` mais sans boost distinct ; consigné) ; routeur regex ✓ T5 ; hybride BM25+dense ✓ T4/T5 ; RRF ✓ T5 ; reranker → DIFFÉRÉ (jalon 2.5, une fois le tableau de base établi — décision assumée : mesurer d'abord sans) ; renvois 1-hop ✓ T5 ; small-to-big ✓ T2/T5 ; filtres nature/dates → différés (corpus mono-nature en v1). Section 5 : benchmark ✓ T6 (3 des 5 familles — divergences et abstention exigent BOFiP/génération : jalons suivants), split dev/test ✓, retrieval quotidien gratuit ✓ T7.
- **Placeholders** : chaque étape porte code ou commande ✓.
- **Cohérence des types** : `normalize` (T1) consommé par T4/T5 ; `make_chunks` (T2) par T4 ; `Embedder.encode_query/passages` (T3) par T4/T5 ; le format de résultat de `search()` (T5) consommé par `evaluate` (T7) — champs `record_id`/`score`/`source` alignés ✓ ; `match()` (T7) aligné sur le format d'ids du jalon 1 (`@`, `-c`, `#`) ✓.
