# Jalon 4 — mesurer la génération : plan d'implémentation

> **Pour les agents :** SOUS-COMPÉTENCE REQUISE — utiliser
> `superpowers:subagent-driven-development` pour exécuter ce plan tâche par tâche.
> Les étapes utilisent des cases à cocher (`- [ ]`) pour le suivi.

**Objectif :** rendre mesurable ce que le système *répond*, et pas seulement ce
qu'il retrouve.

**Architecture :** un générateur contraint par sortie structurée (réponse citée ou
abstention explicite) ; une vérification des citations purement programmatique
(SQL et comparaison de chaînes, aucun LLM) ; un LLM-juge dont l'accord avec des
notes humaines est mesuré *avant* qu'il ne publie quoi que ce soit ; et une
extension du benchmark limitée à ce que le corpus actuel couvre déjà.

**Pile technique :** Python 3.13, `uv`, SQLite (+ FTS5, sqlite-vec), SDK
`anthropic`, `pytest`. Aucune dépendance nouvelle.

**Spec :** `docs/superpowers/specs/2026-08-17-jalon4-mesure-generation-design.md`
— l'exécutant lit les deux.

## Contraintes globales

Copiées de la spec et du `CLAUDE.md`. Elles s'appliquent à **chaque** tâche.

- **Aucune dépendance nouvelle.** `anthropic` est déjà là.
- **Aucune nouvelle source de corpus**, aucune reconstruction d'index, aucun
  changement du retrieval adopté au jalon 3. `SYNONYMES` est gelé.
- **Défauts neutres :** `Searcher()` sans argument doit continuer à rendre
  recall@10 = 0,672 sur dev. Rien dans ce jalon ne touche `src/accounting_rag/search.py`.
- **Tout chiffre publié est recalculable** depuis un JSON de `docs/mesures/jalon4/`
  produit par un script versionné. Y compris les chiffres dérivés.
- **`docs/mesures/**` est en lecture seule à l'exécution.** Tout `Rewriter` ou
  `Generator` pointant un cache versionné se construit avec `ecrire_cache=False`.
  Les caches d'exécution vont dans `data/`, gitignorés.
- **Aucun modèle ne voit le gold.** Générateur et juge reçoivent la question et les
  passages ; jamais `citations`, jamais le corpus entier. Test structurel obligatoire.
- **Tout contrôle doit être falsifiable :** un test qui le fait échouer, appliqué
  puis restauré, et mentionné dans le rapport de tâche.
- **Critère d'adoption inchangé** pour toute comparaison de configuration :
  `p_amelioration >= 0,95` sur la métrique déclarée *avant* la mesure, et aucune
  catégorie perdant plus de 0,05. Ne jamais substituer une métrique après coup.
- **Split gelé :** `benchmark/test.jsonl` ne sert à choisir aucun paramètre, et
  n'est exécuté qu'à la clôture.
- **Le benchmark change d'effectif** (90 → 150+). Les chiffres du jalon 3 ne sont
  **pas** comparables à ceux du jalon 4 ; toute publication doit le dire.
- **Commits :** gitmoji sur la branche, titre de PR conventionnel anglais ≤ 72
  caractères. Aucune attribution d'outil.

## Structure des fichiers

| fichier | responsabilité |
|---|---|
| `src/accounting_rag/generate.py` | `Generator` : question + passages → réponse citée ou abstention. Calque de `rewrite.py`. |
| `src/accounting_rag/citations.py` | Vérification programmatique des citations. **Fonctions pures**, aucun LLM, aucun réseau. |
| `src/accounting_rag/judge.py` | `Judge` : note une réponse contre un barème. Et `accord()`, qui mesure l'accord juge/humain. |
| `scripts/eval_generation.py` | Campagne : génère, vérifie les citations, agrège, persiste le JSON. |
| `scripts/calibrer_juge.py` | Mesure l'accord juge/humain sur le jeu de calibration et **échoue** sous le seuil. |
| `benchmark/abstention.jsonl` | Famille d'abstention, écrite à la main. |
| `docs/mesures/jalon4/calibration_juge.json` | Jeu de calibration : réponses, notes humaines, notes du juge. |
| `tests/test_generate.py`, `tests/test_citations.py`, `tests/test_judge.py` | Tests unitaires, clients factices, aucun réseau. |

---

## Task 1 : le générateur contraint

**Files:**
- Create: `src/accounting_rag/generate.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consomme : `Searcher.search(query, k, mode)` rend une liste de dicts
  `{"record_id", "article", "chemin", "texte", "score", "source"}`.
- Produit : `Generator(cache_path, modele=None, client=None, ecrire_cache=True)`
  avec `repondre(question: str, passages: list[dict]) -> dict` rendant
  `{"abstention": bool, "reponse": str, "citations": list[dict]}` où chaque
  citation est `{"record_id": str, "extrait": str}`. Attributs de comptage
  `appels`, `tokens_entree`, `tokens_sortie`, comme `Rewriter`.

- [ ] **Step 1 : écrire les tests d'abord** (`tests/test_generate.py`, aucun réseau)

```python
import json
import pytest
from accounting_rag.generate import Generator

PASSAGES = [
    {"record_id": "pcg-214-13@2026-01-01", "article": "214-13", "chemin": "Livre II",
     "texte": "Le mode d'amortissement linéaire est appliqué à défaut de mode mieux adapté.",
     "score": 1.0, "source": "fusion"},
]


class FauxBloc:
    def __init__(self, texte):
        self.type, self.text = "text", texte


class FauxMessage:
    def __init__(self, payload):
        self.content = [FauxBloc(json.dumps(payload, ensure_ascii=False))]
        self.stop_reason = "end_turn"
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class FauxClient:
    """Client Anthropic factice : enregistre les appels, ne sort jamais sur le réseau."""

    def __init__(self, payload=None):
        self.payload = payload or {
            "abstention": False,
            "reponse": "Le mode linéaire s'applique à défaut de mode mieux adapté.",
            "citations": [{"record_id": "pcg-214-13@2026-01-01",
                           "extrait": "Le mode d'amortissement linéaire est appliqué"}],
        }
        self.appels = []
        self.messages = self

    def create(self, **kwargs):
        self.appels.append(kwargs)
        return FauxMessage(self.payload)


def test_repondre_rend_une_reponse_citee(tmp_path):
    client = FauxClient()
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    out = g.repondre("comment amortir ?", PASSAGES)
    assert out["abstention"] is False
    assert out["citations"][0]["record_id"] == "pcg-214-13@2026-01-01"
    assert len(client.appels) == 1


def test_le_cache_evite_un_second_appel(tmp_path):
    client = FauxClient()
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    a = g.repondre("comment amortir ?", PASSAGES)
    b = g.repondre("comment amortir ?", PASSAGES)
    assert a == b
    assert len(client.appels) == 1


def test_abstention_est_transmise_telle_quelle(tmp_path):
    client = FauxClient({"abstention": True, "reponse": "Le corpus ne le dit pas.",
                         "citations": []})
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    out = g.repondre("quel est le taux d'IS ?", PASSAGES)
    assert out["abstention"] is True and out["citations"] == []


def test_le_generateur_ne_recoit_que_question_et_passages(tmp_path):
    """Intégrité du benchmark : ni gold, ni citations attendues dans le prompt."""
    client = FauxClient()
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    g.repondre("comment amortir ?", PASSAGES)
    envoye = json.dumps(client.appels[0], default=str)
    assert "comment amortir ?" in envoye
    for interdit in ("citations_attendues", "gold", "expected", "benchmark"):
        assert interdit not in envoye


def test_lecture_seule_leve_et_nappelle_pas_lapi(tmp_path):
    """`ecrire_cache=False` : question absente -> lève, sans appel ni écriture."""
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({}), encoding="utf-8")
    avant = cache.read_bytes()
    client = FauxClient()
    g = Generator(cache_path=cache, client=client, ecrire_cache=False)
    with pytest.raises(RuntimeError, match="lecture seule"):
        g.repondre("inconnue", PASSAGES)
    assert client.appels == [] and cache.read_bytes() == avant


def test_reponse_vide_leve_au_lieu_detre_mise_en_cache(tmp_path):
    client = FauxClient({"abstention": False, "reponse": "   ", "citations": []})
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    with pytest.raises(RuntimeError, match="réponse vide"):
        g.repondre("comment amortir ?", PASSAGES)
    assert not (tmp_path / "cache.json").exists()
```

- [ ] **Step 2 : lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_generate.py -q`
Attendu : FAIL, `ModuleNotFoundError: No module named 'accounting_rag.generate'`

- [ ] **Step 3 : implémenter `src/accounting_rag/generate.py`**

```python
"""Génération d'une réponse comptable citée, à partir des passages retrouvés.

Le générateur ne produit que DEUX formes de sortie : une réponse dont chaque
affirmation porte une citation, ou une abstention explicite. La contrainte est
appliquée par sortie structurée (`output_config.format`), pas par prompt : un
schéma JSON garantit que la réponse est analysable programmatiquement, ce qui est
la condition pour que la vérification des citations (citations.py) soit possible.

Le modèle ne reçoit QUE la question et les passages — jamais les citations
attendues du benchmark, jamais le corpus entier.
"""
import json
import os
from pathlib import Path

_DEFAUT = "claude-opus-5"

_SCHEMA = {
    "type": "object",
    "properties": {
        "abstention": {
            "type": "boolean",
            "description": "true si les passages fournis ne permettent pas de répondre",
        },
        "reponse": {
            "type": "string",
            "description": "La réponse, ou la raison de l'abstention.",
        },
        "citations": {
            "type": "array",
            "description": "Une entrée par affirmation. Vide si abstention.",
            "items": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "L'identifiant exact du passage cité, recopié tel quel.",
                    },
                    "extrait": {
                        "type": "string",
                        "description": (
                            "Un extrait VERBATIM du texte du passage, recopié "
                            "caractère pour caractère, qui soutient l'affirmation."
                        ),
                    },
                },
                "required": ["record_id", "extrait"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["abstention", "reponse", "citations"],
    "additionalProperties": False,
}

_SYSTEME = (
    "Tu réponds à une question de comptabilité française en t'appuyant EXCLUSIVEMENT "
    "sur les passages du Plan comptable général qui te sont fournis. "
    "Chaque affirmation de ta réponse doit être soutenue par une citation : le "
    "record_id du passage, recopié exactement, et un extrait VERBATIM de son texte. "
    "N'invente jamais un numéro d'article ni un extrait. "
    "Si les passages fournis ne permettent pas de répondre — parce que la réponse "
    "n'y figure pas, ou parce que la question relève de la fiscalité et non de la "
    "comptabilité — mets abstention à true et explique en une phrase ce qui manque. "
    "Une abstention correcte vaut mieux qu'une réponse inventée."
)


class Generator:
    def __init__(self, cache_path: str | Path, modele: str | None = None, client=None,
                 ecrire_cache: bool = True):
        self.cache_path = Path(cache_path)
        if modele is None:
            # charge_env() AVANT de lire la variable : sinon une valeur placée dans
            # .env n'aurait aucun effet, le module étant importé avant tout
            # chargement du fichier (défaut corrigé sur Rewriter au jalon 3).
            from .config import charge_env
            charge_env()
            modele = os.environ.get("ACCRAG_GEN_MODEL", _DEFAUT)
        self.modele = modele
        self.ecrire_cache = ecrire_cache
        self._client = client
        self._cache: dict[str, dict] = {}
        if self.cache_path.is_file():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.appels = 0
        self.tokens_entree = 0
        self.tokens_sortie = 0

    @property
    def client(self):
        if self._client is None:
            import anthropic  # import paresseux : le module s'importe sans clé ni réseau
            from .config import charge_env
            charge_env()
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _cle(question: str, passages: list[dict]) -> str:
        """Clé de cache : la question ET les passages, puisque la réponse dépend des deux."""
        return json.dumps([question, [p["record_id"] for p in passages]],
                          ensure_ascii=False, sort_keys=True)

    def repondre(self, question: str, passages: list[dict]) -> dict:
        cle = self._cle(question, passages)
        if cle in self._cache:
            return self._cache[cle]
        if not self.ecrire_cache:
            raise RuntimeError(
                f"Generator en lecture seule (ecrire_cache=False) et entrée absente du "
                f"cache {self.cache_path} : {question!r}. Aucun appel API ni écriture.")
        if self.appels >= 400:
            raise RuntimeError("garde-fou : plus de 400 appels API dans une exécution")

        contexte = "\n\n".join(
            f"[{p['record_id']}] ({p['chemin']}, article {p['article']})\n{p['texte']}"
            for p in passages)
        reponse = self.client.messages.create(
            model=self.modele,
            max_tokens=2000,
            system=_SYSTEME,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user",
                       "content": f"Question : {question}\n\nPassages :\n{contexte}"}],
        )
        self.appels += 1
        usage = getattr(reponse, "usage", None)
        if usage is not None:
            self.tokens_entree += getattr(usage, "input_tokens", 0) or 0
            self.tokens_sortie += getattr(usage, "output_tokens", 0) or 0

        texte = "".join(b.text for b in reponse.content
                        if getattr(b, "type", None) == "text").strip()
        if not texte:
            raise RuntimeError(
                f"réponse vide renvoyée par {self.modele} pour : {question!r} "
                f"(blocs : {[getattr(b, 'type', None) for b in reponse.content]}, "
                f"stop_reason : {getattr(reponse, 'stop_reason', None)!r})")
        out = json.loads(texte)
        if not out.get("reponse", "").strip():
            # Ne JAMAIS mettre en cache une réponse vide : le cache étant committé,
            # elle serait rejouée silencieusement par toutes les campagnes suivantes.
            raise RuntimeError(
                f"réponse vide dans le JSON renvoyé par {self.modele} pour : {question!r}")

        self._cache[cle] = out
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        return out
```

- [ ] **Step 4 : lancer les tests pour vérifier qu'ils passent**

Run : `uv run pytest tests/test_generate.py -q`
Attendu : PASS, 6 tests.

- [ ] **Step 5 : vérifier que les tests savent échouer**

Casser volontairement, un à la fois, puis restaurer :
1. remplacer `if not self.ecrire_cache:` par `if False:` → `test_lecture_seule…` doit échouer ;
2. supprimer le `raise` de la réponse vide → `test_reponse_vide…` doit échouer ;
3. ajouter `"gold": "pcg-214-13"` au dict `messages` → `test_le_generateur_ne_recoit_que…` doit échouer.

Run après chaque mutation : `uv run pytest tests/test_generate.py -q`
Attendu : le test correspondant ÉCHOUE. Restaurer, puis re-vérifier PASS.
**Consigner dans le rapport de tâche que les trois mutations ont été appliquées.**

- [ ] **Step 6 : lancer la suite complète**

Run : `uv run pytest -q`
Attendu : 162 passed (156 + 6), aucune régression.

- [ ] **Step 7 : commit**

```bash
git add src/accounting_rag/generate.py tests/test_generate.py
git commit -m "✨ Add the constrained generator with structured citation output"
```

---

## Task 2 : la vérification programmatique des citations

**Files:**
- Create: `src/accounting_rag/citations.py`
- Create: `tests/test_citations.py`

**Interfaces:**
- Consomme : la sortie de `Generator.repondre()`, et `data/corpus.db` (table
  `records`, colonnes `id`, `texte`).
- Produit :
  - `normaliser_pour_comparaison(texte: str) -> str`
  - `verifier_citation(con, record_id: str, extrait: str) -> str` rendant
    `"ok"` | `"record_inexistant"` | `"extrait_absent"` | `"extrait_trop_court"`
  - `metriques(reponses: dict[str, dict], db_path: Path) -> dict` rendant
    `{"n", "taux_citations_inexistantes", "taux_citations_non_portantes",
      "taux_reponses_sans_citation", "taux_abstention", "par_question", "details"}`

- [ ] **Step 1 : écrire les tests d'abord** (`tests/test_citations.py`)

```python
import sqlite3
import pytest
from accounting_rag.citations import (
    EXTRAIT_MINIMUM, metriques, normaliser_pour_comparaison, verifier_citation)


@pytest.fixture
def con(tmp_path):
    """Base minimale : deux records, dont un au texte à espaces irréguliers."""
    db = tmp_path / "mini.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE records (id TEXT PRIMARY KEY, texte TEXT)")
    c.executemany("INSERT INTO records VALUES (?, ?)", [
        ("pcg-214-13@2026-01-01",
         "Le mode  d'amortissement\nlinéaire est appliqué à défaut de mode mieux adapté."),
        ("pcg-212-3@2026-01-01", "Un actif est un élément identifiable du patrimoine."),
    ])
    c.commit()
    return c


def test_normalisation_replie_les_espaces_et_la_casse():
    assert (normaliser_pour_comparaison("Le mode  d'amortissement\nLINÉAIRE")
            == normaliser_pour_comparaison("le mode d'amortissement linéaire"))


def test_extrait_verbatim_est_ok_malgre_les_espaces(con):
    assert verifier_citation(
        con, "pcg-214-13@2026-01-01",
        "Le mode d'amortissement linéaire est appliqué à défaut") == "ok"


def test_record_inexistant_est_detecte(con):
    assert verifier_citation(
        con, "pcg-999-99@2026-01-01",
        "Le mode d'amortissement linéaire est appliqué à défaut") == "record_inexistant"


def test_extrait_absent_du_record_est_detecte(con):
    """Le record existe mais ne contient pas l'extrait : citation non portante."""
    assert verifier_citation(
        con, "pcg-212-3@2026-01-01",
        "Le mode d'amortissement linéaire est appliqué à défaut") == "extrait_absent"


def test_extrait_trop_court_est_refuse(con):
    """Un extrait de quelques caractères matcherait n'importe quoi : il ne prouve rien."""
    assert len("amortir") < EXTRAIT_MINIMUM
    assert verifier_citation(con, "pcg-214-13@2026-01-01", "amortir") == "extrait_trop_court"


def test_metriques_agrege_les_trois_taux(tmp_path, con):
    db = tmp_path / "mini.db"
    reponses = {
        "q1": {"abstention": False, "reponse": "a", "citations": [
            {"record_id": "pcg-214-13@2026-01-01",
             "extrait": "Le mode d'amortissement linéaire est appliqué à défaut"}]},
        "q2": {"abstention": False, "reponse": "b", "citations": [
            {"record_id": "pcg-999-99@2026-01-01", "extrait": "un extrait inventé de longueur suffisante"}]},
        "q3": {"abstention": False, "reponse": "c", "citations": []},
        "q4": {"abstention": True, "reponse": "le corpus ne le dit pas", "citations": []},
    }
    m = metriques(reponses, db)
    assert m["n"] == 4
    assert m["taux_citations_inexistantes"] == 0.5      # 1 citation sur 2 vérifiables
    assert m["taux_reponses_sans_citation"] == 1 / 3    # q3 sur les 3 non-abstentions
    assert m["taux_abstention"] == 0.25
    assert m["par_question"]["q1"] == "ok"
```

- [ ] **Step 2 : lancer les tests pour vérifier qu'ils échouent**

Run : `uv run pytest tests/test_citations.py -q`
Attendu : FAIL, `ModuleNotFoundError: No module named 'accounting_rag.citations'`

- [ ] **Step 3 : implémenter `src/accounting_rag/citations.py`**

```python
"""Vérification programmatique des citations : aucun LLM, aucun réseau.

C'est la brique la moins truquable du projet, et elle passe volontairement AVANT
le LLM-juge : le juge mesure la qualité d'une réponse, ceci mesure son honnêteté.
La faute la plus grave qu'un RAG comptable puisse commettre est de citer un
article qui n'existe pas, et elle se détecte en SQL.

Trois verdicts possibles par citation, plus un refus de principe :
  - "ok"                  : le record existe et contient l'extrait
  - "record_inexistant"   : citation hallucinée
  - "extrait_absent"      : le record existe mais ne porte pas l'affirmation
  - "extrait_trop_court"  : l'extrait ne prouve rien, on refuse de le valider
"""
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

# Un extrait plus court que cela matcherait du texte par accident : le valider
# donnerait un faux « ok » et rendrait tout le contrôle inutile.
EXTRAIT_MINIMUM = 30

_ESPACES = re.compile(r"\s+")


def normaliser_pour_comparaison(texte: str) -> str:
    """Replie espaces, casse et forme Unicode — RIEN d'autre.

    Volontairement plus strict que `normalize.normalize()` : on ne déplie pas les
    élisions, on ne stemme pas, on ne replie pas les accents. Un extrait doit être
    verbatim ; seules les différences de mise en page (retours à la ligne, espaces
    doubles du PDF) sont tolérées.
    """
    return _ESPACES.sub(" ", unicodedata.normalize("NFC", texte)).strip().casefold()


def verifier_citation(con: sqlite3.Connection, record_id: str, extrait: str) -> str:
    if len(extrait.strip()) < EXTRAIT_MINIMUM:
        return "extrait_trop_court"
    row = con.execute("SELECT texte FROM records WHERE id = ?", (record_id,)).fetchone()
    if row is None:
        return "record_inexistant"
    if normaliser_pour_comparaison(extrait) in normaliser_pour_comparaison(row[0]):
        return "ok"
    return "extrait_absent"


def metriques(reponses: dict[str, dict], db_path: str | Path) -> dict:
    """Agrège les trois taux sur un dict {question_id: sortie de Generator.repondre}."""
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    verdicts: Counter[str] = Counter()
    par_question: dict[str, str] = {}
    details: dict[str, list[dict]] = {}
    n_sans_citation = n_abstentions = n_non_abstentions = 0

    for qid, r in sorted(reponses.items()):
        if r.get("abstention"):
            n_abstentions += 1
            par_question[qid] = "abstention"
            continue
        n_non_abstentions += 1
        cits = r.get("citations") or []
        if not cits:
            n_sans_citation += 1
            par_question[qid] = "sans_citation"
            continue
        vus = []
        for c in cits:
            v = verifier_citation(con, c["record_id"], c["extrait"])
            verdicts[v] += 1
            vus.append({"record_id": c["record_id"], "verdict": v})
        details[qid] = vus
        # Le verdict de la question est le pire de ses citations : une seule
        # citation hallucinée suffit à rendre la réponse malhonnête.
        for pire in ("record_inexistant", "extrait_absent", "extrait_trop_court", "ok"):
            if any(v["verdict"] == pire for v in vus):
                par_question[qid] = pire
                break
    con.close()

    total_cit = sum(verdicts.values())
    def taux(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    return {
        "n": len(reponses),
        "n_citations": total_cit,
        "verdicts": dict(verdicts),
        "taux_citations_inexistantes": taux(verdicts["record_inexistant"], total_cit),
        "taux_citations_non_portantes": taux(
            verdicts["extrait_absent"] + verdicts["extrait_trop_court"], total_cit),
        "taux_reponses_sans_citation": taux(n_sans_citation, n_non_abstentions),
        "taux_abstention": taux(n_abstentions, len(reponses)),
        "par_question": par_question,
        "details": details,
    }
```

- [ ] **Step 4 : lancer les tests pour vérifier qu'ils passent**

Run : `uv run pytest tests/test_citations.py -q`
Attendu : PASS, 6 tests.

- [ ] **Step 5 : vérifier que les tests savent échouer**

1. remplacer `EXTRAIT_MINIMUM = 30` par `= 0` → `test_extrait_trop_court…` doit échouer ;
2. dans `verifier_citation`, renvoyer `"ok"` avant le test d'appartenance →
   `test_extrait_absent…` doit échouer.

Restaurer après chaque mutation. **Consigner les deux dans le rapport.**

- [ ] **Step 6 : contrôle sur le corpus réel** (aucune écriture)

```bash
uv run python -c "
import sqlite3
from accounting_rag.citations import verifier_citation
con = sqlite3.connect('file:data/corpus.db?mode=ro', uri=True)
rid, texte = con.execute(\"SELECT id, texte FROM records WHERE length(texte) > 200 LIMIT 1\").fetchone()
print('verbatim  ->', verifier_citation(con, rid, texte[50:140]))
print('inventé   ->', verifier_citation(con, rid, 'ceci est un extrait totalement inventé de longueur suffisante'))
print('inexistant->', verifier_citation(con, 'pcg-999-99@2026-01-01', texte[50:140]))
"
```
Attendu : `ok`, `extrait_absent`, `record_inexistant`.

- [ ] **Step 7 : commit**

```bash
git add src/accounting_rag/citations.py tests/test_citations.py
git commit -m "✨ Add programmatic citation verification"
```

---

## Task 3 : la campagne de génération et sa première mesure

**Files:**
- Create: `scripts/eval_generation.py`
- Create: `docs/mesures/jalon4/generation_dev.json` (produit par le script)

**Interfaces:**
- Consomme : `load_benchmark`, `Searcher`, `Reranker`, `Rewriter`, `Generator`,
  `citations.metriques`.
- Produit : `docs/mesures/jalon4/generation_{split}.json` contenant
  `{"split", "n", "config", "metriques", "reponses", "cout"}`.

- [ ] **Step 1 : écrire le script**

Configuration mesurée = **celle livrée au jalon 3**, sans y toucher : réécriture
`etend` + `hybrid+rerank` (`bge-reranker-v2-m3`, `n_rerank=25`, `pool=50`).

Points obligatoires, chacun repris d'une leçon du jalon 3 :
- `Rewriter(cache_path="docs/mesures/jalon3/reecritures.json", ecrire_cache=False)`
  — l'ancrage de reproductibilité ne doit pas être réécrit ;
- `Generator(cache_path="docs/mesures/jalon4/reponses_{split}.json")`, en écriture,
  puisque c'est l'artefact que ce jalon produit ;
- un **contrôle de fraîcheur** avant tout appel payant : `Searcher()` neutre sur dev
  doit rendre `recall@10 == 0.672`, sinon `SystemExit` avec `STATUS: BLOCKED` ;
- persistance des `reponses` brutes ET des métriques, comptage des tokens.

- [ ] **Step 2 : lancer le contrôle de fraîcheur seul** (gratuit)

Run : `uv run python scripts/eval_generation.py --split dev --controle-seul`
Attendu : `contrôle OK : recall@10 = 0.672`.

- [ ] **Step 3 : lancer la campagne sur dev** (61 appels API)

Run : `uv run python scripts/eval_generation.py --split dev`
Attendu : les quatre taux affichés, `docs/mesures/jalon4/generation_dev.json` écrit.

- [ ] **Step 4 : inspecter à la main les 5 premières réponses**

Vérifier qu'aucune ne cite un `record_id` absent des passages fournis, et que les
extraits sont bien verbatim. **Reporter les cinq dans le rapport de tâche.**

- [ ] **Step 5 : commit**

```bash
git add scripts/eval_generation.py docs/mesures/jalon4/
git commit -m "✨ Add the generation campaign and its first dev measurement"
```

---

## Task 4 : la famille d'abstention

**Files:**
- Create: `benchmark/abstention.jsonl`
- Modify: `tests/test_benchmark_format.py` (ajouter les gardes de format)
- Modify: `scripts/eval_generation.py` (`--split abstention`)

**Interfaces:**
- Produit : `benchmark/abstention.jsonl`, un objet JSON par ligne, mêmes clés que
  `dev.jsonl` plus `attendu` : `{"id", "question", "categorie", "citations": [],
  "attendu": "abstention", "raison", "notes"}`.

- [ ] **Step 1 : écrire 30 questions, réparties en trois raisons**

10 par raison, chacune avec sa `raison` explicite :

| `raison` | ce qui est testé | exemple d'énoncé |
|---|---|---|
| `hors_corpus` | la réponse n'est dans aucun texte du corpus | « Quel est le taux normal de l'impôt sur les sociétés en 2026 ? » |
| `fiscal_pas_comptable` | la question relève du fiscal | « La provision pour indemnités de fin de carrière est-elle déductible ? » |
| `hors_perimetre` | ni comptable ni fiscal | « Quel est le délai de prescription d'une action en responsabilité civile ? » |

Règle de rédaction : chaque question doit être **plausible pour un praticien** et
**réellement sans réponse dans le corpus** — vérifié par une recherche avant de
l'écrire. `citations` reste vide, par construction.

- [ ] **Step 2 : ajouter les gardes de format**

```python
def test_abstention_format():
    lignes = [json.loads(l) for l in open("benchmark/abstention.jsonl")]
    assert len(lignes) >= 30
    assert all(q["citations"] == [] for q in lignes), "une question d'abstention n'a pas de gold"
    assert all(q["attendu"] == "abstention" for q in lignes)
    raisons = collections.Counter(q["raison"] for q in lignes)
    assert set(raisons) == {"hors_corpus", "fiscal_pas_comptable", "hors_perimetre"}
    assert min(raisons.values()) >= 8, f"répartition déséquilibrée : {raisons}"
    assert len({q["id"] for q in lignes}) == len(lignes), "ids dupliqués"
```

- [ ] **Step 3 : vérifier qu'aucune question d'abstention n'a en fait une réponse**

Pour chaque question, lancer le retrieval et vérifier à la main qu'aucun des 10
passages ne répond réellement. Toute question qui trouve sa réponse est **retirée**
— une question d'abstention à laquelle le corpus répond est un piège invalide.

Run : `uv run python scripts/run_eval.py --split abstention --mode hybrid+rerank`
(le recall n'a pas de sens ici — c'est la lecture des passages qui compte).

- [ ] **Step 4 : mesurer le taux d'abstention correcte**

Run : `uv run python scripts/eval_generation.py --split abstention`
Métrique : `taux_abstention` sur ce split = **taux d'abstention correcte**. Toute
non-abstention est une réponse inventée, à lister nommément dans le rapport.

- [ ] **Step 5 : commit**

```bash
git add benchmark/abstention.jsonl tests/test_benchmark_format.py \
        scripts/eval_generation.py docs/mesures/jalon4/
git commit -m "✨ Add the abstention family and measure correct abstention"
```

---

## Task 5 : le LLM-juge et sa calibration

**Files:**
- Create: `src/accounting_rag/judge.py`
- Create: `scripts/calibrer_juge.py`
- Create: `docs/mesures/jalon4/calibration_juge.json`
- Create: `tests/test_judge.py`

**Interfaces:**
- Produit :
  - `Judge(cache_path, modele=None, client=None, ecrire_cache=True)` avec
    `noter(question: str, reponse: dict, bareme: list[str]) -> dict` rendant
    `{"note": int, "sur": int, "par_critere": list[dict]}` ;
  - `accord(humaines: dict[str, int], juge: dict[str, int]) -> dict` rendant
    `{"n", "exact", "ecart_moyen", "kappa_pondere"}`.

**SEUIL D'ACCEPTATION, FIXÉ AVANT TOUTE MESURE :** `kappa_pondere >= 0,60`. Sous
ce seuil, **le juge ne publie aucun chiffre** : l'échec est documenté comme
résultat négatif, exactement comme une ablation rejetée. Ce seuil ne doit pas être
révisé après avoir vu le résultat.

- [ ] **Step 1 : écrire les tests du juge et de l'accord**

Tests de `accord()` d'abord — c'est une fonction pure, donc le cœur testable :

```python
from accounting_rag.judge import accord

def test_accord_parfait():
    a = {"c1": 3, "c2": 1, "c3": 2}
    r = accord(a, a)
    assert r["exact"] == 1.0 and r["ecart_moyen"] == 0.0 and r["kappa_pondere"] == 1.0

def test_accord_nul_quand_le_juge_note_a_lenvers():
    r = accord({"c1": 0, "c2": 3}, {"c1": 3, "c2": 0})
    assert r["kappa_pondere"] < 0.0 and r["exact"] == 0.0

def test_ecart_moyen_est_bien_une_moyenne():
    r = accord({"c1": 3, "c2": 3}, {"c1": 2, "c2": 3})
    assert r["ecart_moyen"] == 0.5 and r["exact"] == 0.5

def test_accord_refuse_des_ensembles_de_cles_differents():
    import pytest
    with pytest.raises(ValueError, match="mêmes questions"):
        accord({"c1": 1}, {"c2": 1})
```

- [ ] **Step 2 : lancer, vérifier l'échec** — `ModuleNotFoundError`.

- [ ] **Step 3 : implémenter `judge.py`**

`accord()` calcule un kappa de Cohen pondéré (poids quadratiques) sur des notes
entières, plus le taux d'accord exact et l'écart moyen absolu. `Judge.noter()`
suit le même patron que `Generator` : sortie structurée, cache disque,
`ecrire_cache`, import paresseux, garde-fou d'appels.

- [ ] **Step 4 : lancer, vérifier que les tests passent.**

- [ ] **Step 5 : construire le jeu de calibration — 30 réponses notées à la main**

Prendre 30 réponses de `docs/mesures/jalon4/generation_dev.json` en couvrant
**délibérément** les cinq cas limites, six de chaque :

| cas | ce qu'il piège |
|---|---|
| juste et bien citée | le juge doit noter haut |
| juste mais mal citée | le juge doit pénaliser la citation, pas le fond |
| fausse mais bien citée | le juge ne doit pas se laisser convaincre par la forme |
| abstention correcte | le juge doit la récompenser, pas la punir |
| abstention excessive | le juge doit la pénaliser (la réponse était dans les passages) |

Noter chacune de 0 à 3 **avant** de faire tourner le juge, et écrire les notes dans
`docs/mesures/jalon4/calibration_juge.json`, champ `notes_humaines`.

- [ ] **Step 6 : mesurer l'accord**

Run : `uv run python scripts/calibrer_juge.py`
Le script écrit `notes_juge` et `accord` dans le même JSON, affiche le kappa, et
**sort en code 1 si `kappa_pondere < 0.60`**.

- [ ] **Step 7 : deux issues, toutes deux acceptables**

- **kappa ≥ 0,60** → le juge sert. Passer à la notation du benchmark complet.
- **kappa < 0,60** → **le juge ne publie rien.** Documenter l'échec dans
  `docs/eval-jalon4.md` comme un résultat négatif, avec le kappa obtenu et les cas
  limites où le désaccord se concentre. Ne PAS réviser le seuil. Un second essai
  avec un prompt différent est permis ; le premier échec reste publié.

- [ ] **Step 8 : commit**

```bash
git add src/accounting_rag/judge.py scripts/calibrer_juge.py tests/test_judge.py \
        docs/mesures/jalon4/calibration_juge.json
git commit -m "✨ Add the LLM judge and measure its agreement with human scores"
```

---

## Task 6 : l'extension du benchmark

**Files:**
- Modify: `benchmark/dev.jsonl`, `benchmark/test.jsonl`
- Create: `benchmark/validation.jsonl` (le second split gelé)
- Modify: `tests/test_benchmark_format.py`
- Modify: `benchmark/README.md`

- [ ] **Step 1 : rédiger les questions depuis le corpus, jamais depuis les corrigés**

Trois sources, dans cet ordre :
1. **Dossier fusions du DSCG UE4** — Titre VII, présent dans le corpus (81 records).
   Golds repérés : prime de fusion → `744-1`, `744-2`, `751-2`, `751-3`, `752-5` ;
   mali de fusion → `745-3`, `745-4`, `751-4`.
2. **DCG UE9 et UE10** — comptabilité et comptabilité approfondie, Livres I à V.
3. **Divergences 2058-A** — écrire la question, golder la **moitié comptable**,
   marquer la moitié fiscale `"gold_fiscal": "a_completer"`.

Les corrigés Compta Online servent de **contre-vérification privée uniquement** :
rédiger le gold depuis le PCG, puis confronter. Rien n'en dérive dans le dépôt.

- [ ] **Step 2 : vérifier chaque gold en SQL**

Pour chaque citation ajoutée : `SELECT id FROM records WHERE id LIKE 'pcg-<num>%'`
doit render au moins une ligne. Un gold inexistant est un bug de benchmark.

- [ ] **Step 3 : geler le second split de validation**

`benchmark/validation.jsonl`, stratifié par catégorie comme dev/test, **jamais
exécuté avant la clôture du jalon 4**. Documenter la date de gel dans
`benchmark/README.md`.

- [ ] **Step 4 : mettre à jour les gardes de format** (comptes, stratification, unicité des ids).

- [ ] **Step 5 : commit**

```bash
git add benchmark/ tests/test_benchmark_format.py
git commit -m "✨ Extend the benchmark to 150 questions and freeze a validation split"
```

---

## Task 7 : la clôture

**Files:**
- Create: `docs/eval-jalon4.md`
- Modify: `README.md`, `JOURNAL.md`
- Create: `scripts/cloture_jalon4.py`

- [ ] **Step 1 : campagne finale** sur dev, puis **UNE seule exécution** de
  `benchmark/validation.jsonl`. Contrôles de fraîcheur avant tout appel payant.

- [ ] **Step 2 : rédiger `docs/eval-jalon4.md`** avec, obligatoirement :
  - les quatre taux de citation, par split ;
  - le taux d'abstention correcte, et le nom des questions où le système a inventé ;
  - le kappa du juge et sa décision (sert / ne publie rien) ;
  - **la mention explicite que les chiffres ne sont pas comparables au jalon 3**,
    le benchmark ayant changé d'effectif ;
  - les réserves, dont celles héritées : dépendance API non déterministe, latences
    dépendantes du device, dettes RRF et réécriture non traitées.

- [ ] **Step 3 : contrôle scripté des chiffres publiés**

Écrire un contrôle qui recalcule chaque chiffre cité depuis
`docs/mesures/jalon4/*.json`, y compris les chiffres dérivés, et échoue sur écart.
C'est la leçon la plus chère du jalon 3 : trois vagues de correctifs y ont été
consacrées.

- [ ] **Step 4 : commit et PR** au titre conventionnel.

---

## Auto-relecture

**1. Couverture de la spec.** §3.1 → Task 1. §3.2 → Task 2. §3.3 → Task 5. §3.4 →
Task 4. §4.1/4.2 → Task 6. §4.3 (second split) → Task 6 Step 3. §7 critères de
clôture → Task 7. **Aucune section sans tâche.**

**2. Placeholders.** Aucun « TBD » ni « ajouter la gestion d'erreurs ». Les seuls
`a_completer` sont des **valeurs de données** volontaires (moitié fiscale d'une
divergence 2058-A), pas des trous de plan.

**3. Cohérence des types.** `Generator.repondre()` rend
`{"abstention", "reponse", "citations"}` — consommé sous cette forme exacte par
`citations.metriques()` (Task 2) et `Judge.noter()` (Task 5). `verifier_citation()`
rend les quatre chaînes listées, et `metriques()` n'en agrège pas d'autres.
`accord()` rend `kappa_pondere`, seul nom utilisé par le seuil de Task 5 Step 7.
