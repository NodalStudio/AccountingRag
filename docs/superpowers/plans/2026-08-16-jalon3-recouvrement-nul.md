# Jalon 3 — Franchir le recouvrement lexical nul

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire entrer dans la fenêtre de candidats les articles qui ne partagent aucun mot avec la question, en mesurant trois leviers : filtrage des tokens peu discriminants, élargissement du pool, reranking sur pool large.

**Architecture:** Le retrieval du jalon 2.5 reste la base (routeur → BM25 + dense → RRF → graphe → reranker optionnel). Les trois leviers sont des paramètres de requête (aucun changement d'index, donc **aucun rebuild dans tout ce jalon**), neutres par défaut, mesurés un à un par bootstrap apparié sur le split dev v2.

**Tech Stack:** Python ≥3.12 via uv, SQLite FTS5 + sqlite-vec, sentence-transformers (Embedder + CrossEncoder déjà présents), pytest. **Aucune dépendance nouvelle.**

**Spec:** docs/superpowers/specs/2026-08-14-accountingrag-design.md (§4-5 ; le protocole d'ablation de la §5 est la loi). Contexte chiffré : docs/eval-jalon25.md.

## Diagnostic fondateur (sondes exécutées avant rédaction du plan)

Deux sondes sur les questions dev en échec (`vocabulaire_courant`) ont établi :

| question | gold | rang lexical (OR de tous les tokens) | rang lexical (tokens df ≤ 2 %) | rang dense |
|---|---|---|---|---|
| q021 | pcg-214-13 | **154/154** (dernier) | absent (40 candidats) | 178/1660 |
| q026 | pcg-212-3 | **46/46** (dernier) | **14/14** | 248/1660 |
| q060 | pcg-1222-74 | **1430/1430** (dernier) | absent (63 candidats) | 528/1660 |
| q023 | pcg-214-22 | — | — | 257/1660 |

Lectures qui fondent les trois ablations :
1. Le gold est **toujours dernier** de sa liste : il n'est retrouvé que par des mots fonctionnels (`le`, `de`, `une`), sans aucun token de contenu partagé. Le canal lexical est structurellement muet sur ces questions.
2. Filtrer les tokens à forte fréquence documentaire **réduit massivement le bruit** (q060 : 1430 → 63 candidats) et peut faire entrer le gold dans la fenêtre utile (q026 : rang 46 → 14). Mais si le gold ne partage aucun token rare, il disparaît du pool (q021, q060) — d'où la nécessité de mesurer, pas de supposer.
3. Le canal dense place le gold aux rangs 178-528 : **hors de la fenêtre de 50, mais atteignable** par un pool de 200-400. C'est le reranker (seul composant qui lit le sens d'une paire question/passage) qui peut ensuite le promouvoir.
4. La conjonction (`AND`) des tokens rares donne **zéro candidat** dans les trois cas : voie morte, exclue de ce plan.

Corollaire de coût qui décide de l'ablation F : le reranker adopté (`bge`, ~4,7 s/candidat) est inutilisable sur pool large (200 candidats ≈ 15 min/question). Le modèle léger rejeté au jalon 2.5 (`mmarco`, ~0,4 s/candidat) redevient le seul viable (~80 s/question) — l'arbitrage « modèle faible sur pool large » contre « modèle fort sur pool étroit » est la vraie question de ce jalon.

## Global Constraints

- Exécution via `uv run ...` ; **aucune dépendance nouvelle** dans pyproject.toml.
- **Aucun rebuild d'index** : les trois leviers agissent au moment de la requête. Toute proposition impliquant un rebuild est hors périmètre de ce jalon.
- Les tables du jalon 1 (`records`, `renvois`, `records_fts`) et l'index (`chunks`, `chunks_norm`, `chunks_vec`) ne sont jamais modifiés.
- `normalize()` reste l'unique fonction de normalisation, identique build/requête ; **`SYNONYMES` n'est pas touché** dans ce jalon (il exigerait un rebuild).
- Rétro-compatibilité : chaque nouveau paramètre de `Searcher` a une valeur par défaut qui reproduit exactement le comportement du jalon 2.5, jusqu'à ce qu'une mesure l'adopte.
- Protocole d'ablation : une variable à la fois ; comparaison au meilleur cumulé précédent ; `paired_bootstrap` (n_boot=10000, seed=42) sur recall@10 par question ; adoption si `p_amelioration ≥ 0,95` **ET** aucune catégorie ne perd plus de 0,05 de recall@10 ; un résultat négatif mesuré est un livrable et doit être documenté avec ses chiffres.
- **Traçabilité (règle de projet, issue du jalon 2.5)** : toute mesure passe par un script versionné (`scripts/ablations_jalon3.py`) et persiste ses dicts `par_question` bruts dans `docs/mesures/jalon3/` ; tout chiffre cité dans un doc doit être recalculable depuis ces JSON. Aucun chiffre issu d'un script jetable.
- Le split test (`benchmark/test.jsonl`, 29 questions) est gelé : **une seule exécution**, à la tâche 4, jamais utilisée pour choisir un paramètre.
- Benchmark inchangé (90 questions v2) : ce jalon mesure le système, pas l'instrument. Aucune comparaison avec des chiffres n=21.
- Messages de commit en français, préfixés `feat(jalon3):` / `fix(jalon3):` / `docs(jalon3):`.
- Toute commande dépassant 5 min (campagnes rerank, suite complète) se lance en arrière-plan (`run_in_background`) ; le reste au premier plan.

---

### Task 1: Statistiques de fréquence documentaire + filtrage des tokens de requête (ablation D)

**Files:**
- Modify: `src/accounting_rag/search.py`
- Modify: `tests/test_search.py`
- Create: `scripts/diagnostic_rangs.py`
- Create: `scripts/ablations_jalon3.py`
- Create: `docs/eval-jalon3.md`

**Interfaces:**
- Consumes: `Searcher(db_path, embedder=None, poids_chemin=1.0, boost_commentaire=1.0)` existant ; `evaluate`/`paired_bootstrap` de `evalrag`.
- Produces:
  - `Searcher(..., df_max: float | None = None)` — `None` = comportement jalon 2.5 (aucun filtrage). Sinon, fraction de `COUNT(*) FROM chunks` au-delà de laquelle un token de requête est écarté du `MATCH`.
  - `Searcher.df(token) -> int` — fréquence documentaire d'un token, mise en cache par instance.
  - `scripts/diagnostic_rangs.py --questions q021,q026 [--split dev]` — rang du gold par canal (lexical filtré ou non, dense), sortie markdown + JSON dans `docs/mesures/jalon3/`. Version versionnée et généralisée des sondes du diagnostic.
  - `scripts/ablations_jalon3.py --ablation D|E|F|cumul [--split dev|test]` — exécute une ablation, écrit `docs/mesures/jalon3/<ablation>_<split>.json` (agrégats **et** `par_question` bruts pour chaque configuration), imprime le tableau bootstrap.

- [ ] **Step 1 : Écrire les tests d'abord** (dans `tests/test_search.py`, sur la fixture de base synthétique existante `db_synthetique` + `FakeEmbedder`) :

```python
def test_df_compte_les_chunks_contenant_le_token(db_synthetique):
    s = Searcher(db_synthetique, embedder=FakeEmbedder())
    # le corpus synthétique contient le terme dans un seul chunk
    assert s.df("amortissement") == 1
    assert s.df("motabsent") == 0


def test_df_est_mis_en_cache(db_synthetique):
    s = Searcher(db_synthetique, embedder=FakeEmbedder())
    s.df("amortissement")
    avant = s.con.total_changes  # inchangé par un SELECT ; on vérifie le cache autrement
    assert "amortissement" in s._df_cache
    # deuxième appel : servi par le cache (pas de requête), valeur identique
    assert s.df("amortissement") == s._df_cache["amortissement"]


def test_df_max_neutre_par_defaut(db_synthetique):
    """df_max=None doit reproduire exactement le comportement jalon 2.5."""
    a = Searcher(db_synthetique, embedder=FakeEmbedder())
    b = Searcher(db_synthetique, embedder=FakeEmbedder(), df_max=None)
    for requete in ("amortissement", "immobilisation corporelle", "que dit l'article 111-1 ?"):
        assert [r["record_id"] for r in a.search(requete, mode="bm25")] == \
               [r["record_id"] for r in b.search(requete, mode="bm25")]


def test_df_max_ecarte_les_tokens_trop_frequents(db_synthetique):
    """Un token présent dans tous les chunks est écarté ; le token rare décide seul."""
    s = Searcher(db_synthetique, embedder=FakeEmbedder(), df_max=0.5)
    termes = s._termes_match("le amortissement")  # 'le' est dans tous les chunks de la fixture
    assert "amortissement" in termes
    assert "le" not in termes


def test_df_max_repli_si_tous_les_tokens_sont_ecartes(db_synthetique):
    """Si le filtrage vide la requête, on retombe sur tous les tokens (jamais zéro résultat gratuit)."""
    s = Searcher(db_synthetique, embedder=FakeEmbedder(), df_max=0.0)
    termes = s._termes_match("le amortissement")
    assert set(termes) == set(normalize("le amortissement").split())
```

(La fixture `db_synthetique` doit contenir au moins un terme présent dans tous les chunks et un terme présent dans un seul ; si ce n'est pas le cas, l'ajuster dans le même commit et vérifier que les tests existants passent toujours.)

- [ ] **Step 2 : Run** `uv run pytest tests/test_search.py -q` → FAIL (`df`, `_termes_match`, `df_max` absents).

- [ ] **Step 3 : Implémenter dans `search.py`** :

```python
    def __init__(self, db_path: Path, embedder=None, poids_chemin: float = 1.0,
                 boost_commentaire: float = 1.0, df_max: float | None = None):
        ...  # existant
        self.df_max = df_max
        self._df_cache: dict[str, int] = {}
        self._n_chunks: int | None = None

    @property
    def n_chunks(self) -> int:
        if self._n_chunks is None:
            self._n_chunks = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return self._n_chunks

    def df(self, token: str) -> int:
        """Fréquence documentaire : nombre de chunks contenant le token (mis en cache)."""
        if token not in self._df_cache:
            try:
                n = self.con.execute(
                    "SELECT COUNT(*) FROM chunks_norm WHERE chunks_norm MATCH ?", (f'"{token}"',)
                ).fetchone()[0]
            except sqlite3.OperationalError:
                n = 0  # token rejeté par la syntaxe FTS5
            self._df_cache[token] = n
        return self._df_cache[token]

    def _termes_match(self, query: str) -> list[str]:
        """Tokens retenus pour le MATCH, dans l'ordre, dédupliqués.

        Avec df_max, les tokens présents dans plus de df_max × n_chunks chunks sont
        écartés : ce sont les mots fonctionnels par lesquels un article sans aucun mot
        de contenu commun se fait quand même retrouver (et se classe alors dernier).
        Repli : si le filtrage ne laisse rien, on garde tous les tokens.
        """
        toks = list(dict.fromkeys(normalize(query).split()))
        if self.df_max is None:
            return toks
        seuil = self.df_max * self.n_chunks
        gardes = [t for t in toks if 0 < self.df(t) <= seuil]
        return gardes or toks
```

et remplacer dans `_bm25` la construction des tokens par un appel à `self._termes_match(query)` (le reste de `_bm25` est inchangé).

- [ ] **Step 4 : Run** `uv run pytest tests/test_search.py -q` → PASS, puis `uv run pytest -q` (arrière-plan si > 5 min) → aucune régression.

- [ ] **Step 5 : Écrire `scripts/diagnostic_rangs.py`** — outil versionné remplaçant les sondes jetables : pour chaque question demandée, imprime et persiste le rang du record gold dans le classement lexical complet (avec et sans `df_max`) et dans le classement dense complet, ainsi que le nombre de candidats. Docstring expliquant l'usage et le coût (chargement du modèle dense ~50 s). Sortie JSON dans `docs/mesures/jalon3/diagnostic_rangs.json`.

- [ ] **Step 6 : Run** `uv run python scripts/diagnostic_rangs.py --questions q021,q026,q060,q023` et vérifier que les rangs reproduisent le tableau du diagnostic fondateur de ce plan (q021 154/154 lexical, 178 dense ; q026 46/46 puis 14/14 filtré, 248 dense ; q060 1430/1430, 528 dense ; q023 257 dense). **Tout écart doit être signalé au rapport** — c'est le contrôle que l'outil versionné dit bien la même chose que les sondes.

- [ ] **Step 7 : Écrire `scripts/ablations_jalon3.py`** — squelette commun aux trois ablations : charge le benchmark, instancie **un seul** `Embedder` partagé entre toutes les configurations (économie de ~50 s par config), exécute `evaluate` par configuration, appelle `paired_bootstrap` contre la référence, imprime un tableau markdown (config, recall@5, recall@10, MRR, ventilation, delta, IC95, p_amelioration, latence/question) et écrit le JSON complet (agrégats + `par_question` de chaque config) dans `docs/mesures/jalon3/`. Pour cette tâche : `--ablation D` compare `df_max ∈ {None, 0.10, 0.05, 0.02}` en mode `hybrid`.

- [ ] **Step 8 : Mesurer l'ablation D** : `uv run python scripts/ablations_jalon3.py --ablation D --split dev` (~5 min, 4 configs × 61 questions). Référence = `df_max=None` (baseline jalon 2.5 : recall@10 = 0,672 — **contrôle de non-régression : si la config neutre ne redonne pas 0,672, arrêter et signaler**).

- [ ] **Step 9 : Créer `docs/eval-jalon3.md`** : conditions exactes (comme docs/eval-jalon25.md), section « Diagnostic fondateur » reprenant le tableau des rangs (avec la commande `diagnostic_rangs.py` qui le régénère), puis section « Ablation D » avec le tableau complet et la décision motivée (valeur adoptée ou rejet).

- [ ] **Step 10 : Commit**

```bash
git add src/accounting_rag/search.py tests/test_search.py scripts/diagnostic_rangs.py scripts/ablations_jalon3.py docs/eval-jalon3.md docs/mesures/jalon3
git commit -m "feat(jalon3): filtrage des tokens peu discriminants (df_max) — mesuré par bootstrap"
```

---

### Task 2: Largeur du pool de candidats (ablation E)

**Files:**
- Modify: `src/accounting_rag/search.py`
- Modify: `tests/test_search.py`
- Modify: `scripts/ablations_jalon3.py`
- Modify: `docs/eval-jalon3.md`

**Interfaces:**
- Consumes: `Searcher(..., df_max=...)` de la tâche 1, avec la valeur adoptée (ou neutre si rejetée).
- Produces: `Searcher(..., pool: int = 50)` — nombre de lignes récupérées par canal avant fusion (`limit` de `_bm25` et `k` de `_dense`). 50 = comportement jalon 2.5.

- [ ] **Step 1 : Tests d'abord** :

```python
def test_pool_neutre_par_defaut(db_synthetique):
    a = Searcher(db_synthetique, embedder=FakeEmbedder())
    b = Searcher(db_synthetique, embedder=FakeEmbedder(), pool=50)
    for requete in ("amortissement", "immobilisation corporelle"):
        assert [r["record_id"] for r in a.search(requete)] == \
               [r["record_id"] for r in b.search(requete)]


def test_pool_est_transmis_aux_deux_canaux(db_synthetique, monkeypatch):
    """Le pool doit piloter la limite lexicale ET le k dense, pas seulement l'un des deux."""
    s = Searcher(db_synthetique, embedder=FakeEmbedder(), pool=7)
    vus = {}
    vrai_bm25, vrai_dense = s._bm25, s._dense
    s._bm25 = lambda q, limit=None: vus.setdefault("bm25", limit) or vrai_bm25(q, limit)
    s._dense = lambda q, limit=None: vus.setdefault("dense", limit) or vrai_dense(q, limit)
    s.search("amortissement", mode="hybrid")
    assert vus["bm25"] == 7 and vus["dense"] == 7
```

(Adapter la mécanique d'espionnage au style des tests existants du fichier si un patron plus simple y est déjà en usage ; l'exigence est de vérifier que les DEUX canaux reçoivent la valeur.)

- [ ] **Step 2 : Run** → FAIL. **Step 3 : Implémenter** : stocker `self.pool = pool`, remplacer les valeurs par défaut `limit: int = 50` de `_bm25`/`_dense` par `limit: int | None = None` avec `limit = self.pool if limit is None else limit`, et laisser `search()` appeler les canaux sans passer de limite explicite. **Step 4 : Run** `uv run pytest tests/test_search.py -q` → PASS.

- [ ] **Step 5 : Mesurer l'ablation E** — ajouter `--ablation E` à `scripts/ablations_jalon3.py` : `pool ∈ {50, 100, 200, 400}` en mode `hybrid`, avec le `df_max` adopté en T1. Référence = meilleure config de T1. Attendu d'après le diagnostic : le gold des questions dures entre dans le pool à 200-400, mais **son rang RRF restera trop bas pour changer recall@10 sans reranker** — l'ablation E est donc probablement négative *seule*, et son intérêt réel est de préparer l'ablation F. Le documenter comme tel : une ablation dont on attend un résultat nul n'est pas une ablation inutile, c'est le prérequis mesuré de la suivante.

- [ ] **Step 6 : Documenter** la section « Ablation E » dans `docs/eval-jalon3.md` : tableau, decision, et — c'est le point important — le **nombre de questions dont le gold entre dans le pool** à chaque largeur (métrique de couverture du pool, à calculer dans le script : part des questions dont au moins une citation gold est présente dans le pool avant fusion). Cette métrique est le vrai résultat de la tâche.

- [ ] **Step 7 : Commit** — `feat(jalon3): largeur du pool de candidats paramétrable (pool) — couverture mesurée`

---

### Task 3: Reranking sur pool élargi (ablation F)

**Files:**
- Modify: `src/accounting_rag/search.py`
- Modify: `src/accounting_rag/rerank.py`
- Modify: `tests/test_rerank.py`
- Modify: `scripts/ablations_jalon3.py`
- Modify: `scripts/run_eval.py`
- Modify: `docs/eval-jalon3.md`

**Interfaces:**
- Produces: `Searcher(..., n_rerank: int = 25)` — nombre de candidats soumis au reranker en mode `hybrid+rerank` (25 = comportement jalon 2.5). Les résultats routés restent épinglés hors reranking.

- [ ] **Step 1 : Tests d'abord** (avec le `FakeCrossEncoder` / `CountingFakeCrossEncoder` existants, aucun téléchargement) :

```python
def test_n_rerank_neutre_par_defaut(searcher_synthetique_avec_fake_reranker):
    """n_rerank=25 par défaut : mêmes candidats soumis qu'au jalon 2.5."""
    ...  # compter les paires soumises au fake, borne 25


def test_n_rerank_elargit_les_candidats_soumis(db_synthetique):
    """n_rerank=200 soumet tous les candidats disponibles quand le pool le permet."""
    ...  # pool=200, n_rerank=200 -> le compteur du fake voit plus de paires qu'à 25


def test_routes_restent_epingles_avec_pool_elargi(db_synthetique):
    """Élargir le pool ne doit pas déloger un résultat routé de la première place."""
    ...  # requête « article 111-1 », pool=200, n_rerank=200 -> source 'route' en position 0
```

- [ ] **Step 2 : Run** → FAIL. **Step 3 : Implémenter** `self.n_rerank` et l'utiliser à la place du 25 codé en dur dans la branche `hybrid+rerank` de `search()`. **Step 4 : Run** `uv run pytest tests/test_rerank.py tests/test_search.py -q` → PASS.

- [ ] **Step 5 : Mesurer l'ablation F** — `--ablation F`, mode `hybrid+rerank`, **deux modèles × deux largeurs**, avec le `df_max`/`pool` adoptés :
  1. `bge` + `n_rerank=25` (config adoptée au jalon 2.5, référence) ;
  2. `mmarco` + `n_rerank=200` (via `ACCRAG_RERANKER`, ~80 s/question) ;
  3. `mmarco` + `n_rerank=25` (contrôle : isole l'effet de la largeur de celui du modèle) ;
  4. `bge` + `n_rerank=100` **seulement si** le temps le permet (~8 min/question × 61 ≈ 8 h : à lancer en arrière-plan uniquement si les configs 1-3 laissent penser que la largeur est le facteur dominant ; sinon documenter « non mesuré, coût », ce qui est une réserve honnête et non un oubli).
  Chaque campagne rerank en **arrière-plan** ; noter la latence par question de chaque config. Le tableau doit permettre de répondre à la question du jalon : **modèle faible sur pool large, ou modèle fort sur pool étroit ?**

- [ ] **Step 6 : Documenter** la section « Ablation F » : tableau des quatre configs (recall@5/@10/MRR, ventilation, delta/IC95/p vs référence, latence), et la décision. Si `mmarco+200` gagne, `Reranker._DEFAULT` et les défauts de `Searcher` changent en conséquence (le défaut suit la mesure, règle du projet) ; sinon la config du jalon 2.5 est conservée.

- [ ] **Step 7 : Mettre `scripts/run_eval.py` en cohérence** : exposer `--df-max`, `--pool`, `--n-rerank` avec les valeurs adoptées comme défauts ; `hybrid+rerank` reste hors de `--mode all`.

- [ ] **Step 8 : Commit** — `feat(jalon3): reranking sur pool élargi (n_rerank) — arbitrage modèle/largeur mesuré`

---

### Task 4: Campagne de clôture + documentation

**Files:**
- Modify: `docs/eval-jalon3.md`
- Modify: `README.md`
- Modify: `docs/mesures/jalon3/` (JSON de clôture)

- [ ] **Step 1 : Suite complète** `uv run pytest -q` (arrière-plan) → tout PASS ; consigner la ligne verbatim.

- [ ] **Step 2 : Campagne dev finale** — `--ablation cumul --split dev` : baseline jalon 2.5 (`df_max=None, pool=50, n_rerank=25, bge`) contre la configuration finale adoptée ; bootstrap global **et par catégorie** ; tableau complet.

- [ ] **Step 3 : Split test, UNE exécution** — `--ablation cumul --split test` : baseline et configuration finale sur les 29 questions gelées. Présenter comme « référence gelée jalon 3 », commenter l'écart dev/test, et rappeler qu'aucune décision du jalon ne s'appuie sur ces chiffres.

- [ ] **Step 4 : Lecture d'ensemble** dans `docs/eval-jalon3.md` — répondre explicitement à la question posée par le diagnostic : le recouvrement lexical nul est-il franchi ? Quelle part des échecs `vocabulaire_courant` reste-t-il ? Et **nommer ce qui reste hors d'atteinte de ce jalon** : les questions dont le gold n'est atteignable ni par le lexical filtré ni par le dense (q060 et ses semblables) ne peuvent l'être que par réécriture de requête (LLM) ou embeddings métier — matière du jalon suivant, avec le chiffre exact du nombre de questions concernées.

- [ ] **Step 5 : README** — tableau des résultats mis à jour (configuration finale, latence honnête, baseline recommandée pour l'interactif), section limitations réécrite (ce qui reste : réécriture de requête, embeddings métier, génération), variables d'environnement documentées.

- [ ] **Step 6 : Cohérence** — mêmes chiffres partout (README ↔ eval-jalon3.md ↔ JSON de `docs/mesures/jalon3/`), et vérifier qu'un exemple de recalcul de bootstrap depuis les JSON est fourni et **exécuté**.

- [ ] **Step 7 : Commit** — `docs(jalon3): campagne de clôture — dev + référence test gelée, décisions d'ablation`

---

## Self-review (effectuée à l'écriture)

- **Couverture du diagnostic** : lecture 2 (filtrage) → T1 ; lecture 3 (pool atteignable) → T2 ; corollaire de coût (modèle léger sur pool large) → T3 ; lecture 1 et 4 (recouvrement nul structurel, `AND` mort) → cadrées comme hors d'atteinte et documentées en T4 step 4.
- **Cohérence des types** : les trois paramètres s'ajoutent au constructeur de `Searcher` avec des défauts neutres ; `_bm25`/`_dense` passent de `limit: int = 50` à `limit: int | None = None` (T2) — T1 ne touche pas leur signature, donc pas de conflit d'ordre entre les tâches. `_termes_match` est introduit en T1 et consommé uniquement par `_bm25`.
- **Pas de rebuild** : vérifié tâche par tâche — les trois leviers n'agissent qu'à la requête ; `SYNONYMES` est explicitement gelé par les Global Constraints.
- **Risque nommé** : l'ablation E est attendue négative seule ; le plan l'assume et lui donne une métrique propre (couverture du pool) pour qu'elle reste informative, au lieu de la présenter comme un échec.
