# Jalon 3 — Franchir le recouvrement lexical nul

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faire entrer dans la fenêtre de candidats les articles qui ne partagent aucun mot avec la question, en mesurant trois leviers : filtrage des tokens peu discriminants, élargissement du pool, reranking sur pool large.

**Architecture:** Le retrieval du jalon 2.5 reste la base (routeur → BM25 + dense → RRF → graphe → reranker optionnel). Les trois leviers sont des paramètres de requête (aucun changement d'index, donc **aucun rebuild dans tout ce jalon**), neutres par défaut, mesurés un à un par bootstrap apparié sur le split dev v2.

**Tech Stack:** Python ≥3.12 via uv, SQLite FTS5 + sqlite-vec, sentence-transformers (Embedder + CrossEncoder déjà présents), pytest. **Aucune dépendance nouvelle.**

**Spec:** docs/superpowers/specs/2026-08-14-accountingrag-design.md (§4-5 ; le protocole d'ablation de la §5 est la loi). Contexte chiffré : docs/eval-jalon25.md.

## Diagnostic fondateur (sondes exécutées avant rédaction du plan)

Deux sondes sur les questions dev en échec (`vocabulaire_courant`) ont établi :

> **Correction du 16 août 2026 (contrôleur).** La première version de ce tableau, produite par une sonde jetable, contenait une erreur de mesure : la sonde retournait dès qu'elle trouvait le gold, si bien que le « total » affiché valait toujours le rang (d'où un trompeur « 154/154, 46/46, 1430/1430 — le gold est toujours dernier »). Les totaux réels sont rétablis ci-dessous ; la conclusion « toujours dernier » était un **artefact de la sonde** et est retirée. Chiffres vérifiés dans la convention de production (sans déduplication des termes), reproductibles par `scripts/diagnostic_rangs.py`.

| question | gold | rang lexical / total | position | rang lexical (tokens df ≤ 2 %) | rang dense |
|---|---|---|---|---|---|
| q021 | pcg-214-13 | 154 / 1585 | 10 % du classement | absent (40 candidats) | 178/1660 |
| q026 | pcg-212-3 | **46 / 1659** | **3 %** | 14 / 67 | 248/1660 |
| q060 | pcg-1222-74 | 1453 / 1659 | 88 % | absent (63 candidats) | 528/1660 |
| q023 | pcg-214-22 | — | — | — | 257/1660 |

Lectures qui fondent les ablations (révisées après correction) :
1. **Deux régimes d'échec distincts, à ne pas confondre.** q021 et q026 sont des *quasi-succès* : leur gold est dans les 3-10 % de tête du classement lexical, juste au-delà de la fenêtre de 50 candidats. q060 est un échec de fond (88 % du classement) : là, le canal lexical est effectivement muet. Une seule ablation ne peut pas traiter les deux régimes.
2. Le régime « quasi-succès » désigne directement **l'élargissement du pool** (ablation E) : à 200 ou 400 candidats, les golds de q021 et q026 entrent dans le champ. Reste à savoir si leur rang dans la fusion suffit — d'où l'ablation F, le reranker étant le seul composant capable de les remonter.
3. Filtrer les tokens à forte fréquence documentaire réduit massivement le bruit (q060 : 1659 → 63 candidats) et améliore le rang quand un token rare est partagé (q026 : 46/1659 → 14/67). Mais quand aucun token rare n'est partagé, le gold **disparaît du pool** (q021, q060) — d'où la nécessité de mesurer. *(Mesuré en T1 : rejeté, effet monotone négatif.)*
4. Le canal dense place le gold aux rangs 178-528 : hors de la fenêtre de 50, atteignable par un pool de 200-400.
5. La conjonction (`AND`) des tokens rares donne **zéro candidat** dans les trois cas : voie morte, exclue de ce plan.

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

- [ ] **Step 5 : Mesurer l'ablation E** — ajouter `--ablation E` à `scripts/ablations_jalon3.py`. `df_max` ayant été **rejeté** en T1, la référence est la baseline neutre (recall@10 = 0,672). Configurations :
  1. `pool ∈ {50, 100, 200, 400}` en mode `hybrid` ;
  2. **`dedup=True` à pool constant (50)** — nouveau paramètre `Searcher(..., dedup_termes: bool = False)` qui déduplique les termes du `MATCH`. Découverte incidente de T1 : `bm25()` de FTS5 pondère la multiplicité des termes, et dédupliquer donne recall@10 = 0,689 contre 0,672 (49/61 questions dev ont des tokens répétés). Mesure de contrôle du relecteur : `delta=+0,0164`, `IC95=(0,0000 ; 0,0492)`, `p_amelioration=0,6319` — **sous le seuil**, donc à mesurer proprement ici avec lecture par catégorie (l'effet est non uniforme : recall@5 baisse de 0,639 à 0,623, `vocabulaire_courant` gagne +0,032) ;
  3. la meilleure largeur de pool **combinée** à `dedup=True`, seulement si l'une des deux est adoptée séparément.

  Attendu, d'après le diagnostic **corrigé** : les golds de q021 (rang 154/1585) et q026 (46/1659) entrent dans le pool dès 200 candidats ; leur rang dans la fusion RRF restera cependant faible, donc un gain de recall@10 sans reranker n'est pas acquis. L'ablation E garde sa valeur même négative : elle établit la **couverture du pool** dont dépend l'ablation F.

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

> **Ordre d'exécution : DERNIÈRE tâche du jalon, après la tâche 5** (ajoutée par amendement une fois la clé API fournie). La « configuration finale » inclut donc l'éventuelle adoption de la réécriture de requête.

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

### Task 5: Réécriture de requête par LLM (ablation G)

> **Amendement du 16 août 2026** : tâche ajoutée après la fourniture d'une clé API par l'utilisateur (`.env`, ignoré par git). Elle amende deux Global Constraints : la dépendance `anthropic` est désormais autorisée (SDK officiel), et le jalon acquiert un coût monétaire — borné ci-dessous. **Exécuter cette tâche AVANT la tâche 4** (clôture).

**Files:**
- Create: `src/accounting_rag/config.py`
- Create: `src/accounting_rag/rewrite.py`
- Create: `tests/test_rewrite.py`
- Create: `.env.example`
- Modify: `pyproject.toml`, `src/accounting_rag/search.py`, `scripts/ablations_jalon3.py`, `scripts/run_eval.py`, `docs/eval-jalon3.md`, `README.md`

**Interfaces:**
- Produces:
  - `charge_env(chemin=".env") -> None` (`config.py`) — charge les paires `CLE=valeur` d'un fichier `.env` dans `os.environ` **sans écraser** une variable déjà définie ; silencieux si le fichier est absent. Aucune dépendance.
  - `Rewriter(cache_path, modele=None, client=None)` (`rewrite.py`) — `modele` par défaut `"claude-sonnet-5"`, surchargeable par `ACCRAG_REWRITE_MODEL` ; `client` injectable (tests). Méthode `reecrire(question: str) -> str`, avec cache disque JSON : un appel API par question **au plus une fois dans la vie du projet**.
  - `Searcher(..., rewriter=None, mode_reecriture="remplace"|"etend")` — `rewriter=None` = comportement actuel. Quand un rewriter est fourni, la requête transmise aux canaux lexical et dense est la réécriture (`remplace`) ou la concaténation `question + " " + réécriture` (`etend`). **Le routeur de références d'articles continue de lire la question ORIGINALE** (une référence explicite ne doit jamais dépendre d'une reformulation).

**Intégrité du benchmark — exigence non négociable :** le rewriter ne reçoit QUE le texte de la question. Il ne voit jamais les citations gold, ni le corpus, ni les résultats de recherche. Toute autre conception ferait fuiter la réponse dans la requête et invaliderait la mesure. Un test structurel doit le garantir.

**Bornes de coût :** un appel par question, ~150 tokens d'entrée et ~80 de sortie, sur 90 questions au plus → coût total de l'ordre de quelques centimes. Le cache JSON est **commité** (`docs/mesures/jalon3/reecritures.json`) : les mesures ultérieures et les revues sont donc gratuites et reproductibles à l'identique. Le script doit refuser de dépasser 200 appels API dans une exécution (garde-fou anti-boucle) et journaliser le total de tokens consommés.

- [ ] **Step 1 : Écrire les tests d'abord** (`tests/test_rewrite.py` — aucun appel réseau) :

```python
import json
from accounting_rag.config import charge_env
from accounting_rag.rewrite import Rewriter


class FauxBloc:
    def __init__(self, texte):
        self.type = "text"
        self.text = texte


class FauxMessage:
    def __init__(self, texte):
        self.content = [FauxBloc(texte)]


class FauxClient:
    """Client Anthropic factice : enregistre les appels, ne sort jamais sur le réseau."""

    def __init__(self, reponse="amortissement immobilisation corporelle"):
        self.reponse = reponse
        self.appels = []
        self.messages = self

    def create(self, **kwargs):
        self.appels.append(kwargs)
        return FauxMessage(self.reponse)


def test_charge_env_sans_ecraser(tmp_path, monkeypatch):
    fichier = tmp_path / ".env"
    fichier.write_text("ANTHROPIC_API_KEY=depuis-le-fichier\nAUTRE=x\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AUTRE", "deja-defini")
    charge_env(fichier)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "depuis-le-fichier"
    assert os.environ["AUTRE"] == "deja-defini"  # jamais écrasé


def test_charge_env_silencieux_si_absent(tmp_path):
    charge_env(tmp_path / "inexistant")  # ne doit pas lever


def test_reecrire_appelle_le_modele_et_met_en_cache(tmp_path):
    client = FauxClient()
    r = Rewriter(cache_path=tmp_path / "cache.json", client=client)
    a = r.reecrire("comment je répartis le coût d'une machine ?")
    b = r.reecrire("comment je répartis le coût d'une machine ?")
    assert a == b == "amortissement immobilisation corporelle"
    assert len(client.appels) == 1  # deuxième appel servi par le cache
    cache = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert cache["comment je répartis le coût d'une machine ?"] == a


def test_cache_relu_depuis_le_disque(tmp_path):
    (tmp_path / "cache.json").write_text(
        json.dumps({"q": "reecriture-en-cache"}), encoding="utf-8"
    )
    client = FauxClient()
    r = Rewriter(cache_path=tmp_path / "cache.json", client=client)
    assert r.reecrire("q") == "reecriture-en-cache"
    assert client.appels == []  # aucun appel API


def test_le_rewriter_ne_recoit_que_la_question(tmp_path):
    """Intégrité du benchmark : ni gold, ni corpus, ni résultats dans le prompt."""
    client = FauxClient()
    r = Rewriter(cache_path=tmp_path / "cache.json", client=client)
    r.reecrire("ma question")
    envoye = json.dumps(client.appels[0], default=str)
    assert "ma question" in envoye
    for interdit in ("pcg-", "citations", "gold", "record_id"):
        assert interdit not in envoye
```

- [ ] **Step 2 : Run** `uv run pytest tests/test_rewrite.py -q` → FAIL (modules absents).

- [ ] **Step 3 : Implémenter `config.py`** :

```python
"""Chargement d'un fichier .env sans dépendance externe (les secrets ne sont jamais versionnés)."""
import os
from pathlib import Path


def charge_env(chemin: str | Path = ".env") -> None:
    """Charge les paires CLE=valeur dans os.environ, sans écraser l'existant.

    Silencieux si le fichier est absent : l'environnement peut déjà porter les variables.
    """
    p = Path(chemin)
    if not p.is_file():
        return
    for ligne in p.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, _, valeur = ligne.partition("=")
        cle, valeur = cle.strip(), valeur.strip().strip('"').strip("'")
        os.environ.setdefault(cle, valeur)
```

- [ ] **Step 4 : Implémenter `rewrite.py`** :

```python
"""Réécriture d'une question en langage courant vers le vocabulaire du PCG (via l'API Claude).

Le diagnostic du jalon 3 a montré que les questions « grand public » ne partagent aucun
token de contenu avec l'article qui y répond : le canal lexical est alors muet. La
réécriture vise ce cas précis. Le modèle ne voit QUE la question — jamais le corpus,
jamais les citations attendues.
"""
import json
import os
from pathlib import Path

_DEFAUT = os.environ.get("ACCRAG_REWRITE_MODEL", "claude-sonnet-5")

_SYSTEME = (
    "Tu traduis une question de comptabilité posée en langage courant vers le vocabulaire "
    "technique du Plan comptable général français. Réponds UNIQUEMENT par une liste de "
    "termes et expressions normalisés, séparés par des espaces, sans phrase, sans "
    "explication, sans ponctuation superflue. N'invente aucun numéro d'article. "
    "Exemple d'entrée : « comment je répartis le coût d'une machine sur plusieurs années ». "
    "Exemple de sortie : amortissement immobilisation corporelle plan d'amortissement "
    "durée d'utilisation base amortissable."
)


class Rewriter:
    def __init__(self, cache_path: str | Path, modele: str | None = None, client=None):
        self.cache_path = Path(cache_path)
        self.modele = modele or _DEFAUT
        self._client = client
        self._cache: dict[str, str] = {}
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

    def reecrire(self, question: str) -> str:
        if question in self._cache:
            return self._cache[question]
        if self.appels >= 200:
            raise RuntimeError("garde-fou : plus de 200 appels API dans une exécution")
        reponse = self.client.messages.create(
            model=self.modele,
            max_tokens=200,
            system=_SYSTEME,
            messages=[{"role": "user", "content": question}],
        )
        self.appels += 1
        usage = getattr(reponse, "usage", None)
        if usage is not None:
            self.tokens_entree += getattr(usage, "input_tokens", 0) or 0
            self.tokens_sortie += getattr(usage, "output_tokens", 0) or 0
        texte = " ".join(
            bloc.text.strip() for bloc in reponse.content if getattr(bloc, "type", None) == "text"
        ).strip()
        self._cache[question] = texte
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return texte
```

- [ ] **Step 5 : Run** `uv run pytest tests/test_rewrite.py -q` → PASS (5 tests).

- [ ] **Step 6 : Brancher sur `Searcher`** — ajouter `rewriter=None, mode_reecriture="remplace"` au constructeur ; dans `search()`, calculer une fois `requete_canaux = query` si `self.rewriter is None`, sinon la réécriture (`remplace`) ou `f"{query} {reecriture}"` (`etend`), et passer `requete_canaux` à `_bm25`/`_dense` **en laissant `_route(query)` sur la question originale**. Test à ajouter dans `tests/test_search.py` avec un faux rewriter (objet à méthode `reecrire`) : (a) `rewriter=None` reproduit le comportement actuel, (b) avec rewriter, une requête sans aucun token commun avec la fixture retrouve le document que la réécriture désigne, (c) une question contenant « article 111-1 » garde son résultat routé en position 0 même si la réécriture est absurde.

- [ ] **Step 7 : Run** `uv run pytest tests/test_search.py tests/test_rewrite.py -q` → PASS ; `uv run pytest -q` (arrière-plan) → aucune régression.

- [ ] **Step 8 : Ajouter la dépendance et l'exemple** — `pyproject.toml` : `"anthropic>=0.40"` dans `dependencies` ; `uv sync` ; créer `.env.example` contenant une seule ligne `ANTHROPIC_API_KEY=sk-ant-votre-cle-ici` avec un commentaire rappelant que `.env` est ignoré par git.

- [ ] **Step 9 : Mesurer l'ablation G** — `--ablation G` dans `scripts/ablations_jalon3.py`, référence = meilleure config cumulée (D/E/F), configs :
  1. réécriture `remplace` ;
  2. réécriture `etend`.
  Les réécritures des 61 questions dev sont produites une fois et mises en cache dans `docs/mesures/jalon3/reecritures.json` (le script journalise le nombre d'appels et les tokens consommés, à reporter dans le doc). Vérifier **avant** la mesure que le cache contient bien 61 entrées et **inspecter manuellement 5 réécritures** pour s'assurer qu'aucune ne contient de numéro d'article inventé (le prompt l'interdit ; une violation invaliderait la catégorie `reference_directe`) — reporter ces 5 exemples dans le rapport.

- [ ] **Step 10 : Documenter** la section « Ablation G » de `docs/eval-jalon3.md` : les deux configs (delta/IC95/p/ventilation/latence/coût), la décision, 5 exemples de réécritures avant/après, et — point clé — le sort des questions que le diagnostic donnait hors d'atteinte (q021, q060 : leur gold entre-t-il enfin dans la fenêtre ?).

- [ ] **Step 11 : Commit**

```bash
git add src/accounting_rag/config.py src/accounting_rag/rewrite.py tests/test_rewrite.py tests/test_search.py src/accounting_rag/search.py scripts/ablations_jalon3.py scripts/run_eval.py pyproject.toml uv.lock .env.example docs/eval-jalon3.md docs/mesures/jalon3
git commit -m "feat(jalon3): réécriture de requête par LLM (ablation G) — mesurée par bootstrap, réécritures mises en cache"
```

---

## Self-review (effectuée à l'écriture)

- **Couverture du diagnostic** : lecture 2 (filtrage) → T1 ; lecture 3 (pool atteignable) → T2 ; corollaire de coût (modèle léger sur pool large) → T3 ; lecture 1 et 4 (recouvrement nul structurel, `AND` mort) → cadrées comme hors d'atteinte et documentées en T4 step 4.
- **Cohérence des types** : les trois paramètres s'ajoutent au constructeur de `Searcher` avec des défauts neutres ; `_bm25`/`_dense` passent de `limit: int = 50` à `limit: int | None = None` (T2) — T1 ne touche pas leur signature, donc pas de conflit d'ordre entre les tâches. `_termes_match` est introduit en T1 et consommé uniquement par `_bm25`.
- **Pas de rebuild** : vérifié tâche par tâche — les trois leviers n'agissent qu'à la requête ; `SYNONYMES` est explicitement gelé par les Global Constraints.
- **Risque nommé** : l'ablation E est attendue négative seule ; le plan l'assume et lui donne une métrique propre (couverture du pool) pour qu'elle reste informative, au lieu de la présenter comme un échec.
