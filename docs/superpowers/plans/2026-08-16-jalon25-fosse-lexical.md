# Jalon 2.5 — Fossé lexical : benchmark étendu + ablations mesurées

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Étendre le benchmark de 30 à 90 questions (puissance statistique + sur-échantillonnage de la catégorie faible), puis mesurer une variable à la fois trois leviers anti-fossé-lexical : pondération par champ, reranker cross-encoder, synonymes pilotés par les échecs.

**Architecture:** Le retrieval du jalon 2 (routeur → BM25 normalisé + dense → RRF → graphe) reste la base. Chaque levier est un paramètre optionnel de `Searcher` (rétro-compatible), mesuré sur le split dev étendu avec bootstrap apparié, adopté seulement si le gain est probable (p ≥ 0,95) sans dégradation d'une autre catégorie.

**Tech Stack:** Python ≥3.12 via uv, SQLite (FTS5 + sqlite-vec), sentence-transformers (Embedder existant + CrossEncoder pour le reranker — AUCUNE dépendance nouvelle), pytest.

**Spec:** docs/superpowers/specs/2026-08-14-accountingrag-design.md (sections 4-5 ; le protocole d'ablation de la §5 est la loi de ce jalon). Contexte chiffré : docs/eval-jalon2.md (baseline jalon 2 : hybrid 0,81/0,81/0,763 sur dev n=21 ; vocabulaire_courant 0,643 lexical / 0,214 dense).

## Global Constraints

- Exécution via `uv run ...` uniquement ; aucune dépendance nouvelle dans pyproject.toml.
- Les tables du jalon 1 (`records`, `renvois`, `records_fts`) ne sont JAMAIS modifiées ; seules `chunks`/`chunks_norm`/`chunks_vec` sont reconstruites par `scripts/build_index.py`.
- `normalize()` reste l'unique fonction de normalisation, identique build/requête ; TOUT changement de `SYNONYMES` (ou de normalize) impose un rebuild d'index AVANT toute mesure.
- Le split test est GELÉ dès la fin de la tâche 1 (re-gel à 29 questions) : il n'est exécuté qu'UNE fois, à la tâche 6, jamais utilisé pour choisir un paramètre.
- Protocole d'ablation (spec §5) : une variable à la fois ; chaque comparaison = paired bootstrap sur recall@10 par question (n_boot=10000, seed=42) ; adoption si p_amelioration ≥ 0,95 ET aucune catégorie ne perd plus de 0,05 de recall@10 ; sinon le paramètre reste à sa valeur neutre et le résultat négatif est documenté (un résultat négatif mesuré est un livrable).
- Les questions q001-q030 existantes gardent leurs ids, leur texte et leur split — on AJOUTE, on ne réécrit pas.
- Citations gold = préfixes d'ids sans suffixes @/-c/# (frontière stricte, logique existante de `evalrag.match`).
- Anciens chiffres (n=21) et nouveaux (n=61) jamais comparés entre eux dans les docs — le changement de benchmark est un changement d'instrument.
- Messages de commit en français, préfixés `feat(jalon25):` / `fix(jalon25):` / `docs(jalon25):`.
- Toute commande > 5 min (rebuild ~14 min, suite complète ~6 min) se lance en arrière-plan (run_in_background) ; le reste au premier plan.

---

### Task 1: Extension du benchmark — 30 → 90 questions, re-gel du split

**Files:**
- Modify: `benchmark/dev.jsonl` (21 → 61 questions)
- Modify: `benchmark/test.jsonl` (9 → 29 questions)
- Modify: `benchmark/README.md` (comptes, re-gel, date)
- Modify: `tests/test_benchmark_format.py` (comptes attendus)

**Interfaces:**
- Produces: benchmark v2 — 90 questions, mêmes champs {id, question, categorie, citations, notes}, ids q031-q090 pour les nouvelles.

**Composition exacte des 60 nouvelles questions :**

| catégorie | nouvelles | → dev | → test | total après (dev/test) |
|---|---|---|---|---|
| reference_directe | 5 | 3 | 2 | 10 (7/3) |
| regle | 20 | 13 | 7 | 35 (23/12) |
| vocabulaire_courant | 35 | 24 | 11 | 45 (31/14) |
| **total** | **60** | **40** | **20** | **90 (61/29)** |

**Exigences de rédaction (chacune vérifiable) :**

1. Chaque citation est choisie en LISANT le corpus (SQL sur `records`), jamais en interrogeant `Searcher` — le gold reste indépendant du système évalué.
2. Chaque citation existe dans `data/corpus.db` ET son texte répond réellement à la question (relire le record avant d'écrire la question).
3. `vocabulaire_courant` : zéro terme PCG exact de l'article visé ; registres variés et étiquetés dans `notes` (étudiant DCG, dirigeant de PME, comptable junior, langage familier) ; les tournures des 3 pires questions du jalon 2 (q021/q026/q022 — paraphrase totale, terme grand public, question à deux volets) servent de gabarits à multiplier.
4. **Au moins 20 des 60 nouvelles questions utilisent l'apostrophe typographique U+2019 (’) et non l'ASCII (')** — le benchmark du jalon 2 était 100 % ASCII et aveugle au bug C1 ; le champ `notes` de ces questions contient le tag `apostrophe:typo`.
5. Couverture thématique : au moins 12 thèmes non couverts par q001-q030, pris dans : stocks et en-cours, subventions d'exploitation, provisions réglementées, changements de méthode comptable, contrats à long terme, frais de développement, écarts de conversion, abandons de créances, effets de commerce, opérations en devises, indemnités d'assurance, fusion/apports, engagement de crédit-bail (annexe), production immobilisée. `notes` mentionne le thème.
6. `reference_directe` : 5 articles jamais cités par q001-q005, dont au moins 2 en dehors du Titre II (ex. classe de comptes du Titre IX, article de consolidation du Titre VII si présent dans le corpus).
7. Split : les nouvelles questions sont réparties AVANT toute mesure (re-gel), stratifiées comme au tableau ; aucune question test n'est jamais citée dans une analyse d'erreurs.

- [ ] **Step 1 : Explorer le corpus** — inventorier par SQL les chapitres/thèmes disponibles et leur volume (`SELECT chemin, COUNT(*) FROM records GROUP BY ...`), lister les articles candidats par thème avec leur texte.
- [ ] **Step 2 : Rédiger les 60 questions** en JSONL (ids q031-q090, ordre d'id = ordre d'ajout), en respectant les 7 exigences.
- [ ] **Step 3 : Mettre à jour `tests/test_benchmark_format.py`** — comptes attendus : total 90, dev 61, test 29, catégories 10/35/45 ; ajouter une assertion : au moins 20 questions contiennent le caractère U+2019.
- [ ] **Step 4 : Run** `uv run pytest tests/test_benchmark_format.py -q` → PASS.
- [ ] **Step 5 : Mettre à jour `benchmark/README.md`** — nouveaux comptes, date de re-gel, phrase explicite : « le split test v2 (29 questions) est gelé le 2026-08-16 ; il ne sera exécuté qu'une fois, à la clôture du jalon 2.5 ».
- [ ] **Step 6 : Commit** — `feat(jalon25): benchmark étendu à 90 questions (61 dev / 29 test), re-gel du split`

---

### Task 2: Bootstrap apparié + résultats par question + baseline 2.5

**Files:**
- Modify: `src/accounting_rag/evalrag.py`
- Modify: `tests/test_evalrag.py`
- Create: `docs/eval-jalon25.md` (squelette : conditions + tableau baseline)

**Interfaces:**
- Consumes: `evaluate(searcher, questions, mode, k=10) -> dict` existant.
- Produces: `evaluate()` retourne EN PLUS `"par_question": {qid: recall10}` ; nouvelle fonction `paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot=10000, seed=42) -> dict` avec clés `delta`, `ic95`, `p_amelioration`.

- [ ] **Step 1 : Test d'abord** — ajouter à `tests/test_evalrag.py` :

```python
def test_evaluate_expose_par_question():
    qs = [
        {"id": "q1", "question": "x", "categorie": "regle", "citations": ["pcg-214-1"]},
        {"id": "q2", "question": "y", "categorie": "regle", "citations": ["pcg-999-9"]},
    ]
    m = evaluate(FakeSearcher(), qs, mode="bm25", k=10)
    assert m["par_question"] == {"q1": 1.0, "q2": 0.0}


def test_paired_bootstrap_deterministe():
    a = {f"q{i}": 0.0 for i in range(20)}
    b = {f"q{i}": 1.0 for i in range(20)}
    r = paired_bootstrap(a, b)
    assert r["delta"] == 1.0
    assert r["p_amelioration"] == 1.0
    assert r["ic95"] == (1.0, 1.0)
    r2 = paired_bootstrap(a, a)
    assert r2["delta"] == 0.0
    assert r2["p_amelioration"] < 0.95
```

- [ ] **Step 2 : Run** → FAIL (fonctions absentes).
- [ ] **Step 3 : Implémenter** dans `evalrag.py` :

```python
def paired_bootstrap(a: dict[str, float], b: dict[str, float],
                     n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap apparié sur les scores par question (b - a). Ids alignés obligatoires."""
    import random
    ids = sorted(a)
    assert sorted(b) == ids, "les deux runs doivent porter sur les mêmes questions"
    deltas = [b[i] - a[i] for i in ids]
    n = len(deltas)
    mean_delta = sum(deltas) / n
    rng = random.Random(seed)
    boots = []
    wins = 0
    for _ in range(n_boot):
        m = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        boots.append(m)
        if m > 0:
            wins += 1
    boots.sort()
    return {"delta": round(mean_delta, 4),
            "ic95": (round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)),
            "p_amelioration": wins / n_boot}
```

et dans `evaluate()` : collecter `par_question[q["id"]] = covered10 / len(q["citations"])` et l'ajouter au dict retourné.

- [ ] **Step 4 : Run** `uv run pytest tests/test_evalrag.py -q` → PASS (tous).
- [ ] **Step 5 : Baseline 2.5** — `uv run python scripts/run_eval.py --mode all --split dev` (nouveau dev, 61 questions) ; consigner le tableau complet (4 modes × métriques + ventilation) dans `docs/eval-jalon25.md` (section « Baseline », conditions exactes comme docs/eval-jalon2.md, avec la mention : instrument changé, non comparable au n=21).
- [ ] **Step 6 : Commit** — `feat(jalon25): bootstrap apparié + scores par question ; baseline sur benchmark v2`

---

### Task 3: Ablation A — pondération par champ (chemin, type de record)

**Files:**
- Modify: `src/accounting_rag/search.py`
- Modify: `tests/test_search.py`
- Modify: `docs/eval-jalon25.md` (section « Ablation A »)

**Interfaces:**
- Produces: `Searcher(db_path, embedder=None, poids_chemin: float = 1.0, boost_commentaire: float = 1.0)` — valeurs neutres par défaut (comportement jalon 2 inchangé) ; `_bm25` utilise `bm25(chunks_norm, ?, ?)` avec (poids_texte=1.0, poids_chemin) LIÉS en paramètres SQL, et multiplie le score agrégé d'un record par `boost_commentaire` si `records.type != 'reglementaire'`.

- [ ] **Step 1 : Tests d'abord** (base synthétique tmp_path existante de tests/test_search.py) :

```python
def test_poids_chemin_neutre_par_defaut(searcher_synthetique):
    # deux Searcher, poids 1.0 explicite vs défaut → mêmes résultats bm25
    ...  # comparer les listes de record_id sur 3 requêtes de la fixture

def test_poids_chemin_favorise_le_chemin(db_synthetique):
    # un doc dont le terme requête n'est QUE dans le chemin, un autre QUE dans le texte ;
    # poids_chemin=3.0 → le premier passe devant ; poids_chemin=1.0 → l'ordre de référence est conservé
    ...

def test_boost_commentaire_penalise(db_synthetique):
    # fixture avec un record commentaire et un réglementaire matchant pareil ;
    # boost_commentaire=0.5 → le réglementaire passe devant
    ...
```

(Écrire les `...` en s'appuyant sur le patron write_db + FakeEmbedder existant ; chaque test doit pouvoir échouer.)

- [ ] **Step 2 : Run** → FAIL. **Step 3 : Implémenter** (requête `_bm25` : `SELECT c.record_id, bm25(chunks_norm, ?, ?) AS b ...` avec `(1.0, self.poids_chemin, match, limit)` ; boost : récupérer en une requête les types des records agrégés puis `s *= self.boost_commentaire if type != 'reglementaire' else 1.0` avant le max par record). **Step 4 : Run** → PASS, plus `uv run pytest tests/test_search.py tests/test_evalrag.py -q` sans régression.
- [ ] **Step 5 : Mesure (une variable à la fois)** — script ad hoc au premier plan (~2 min par run, modèle chargé une fois) :
  1. run A = hybrid baseline (poids neutres) sur dev — réutiliser `par_question` de la baseline T2 si le mode est identique, sinon re-runner ;
  2. run B1 = hybrid avec `poids_chemin=2.0` seul → `paired_bootstrap(A, B1)` ;
  3. run B2 = meilleur de {A, B1} avec `boost_commentaire=0.7` seul en plus → bootstrap contre le meilleur précédent ;
  4. si un paramètre est adopté (p ≥ 0,95, aucune catégorie ne perd > 0,05), essayer UNE valeur voisine (3.0 pour le poids, 0.5 pour le boost) pour vérifier la robustesse — pas de grid search.
- [ ] **Step 6 : Documenter** dans `docs/eval-jalon25.md` (« Ablation A ») : tableau A/B1/B2 avec delta, IC95, p, ventilation ; décision motivée (adopté à telle valeur / rejeté). Les valeurs ADOPTÉES deviennent les défauts de `scripts/run_eval.py` via deux nouveaux flags `--poids-chemin` / `--boost-commentaire` (défaut = valeur adoptée ; neutre si rejeté).
- [ ] **Step 7 : Commit** — `feat(jalon25): pondération par champ (chemin, type) — mesurée par bootstrap, décision documentée`

---

### Task 4: Ablation B — reranker cross-encoder

**Files:**
- Create: `src/accounting_rag/rerank.py`
- Modify: `src/accounting_rag/search.py` (mode `hybrid+rerank`)
- Create: `tests/test_rerank.py`
- Modify: `scripts/run_eval.py` (choices + `--mode all` inclut le nouveau mode)
- Modify: `docs/eval-jalon25.md` (section « Ablation B »)

**Interfaces:**
- Produces: `Reranker(model_name=None)` — env `ACCRAG_RERANKER`, défaut `"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"` (multilingue, léger) ; méthode `rerank(query: str, results: list[dict], top_k: int) -> list[dict]` (ajoute `score_rerank`, trie décroissant, tronque).
- `search(mode="hybrid+rerank")` : les résultats routés (source `route`) restent épinglés en tête SANS passer au reranker (une référence d'article exacte ne se re-juge pas) ; la fusion est récupérée à 25 candidats puis rerankée à `k - len(routed)`.

- [ ] **Step 1 : Tests d'abord** (`tests/test_rerank.py`) avec un FakeCrossEncoder (attribut injecté, pas de téléchargement) :

```python
class FakeCrossEncoder:
    def predict(self, pairs):
        # score = longueur du recouvrement lexical naïf, déterministe
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]

def test_rerank_trie_et_tronque():
    r = Reranker.__new__(Reranker)
    r.model = FakeCrossEncoder()
    results = [{"texte": "rien ici", "record_id": "a"},
               {"texte": "amortissement des logiciels", "record_id": "b"}]
    out = r.rerank("amortissement logiciels", results, top_k=1)
    assert [x["record_id"] for x in out] == ["b"]
    assert "score_rerank" in out[0]

def test_mode_hybrid_rerank_epingle_le_route(searcher_synthetique_avec_fake_reranker):
    # requête avec référence directe : le résultat source='route' reste en position 1
    ...
```

- [ ] **Step 2 : Run** → FAIL. **Step 3 : Implémenter** `rerank.py` (import CrossEncoder DANS `__init__`, lazy comme Embedder ; tronquer chaque texte à 1000 caractères pour borner la latence) et le mode dans `search.py` (`_MODES` + propriété lazy `self.reranker` + branche : fusion à 25, rerank, épinglage des routés). **Step 4 : Run** ciblé → PASS ; suite `tests/test_search.py tests/test_rerank.py` → PASS.
- [ ] **Step 5 : Mesure** — télécharger le modèle (premier appel), runner `hybrid+rerank` sur dev avec les paramètres adoptés en T3 ; `paired_bootstrap(meilleure_config_T3, rerank)` ; noter aussi la latence par question (le doc de campagne doit dire ce que coûte le gain). Si rejet, essayer UNE alternative : `ACCRAG_RERANKER=BAAI/bge-reranker-v2-m3` (2,2 Go — seulement si la machine le permet, sinon documenter « non testé, taille »).
- [ ] **Step 6 : Documenter** (« Ablation B ») : delta, IC95, p, ventilation, latence, décision. Si adopté, `--mode all` de run_eval inclut `hybrid+rerank` et le README le mentionnera en T6.
- [ ] **Step 7 : Commit** — `feat(jalon25): reranker cross-encoder optionnel (mode hybrid+rerank) — mesuré par bootstrap`

---

### Task 5: Ablation C — synonymes pilotés par les échecs (avec rebuild)

**Files:**
- Create: `scripts/analyse_echecs.py`
- Modify: `src/accounting_rag/normalize.py` (SYNONYMES uniquement)
- Modify: `tests/test_normalize.py`
- Modify: `docs/eval-jalon25.md` (section « Ablation C »)

**Interfaces:**
- Consumes: `evaluate(...)["par_question"]`, corpus.db, SYNONYMES existant.
- Produces: `scripts/analyse_echecs.py --split dev --mode <meilleur>` → pour chaque question dev à recall@10 < 1 : question, citations gold, top-10 obtenu, et le diff des tokens normalisés (tokens de la question absents des records gold et réciproquement).

- [ ] **Step 1 : Écrire `scripts/analyse_echecs.py`** (réutilise Searcher + evalrag ; sortie markdown lisible). Pas de test dédié (script d'analyse), mais un run réel vérifié.
- [ ] **Step 2 : Run réel** sur dev avec la meilleure config T3/T4 ; consigner la sortie (fichier `docs/echecs-dev-jalon25.md`, committé — c'est le matériau du choix des synonymes).
- [ ] **Step 3 : Proposer un LOT de candidats** (max 10 entrées) à partir des échecs : chaque entrée cite la ou les questions qu'elle vise et le token gold qu'elle doit atteindre. RÈGLE (ruling J2-5) : une entrée relie un terme courant à SON équivalent PCG exact — jamais un rapprochement de concepts distincts. Chaque entrée douteuse sur le plan comptable est écartée.
- [ ] **Step 4 : Appliquer le lot** dans SYNONYMES + tests unitaires des nouvelles entrées (normalize("terme courant") contient le stem attendu).
- [ ] **Step 5 : REBUILD** — `cp data/corpus.db <scratchpad>/corpus.db.bak-j25` puis `uv run python scripts/build_index.py` en arrière-plan (~14 min) ; à la fin, vérifier `records`=1660 / `renvois`=981 inchangés.
- [ ] **Step 6 : Mesure** — re-runner la meilleure config sur dev ; `paired_bootstrap(avant, après)`. Adoption par le critère global ; si dégradation, retirer les entrées incriminées (identifiables par leurs questions cibles), re-rebuild, re-mesurer — UN seul cycle de retrait maximum.
- [ ] **Step 7 : Documenter** (« Ablation C ») : lot proposé, lot retenu, delta/IC/p, et la liste des échecs RESTANTS (matériau du jalon 3 : ce que le lexical ne peut pas combler).
- [ ] **Step 8 : Commit** — `feat(jalon25): synonymes pilotés par les échecs mesurés (+ analyse des échecs restants)`

---

### Task 6: Campagne de clôture — test split (une fois) + docs + README

**Files:**
- Modify: `docs/eval-jalon25.md` (sections « Clôture » et « Lecture »)
- Modify: `README.md`

**Interfaces:**
- Consumes: la configuration finale adoptée (paramètres T3, mode T4, SYNONYMES T5).

- [ ] **Step 1 : Suite complète** `uv run pytest -q` (arrière-plan si > 5 min) → tout PASS.
- [ ] **Step 2 : Campagne finale dev** — baseline 2.5 vs configuration finale, tableau + bootstrap global et par catégorie.
- [ ] **Step 3 : Test split, UNE exécution** — baseline 2.5 ET configuration finale sur test (29 questions) ; présentées comme « référence gelée jalon 2.5 » ; constat dev/test commenté (écart = signal de sur-ajustement au dev).
- [ ] **Step 4 : README** — tableau mis à jour (benchmark v2, config finale), section limitations ajustée (ce qui reste : réécriture de requête, embeddings métier — jalon 3), mention `ACCRAG_RERANKER` si le reranker est adopté.
- [ ] **Step 5 : Cohérence** — mêmes chiffres partout (README ↔ eval-jalon25.md), aucun chiffre n=21 comparé à n=61.
- [ ] **Step 6 : Commit** — `docs(jalon25): campagne de clôture — dev + référence test gelée, décisions d'ablation`

---

## Self-review (effectuée à l'écriture)

- Couverture spec : §4 item 4 (pondération par champ) → T3 ; §5 protocole d'ablation → Global Constraints + T2 (outillage) + T3/T4/T5 (application) ; reranker (ruling J2-3) → T4 ; « agrandir le benchmark avant tout ajustement lexical » (réserve de la vague finale du jalon 2) → T1 avant tout, T5 après.
- Cohérence de types : `evaluate()` étend son dict sans casser les clés existantes (T7 jalon 2 n'y lit que des clés stables) ; `Searcher` n'ajoute que des kwargs à défaut neutre ; `paired_bootstrap` consomme exactement le format `par_question` produit par `evaluate`.
- Pièges nommés : poids bm25() liés en paramètres SQL (pas d'interpolation) ; routés épinglés hors reranker ; rebuild obligatoire après SYNONYMES ; un seul passage sur le split test.
