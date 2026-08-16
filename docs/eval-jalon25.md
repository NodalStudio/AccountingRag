# Jalon 2.5 — Fossé lexical : baseline v2 et bootstrap apparié

Campagne exécutée le **16 août 2026** sur la branche `jalon-25-fosse-lexical`, à l'issue des tâches T1-T3 (bootstrap apparié, scores par question, mesure baseline 2.5). Objectif : établir la baseline sur le benchmark v2 (61 questions) avec décomposition par question et test statistique par bootstrap apparié (en préparation pour les comparaisons de tâches T4+).

**Important** : le benchmark a changé depuis le jalon 2 — passage de n=21 (`dev` jalon 2) à n=61 (`dev` v2). Les chiffres ci-dessous ne sont **pas comparables** directement aux résultats du jalon 2 (instrument et population différents). Voir section « Conditions exactes ».

## Conditions exactes

| Paramètre | Valeur |
|---|---|
| Corpus | `data/corpus.db`, 1 660 `records` (739 réglementaires, 921 commentaires ANC), 981 `renvois` |
| Index de recherche | 2 160 chunks, table FTS5 `chunks_norm` (texte normalisé : élisions, refs atomiques, synonymes, stemming FR), table vectorielle `chunks_vec` (sqlite-vec 0.1.9) |
| Modèle d'embeddings | `intfloat/multilingual-e5-small` (défaut de `Embedder`, 384 dimensions), préfixes `query:` / `passage:` |
| Fusion hybride | RRF (`k=60`) sur BM25 + dense ; `hybrid+graph` ajoute une expansion 1-hop via `renvois` (famille `interne`) sur les 5 premiers résultats, score pondéré ×0,5 |
| Benchmark | `benchmark/dev.jsonl` v2 — 61 questions (7 `reference_directe`, 23 `regle`, 31 `vocabulaire_courant`) |
| Commande | `uv run python scripts/run_eval.py --mode all --split dev` |
| Machine | Linux 6.19.8-arch1-3-surface, x86_64, 8 threads, CPU uniquement (pas de GPU exploité — `torch` signale l'absence de driver CUDA compatible, sans conséquence : e5-small tourne en CPU) |
| Environnement | Python 3.13.14, `sentence-transformers` 5.7.0, `torch` 2.13.0+cu130 (CPU), `sqlite-vec` 0.1.9 — versions résolues par `uv.lock` |

## Résultats — split dev (61 questions), benchmark v2

| mode | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| bm25 | 0,648 | 0,664 | 0,547 | 61 |
| dense | 0,475 | 0,566 | 0,426 | 61 |
| hybrid | 0,639 | 0,672 | 0,565 | 61 |
| hybrid+graph | 0,639 | 0,672 | 0,565 | 61 |

### Ventilation par catégorie (recall@10)

| mode | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|
| bm25 | 1,0 | 0,891 | 0,419 |
| dense | 1,0 | 0,913 | 0,21 |
| hybrid | 1,0 | 0,935 | 0,403 |
| hybrid+graph | 1,0 | 0,935 | 0,403 |

Sortie brute de `scripts/run_eval.py --mode all --split dev` :

```
| mode | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| bm25 | 0.648 | 0.664 | 0.547 | 61 |
|   ↳ reference_directe | | 1.0 | | 7 |
|   ↳ regle | | 0.891 | | 23 |
|   ↳ vocabulaire_courant | | 0.419 | | 31 |
| dense | 0.475 | 0.566 | 0.426 | 61 |
|   ↳ reference_directe | | 1.0 | | 7 |
|   ↳ regle | | 0.913 | | 23 |
|   ↳ vocabulaire_courant | | 0.21 | | 31 |
| hybrid | 0.639 | 0.672 | 0.565 | 61 |
|   ↳ reference_directe | | 1.0 | | 7 |
|   ↳ regle | | 0.935 | | 23 |
|   ↳ vocabulaire_courant | | 0.403 | | 31 |
| hybrid+graph | 0.639 | 0.672 | 0.565 | 61 |
|   ↳ reference_directe | | 1.0 | | 7 |
|   ↳ regle | | 0.935 | | 23 |
|   ↳ vocabulaire_courant | | 0.403 | | 31 |
```

## Lecture

- **`bm25` seul reste leader sur `reference_directe` et `regle`** (0,664 et 0,891 de recall@10, vs 0,566 et 0,913 pour dense), cohérent avec le jalon 2, mais le fossé lexical sur `vocabulaire_courant` s'aggrave légèrement (0,419 sur n=31 questions, vs 0,571 sur n=7 au jalon 2) — population plus large et plus hétérogène du benchmark v2 révèle la fragilité du vocabulaire courant.
- **`dense` seul s'effondre sur `vocabulaire_courant`** (0,21, pire que bm25) conformément au diagnostic du jalon 2 — les embeddings génériques e5-small ne comblent pas le fossé lexical entre langage réglementaire et courant.
- **`hybrid` (RRF bm25+dense) améliore légèrement le MRR** (0,565 vs 0,547 pour bm25 seul) mais perd en recall@10 (0,672 vs 0,664), symptôme que la fusion RRF n'optimise pas recall@10 par rapport au meilleur canal seul sur ce type de question.
- **`hybrid+graph` reste strictement égal à `hybrid`** sur ce split v2 également — l'expansion 1-hop ne comble aucun manque supplémentaire.
- **Le routeur regex fonctionne toujours** : `reference_directe` = 1,0 dans tous les modes, signalant que le routeur regex résout ces questions avant même les heuristiques lexicales.

## Réserves

- n=61 (dev v2) vs n=21 (dev jalon 2) : instruments différents, populations différentes — **comparaison directe interdite** (cf. introduction).
- Les catégories `vocabulaire_courant` (n=31) et `regle` (n=23) sont suffisamment larges pour un diagnostic qualitatif basique, mais un écart d'une question déplace le recall de quelques points ; pas de test statistique de significativité sur ces sous-groupes.
- Les durées de chargement du modèle (~50 s) varient selon l'état du cache Hugging Face local ; mesures indicatives dans les conditions exécutées ici.
- Scores par question collectés mais non détaillés dans ce document (table trop volumineuse) — disponibles via `evaluate()` retournant `"par_question": {qid: recall@10}`, préparation pour bootstrap apparié en tâches suivantes.

## Bootstrap apparié (préparation T4+)

La fonction `paired_bootstrap(a: dict[str, float], b: dict[str, float])` est implémentée et testée. Elle calcule sur les scores par question :
- `delta` : moyenne des différences (b - a)
- `ic95` : intervalle de confiance 95% par rééchantillonnage avec replacement (n_boot=10000, seed=42)
- `p_amelioration` : proportion de bootstraps où la différence > 0

Exemple d'utilisation future pour comparer deux modes :
```python
from accounting_rag.evalrag import evaluate, paired_bootstrap
a = evaluate(searcher, qs, mode="bm25")
b = evaluate(searcher, qs, mode="hybrid")
comp = paired_bootstrap(a["par_question"], b["par_question"])
print(f"hybrid - bm25 : delta={comp['delta']}, ic95={comp['ic95']}, p={comp['p_amelioration']:.3f}")
```

## Reproduction exacte

```sh
# Depuis la racine du dépôt, avec uv installé
uv run python scripts/build_corpus.py     # si data/corpus.db n'existe pas encore
uv run python scripts/build_index.py      # construit chunks + FTS + vecteurs (~14 min CPU)
uv run python scripts/run_eval.py --mode all --split dev
uv run pytest tests/test_evalrag.py -q
```

Le split `test` ne doit être relancé qu'en fin de jalon suivant, jamais pendant le développement.

## Ablation A — pondération par champ (chemin, type de record) (T3)

Deux nouveaux paramètres sur `Searcher(db_path, embedder=None, poids_chemin: float = 1.0, boost_commentaire: float = 1.0)`, neutres par défaut (comportement jalon 2 inchangé) :
- `poids_chemin` : poids de la colonne `chemin_norm` dans `bm25(chunks_norm, 1.0, poids_chemin)` (poids_texte fixé à 1.0). Lié en paramètre SQL — testé et validé sur SQLite 3.53.1 (les arguments de `bm25()` acceptent des paramètres liés sur cette version ; pas de repli par interpolation de littéral nécessaire, cf. Ruling J25-2).
- `boost_commentaire` : multiplicateur appliqué au score bm25 agrégé d'un chunk quand `records.type != 'reglementaire'` (donc `commentaire_ANC`), avant le max par `record_id`. Aucune garde supplémentaire n'est nécessaire sur le signe : `bm25()` de SQLite retourne toujours `b ≤ 0` par construction (plus petit = meilleur match), donc `s = -b ≥ 0` systématiquement — un multiplicateur `< 1` pénalise donc toujours (jamais d'inversion de signe qui transformerait une pénalité en bonus), quel que soit le score brut.

### Méthode

Mesure une variable à la fois, par bootstrap apparié contre le run précédent, sur `benchmark/dev.jsonl` (61 questions), mode `hybrid`, k=10. Commande de mesure : script ad hoc (non versionné) instanciant plusieurs `Searcher` dans le même processus python (embedder e5 partagé) pour comparer :

1. **A** = hybrid baseline, poids neutres (= baseline T2, re-mesurée dans le même processus pour garantir l'alignement des ids `par_question`).
2. **B1** = hybrid avec `poids_chemin=2.0` seul → `paired_bootstrap(A, B1)`.
3. **B2** = meilleur de {A, B1} (selon critère d'adoption) + `boost_commentaire=0.7` seul en plus → `paired_bootstrap(config_de_base, B2)`.
4. Valeur voisine de robustesse (`poids_chemin=3.0` ou `boost_commentaire=0.5`) **seulement si le paramètre correspondant a été adopté** à l'étape précédente — non déclenchée ici (voir Résultats).

Critère d'adoption (contrainte globale du plan) : `p_amelioration ≥ 0,95` **ET** aucune catégorie ne perd plus de 0,05 de recall@10 vs la configuration de référence.

### Résultats

| run | config | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| A | neutre (poids_chemin=1.0, boost_commentaire=1.0) | 0,639 | 0,672 | 0,565 |
| B1 | poids_chemin=2.0 | 0,639 | 0,672 | 0,557 |
| B2 | poids_chemin=1.0 (A retenu, B1 non adopté) + boost_commentaire=0.7 | 0,623 | 0,664 | 0,563 |

Ventilation par catégorie (recall@10) :

| run | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|
| A | 1,0 | 0,935 | 0,403 |
| B1 | 1,0 | 0,935 | 0,403 |
| B2 | 1,0 | 0,935 | 0,387 |

Bootstrap apparié (`n_boot=10000`, `seed=42`, sur les scores `par_question` de `evaluate(..., mode="hybrid", k=10)`) :

| comparaison | delta | IC95 | p_amelioration | pire perte par catégorie | adopté ? |
|---|---|---|---|---|---|
| A vs B1 (poids_chemin=2.0) | 0,0000 | (0,0000 ; 0,0000) | 0,000 | 0,000 | **non** |
| A vs B2 (poids_chemin=1.0 + boost_commentaire=0.7, effet cumulé) | -0,0082 | (-0,0656 ; 0,0492) | 0,337 | -0,016 (vocabulaire_courant) | **non** |

`A vs B2` ci-dessus sert aussi de comparaison « config de base retenue (A) vs B2 » puisque B1 n'a pas été adopté et que la config de base pour B2 est donc A elle-même.

Aucune valeur voisine de robustesse (`poids_chemin=3.0`, `boost_commentaire=0.5`) n'a été mesurée : le brief ne la prévoit que pour un paramètre déjà adopté à l'étape précédente, et aucun des deux paramètres ne franchit le seuil `p_amelioration ≥ 0,95`.

### Décision

**Rejeté — les deux paramètres restent à leurs valeurs neutres par défaut** (`poids_chemin=1.0`, `boost_commentaire=1.0`), qui deviennent les défauts des flags `--poids-chemin` / `--boost-commentaire` de `scripts/run_eval.py`.

Motivation :
- `poids_chemin=2.0` produit un delta et un IC95 **exactement nuls** (`p_amelioration=0,000`) sur recall@10 — mais **pas** parce que l'ordre bm25 serait insensible à ce poids : contrôle direct sur les 61 questions, l'ordre top-10 bm25 change effectivement sur 45/61 questions, et le *set* top-10 bm25 change sur 21/61. L'explication correcte est en aval de bm25 : la fusion RRF (`_rrf`) combine les canaux bm25/dense par **rang** (`1/(60+rang+1)`), et sur ce split, les permutations induites par `poids_chemin=2.0` touchent des rangs bas ou des documents non-gold — elles ne changent jamais, sur aucune des 61 questions, le fait qu'une citation gold soit ou non dans le top-10 de la liste **fusionnée**. Le canal dense, indépendant du poids_chemin, amortit encore l'effet en stabilisant la position des documents pertinents dans la fusion. Le MRR qui bouge légèrement (0,565 → 0,557) est justement la trace de ces permutations bm25 réelles : il est sensible au rang fin du premier résultat pertinent, contrairement à recall@10 qui est un simple indicateur in/out à k=10. Un poids de chemin plus agressif (3.0) n'a pas été testé puisque le critère d'adoption échoue déjà nettement à 2.0 (le brief n'exige la valeur voisine que si le paramètre est adopté).
- `boost_commentaire=0.7` va dans le sens **opposé** à l'amélioration (delta négatif, -0,0082) avec `p_amelioration=0,337`, largement sous le seuil de 0,95 — la pénalisation des commentaires ANC dégrade légèrement `vocabulaire_courant` (-0,016, dans la limite de -0,05 mais sans gain compensatoire ailleurs). Contrairement à `poids_chemin`, ce paramètre agit avant la fusion RRF (multiplicateur sur le score bm25 brut) et peut donc réordonner le canal bm25 — l'effet mesuré, bien que faible, est cohérent et défavorable.

### Réserves

- Les deux ablations ont été mesurées **une à la fois** (méthode imposée par le brief), pas en grid search ; une interaction positive entre `poids_chemin` et `boost_commentaire` à d'autres valeurs n'est pas exclue mais sort du périmètre de cette tâche.
- `poids_chemin` n'a d'effet mesurable que via la position relative dans le classement bm25 **avant** fusion RRF ; sur un corpus où le champ `chemin` porte plus d'information distinctive (hiérarchie de plan plus profonde/variée), l'effet pourrait différer — ce résultat est spécifique à `data/corpus.db` (PCG 2026) tel qu'indexé aujourd'hui.
- Les tests unitaires (`tests/test_search.py`) valident le comportement des deux paramètres sur une base synthétique conçue pour isoler chaque effet (voir `test_poids_chemin_favorise_le_chemin`, `test_boost_commentaire_penalise`) — ils démontrent que le mécanisme fonctionne, indépendamment de la décision de ne pas l'activer par défaut sur le corpus réel.

## Ablation B — reranker cross-encoder (T4)

Nouveau module `src/accounting_rag/rerank.py` (`Reranker(model_name=None)`, env `ACCRAG_RERANKER`) et nouveau mode `Searcher.search(mode="hybrid+rerank")` : les résultats routés (référence d'article exacte, source `route`) restent épinglés en tête **sans** repasser par le reranker ; la fusion RRF (bm25+dense, canaux à leurs limites par défaut) est récupérée à 25 candidats non routés, rerankée par le cross-encoder (`score_rerank` ajouté, tri décroissant), puis tronquée à `k - len(routed)`. Chaque texte est tronqué à 1000 caractères avant l'appel `predict()` pour borner la latence par paire.

### Méthode

Référence de mesure : le **hybrid baseline pur** (paramètres neutres `poids_chemin=1.0`, `boost_commentaire=1.0`), re-runné dans le même processus python que les runs `hybrid+rerank` (embedder e5 partagé) pour garantir l'alignement des ids `par_question` — la pondération par champ ayant été rejetée en T3, ce n'est pas une config candidate ici (ruling J25-1). Deux modèles de reranker mesurés indépendamment contre cette même référence A (une comparaison par modèle, pas de comparaison directe modèle-à-modèle) :

1. **B1** = `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingue, léger — pressenti comme défaut dans le brief avant mesure).
2. **B2** = `BAAI/bge-reranker-v2-m3` (2,2 Go — alternative prévue par le brief en cas de rejet de B1 ; testée ici car la machine dispose de 30 Gio de RAM et 69 Gio de disque libre, largement suffisant).

`benchmark/dev.jsonl` (61 questions), `k=10`, script ad hoc (non versionné, scratchpad) instanciant un `Embedder` unique partagé entre les deux `Searcher` (A et B) de chaque run, et un `Reranker` par modèle testé. Critère d'adoption (contrainte globale du plan) : `p_amelioration ≥ 0,95` **ET** aucune catégorie ne perd plus de 0,05 de recall@10 vs A.

### Résultats

| run | config | recall@5 | recall@10 | MRR | latence/question |
|---|---|---|---|---|---|
| A | hybrid baseline neutre (re-mesuré) | 0,639 | 0,672 | 0,565 | 0,20 s |
| B1 | hybrid+rerank, `mmarco-mMiniLMv2-L12-H384-v1` | 0,672 | 0,713 | 0,610 | 10,0 s |
| B2 | hybrid+rerank, `bge-reranker-v2-m3` | 0,680 | 0,738 | 0,642 | 117,1 s |

A re-mesuré ici (même processus python, embedder e5 partagé avec B1/B2) est identique au A de T3 et à la baseline hybrid du corps principal de ce document (recall@10=0,672, mrr=0,565) — alignement des ids `par_question` garanti (ruling J25-1).

Ventilation par catégorie (recall@10) :

| run | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|
| A | 1,0 | 0,935 | 0,403 |
| B1 (mmarco) | 1,0 | 1,0 | 0,435 |
| B2 (bge) | 1,0 | 1,0 | 0,484 |

Bootstrap apparié (`n_boot=10000`, `seed=42`, sur `evaluate(..., mode=..., k=10)["par_question"]`) :

| comparaison | delta | IC95 | p_amelioration | pire perte par catégorie | seuil (p≥0,95 et perte≤0,05) | adopté ? |
|---|---|---|---|---|---|---|
| A vs B1 (mmarco-mMiniLMv2-L12-H384-v1) | 0,0410 | (-0,0246 ; 0,1148) | 0,858 | 0,000 (aucune perte, seulement des gains) | non atteint | **non** |
| A vs B2 (bge-reranker-v2-m3) | 0,0656 | (-0,0082 ; 0,1393) | 0,952 | 0,000 (aucune perte, seulement des gains) | **atteint** | **oui** |

### Décision

**Adopté — avec un coût de latence majeur à documenter.** Le mode `hybrid+rerank` est livré et son critère d'adoption est jugé sur le **meilleur des deux rerankers mesurés** (consigne explicite reçue en cours de tâche) :

- **`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`** (modèle initialement pressenti comme défaut dans le brief, avant mesure) : `p_amelioration=0,858`, sous le seuil de 0,95 — **REJETÉ** malgré un gain de recall@10 réel et positif (0,672→0,713) et l'absence de toute perte de catégorie. Le gain existe mais n'est pas assez robuste statistiquement sur n=61 pour franchir le seuil fixé par le plan.
- **`BAAI/bge-reranker-v2-m3`** : `p_amelioration=0,952`, **franchit** le seuil de 0,95 (de peu — IC95 = (-0,0082 ; 0,1393), la borne basse reste très légèrement négative) ; aucune catégorie ne perd de recall@10 (les trois s'améliorent ou restent égales) — **ADOPTÉ** au sens strict du critère.

Conséquence de ce résultat : le défaut de `Reranker` (`_DEFAULT` dans `rerank.py`, utilisé si `ACCRAG_RERANKER` n'est pas positionné) devient `BAAI/bge-reranker-v2-m3` — et non `mmarco-mMiniLMv2-L12-H384-v1` comme initialement pressenti dans le brief avant mesure, ce dernier ayant été rejeté par la mesure elle-même.

**Coût mesuré du gain (le point central de cette décision) :** `bge-reranker-v2-m3` coûte **117,1 s par question** sur cette machine (CPU, 8 threads, aucun GPU exploitable), soit **≈585× la latence du hybrid baseline** (0,20 s/question) et **≈11,7× celle de `mmarco-mMiniLMv2-L12-H384-v1`** (10,0 s/question, lui-même déjà rejeté). Ce coût n'entre pas dans le critère d'adoption formel (purement statistique), mais doit être visible pour tout appelant de `hybrid+rerank` en contexte interactif — ce mode n'est **pas** adapté à une latence de requête utilisateur en l'état (117 s/requête), seulement à des campagnes d'évaluation batch ou du re-classement asynchrone hors ligne.

**Révision suite revue T4 (fix round 1) : `hybrid+rerank` N'EST PAS ajouté à `--mode all` de `scripts/run_eval.py`.** Le mode reste sélectionnable explicitement (`--mode hybrid+rerank`, présent dans `choices`) mais `--mode all` — la campagne par défaut, censée rester rapide (~1 min sur dev) — ne l'exécute plus, pour ne pas faire passer sa durée à ~2h à l'insu de l'appelant. `Reranker.__init__` émet désormais un avertissement (`stderr`) rappelant le nom du modèle, la taille au premier téléchargement (~2,2 Go) et la latence CPU indicative avant de charger le cross-encoder ; `rerank()` court-circuite immédiatement (`if top_k <= 0: return []`) sans appeler `predict()` quand les résultats routés remplissent déjà `k`, évitant jusqu'à ~2 min de calcul cross-encoder inutile sur ces requêtes.

`mmarco-mMiniLMv2-L12-H384-v1` reste documenté et disponible via `ACCRAG_RERANKER=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` pour les contextes où une latence de ~10 s/question est acceptable mais 117 s/question ne l'est pas — avec le rappel explicite que ce choix n'a **pas** franchi le seuil d'adoption statistique (rejeté, gain non garanti au sens du critère).

### Réserves

1. `p_amelioration=0,952` pour `bge-reranker-v2-m3` est une réussite **marginale** du seuil (0,95 exigé) — l'IC95 inclut encore une borne basse légèrement négative (-0,0082), signe qu'une petite fraction (~4,8 %) des tirages bootstrap ne montre pas d'amélioration. Ce n'est pas un résultat écrasant ; une mesure sur `test` (29 questions, en fin de jalon suivant, jamais avant) pourrait confirmer ou nuancer ce verdict.
2. Le coût de latence (117 s/question) n'a pas été optimisé (pas de quantification, pas d'ONNX Runtime, pas de GPU, batch naïf de 25 paires par requête via `CrossEncoder.predict()` par défaut) — une latence bien plus faible est probablement atteignable avec des optimisations d'inférence hors périmètre de cette tâche. Le chiffre mesuré ici est celui d'une intégration directe, pas une borne physique du modèle.
3. Les deux rerankers ont été mesurés **une fois chacun** contre la même référence A (pas de répétition ni de validation croisée) ; `n_boot=10000` avec `seed=42` fixe la reproductibilité du bootstrap, pas la variance d'échantillonnage du benchmark lui-même (n=61).
4. Le chargement du modèle (téléchargement HuggingFace au premier appel, ~2,2 Go pour `bge-reranker-v2-m3`, ~470 Mo pour `mmarco-mMiniLMv2-L12-H384-v1`) n'est pas comptabilisé dans la latence par question ci-dessus (mesurée après chargement, sur les 61 appels `search()`) — chargé une fois en ~25-55 s dans le script de mesure.
5. Aucune valeur voisine de robustesse n'a été mesurée pour un troisième modèle : le brief ne prévoit qu'UNE alternative en cas de rejet du défaut pressenti, ce qui a été fait (`bge-reranker-v2-m3`).
6. Les tests (`tests/test_rerank.py`) utilisent exclusivement un `FakeCrossEncoder`/`FakeReranker` injecté — aucun téléchargement, aucune dépendance à la mesure ci-dessus pour passer. Le comportement d'épinglage du routeur (`test_mode_hybrid_rerank_epingle_le_route`) est vérifié sur la base synthétique déjà existante dans `tests/test_search.py` (fixture `searcher_synthetique` réutilisée par import direct).
