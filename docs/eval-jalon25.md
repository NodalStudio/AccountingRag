# Jalon 2.5 — Fossé lexical : baseline v2, ablations et clôture

Campagne exécutée le **16 août 2026** sur la branche `jalon-25-fosse-lexical`, à l'issue des tâches T1-T6 (bootstrap apparié, scores par question, mesure baseline 2.5, ablations A/B/C, clôture dev/test). Objectif : établir la baseline sur le benchmark v2 (61 questions) avec décomposition par question et test statistique par bootstrap apparié, mesurer trois pistes d'amélioration (pondération par champ, reranker cross-encoder, synonymes pilotés par les échecs), puis clôturer le jalon avec la campagne dev finale et la référence test gelée.

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

- **`bm25` domine `dense` sur `vocabulaire_courant`** (0,419 vs 0,21 de recall@10) **et globalement** (0,664 vs 0,566), **`dense` étant légèrement meilleur sur `regle`** (0,913 vs 0,891) — le seul sous-ensemble où le canal dense dépasse le lexical sur ce split.
- **`dense` seul s'effondre sur `vocabulaire_courant`** (0,21, pire que bm25) conformément au diagnostic du jalon 2 — les embeddings génériques e5-small ne comblent pas le fossé lexical entre langage réglementaire et courant.
- **`hybrid` (RRF bm25+dense) améliore le MRR et le recall@10 par rapport à `bm25` seul** (MRR 0,565 vs 0,547 ; recall@10 0,672 vs 0,664), mais marginalement (+0,008, soit une demi-question sur 61).
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
uv run python scripts/download_data.py    # télécharge le Recueil ANC, si data/raw n'existe pas encore
uv run python scripts/build_corpus.py     # si data/corpus.db n'existe pas encore
uv run python scripts/build_index.py      # construit chunks + FTS + vecteurs (~14 min CPU)
uv run python scripts/run_eval.py --mode all --split dev
uv run pytest tests/test_evalrag.py -q
```

Le split `test` n'est relancé qu'une seule fois, à la clôture du présent jalon — voir section « Clôture » ci-dessous — jamais pendant le développement des tâches intermédiaires.

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

`benchmark/dev.jsonl` (61 questions), `k=10`, script ad hoc instanciant un `Embedder` unique partagé entre les deux `Searcher` (A et B) de chaque run, et un `Reranker` par modèle testé. Critère d'adoption (contrainte globale du plan) : `p_amelioration ≥ 0,95` **ET** aucune catégorie ne perd plus de 0,05 de recall@10 vs A. Les dicts `par_question` bruts de cette mesure T4 n'ont pas été persistés à l'époque (seuls les agrégats l'ont été, cf. section Clôture) ; pour `bge-reranker-v2-m3` (B2, le modèle finalement adopté), les mêmes chiffres sont reproduits à l'identique et persistés dans `docs/mesures/jalon25/cloture_dev.json` (champ `b`, cf. section Clôture, T6) — pour `mmarco-mMiniLMv2-L12-H384-v1` (B1, rejeté), aucun dict `par_question` brut n'est versionné, seuls les agrégats ci-dessous.

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

## Ablation C — synonymes pilotés par les échecs (T5)

Nouveau script `scripts/analyse_echecs.py` (réutilise `Searcher` + `evalrag`, aucune logique dupliquée) : pour chaque question d'un split dont `recall@10 < 1`, affiche la question, ses citations gold, le top-10 obtenu, le texte des records gold et le diff des tokens normalisés (question ↔ gold). Sortie complète sur `benchmark/dev.jsonl` committée dans `docs/echecs-dev-jalon25.md` — c'est le matériau brut d'où est tiré le lot de synonymes candidats ci-dessous.

**Config (ruling J25-7)** : mode `hybrid` (RRF bm25+dense, paramètres neutres) pour l'analyse des échecs (step 2) **et** pour la mesure avant/après (step 6) — pas `hybrid+rerank` : le canal synonymes n'agit que sur `_bm25()` (`normalize()` est appliqué avant l'appel `bm25()` de SQLite, jamais avant l'encodage dense), et `hybrid` tourne en ~1 min contre ~2h pour `hybrid+rerank` sur cette machine.

### Analyse des échecs dev (61 questions, mode hybrid)

21/61 questions à `recall@10 < 1` (`reference_directe` 1,0 ; `regle` 0,935 ; `vocabulaire_courant` 0,403 — identique à la baseline hybrid du corps principal de ce document). Détail complet dans `docs/echecs-dev-jalon25.md`.

En lisant les diffs de tokens normalisés, trois familles d'échecs se distinguent :

1. **Fossé lexical réel** — la question emploie un terme courant absent du vocabulaire réglementaire du record gold (ex. q059/q060/q086 : « aides » vs « subventions » ; q079/q080/q089 : décrivent un scénario de fusion sans jamais nommer « boni »/« mali » de fusion).
2. **Pas de fossé lexical, problème de rang** — q008 et q054 partagent déjà presque tout leur vocabulaire normalisé avec le record gold (diff de 3-4 tokens seulement, dont une quasi-citation littérale pour q054 : « immobilisation créée par les moyens propres de l'entité ») mais le record gold ne sort pas dans le top-10 fusionné RRF. Aucun synonyme ne peut agir ici : il n'y a rien à rapprocher lexicalement, le problème est en aval (fusion bm25+dense, ou dilution par des tokens trop fréquents).
3. **Distinction comptable fine requise dans la MÊME question** — q022 (« ne paie pas depuis des mois » = créance douteuse, pcg-1214-41, vs « ne paiera plus jamais » = créance irrécouvrable, pcg-1221-65) et q085 (immobilisation en cours de construction vs juste terminée, deux comptes différents 1212-23/1222-72). Un rapprochement lexical générique risquerait ici de conflater deux concepts comptables distincts — exactement l'écueil visé par le ruling J2-5 (précédent degressif/derogatoire du jalon 2).

### Lot de candidats proposé (step 3)

Règle appliquée (ruling J2-5) : une entrée relie un terme courant à SON équivalent PCG exact — jamais un rapprochement de deux concepts distincts, et jamais une clause/scénario entier substitué à un terme (risque de sur-généralisation et de sur-ajustement au benchmark). Sur les 21 échecs, seules deux relations terme-à-terme ont passé ce filtre :

| # | clé (forme trouvée) | valeur (forme canonique) | questions ciblées | token gold visé |
|---|---|---|---|---|
| 1 | `aides` (pluriel uniquement) | `subventions` | q059 (verbe « aider », non couvert), q060, q086 | `subvent` |
| 2 | `la boite` | `l'entite` | q059 | `entit` |
| 3 | `une boite` | `une entite` | q080, q089 | `entit` |

Justification comptable et garde-fous techniques (vérifiés sur tout `data/corpus.db`, 1660 records) :
- **« aides » → « subventions »** : « aide » (financière) est l'usage courant pour ce que le PCG nomme uniformément « subvention » (comptes 74, 312-1) — relation terme-à-terme stable, aucun concept distinct conflaté. Clé au **pluriel uniquement** : le corpus contient l'idiome « à l'aide de » (= au moyen de, ex. `pcg-324-1-c15` : « actualisées à l'aide du taux ») qui n'a **aucun** rapport avec les subventions — un remplacement sur la forme singulière l'aurait corrompu (`normalize()` fait un remplacement de sous-chaîne sans limite de mot). La forme plurielle est, sur ce corpus, exclusivement utilisée au sens subvention/aide d'État (vérifié par balayage exhaustif des occurrences).
- **« boîte » → « entité »** : argot très courant pour « entreprise/société », dont le PCG utilise partout le terme constant « entité » — relation non ambiguë, aucun autre référent possible pour « boîte » dans ce registre. Clé **avec déterminant** (`la boite`/`une boite`, jamais `boite` nu) : le corpus contient « boîtes aux lettres » (`pcg-191-1-c23`, équipement d'immeuble), qui aurait été corrompu par un remplacement du seul radical (« boite » est une sous-chaîne de « boites »).

### Entrées écartées par prudence (step 3)

| question(s) | rapprochement envisagé | motif de l'écart |
|---|---|---|
| q021, q023, q056, q063, q065, q068 | (aucun terme fixe identifiable) | la question décrit un scénario complet sans nommer de terme métier alternatif — il n'y a pas de paire terme-à-terme à ajouter, seulement une lacune de compréhension de requête (hors périmètre lexical) |
| q022 | « ne paie plus jamais » / « ne paie pas depuis des mois » → un terme unique de créance | **écarté par prudence** (ruling J2-5) : la question mélange deux concepts PCG distincts dans la même phrase (créance douteuse, pcg-1214-41 ≠ créance irrécouvrable, pcg-1221-65) ; tout rapprochement générique risquerait de les conflater |
| q057 | « en train d'être fabriqué » → « en cours de production » | **écarté par prudence** : correspondance plausible mais très spécifique à cette formulation exacte du benchmark — risque de sur-ajustement (pas un terme métier réutilisable comme « leasing »/« credit-bail »), pas une vraie synonymie de vocabulaire courant |
| q070, q071, q074 | « dollar » → « devise étrangère » / « monnaie étrangère » | **écarté par prudence** : « dollar » n'est qu'UN exemple de devise parmi d'autres (yen, livre…) — c'est une relation d'appartenance à une catégorie, pas une synonymie terme-à-terme stable ; ne couvrir qu'une devise nommée serait arbitraire |
| q079, q080, q089 | scénario de fusion → « boni de fusion » / « mali de fusion » / « mali technique » | **écarté par prudence** : ces termes sont déjà les termes techniques eux-mêmes ; il n'existe pas de synonyme courant unique et stable qui les désigne (contrairement à « leasing »=« crédit-bail ») — la question décrit une clause entière, pas un terme à rapprocher. Le rapprochement « boîte »→« entité » (lot retenu) reste appliqué à ces questions mais n'atteint pas le cœur du fossé (boni/mali) |
| q008, q054, q085 | (aucun rapprochement pertinent) | pas un fossé lexical (diff de tokens minuscule, cf. § précédent) — problème de rang dans la fusion RRF, hors du périmètre "synonymes" de cette tâche |

### Mesure (step 6)

Lot appliqué dans `SYNONYMES`, tests unitaires ajoutés, **rebuild complet** (`uv run python scripts/build_index.py`, ~14 min), puis re-mesure sur `benchmark/dev.jsonl`, mode `hybrid`, k=10, avec persistance des dicts `par_question` bruts en JSON (ruling J25-6 — fichiers `docs/mesures/jalon25/t5_avant.json`/`t5_apres.json`, reproduits ci-dessous).

| run | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| avant (SYNONYMES original, 9 entrées) | 0,639 | 0,672 | 0,565 | 61 |
| après (SYNONYMES + lot de 3 entrées) | 0,639 | 0,672 | 0,565 | 61 |

Ventilation par catégorie (recall@10) — **strictement identique avant/après** :

| run | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|
| avant | 1,0 | 0,935 | 0,403 |
| après | 1,0 | 0,935 | 0,403 |

Bootstrap apparié (`n_boot=10000`, `seed=42`) :

| comparaison | delta | IC95 | p_amelioration | pire perte par catégorie | adopté ? |
|---|---|---|---|---|---|
| avant vs après (lot de 3 entrées) | 0,0000 | (0,0000 ; 0,0000) | 0,000 | 0,000 (aucune perte, mais aucun gain non plus) | **non** |

Contrôle direct : les dicts `par_question` `avant` et `après` sont **identiques bit à bit** (`avant == après` → `True` en Python) — pas une seule des 61 questions ne change de score, y compris les 5 questions ciblées par le lot (q059, q060, q080, q086, q089, toutes à `recall@10=0.0` avant **et** après).

**Cause racine** — contrôle reproduit avec `_bm25()` **sans limite** (`limit` = nombre total de lignes de `chunks_norm`, soit 2160, pour ne tronquer aucun candidat) et le lot rejeté appliqué **en mémoire** (monkeypatch in-place de `accounting_rag.normalize.SYNONYMES`, sans toucher au fichier ni à l'index), sur l'état actuel de `data/corpus.db` — script `controle_rang_q060.py` (scratchpad ; sortie persistée dans `docs/mesures/jalon25/controle_rang_q060.json`) :

| | rang du gold `pcg-1222-74` (1-indexé) | candidats bm25 (records distincts) |
|---|---|---|
| sans le lot (9 entrées d'origine) | 1453 | 1659 |
| avec le lot (3 entrées, en mémoire) | 210 | 1659 |

Le token `subvent` (issu de « aides »→« subventions ») est bien présent dans la requête normalisée (vérifié) et fait remonter le gold de ~1200 rangs (1453→210) — un effet réel, mais qui reste très en-dessous de la fenêtre `limit=50` transmise par défaut à la fusion RRF dans le pipeline `hybrid`, ce qui explique l'absence totale d'effet observée sur `recall@10`. `subvent`/`entit` restent des tokens à pouvoir discriminant insuffisant à ce niveau de correction (« entité » apparaît dans la quasi-totalité des 1660 records ; « subvention » dans plusieurs dizaines de records liés à des comptes différents) : le gain de rang est réel mais d'un ordre de grandeur (~10²-10³) trop faible pour atteindre la fenêtre effectivement utilisée par `hybrid`. C'est un problème structurel de discriminance lexicale (IDF), pas un problème de couverture de vocabulaire : ajouter le bon token gold ne suffit pas s'il est déjà quasi-omniprésent dans le corpus et que l'écart de rang initial se compte en centaines.

*(Note de méthode : une première version de cette analyse, avant relecture, avait rapporté un contrôle avec `limit=2000` plutôt que « sans limite » et sans isoler proprement l'effet du lot en mémoire, ce qui avait produit des chiffres différents et non reproductibles — rang 244/1511. La méthode et les chiffres ci-dessus sont ceux reproduits et validés par la revue.)*

Dicts `par_question` bruts persistés (identiques avant/après — ruling J25-6, recalculables sans re-runner) :

```json
{
 "q001": 1.0, "q002": 1.0, "q003": 1.0, "q004": 1.0, "q006": 1.0, "q007": 1.0,
 "q008": 0.5, "q009": 1.0, "q010": 1.0, "q011": 1.0, "q012": 1.0, "q013": 1.0,
 "q014": 1.0, "q015": 1.0, "q021": 0.0, "q022": 0.5, "q023": 0.0, "q024": 1.0,
 "q025": 1.0, "q026": 0.0, "q027": 1.0, "q031": 1.0, "q032": 1.0, "q034": 1.0,
 "q036": 1.0, "q038": 1.0, "q040": 1.0, "q041": 1.0, "q043": 1.0, "q044": 1.0,
 "q046": 1.0, "q047": 1.0, "q049": 1.0, "q050": 1.0, "q052": 1.0, "q053": 1.0,
 "q054": 0.0, "q056": 0.0, "q057": 0.0, "q059": 0.0, "q060": 0.0, "q061": 1.0,
 "q063": 0.0, "q064": 1.0, "q065": 0.0, "q067": 1.0, "q068": 0.0, "q070": 0.0,
 "q071": 0.0, "q072": 1.0, "q074": 0.0, "q075": 1.0, "q077": 1.0, "q079": 0.0,
 "q080": 0.0, "q082": 1.0, "q083": 1.0, "q085": 0.0, "q086": 0.0, "q088": 1.0,
 "q089": 0.0
}
```

### Décision

**REJETÉ globalement.** `p_amelioration=0,000`, très loin du seuil de 0,95 — le lot de 3 entrées n'a d'effet mesurable sur **aucune** des 61 questions du split dev. Conformément à la règle du contrôleur pour ce cas (rejet global, pas de régression par catégorie mais pas de gain non plus), les 3 entrées ont été **intégralement retirées** de `SYNONYMES` (retour exact aux 9 entrées héritées du jalon 2/T1-T4), les tests dédiés retirés de `tests/test_normalize.py`, et **l'index a été re-rebuilti** une seconde fois pour restaurer l'état antérieur (`records=1660`, `renvois=981`, `chunks=2160` — vérifiés inchangés après les deux rebuilds). `data/corpus.db` est donc, à l'issue de cette tâche, fonctionnellement identique à son état d'avant T5.

Motivation détaillée : contrairement aux ablations A et B (T3/T4), où le rejet reflétait un compromis (gain insuffisant, coût trop élevé, régression), ici le rejet est **inertie totale au niveau du résultat final** (`recall@10`) — le mécanisme de synonymes fonctionne correctement au niveau lexical (les tokens `subvent`/`entit` apparaissent bien dans les requêtes normalisées, vérifié) et produit même un gain de rang bm25 réel et non négligeable (~1200 rangs sur le cas contrôlé q060, cf. ci-dessus), mais ce gain reste d'un ordre de grandeur insuffisant pour franchir la fenêtre `limit=50` transmise à la fusion RRF dans le pipeline `hybrid` : les tokens ajoutés sont soit trop génériques (`entit`, quasi-omniprésent) soit insuffisants à eux seuls pour combler un écart de rang qui se compte en centaines même après correction. Ce résultat négatif est en soi une information utile pour le jalon 3 : **le canal lexical (bm25 + synonymes) a atteint ses limites structurelles sur ce sous-ensemble de `vocabulaire_courant`** ; améliorer le rang de ces documents demandera soit une expansion de requête plus agressive (hors périmètre "dictionnaire de synonymes"), soit un mécanisme qui n'opère pas uniquement par ajout de tokens bm25 (reranking sémantique — déjà mesuré et adopté en T4 avec un gain réel sur `vocabulaire_courant`, 0,403→0,484 avec `bge-reranker-v2-m3` ; ou une révision du score bm25/IDF).

### Échecs restants (matériau jalon 3)

Les 21 échecs dev identifiés en début de section restent **entièrement non résolus** (le lot rejeté n'a rien changé). Catégorisation pour le jalon 3 :

- **Nécessitent une compréhension de requête au-delà du lexique** (10 groupes / 14 questions) : q021, q023, q026, q056, q057, q063, q065, q068, q070/071/074 (regroupées, même record gold sous-jacent sur la conversion de devises), q079/080/089 (regroupées, famille boni/mali de fusion) — la question décrit un scénario métier sans jamais nommer le terme PCG correspondant ; aucun dictionnaire de synonymes phrase-à-phrase ne peut combler cet écart sans dégénérer en paraphrase générale (hors périmètre, risque de sur-ajustement démontré par le rejet ci-dessus).
- **Nécessitent une distinction comptable fine dans la même question** (2) : q022 (créance douteuse vs irrécouvrable), q085 (immobilisation en cours vs production immobilisée) — un rapprochement lexical générique risquerait ici une erreur de conflation (ruling J2-5) ; ces cas demandent un raisonnement contextuel, pas un lexique.
- **Ne sont pas un fossé lexical mais un problème de rang dans la fusion** (2) : q008, q054 — le vocabulaire est déjà quasi identique entre la question et le record gold ; le reranker cross-encoder (T4, adopté) est le mécanisme déjà mesuré qui adresse ce type de problème (il opère après la fusion RRF, sur le contenu sémantique complet, pas sur des tokens bm25).
- **q059, q060, q080, q086, q089** : le lot rejeté ciblait ces questions ; elles restent à `recall@10` inchangé (0,0 pour toutes). La cause racine (tokens ajoutés trop peu discriminants pour ce corpus à `limit=50`) suggère qu'une future tentative devrait soit augmenter `limit` dans `_bm25()` (hors périmètre SYNONYMES de cette tâche), soit s'appuyer sur le reranker plutôt que sur bm25 seul pour ce sous-ensemble.

### Réserves

1. Le rejet porte sur le split `dev` (61 questions) — `test` (29 questions) n'a pas été et ne doit pas être utilisé pendant le développement (règle constante du jalon 2.5).
2. Les deux mesures (avant/après) ont été exécutées dans des processus python séparés (avant : `normalize.py` remis temporairement à l'état HEAD via `git checkout`, corpus non reconstruit ; après : `normalize.py` édité + corpus reconstruit) — l'alignement des ids `par_question` est garanti par construction (mêmes questions du même fichier benchmark), pas par un partage d'objet en mémoire comme en T3/T4 ; le résultat (`avant == après` bit à bit) constitue en lui-même une preuve solide qu'aucune dérive d'alignement ne s'est produite (un décalage aurait presque certainement produit des différences).
3. `_bm25()` utilise `limit=50` par défaut (`Searcher._bm25(self, query, limit=50)`) — ce paramètre n'a pas été modifié (hors périmètre de cette tâche, restreint à `SYNONYMES`) ; c'est pourtant la cause racine identifiée de l'inertie du lot. Une tâche future pourrait mesurer l'effet d'un `limit` plus large, indépendamment de tout ajout de synonyme.
4. Le lot de candidats a été volontairement restreint à 3 entrées sur un maximum de 10 autorisé par le brief — la plupart des 21 échecs analysés ne présentaient aucune paire terme-à-terme légitime (ruling J2-5), et forcer le compte à 10 aurait signifié accepter des rapprochements plus risqués (paraphrases de clauses, catégories de devises) explicitement écartés par prudence ci-dessus.
5. `docs/echecs-dev-jalon25.md` documente l'état des échecs **avant** le lot (matériau de la proposition) — comme le lot a été intégralement rejeté et retiré, cet état est aussi l'état **final** du split dev à l'issue de T5 ; aucune mise à jour de ce fichier n'était nécessaire après la mesure.

## Clôture — dev final et référence test gelée (T6)

Campagne exécutée le 16 août 2026, dernière tâche du jalon 2.5. **Configuration finale** évaluée (celle livrée par ce jalon) :

- Mode `Searcher.search(mode="hybrid+rerank")` — reranker `BAAI/bge-reranker-v2-m3` (défaut de code, adopté en Ablation B / T4).
- `Searcher` à paramètres neutres : `poids_chemin=1.0`, `boost_commentaire=1.0` (Ablation A / T3, **rejetée** — les deux paramètres restent à leur valeur neutre par défaut).
- `SYNONYMES` d'origine, 9 entrées héritées du jalon 2 (Ablation C / T5, lot de 3 entrées candidates **rejeté** et retiré, index restauré à l'identique).

Comparée systématiquement à la **baseline hybrid pure** (même `Searcher`, mode `hybrid`, mêmes paramètres neutres, mêmes synonymes) — c'est la config recommandée pour l'usage interactif (latence ~0,2 s/question), `hybrid+rerank` restant réservé aux campagnes batch ou au re-classement asynchrone (~130 s/question mesurés ici, cf. tableau ci-dessous).

### Étape 1 — suite de tests complète

```
uv run pytest -q
```

→ **117 passed**, 3 warnings (avertissements CUDA/FutureWarning déjà connus, sans conséquence), en **167,56 s**. Aucune régression.

### Étape 2 — campagne dev finale (61 questions)

**Contrôle de fraîcheur préalable** (avant tout engagement sur la mesure coûteuse) : les dicts `par_question` bruts de la mesure T4 (`hybrid+rerank` sur dev) n'avaient **pas** été persistés sur disque — seuls les agrégats l'ont été, dans `resultat_BAAI_bge-reranker-v2-m3.json` (scratchpad, non versionné). La persistance systématique des dicts `par_question` bruts (ruling J25-6) n'a été instaurée qu'à partir de T5, après la mesure T4. Or le bootstrap par catégorie demandé à cette étape nécessite ces dicts bruts filtrés par catégorie : ils ont donc été **régénérés** dans un script unique (`scripts/mesure_cloture.py`, versionné) qui exécute, dans le même processus (embedder + reranker partagés, alignement des ids garanti comme en T3/T4) :

1. Un contrôle rapide — `hybrid` seul sur dev (13,3 s) — dont le `recall@10` est comparé à la référence T2/T3/T4 (0,672). **Résultat : 0,672, identique** (mrr=0,565 également identique) → l'index (`chunks=2160`, `records=1660`, `renvois=981`) et le code sont dans le même état qu'aux mesures précédentes ; la campagne complète peut procéder en confiance sur la même référence A que T3/T4/Ablation C.
2. La campagne complète dev (baseline hybrid + config finale hybrid+rerank), avec persistance des dicts `par_question` bruts en JSON dans `docs/mesures/jalon25/cloture_dev.json` (ruling J25-6).

| run | config | recall@5 | recall@10 | MRR | latence/question |
|---|---|---|---|---|---|
| baseline | hybrid, neutre | 0,639 | 0,672 | 0,565 | 0,22 s |
| finale | hybrid+rerank (bge-reranker-v2-m3) | 0,680 | 0,738 | 0,642 | 129,5 s |

Ventilation par catégorie (recall@10) :

| run | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|
| baseline | 1,0 | 0,935 | 0,403 |
| finale | 1,0 | 1,0 | 0,484 |

Bootstrap apparié global (`n_boot=10000`, `seed=42`) :

| comparaison | delta | IC95 | p_amelioration |
|---|---|---|---|
| baseline vs finale (dev) | 0,0656 | (-0,0082 ; 0,1393) | 0,9524 |

Ces chiffres sont **strictement identiques** à ceux mesurés en T4 (`resultat_BAAI_bge-reranker-v2-m3.json` : recall@5=0,68, recall@10=0,738, mrr=0,642, delta=0,0656, ic95=(-0,0082 ; 0,1393), p_amelioration=0,9524) — reproduction exacte, aucune dérive entre les deux mesures malgré la reconstruction complète des dicts `par_question`.

Bootstrap apparié par catégorie (mêmes `n_boot`/`seed`, sous-ensemble des ids par catégorie) :

| catégorie | n | delta | IC95 | p_amelioration |
|---|---|---|---|---|
| reference_directe | 7 | 0,0 | (0,0 ; 0,0) | 0,0 |
| regle | 23 | 0,0652 | (0,0 ; 0,1739) | 0,8758 |
| vocabulaire_courant | 31 | 0,0806 | (-0,0484 ; 0,2097) | 0,8642 |

**Point notable** : le critère d'adoption formel (`p_amelioration ≥ 0,95`) n'est franchi par **aucune catégorie prise isolément** (`regle` 0,8758, `vocabulaire_courant` 0,8642 ; `reference_directe` reste à un plancher de 0 par effet de plafond, les deux runs valant déjà 1,0 sur les 7 questions) — seul le pool complet (n=61) franchit le seuil (0,9524). C'est un effet de puissance statistique attendu (n par catégorie 3 à 4 fois plus petit que n global), pas une contradiction : l'adoption T4 a toujours été jugée sur le critère global, avec la garde additionnelle « aucune catégorie ne perd de recall@10 » (vérifiée : les trois catégories sont à égalité ou en progrès).

### Étape 3 — référence test gelée (29 questions), une seule exécution

**Ce split ne sert à aucun choix.** Il est exécuté une unique fois, dans le même run que l'étape 2 (`mesure_cloture.py`), immédiatement après la campagne dev, sur `benchmark/test.jsonl` (29 questions : 3 `reference_directe`, 12 `regle`, 14 `vocabulaire_courant`), gelé depuis le 2026-08-16 (voir `benchmark/README.md`). Il est présenté ici comme **référence gelée jalon 2.5**, pas comme un signal d'ajustement — aucune décision de ce jalon ne s'appuie sur ces chiffres.

| run | config | recall@5 | recall@10 | MRR | latence/question |
|---|---|---|---|---|---|
| baseline | hybrid, neutre | 0,621 | 0,690 | 0,470 | 0,20 s |
| finale | hybrid+rerank (bge-reranker-v2-m3) | 0,707 | 0,759 | 0,626 | 120,6 s |

Ventilation par catégorie (recall@10) :

| run | reference_directe (n=3) | regle (n=12) | vocabulaire_courant (n=14) |
|---|---|---|---|
| baseline | 1,0 | 1,0 | 0,357 |
| finale | 1,0 | 1,0 | 0,5 |

Bootstrap apparié global (test) :

| comparaison | delta | IC95 | p_amelioration |
|---|---|---|---|
| baseline vs finale (test) | 0,069 | (0,0 ; 0,1724) | 0,8773 |

Bootstrap apparié par catégorie (test) :

| catégorie | n | delta | IC95 | p_amelioration |
|---|---|---|---|---|
| reference_directe | 3 | 0,0 | (0,0 ; 0,0) | 0,0 |
| regle | 12 | 0,0 | (0,0 ; 0,0) | 0,0 |
| vocabulaire_courant | 14 | 0,1429 | (0,0 ; 0,3571) | 0,8884 |

Dicts `par_question` bruts persistés (dev et test, baseline et finale) dans `docs/mesures/jalon25/` : `cloture_dev.json`, `cloture_test.json` (ruling J25-6).

Pour recalculer un bootstrap depuis ces JSON versionnés (ex. baseline vs finale sur dev) :

```sh
uv run python -c "import json; from accounting_rag.evalrag import paired_bootstrap as pb; d = json.load(open('docs/mesures/jalon25/cloture_dev.json')); print(pb(d['a']['par_question'], d['b']['par_question']))"
```

Résultat attendu : `{'delta': 0.0656, 'ic95': (-0.0082, 0.1393), 'p_amelioration': 0.9524}` — identique à la table « Bootstrap apparié global » ci-dessus.

### Lecture

- **Le gain de recall@10 réplique, voire s'accentue légèrement, sur test** : delta=0,069 (test) vs 0,0656 (dev) — le point d'estimation ne s'effondre pas hors dev, ce qui n'est pas le profil typique d'un sur-ajustement classique (un paramètre réglé sur le bruit du dev perd généralement de la valeur sur un split indépendant). Cohérent avec le fait que le reranker n'a reçu **aucun réglage** dérivé du dev (pas de seuil, pas d'hyperparamètre appris) — seul le **choix discret** du modèle (`bge-reranker-v2-m3` retenu contre `mmarco-mMiniLMv2-L12-H384-v1` rejeté, T4) a été décidé sur dev, ce qui reste une forme résiduelle et limitée de conditionnement au dev à garder en tête.
- **Le `p_amelioration` global recule pourtant sur test** (0,8773 contre 0,9524 sur dev), sous le seuil d'adoption de 0,95. Lecture privilégiée : un effet de **taille d'échantillon**, pas de sur-ajustement — n=29 (test) contre n=61 (dev) réduit mécaniquement la puissance du bootstrap apparié pour un delta de magnitude comparable (voire supérieure) ; l'IC95 du test ((0,0 ; 0,1724)) est plus large en proportion et sa borne basse touche exactement 0. Si ce test avait dû arbitrer l'adoption (il ne l'a pas fait, cf. règle du jalon 2.5), le mode `hybrid+rerank` n'aurait pas franchi le seuil formel sur ce split seul — signal de fragilité statistique déjà noté en réserve de l'Ablation B (T4, réserve 1 : « réussite marginale du seuil »), maintenant partiellement corroboré par une deuxième mesure indépendante.
- **`regle` est déjà à 1,0 en baseline sur test** (n=12, contre 0,935 sur dev, n=23) : la population test de cette catégorie ne laisse aucune marge de progression au reranker (`delta=0`, effet de plafond, comme `reference_directe` dans les deux splits) — c'est une différence de composition entre les tirages dev/test de cette catégorie, pas un signe que le reranker serait inefficace sur `regle` en général (il l'améliore nettement sur dev, 0,935→1,0).
- **`vocabulaire_courant` confirme le bénéfice du reranker sur les deux splits** : dev 0,403→0,484 (delta catégorie 0,0806, p=0,8642), test 0,357→0,5 (delta catégorie 0,1429, p=0,8884) — la catégorie la plus difficile pour bm25/dense (fossé lexical, cf. Ablation C) reste celle qui bénéficie le plus du reranker sémantique, sur dev **et** sur test, ce qui est le signal le plus rassurant de cette clôture : le mécanisme (reclassement sémantique après fusion, indépendant des tokens bm25) répond bien au diagnostic structurel documenté en Ablation C, sur une population qu'il n'a jamais vue.
- **La baseline hybrid elle-même diffère entre dev et test** (recall@10 0,672 vs 0,690, mrr 0,565 vs 0,470) sans qu'aucun réglage n'ait jamais touché ce mode pendant le jalon (paramètres neutres constants) : une partie de l'écart dev/test observé sur la config finale est donc de la variance d'échantillonnage inhérente à des populations de 61 et 29 questions tirées une fois, pas uniquement un effet du reranker ou de son choix de modèle.

### Réserves

1. Les dicts `par_question` bruts de `hybrid+rerank` sur dev n'existaient pas avant cette tâche (T4 n'avait persisté que les agrégats, ruling J25-6 instauré après T4) — ils ont été régénérés ici avec un contrôle de fraîcheur explicite (recall@10 et mrr baseline strictement identiques à T2/T3/T4) avant d'engager la mesure coûteuse ; la reproduction à l'identique des agrégats B face à T4 est une preuve supplémentaire d'absence de dérive entre les deux mesures (index, code, dépendances).
2. La latence `hybrid+rerank` mesurée ici (129,5 s/question sur dev, 120,6 s/question sur test) diffère légèrement de celle mesurée en T4 (117,1 s/question) — variation attribuée à la charge de la machine au moment de la mesure (CPU partagé, pas de contrôle d'isolation), sans conséquence sur les conclusions (ordre de grandeur identique, ~600× le baseline).
3. `test` (n=29) est une population presque deux fois plus petite que `dev` (n=61) : tout écart de quelques points de recall@10 doit être lu avec prudence, en particulier par catégorie (`reference_directe`, n=3 seulement sur test ; `regle`, n=12).
4. `test` n'a été exécuté qu'une seule fois, jamais pendant le développement des tâches T1-T5 — conformément à la règle constante du jalon 2.5 ; ces chiffres ne doivent pas être réutilisés pour un nouvel ajustement, seulement cités comme référence gelée du jalon 2.5 dans les jalons suivants.
5. Aucune comparaison n'est faite ici avec le benchmark v1 du jalon 2 (n=21 dev / n=9 test) : instruments différents, non comparables (cf. introduction de ce document).
