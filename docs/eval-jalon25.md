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
