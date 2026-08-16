# Jalon 3 — Recouvrement lexical nul : diagnostic fondateur et ablation D (T1)

Campagne exécutée le **16 août 2026** sur la branche `jalon-3-recouvrement-nul`, à l'issue de la tâche T1 (filtrage des tokens peu discriminants de la requête lexicale, `df_max`, plus les deux outils de mesure versionnés du jalon : `scripts/diagnostic_rangs.py`, `scripts/ablations_jalon3.py`). Objectif : reproduire par un outil versionné le diagnostic fondateur du jalon (deux sondes jetables du contrôleur, exécutées avant la rédaction du plan) puis mesurer par bootstrap apparié si le filtrage par fréquence documentaire (`df_max`) répare, seul, une partie du recouvrement lexical nul observé sur `vocabulaire_courant`.

## Conditions exactes

| Paramètre | Valeur |
|---|---|
| Corpus | `data/corpus.db`, 1 660 `records` (739 réglementaires, 921 commentaires ANC), 981 `renvois` — inchangé depuis le jalon 2.5 (vérifié, cf. § Réserves) |
| Index de recherche | 2 160 chunks, table FTS5 `chunks_norm` (texte normalisé), table vectorielle `chunks_vec` (sqlite-vec 0.1.9) — **aucun rebuild** dans ce jalon (le filtrage `df_max` agit à la requête, pas à l'ingestion) |
| Modèle d'embeddings | `intfloat/multilingual-e5-small` (défaut de `Embedder`, 384 dimensions) |
| Fusion hybride | RRF (`k=60`) sur BM25 + dense, paramètres neutres (`poids_chemin=1.0`, `boost_commentaire=1.0` — rejetés au jalon 2.5, conservés neutres) |
| Benchmark | `benchmark/dev.jsonl` v2 — 61 questions (7 `reference_directe`, 23 `regle`, 31 `vocabulaire_courant`) |
| Machine | Linux 6.19.8-arch1-3-surface, x86_64, CPU uniquement (pas de GPU exploité) |
| Environnement | Python 3.13.14, `sentence-transformers` 5.7.0, `torch` 2.13.0+cu130 (CPU), `sqlite-vec` 0.1.9 |

## Diagnostic fondateur

Deux sondes jetables exécutées par le contrôleur avant la rédaction du plan de ce jalon (`sonde_discriminance.py`, `sonde_dense.py`, non versionnées) ont établi que le gold des questions dures de `vocabulaire_courant` se classe **systématiquement dernier** de sa liste de candidats lexicaux — retrouvé uniquement par des mots fonctionnels (aucun token de contenu partagé avec le record gold) — tandis que le canal dense le place dans une fenêtre atteignable (178-528 sur 1660) mais hors de la fenêtre `limit=50` utilisée par `hybrid` en production.

`scripts/diagnostic_rangs.py` reproduit ce diagnostic à l'identique, sans reconstruire l'index, en calculant le rang du gold dans TROIS classements complets (sans troncature `LIMIT`) : lexical non filtré, lexical filtré à un `df_max` donné, dense.

Reproduction (`uv run python scripts/diagnostic_rangs.py --questions q021,q026,q060,q023`) :

| question | catégorie | gold | lexical (non filtré) | lexical (df≤2%) | dense |
|---|---|---|---|---|---|
| q021 | vocabulaire_courant | pcg-214-13 | 154/154 | ABSENT/38 | 178/1660 |
| q026 | vocabulaire_courant | pcg-212-3 | 46/46 | **14/14** | 248/1660 |
| q060 | vocabulaire_courant | pcg-1222-74 | 1430/1430 | ABSENT/63 | 528/1660 |
| q023 | vocabulaire_courant | pcg-214-22 | 2/2 | ABSENT/52 | 257/1660 |

**Contrôle de l'outil (fidélité à la sonde fondatrice) — reproduction exacte, aucun écart** :
- q021 : lexical **154/154** (attendu 154/154 ✓), dense **178**/1660 (attendu 178 ✓).
- q026 : lexical **46/46** (attendu 46/46 ✓), filtré df≤2% **14/14** (attendu 14/14 ✓), dense **248** (attendu 248 ✓).
- q060 : lexical **1430/1430** (attendu 1430/1430 ✓), dense **528** (attendu 528 ✓).
- q023 : dense **257** (attendu 257 ✓) — aucun chiffre lexical fondateur n'existait pour q023 (la sonde `sonde_discriminance.py` n'exécutait que q021/q026/q060) ; le rang lexical non filtré de q023 (2/2) est une donnée nouvelle, produite par l'outil versionné, pas une reproduction.

**Note de convention (documentée dans le code, `_rang_gold_lexical`/`_rang_gold_dense`)** : les deux sondes fondatrices avaient des conventions DIFFÉRENTES pour le second nombre de la paire "rang/total" — `sonde_discriminance.py` s'arrêtait dès le gold trouvé, si bien que ce second nombre est TOUJOURS égal au rang lui-même quand le gold est trouvé (ex. "154/154" ne signifie pas "154 candidats au total", mais "le gold est le 154e et dernier record distinct scanné avant l'arrêt") ; `sonde_dense.py` ne s'arrêtait pas et comptait la taille réelle du classement complet (1660, le corpus entier). `scripts/diagnostic_rangs.py` reproduit fidèlement les deux conventions, une par canal — sciemment conservées telles quelles pour que le contrôle ci-dessus soit un contrôle bit à bit, pas une réinterprétation.

**Lecture** : le filtrage lexical (`df_max`) ne repêche le gold que lorsqu'il partage au moins un token RARE avec la question (cas q026, rang 46→14, entre dans une fenêtre `limit` réaliste) ; quand aucun token de contenu n'est commun (q021, q060), le gold disparaît purement et simplement du pool filtré (`ABSENT`) — le filtrage ne peut pas inventer un signal qui n'existe pas. Le canal dense reste, dans tous les cas mesurés ici, le seul à placer le gold dans une fenêtre non triviale (178-528).

## Ablation D — filtrage par fréquence documentaire (`df_max`)

Nouveau paramètre `Searcher(..., df_max: float | None = None)` et méthodes `Searcher.df(token)` (fréquence documentaire, mise en cache par instance) et `Searcher._termes_match(query)` (tokens retenus pour le `MATCH` lexical, remplaçant la construction `" OR ".join(...)` directe de `_bm25`). `df_max=None` doit reproduire exactement le comportement jalon 2.5 (aucun filtrage) — voir § Défaut constaté ci-dessous pour un écart trouvé et corrigé à la mesure.

### Défaut constaté à la mesure (distinct de la ruling J3-1)

Le code de référence du brief déduplique inconditionnellement les tokens de `_termes_match` (`toks = list(dict.fromkeys(normalize(query).split()))`), y compris quand `df_max=None`. **`bm25()` de SQLite FTS5 est sensible à la MULTIPLICITÉ des termes dans l'expression `MATCH`, pas seulement à l'ensemble des lignes qu'elle sélectionne** : sur `data/corpus.db`, mode `hybrid`, split `dev`, dédupliquer même à `df_max=None` fait passer `recall@10` de 0,672 (baseline jalon 2.5) à **0,689** — un écart découvert au contrôle de non-régression du step 8 (49/61 questions dev contiennent au moins un token répété après `normalize()`, ex. « la »/« le » répétés dans une même question).

Ceci viole l'exigence explicite de l'interface (« `None` = comportement jalon 2.5, aucun filtrage ») sur le corpus réel — la fixture synthétique des tests unitaires ne l'a jamais détecté car ses 3 requêtes de contrôle (`test_df_max_neutre_par_defaut`) ne contiennent aucun token répété. **Corrigé** : `_termes_match` retourne la liste BRUTE (non dédupliquée) quand `df_max=None`, et ne déduplique qu'à l'intérieur du chemin de filtrage actif (nécessaire pour raisonner sur des tokens uniques face à `df_max`, mais sans jamais changer la construction du `MATCH` neutre). Contrôle après correction :

```
recall@10 (hybrid, dev, df_max=None) = 0,672 — identique à la référence jalon 2.5
```

### Méthode

`scripts/ablations_jalon3.py --ablation D --split dev` : un seul `Embedder` partagé entre les 4 configurations (`df_max ∈ {None, 0.10, 0.05, 0.02}`), mode `hybrid`, k=10, sur les 61 questions dev. Référence = `df_max=None`. Bootstrap apparié (`paired_bootstrap`, `n_boot=10000`, `seed=42`) de chaque configuration filtrée contre la référence. Critère d'adoption (contrainte globale du plan) : `p_amelioration ≥ 0,95` **et** aucune catégorie ne perd plus de 0,05 de recall@10.

### Résultats

Sortie brute de `uv run python scripts/ablations_jalon3.py --ablation D --split dev` (persistée intégralement, agrégats **et** `par_question` bruts des 4 configurations, dans `docs/mesures/jalon3/D_dev.json`) :

| config | recall@5 | recall@10 | MRR | latence/question |
|---|---|---|---|---|
| df_max=None (neutre, jalon 2.5) | 0,639 | **0,672** | 0,565 | 0,261 s |
| df_max=0,10 | 0,639 | 0,664 | 0,548 | 0,189 s |
| df_max=0,05 | 0,623 | 0,656 | 0,514 | 0,186 s |
| df_max=0,02 | 0,516 | 0,590 | 0,474 | 0,190 s |

Ventilation par catégorie (recall@10) :

| config | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|
| df_max=None | 1,0 | 0,935 | 0,403 |
| df_max=0,10 | 1,0 | 0,935 | 0,387 |
| df_max=0,05 | 1,0 | 0,935 | 0,371 |
| df_max=0,02 | 1,0 | 0,870 | 0,290 |

Bootstrap apparié (`n_boot=10000`, `seed=42`, contre la référence `df_max=None`) :

| comparaison | delta | IC95 | p_amelioration | pire perte par catégorie | adopté ? |
|---|---|---|---|---|---|
| df_max=0,10 | -0,0082 | (-0,0492 ; 0,0246) | 0,2643 | -0,0161 (vocabulaire_courant) | **non** |
| df_max=0,05 | -0,0164 | (-0,0492 ; 0,0000) | 0,0000 | -0,0323 (vocabulaire_courant) | **non** |
| df_max=0,02 | -0,0820 | (-0,1639 ; -0,0164) | 0,0030 | -0,1129 (vocabulaire_courant) | **non** |

### Décision

**REJETÉ — `df_max` reste à sa valeur neutre par défaut (`None`, aucun filtrage).** Les trois seuils mesurés dégradent `recall@10` de façon monotone et croissante avec l'agressivité du filtrage (0,672 → 0,664 → 0,656 → 0,590), `p_amelioration` restant très en-dessous du seuil de 0,95 dans les trois cas (0,2643 ; 0,0000 ; 0,0030) — et pour `df_max=0,10`/`0,05`/`0,02`, la perte sur `vocabulaire_courant` (-0,0161 ; -0,0323 ; -0,1129) dépasse déjà la garde de -0,05 dès `df_max=0,05`. Aucun seuil ne franchit ni la moitié du critère d'adoption.

Motivation, à la lumière du diagnostic fondateur (§ précédente) : le filtrage par fréquence documentaire ne peut REPÊCHER un gold que lorsqu'il partage déjà un token RARE avec la question — c'est exactement ce qui se passe pour q026 (rang 46→14, isolé au diagnostic), mais ce cas est rare dans le split dev complet. Pour la majorité des autres questions (`regle` et `vocabulaire_courant` confondus), retirer les tokens fréquents de la requête **retire aussi du signal utile** : un mot fonctionnel ou moyennement fréquent (au sens `df_max`) peut rester discriminant à l'intérieur de la fenêtre `limit=50` transmise à `_bm25()` en production — le supprimer ne fait alors que réduire le nombre de candidats retrouvés, sans jamais faire remonter un gold qui n'a de toute façon aucun token de contenu partagé (q021, q060, cf. diagnostic). Le mécanisme fonctionne exactement comme prévu (vérifié par les tests unitaires sur la fixture synthétique, `test_df_max_ecarte_les_tokens_trop_frequents`), mais son effet net sur le split dev réel est négatif : il gagne sur une poignée de questions à recouvrement lexical partiel (comme q026) et perd sur bien davantage de questions où les tokens filtrés portaient encore un signal exploitable dans la fenêtre `limit=50`.

Ce résultat négatif est informatif pour la suite du jalon : le filtrage de tokens, seul, n'est pas le levier qui comble le recouvrement lexical nul — cohérent avec le diagnostic fondateur (q021/q060 restent `ABSENT` du pool filtré à tout seuil testé, faute de tout token de contenu commun). Les leviers restants (élargissement du pool de candidats avant fusion, ablation E ; reranking sur pool élargi, ablation F ; réécriture de requête par LLM, ablation G) sont hors périmètre de cette tâche T1.

### Réserves

- Le diagnostic fondateur (§ précédente) montre que `df_max` ne peut repêcher que les questions partageant déjà un token rare avec leur gold (cas q026) — les cas de recouvrement lexical STRICTEMENT nul (q021, q060, et vraisemblablement leurs semblables) restent hors d'atteinte de ce mécanisme par construction, quel que soit le seuil choisi. Ce n'est pas une limite de l'implémentation mais une limite structurelle du signal disponible sans réécriture de requête ni rebuild — matière des tâches suivantes du jalon (ablations E/F, pool + reranking large ; ablation G, réécriture LLM).
- Le défaut de dédup constaté ci-dessus n'a été détecté qu'au contrôle de non-régression sur le corpus RÉEL (step 8) — la fixture synthétique des tests unitaires ne l'exerçait pas. Aucune garantie que d'autres écarts entre comportement synthétique et réel n'existent ailleurs dans ce jalon ; seul le contrôle explicite prévu par le brief (recall@10=0,672 attendu) l'a fait apparaître.
- `n=61` (dev) : tout écart d'une à deux questions déplace visiblement le recall par catégorie — pas de test de significativité par catégorie ici (seulement au global, via `paired_bootstrap`).

## Reproduction exacte

```sh
uv run pytest tests/test_search.py -q                     # tests unitaires df_max/df/_termes_match
uv run python scripts/diagnostic_rangs.py --questions q021,q026,q060,q023   # § Diagnostic fondateur
uv run python scripts/ablations_jalon3.py --ablation D --split dev          # § Ablation D
```

`docs/mesures/jalon3/diagnostic_rangs.json` et `docs/mesures/jalon3/D_dev.json` sont versionnés — tout chiffre publié ci-dessus est recalculable sans re-runner ces commandes (les `par_question` bruts des 4 configurations de l'ablation D sont dans `D_dev.json`, champ `configs[i].par_question`).
