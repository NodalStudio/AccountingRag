# Jalon 3 — Recouvrement lexical nul : diagnostic fondateur, ablations D, E, F, G et clôture

Campagne exécutée le **16 août 2026** sur la branche `jalon-3-recouvrement-nul`, à l'issue de la tâche T1 (filtrage des tokens peu discriminants de la requête lexicale, `df_max`, plus les deux outils de mesure versionnés du jalon : `scripts/diagnostic_rangs.py`, `scripts/ablations_jalon3.py`) puis de la tâche T2 (largeur du pool de candidats avant fusion, `pool`, et déduplication des termes du `MATCH`, `dedup_termes`). Objectif T1 : reproduire par un outil versionné le diagnostic fondateur du jalon (deux sondes jetables du contrôleur, exécutées avant la rédaction du plan) puis mesurer par bootstrap apparié si le filtrage par fréquence documentaire (`df_max`) répare, seul, une partie du recouvrement lexical nul observé sur `vocabulaire_courant`. Objectif T2 : mesurer si élargir le pool de candidats avant fusion RRF (ou dédupliquer les termes du `MATCH`) répare une partie de ce recouvrement nul, et — livrable central de T2 — établir la **couverture du pool** (part des questions dont au moins une citation gold est présente dans le pool avant fusion) à chaque largeur, la métrique qui conditionne l'ablation F (reranking). Objectif T3 : mesurer si un cross-encoder sait exploiter cette couverture (`n_rerank`). Objectif T5 : mesurer la réécriture de la question par un LLM (ablation G) — le seul levier du jalon capable de créer un lien de vocabulaire qui n'existe pas. Objectif T4 : la clôture, dont l'unique exécution du split de test gelé.

## Conditions exactes

| Paramètre | Valeur |
|---|---|
| Corpus | `data/corpus.db`, 1 660 `records` (739 réglementaires, 921 commentaires ANC), 981 `renvois` — inchangé depuis le jalon 2.5 (vérifié, cf. § Réserves) |
| Index de recherche | 2 160 chunks, table FTS5 `chunks_norm` (texte normalisé), table vectorielle `chunks_vec` (sqlite-vec 0.1.9) — **aucun rebuild** dans ce jalon (le filtrage `df_max` agit à la requête, pas à l'ingestion) |
| Modèle d'embeddings | `intfloat/multilingual-e5-small` (défaut de `Embedder`, 384 dimensions) |
| Fusion hybride | RRF (`k=60`) sur BM25 + dense, paramètres neutres (`poids_chemin=1.0`, `boost_commentaire=1.0` — rejetés au jalon 2.5, conservés neutres) |
| Convention de rang | **1-indexée** partout dans ce rapport (le meilleur candidat d'un canal est au rang 1), comme `scripts/diagnostic_rangs.py` ; les scores RRF cités valent donc `1/(60 + rang)` |
| Benchmark | `benchmark/dev.jsonl` v2 — 61 questions (7 `reference_directe`, 23 `regle`, 31 `vocabulaire_courant`) |
| Machine | Linux 6.19.8-arch1-3-surface, x86_64, 8 cœurs (`torch.get_num_threads() == 4`), **GPU Quadro RTX 3000 Max-Q, 6 Go de VRAM** |
| Device des modèles | `cuda:0` pour `Embedder` ET `Reranker` (auto-détection de `sentence-transformers` ; ce projet ne passe jamais `device` explicitement), dtype fp32 — vérifié en T3, et déterminant pour toute latence de ce rapport (cf. § ci-dessous) |
| Environnement | Python 3.13.14, `sentence-transformers` 5.7.0, `torch` 2.13.0+cu130 (build CUDA, GPU effectivement utilisé — cf. ligne « Device des modèles »), `sqlite-vec` 0.1.9 |

### Le reranking du jalon 2.5 était mesuré sur CPU : correction d'un facteur ~70

La configuration adoptée au jalon 2.5 (`hybrid+rerank`, `bge-reranker-v2-m3`, 25 candidats) y était publiée à **129,5 s/question** (7 899 s pour 61 questions). Re-mesurée à l'identique en T3, elle coûte **1,8 s/question** — facteur ~70. Le code n'a pas changé d'une ligne (`src/accounting_rag/rerank.py`, commit d'origine `7bce96a` ; `_MAX_CHARS=1000` y figure dès l'origine) et le témoin re-mesuré reproduit les scores **question par question**, pas seulement l'agrégat (`temoin_reproduction` dans `docs/mesures/jalon3/F_dev.json`).

Cause établie par contrôle direct — la même configuration, sur une question, le modèle déplacé explicitement d'un device à l'autre :

| device du cross-encoder | latence (bge, 25 candidats, 1 question) |
|---|---|
| `cuda:0` (Quadro RTX 3000 Max-Q, 6 Go) | **2,29 s** |
| `cpu` (8 cœurs, `torch.get_num_threads() == 4`) | **178,1 s** |

Ratio **×77,8** (`docs/mesures/jalon3/sondes.json`, produit par `scripts/sondes_jalon3.py`). Une version antérieure de cette section publiait 2,1 s / 149,5 s / ×71 : c'était la même sonde, exécutée sous une charge machine différente, avant d'être versionnée. L'écart entre les deux exécutions (×71 contre ×77,8, soit ~10 % sur le ratio et ~19 % sur la valeur CPU) illustre le point même de cette section — une latence à une seule question sur une machine partagée n'est pas un chiffre stable. **C'est l'ordre de grandeur du ratio qui est le résultat, pas ses décimales** : le reranking coûte deux ordres de grandeur de plus sur CPU que sur GPU.

178 s encadre le chiffre publié au jalon 2.5 : **la campagne du jalon 2.5 exécutait le cross-encoder sur CPU**, alors que la machine disposait déjà du GPU (même `uv.lock`, même `torch` 2.13.0 avec les wheels CUDA — vérifié par `git show`). En régime nominal, `Embedder` comme `Reranker` s'initialisent sur `cuda:0` (auto-détection de `sentence-transformers`, aucun `device` passé explicitement dans ce projet, vérifié en T3). **Ce qui a privé la campagne du jalon 2.5 du GPU n'est pas établi** — hypothèse la plus probable : plusieurs sous-agents travaillaient en parallèle et se disputaient les 6 Go de VRAM. Faute de preuve, c'est une hypothèse et pas une conclusion.

Trois conséquences assumées :

1. **Aucune latence absolue de ce projet n'est une caractéristique du système mesuré** — c'est une caractéristique du couple (machine, device, charge). Seuls les RAPPORTS entre configurations mesurées dans la MÊME campagne sont interprétables. Le device et le dtype rejoignent donc le tableau « Conditions exactes » de tout rapport à venir.
2. Les latences du rapport du jalon 2.5 restent telles quelles dans leur propre document — elles ont bien été mesurées, dans leurs conditions — mais ne sont **pas comparables** à celles de ce rapport-ci.
3. La décision de conception prise au jalon 2.5 de garder `hybrid+rerank` hors de la campagne par défaut (« 2 h de calcul surprise pour un nouveau contributeur ») reposait sur ce chiffre. Elle reste **juste pour un contributeur sans GPU** (149,5 s/question × 61 ≈ 2 h 30, le scénario du jalon 2.5) et devient discutable dès qu'une carte est disponible (~2 min). Réexaminée à la clôture de ce jalon : c'est la disponibilité d'un GPU, pas le mode lui-même, qui doit conditionner le défaut.

Reproduction : `docs/mesures/jalon3/sondes.json` (champ `latence_par_device`, produit par `scripts/sondes_jalon3.py`) et `docs/mesures/jalon3/F_dev.json`, champ `temoin_reproduction` (facteur de surestimation calculé automatiquement à chaque exécution de `--ablation F`).

## Diagnostic fondateur

Deux sondes jetables exécutées par le contrôleur avant la rédaction du plan de ce jalon (`sonde_discriminance.py`, `sonde_dense.py`, non versionnées) ont établi que le gold des questions dures de `vocabulaire_courant` se classe mal dans son classement lexical — retrouvé sans partager de token de contenu discriminant avec le record gold sur certaines questions — tandis que le canal dense le place dans une fenêtre atteignable (178-528 sur 1660) mais hors de la fenêtre `limit=50` utilisée par `hybrid` en production.

`scripts/diagnostic_rangs.py` reproduit ce diagnostic à l'identique, sans reconstruire l'index, en calculant le rang du gold dans TROIS classements complets (sans troncature `LIMIT`) : lexical non filtré, lexical filtré à un `df_max` donné, dense.

**Correction (revue T2, 16 août 2026)** : la première version de cet outil avait hérité, sans le savoir, un bug de la sonde jetable fondatrice qu'il remplaçait (`sonde_discriminance.py`) : `_rang_gold_lexical` retournait DÈS la découverte du gold (`return rang, len(vus)` dans la boucle), si bien que le second nombre de la paire « rang/total » valait mécaniquement le rang lui-même — jamais la taille réelle du classement. D'où les paires "154/154", "46/46", "1430/1430" publiées dans une version antérieure de cette section : ce n'étaient PAS des données (« le gold est toujours dernier de son classement »), mais un pur artefact de sonde (« le gold est le n-ième et dernier record scanné avant l'arrêt anticipé de la boucle »). Corrigé : le total est désormais la taille RÉELLE du classement complet, calculée indépendamment de la position où le gold est trouvé (cf. docstring de `_rang_gold_lexical` dans `scripts/diagnostic_rangs.py`, et `tests/test_diagnostic_rangs.py` pour la non-régression). Le JSON committé (`docs/mesures/jalon3/diagnostic_rangs.json`) était en outre périmé : régénéré avec l'outil corrigé, et avec les tokens non dédupliqués (correction du défaut 3 de T1, cf. § Ablation D) — d'où 1453 pour q060 (au lieu de 1430 dans la version périmée).

Reproduction (`uv run python scripts/diagnostic_rangs.py --questions q021,q026,q060,q023`), chiffres corrigés et vérifiés par SQL direct (contrôle indépendant) :

| question | catégorie | gold | lexical (non filtré) | lexical (df≤2%) | dense |
|---|---|---|---|---|---|
| q021 | vocabulaire_courant | pcg-214-13 | 154/1585 | ABSENT/38 | 178/1660 |
| q026 | vocabulaire_courant | pcg-212-3 | 46/1659 | **14/67** | 248/1660 |
| q060 | vocabulaire_courant | pcg-1222-74 | 1453/1659 | ABSENT/63 | 528/1660 |
| q023 | vocabulaire_courant | pcg-214-22 | 2/1653 | ABSENT/52 | 257/1660 |

**Lecture — quatre profils distincts, pas un seul cas** : le diagnostic initial groupait ces quatre questions sous un même verdict (« le gold se classe mal en lexical ») ; les rangs corrigés montrent qu'il s'agit en réalité de quatre profils différents :
- **q023** : rang lexical **2**/1653 — quasi immédiat, largement dans toute fenêtre réaliste (`limit=50` suffit déjà en production).
- **q026** : rang lexical **46**/1659 — juste EN-DESSOUS de la fenêtre par défaut (`limit=50`) : son gold entre donc déjà dans le pool bm25 au réglage neutre du jalon 2.5, sans qu'aucun levier de ce jalon soit nécessaire pour cette question précise (confirmé à la mesure T2, § Ablation E : couvert dès `pool=50`). Le filtrage `df_max` l'améliore encore (46→14) en écartant les mots fonctionnels qui diluaient son score.
- **q021** : rang lexical **154**/1585 — hors de la fenêtre par défaut, mais à portée d'un pool élargi (`pool≥154`, donc `pool=200` suffit) : confirmé à la mesure T2 (couvert dès `pool=200`, absent à 50 et 100). *Ces comparaisons rang↔`pool` sont approximatives* : le rang est calculé au niveau **record** (après agrégation) tandis que `pool` borne les lignes de `chunks_norm` au niveau **chunk** (2 160 chunks pour 1 660 records, ~1,3 chunk par record). C'est la couverture mesurée du § Ablation E, pas cette arithmétique, qui tranche.
- **q060** : rang lexical **1453**/1659 (88 % du classement) — le seul véritablement muet côté lexical, hors d'atteinte de tout pool réaliste testé dans ce jalon (`pool=400` encore insuffisant, § Ablation E) et du filtrage `df_max` (disparaît du pool filtré, `ABSENT`).

Le filtrage lexical (`df_max`) ne repêche donc le gold que lorsqu'il partage déjà au moins un token RARE avec la question (cas q026, rang 46→14) ; quand aucun token de contenu n'est commun (q021, q060), le gold disparaît purement et simplement du pool filtré (`ABSENT`) — le filtrage ne peut pas inventer un signal qui n'existe pas. Le canal dense reste, dans tous les cas mesurés ici, le seul à placer TOUS ces golds dans une fenêtre non triviale (178-528 sur 1660), y compris q060.

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

## Ablation E — largeur du pool de candidats (`pool`) et déduplication des termes du `MATCH` (`dedup_termes`)

Nouveaux paramètres `Searcher(..., pool: int = 50, dedup_termes: bool = False)`. `pool` fixe le nombre de lignes récupérées par CHAQUE canal (`_bm25`, `_dense`) AVANT la fusion RRF — 50 reproduit le comportement jalon 2.5. `dedup_termes` rend explicite et mesurable un comportement déjà découvert incidemment en T1 : `_termes_match` ne dédupliquait jamais les tokens du chemin neutre (`df_max=None`), car `bm25()` de SQLite FTS5 pondère leur MULTIPLICITÉ dans l'expression `MATCH`, pas seulement l'ensemble des lignes sélectionnées ; `dedup_termes=False` (défaut) reproduit ce comportement exactement, `dedup_termes=True` déduplique.

**Sémantique retenue quand `df_max` et `dedup_termes` sont actifs simultanément** (cas non mesuré ici, `df_max` ayant été rejeté en T1, mais qui doit rester cohérent) : le chemin de filtrage actif de `_termes_match` déduplique déjà ses tokens en interne, indépendamment de `dedup_termes` — il ne peut donc jamais être MOINS déduplique que `dedup_termes=True`. Documenté dans le docstring de `_termes_match` (`src/accounting_rag/search.py`).

### Défaut constaté dans le brief T2 (distinct des défauts de T1)

Le test verbatim du brief (`test_pool_est_transmis_aux_deux_canaux`) espionne `_bm25`/`_dense` en substituant une fonction sur l'instance (`s._bm25 = lambda q, limit=None: ...`) — substitution qui n'est PAS liée comme méthode (contrairement à un attribut de classe), donc le brief demandait dans le même temps que `search()` n'appelle les canaux « sans passer de limite explicite » : sous cette consigne, la fonction de substitution reçoit toujours `limit=None`, et le test échoue nécessairement (`vus["bm25"] == None != 7`). Corrigé : `search()` transmet désormais explicitement `self.pool` aux deux canaux (`self._bm25(query, self.pool)`, `self._dense(query, self.pool)`) — comportement neutre inchangé (pool=50 par défaut), mais vérifiable par un test qui espionne un attribut d'instance. Deuxième défaut, dans le corps même du test fourni : `vus.setdefault("bm25", limit) or vrai_bm25(q, limit)` court-circuite dès que `limit` est une valeur entière vraie (tout pool non nul) — `setdefault` renvoie alors `limit`, l'opérateur `or` ne rappelle jamais `vrai_bm25`, et la valeur renvoyée à `search()` devient l'entier `limit` lui-même au lieu d'un dict de scores (`AttributeError` dans `_rrf`). Corrigé en séparant la capture de l'espionnage du calcul du retour réel (`tests/test_search.py::test_pool_est_transmis_aux_deux_canaux`, docstring détaillée). Les deux défauts sont dans le mécanisme de test lui-même, pas dans le mécanisme mesuré — le brief l'anticipait explicitement (« adapter la mécanique d'espionnage... si un patron plus simple y est déjà en usage »).

### Méthode

`scripts/ablations_jalon3.py --ablation E --split dev` — chemin dédié `run_ablation_E()` (pas le squelette générique `run_ablation()` utilisé par D), car cette ablation a deux particularités : (1) la métrique de couverture du pool (`_couverture_pool`), calculée directement via `Searcher._bm25`/`_dense` sur les MÊMES instances que celles utilisées pour `evaluate()` (embedder partagé, aucune reconstruction) ; (2) une configuration combinée (meilleure largeur + dedup) mesurée SEULEMENT si l'une des deux mesures séparées est adoptée (brief T2, step 5.3). Cinq configurations, mode `hybrid`, k=10, 61 questions dev : `pool=50` (référence, neutre), `pool=100`, `pool=200`, `pool=400`, `dedup_termes=True` (pool=50 constant). Contrôle de non-régression identique à D : la config neutre doit redonner recall@10=0,672 exactement.

### Résultats

Sortie brute de `uv run python scripts/ablations_jalon3.py --ablation E --split dev` (persistée intégralement dans `docs/mesures/jalon3/E_dev.json`) :

```
[ablations_jalon3] contrôle de non-régression OK : recall@10 neutre = 0.672 == 0.672.
```

| config | recall@5 | recall@10 | MRR | latence/question |
|---|---|---|---|---|
| pool=50 (neutre, jalon 2.5) | 0,639 | **0,672** | 0,565 | 0,201 s |
| pool=100 | 0,607 | 0,639 | 0,557 | 0,230 s |
| pool=200 | 0,623 | 0,639 | 0,56 | 0,228 s |
| pool=400 | 0,623 | 0,639 | 0,56 | 0,254 s |
| dedup_termes=True (pool=50) | 0,623 | 0,689 | 0,568 | 0,216 s |

Ventilation par catégorie (recall@10) :

| config | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|
| pool=50 (neutre) | 1,0 | 0,935 | 0,403 |
| pool=100 | 1,0 | 0,978 | 0,306 |
| pool=200 | 1,0 | 0,978 | 0,306 |
| pool=400 | 1,0 | 0,978 | 0,306 |
| dedup_termes=True | 1,0 | 0,935 | 0,435 |

Bootstrap apparié (`n_boot=10000`, `seed=42`, contre la référence `pool=50`) :

| comparaison | delta | IC95 | p_amelioration | pire perte catégorie | adopté ? |
|---|---|---|---|---|---|
| pool=100 | -0,0328 | (-0,0984 ; 0,0328) | 0,0944 | -0,0968 (vocabulaire_courant) | **non** |
| pool=200 | -0,0328 | (-0,0984 ; 0,0328) | 0,0944 | -0,0968 (vocabulaire_courant) | **non** |
| pool=400 | -0,0328 | (-0,0984 ; 0,0328) | 0,0944 | -0,0968 (vocabulaire_courant) | **non** |
| dedup_termes=True | +0,0164 | (0,0000 ; 0,0492) | 0,6319 | 0,0000 (aucune) | **non** |

**Contrôle croisé (mesure indépendante du relecteur de T1)** : la mesure de `dedup_termes=True` reproduit EXACTEMENT les chiffres de contrôle fournis (`delta=+0,0164`, `IC95=(0,0000 ; 0,0492)`, `p_amelioration=0,6319`) — aucun écart.

### Couverture du pool avant fusion (livrable central de cette tâche)

Part des questions dont au moins une citation gold est présente dans l'union bm25 ∪ dense, calculée directement (sans troncature au top-10, sans fusion RRF) à chaque largeur — c'est la fenêtre de candidats dont dépend l'ablation F (reranking) :

| pool | couverture globale (n=61) | reference_directe (n=7) | regle (n=23) | vocabulaire_courant (n=31) |
|---|---|---|---|---|
| 50 | 0,820 | 1,0 | 1,0 | 0,645 |
| 100 | 0,852 | 1,0 | 1,0 | 0,710 |
| 200 | 0,918 | 1,0 | 1,0 | 0,839 |
| 400 | 0,934 | 1,0 | 1,0 | 0,871 |

Vérification ciblée sur les trois questions du diagnostic fondateur (script ad hoc, non versionné, à partir de `Searcher._bm25`/`_dense`) :

| question | pool=50 | pool=100 | pool=200 | pool=400 |
|---|---|---|---|---|
| q021 (gold rang lexical 154/1585) | absent | absent | **présent** | présent |
| q026 (gold rang lexical 46/1659) | **présent** | présent | présent | présent |
| q060 (gold rang lexical 1453/1659) | absent | absent | absent | absent |

Conforme au diagnostic corrigé sur deux points exacts : q021 entre dans le pool précisément à la largeur 200 (rang 154, donc hors de 50 et 100, dedans dès que la largeur ≥154) ; q060 reste hors d'atteinte à toute largeur testée (rang 1453, hors de portée même à 400). **Précision par rapport au texte du plan** : q026 (rang lexical 46/1659) est en réalité déjà couvert au pool NEUTRE (50), pas seulement « dès 200 » — son rang lexical (46) est inférieur à la limite par défaut du canal bm25 (50), donc son gold entre déjà dans le pool bm25 avant même tout élargissement. Ce n'est pas un écart de mesure : le rang 46 mesuré par `diagnostic_rangs.py` en T1 (§ Diagnostic fondateur ci-dessus) est cohérent avec cette observation — 46 ≤ 50. Le texte du plan groupait q021 et q026 sous le même seuil « 200 » par simplification ; la mesure ici est plus précise et ne contredit aucun chiffre déjà publié.

### Le découplage couverture / recall — pourquoi la couverture monte alors que le recall baisse

La couverture du pool progresse fortement avec la largeur (globale 0,820 → 0,918 à pool=200, +0,098 ; `vocabulaire_courant` 0,645 → 0,839, +0,194) alors que le recall@10 APRÈS fusion RRF, lui, **baisse** (0,672 → 0,639, `vocabulaire_courant` -0,097). L'écart entre 83,9 % de couverture et 30,6 % de recall sur `vocabulaire_courant` à pool=200 est un déficit de **classement**, pas de rappel : le gold entre bien dans le pool (couverture), mais la fusion RRF ne le remonte pas dans le top-10 fusionné (recall).

Mécanisme, vérifié par deux contrôles :

1. **Pourquoi pool=100, 200 et 400 donnent des métriques strictement identiques** (recall@5/10/MRR, delta, IC95 et p_amelioration bit à bit égaux dans le tableau ci-dessus). Contrôle ad hoc sur les 61 questions dev, config `pool=200` : sur les 610 entrées du top-10 fusionné (61 questions × 10), **aucune** (0/610, 0,0 %) n'a son meilleur rang par canal (bm25 ou dense) au-delà de 100. Le score RRF d'un candidat au rang r dans un canal est `1/(60+r+1)` (`_RRF_K=60`) : au rang 100, ce plafond vaut `1/161 ≈ 0,0062`, systématiquement dominé par les candidats déjà présents dans le pool à 100 (rang < 100 dans au moins un canal). Élargir le pool de 100 à 200 ou 400 n'ajoute donc, empiriquement sur ce split, JAMAIS de candidat capable d'entrer dans le top-10 fusionné — la fusion RRF s'est déjà « saturée » à 100.

2. **Pourquoi le recall BAISSE en passant de 50 à 100** (contre-intuitif : plus de candidats devrait, au minimum, ne pas nuire). Diff question par question (`docs/mesures/jalon3/E_dev.json`, champs `configs[i].par_question`) : 3 questions régressent (q061, q072, q082 : 1,0 → 0,0) et 1 s'améliore (q054 : 0,0 → 1,0), net -2/61 = -0,0328 — exactement le delta mesuré. Exemple mécanistique concret, q061 (« Si je décide de compter mes marchandises d'une autre façon... », gold `pcg-122-3`) :
   - À pool=50 : le record réglementaire `pcg-122-3` (le gold) est retrouvé UNIQUEMENT par bm25 (rang 1, absent du top-50 dense) → score RRF = `1/61 ≈ 0,01639`, 3e position du top-10 fusionné. Un chunk de commentaire du même article (`pcg-122-3-c1`, qui matche aussi la citation gold) est présent côté dense (rang 22, hors bm25) mais son score RRF solo (`1/82 ≈ 0,01220`) est trop faible pour entrer dans le top-10 — sans effet sur le résultat.
   - À pool=100 : le score RRF de `pcg-122-3` reste `0,01639` (aucun changement de ses rangs individuels) et celui de `pcg-122-3-c1` reste `0,01220` (toujours sous le seuil, vérifié) — mais deux nouveaux candidats, absents à pool=50, entrent maintenant dans LES DEUX canaux à des rangs médiocres (ex. `pcg-na-18` : bm25 rang 71, dense rang 3 → RRF = `1/131 + 1/63 ≈ 0,02351` ; `pcg-na-24` : bm25 rang 49, dense rang 10 → RRF = `1/109 + 1/70 ≈ 0,02346`) — supérieurs à la fois à `pcg-122-3` et à `pcg-122-3-c1`. Le 10e score du top-10 à pool=100 vaut `0,01740` : `pcg-122-3` (0,01639) en sort, `pcg-122-3-c1` (0,01220) n'y était déjà pas. Leur somme de deux contributions MÉDIOCRES (un canal moyen + un canal moyen) dépasse la contribution UNIQUE mais meilleure du gold (un canal excellent, l'autre absent), qui sort du top-10.

   C'est le mécanisme générique de la fusion RRF par sommation : un document moyen dans les DEUX canaux peut accumuler plus de « masse de rang » qu'un document excellent dans un seul — élargir le pool expose davantage de candidats à ce risque de double comptage, sans qu'aucun mécanisme ne privilégie la pertinence absolue plutôt que la présence conjointe.

**Lecture pour la suite du jalon** : le pool élargi RÉPARE bien le problème diagnostiqué en amont (le gold entre dans le pool — couverture +0,194 sur `vocabulaire_courant`), mais la fusion RRF, sans discrimination de pertinence fine, ne sait pas exploiter ce pool élargi et se dégrade même légèrement. C'est très précisément le plafond qu'un reranker cross-encoder est censé lever (ablation F, T3) : le reranker note chaque candidat individuellement (pertinence absolue vis-à-vis de la question), au lieu de sommer des rangs relatifs — jusqu'à 0,839 de recall@10 potentiellement atteignable sur `vocabulaire_courant` (couverture du pool à 200) contre 0,403 aujourd'hui (pool=50, sans reranker), à condition que le reranker sache effectivement remonter le bon candidat une fois dans le pool.

### Décisions

**`pool` : REJETÉ — reste à sa valeur neutre par défaut (`50`).** Aucune des trois largeurs testées ne franchit le critère d'adoption (`p_amelioration ≥ 0,95` ; mesuré 0,0944 dans les trois cas, très en-dessous), et la perte sur `vocabulaire_courant` (-0,0968) dépasse déjà la garde de -0,05. L'effet est négatif malgré la couverture du pool qui, elle, progresse nettement — cohérent avec le découplage couverture/recall expliqué ci-dessus : élargir le pool sans améliorer la fusion ne peut pas améliorer le recall, il peut même le dégrader. Résultat négatif mais informatif : il établit la couverture du pool (le vrai livrable) et motive directement l'ablation F.

**`dedup_termes` : REJETÉ — reste à sa valeur neutre par défaut (`False`).** `p_amelioration=0,6319` reste nettement sous le seuil de 0,95, malgré un delta global positif (+0,0164) et AUCUNE perte par catégorie (`pire_perte_categorie` = 0,0 — dedup ne fait jamais moins bien qu'une catégorie de la référence en recall@10). L'effet est non uniforme entre les deux métriques de rappel : recall@10 progresse (0,672 → 0,689, +0,017) et `vocabulaire_courant` gagne +0,032 en recall@10 (0,403 → 0,435), mais recall@5 RECULE (0,639 → 0,623, -0,016) — la déduplication remonte certains golds entre les rangs 6 et 10 (gain visible seulement à k=10) tout en en faisant sortir d'autres du top-5. Le critère d'adoption du plan porte sur `p_amelioration` du recall@10 (pas du recall@5) et sur la perte par catégorie (pas sur recall@5 non plus) : sur ce seul critère, `dedup_termes=True` échoue franchement (0,6319 très inférieur à 0,95), quel que soit le signe du delta global. Rejeté.

**Configuration combinée (meilleure largeur + `dedup_termes=True`) : NON MESURÉE**, conformément à la règle du brief T2 (step 5.3) — ni `pool=100` (meilleure largeur parmi 100/200/400 par recall@10, toutes trois à égalité) ni `dedup_termes=True` n'est adopté séparément. C'est une réserve honnête, pas un oubli : mesurer une config combinée de deux leviers rejetés séparément n'aurait apporté aucune information supplémentaire pour ce jalon.

### Réserves

- **Le vrai résultat de cette tâche est négatif pour `recall@10` mais positif pour la couverture du pool** : le pool élargi répare le problème structurel diagnostiqué (`vocabulaire_courant`, +0,194 de couverture à pool=200) sans que cela se traduise en gain de recall, faute d'un mécanisme de fusion capable d'exploiter ce pool. Ce résultat conditionne directement l'ablation F : le reranking sur pool élargi est maintenant justifié par une mesure, pas seulement par une intuition.
- Le contrôle du § « découplage couverture/recall » (0/610 entrées du top-10 au-delà du rang 100) est produit par un script ad hoc, non versionné (exécuté pour ce rapport, reproductible à partir de `Searcher._bm25`/`_dense`/`_rrf`, mais pas committé) — contrairement aux chiffres agrégés, qui sont tous dans `docs/mesures/jalon3/E_dev.json`.
- `n=61` (dev) : le delta pool=50→100 (-0,0328) tient sur seulement 4 questions qui basculent (3 régressions, 1 amélioration) — un split dev plus grand pourrait déplacer ce chiffre sensiblement.
- Deux défauts trouvés et corrigés dans le test fourni par le brief T2 (§ ci-dessus) — signalés en toute transparence, dans la continuité de la pratique établie en T1.
- **La couverture est un plafond d'ATTEIGNABILITÉ, pas un recall promis.** `0,839` sur `vocabulaire_courant` à `pool=200` signifie : dans 83,9 % de ces questions, au moins une citation attendue se trouve quelque part dans l'union `bm25 ∪ dense` AVANT fusion. Aucun mécanisme connu ne garantit qu'un reranker saura la remonter dans le top-10 ; le chiffre borne ce qui est possible sans reconstruire l'index, il ne prédit pas ce qui sera obtenu. Toute communication de ce chiffre doit porter cette distinction.
- **Le coût du reranking sur un pool élargi n'est pas mesuré dans cette tâche.** Un pool de 200 par canal expose jusqu'à 2×200 candidats à la fusion ; soumettre cet ordre de grandeur à un cross-encoder change complètement le budget latence par question (le reranker adopté au jalon 2.5 coûte ~4,7 s/candidat). C'est précisément l'arbitrage que l'ablation F mesure — tant qu'il ne l'est pas, le plafond de 0,839 est un plafond sans prix affiché.

## Reproduction exacte

```sh
uv run pytest tests/test_search.py -q                     # tests unitaires df_max/df/_termes_match/pool/dedup_termes
uv run python scripts/diagnostic_rangs.py --questions q021,q026,q060,q023   # § Diagnostic fondateur
uv run python scripts/ablations_jalon3.py --ablation D --split dev          # § Ablation D
uv run python scripts/ablations_jalon3.py --ablation E --split dev          # § Ablation E
```

`docs/mesures/jalon3/diagnostic_rangs.json`, `docs/mesures/jalon3/D_dev.json` et `docs/mesures/jalon3/E_dev.json` sont versionnés — tout chiffre publié ci-dessus est recalculable sans re-runner ces commandes (les `par_question` bruts de chaque configuration sont dans `configs[i].par_question` de chaque JSON ; la couverture du pool par largeur est dans `E_dev.json`, champ `couverture_pool`).

## Ablation F — largeur du pool soumis au reranker (`n_rerank`)

Nouveau paramètre `Searcher(..., n_rerank: int = 25)` : le nombre de candidats issus de la fusion soumis au cross-encoder en mode `hybrid+rerank`, jusqu'ici codé en dur à 25 dans `search()`. `n_rerank` est indépendant de `pool` (largeur récupérée par canal AVANT fusion) : on peut donc élargir le vivier et ce qu'on en soumet séparément. Les résultats routés (référence d'article exacte) restent épinglés hors reranking à toute largeur (test dédié, `tests/test_rerank.py`). Un `n_rerank` supérieur au nombre de candidats disponibles ne casse rien : la troncature par slice renvoie ce qui existe.

**Question posée par le jalon** : le § Ablation E a établi que le pool contient la bonne réponse bien plus souvent que la fusion ne la restitue (couverture 0,918 global / 0,839 sur `vocabulaire_courant` à `pool=200`, contre 0,672 / 0,403 de recall@10). Un cross-encoder sait-il aller chercher cet écart ? Et si oui, faut-il un modèle **faible sur beaucoup** de candidats ou un modèle **fort sur peu** ?

### Méthode

Référence = la configuration ADOPTÉE au jalon 2.5 (`bge-reranker-v2-m3`, `n_rerank=25`, `pool=50`), réutilisée depuis `docs/mesures/jalon25/cloture_dev.json` (champ `b`) pour le bootstrap — avec contrôle de non-régression sur `recall@10 = 0,738` avant toute comparaison — ET re-mesurée à l'identique comme **témoin**, pour deux raisons : vérifier que la référence se reproduit question par question (elle le fait : 61/61, `temoin_reproduction.identique = true`), et disposer d'une latence comparable aux autres lignes du tableau (celle du jalon 2.5 étant une latence CPU, cf. § ci-dessus).

Grille complète : les deux modèles × largeur étroite / large. Le premier jet de cette tâche plaçait `bge` sur pool large derrière une option désactivée par défaut, sur une estimation de coût (~8 min/question, ~8 h pour le split) qui s'est révélée fausse d'un facteur ~50 — la configuration décisive du jalon était donc écartée sans avoir été mesurée. Corrigé : les cinq configurations ci-dessous tournent sans option, en ~15 min au total.

### Résultats

| config | recall@5 | recall@10 | MRR | `vocabulaire_courant` recall@10 | latence/question |
|---|---|---|---|---|---|
| bge, `n_rerank=25`, `pool=50` (référence jalon 2.5, réutilisée) | 0,680 | **0,738** | 0,642 | 0,484 | 129,5 s (**CPU**, jalon 2.5) |
| bge, `n_rerank=25`, `pool=50` (témoin re-mesuré ici) | 0,680 | **0,738** | 0,642 | 0,484 | 1,71 s (GPU) |
| mmarco, `n_rerank=25`, `pool=50` (contrôle modèle) | 0,672 | 0,713 | 0,610 | 0,435 | 0,40 s |
| mmarco, `n_rerank=200`, `pool=200` | 0,656 | 0,713 | 0,581 | 0,435 | 1,24 s |
| **bge, `n_rerank=100`, `pool=50`** (isole `n_rerank`) | 0,697 | 0,705 | 0,646 | 0,419 | 4,44 s |
| bge, `n_rerank=100`, `pool=100` | 0,697 | 0,705 | 0,646 | 0,419 | 5,46 s |
| bge, `n_rerank=200`, `pool=200` | **0,730** | **0,738** | **0,675** | 0,484 | 10,16 s |

Les deux premières lignes sont la même configuration : la référence relue depuis le JSON du jalon 2.5 et son témoin re-mesuré ici. Elles sont identiques sur toutes les métriques — c'est le contrôle de reproduction — et diffèrent **uniquement** par la latence, qui est l'objet du § « Le reranking du jalon 2.5 était mesuré sur CPU ».

Bootstrap apparié (10 000 tirages, seed 42) sur le recall@10 par question, contre la référence :

| config | delta | IC95 | p_amelioration | pire perte catégorie | adopté ? |
|---|---|---|---|---|---|
| témoin re-mesuré | 0,0000 | (0,000 ; 0,000) | — (identique) | 0,0 | — (contrôle) |
| mmarco, 25 | −0,0246 | (−0,0656 ; 0,0000) | 0,000 | −0,048 (`vocabulaire_courant`) | non |
| mmarco, 200 | −0,0246 | (−0,1066 ; 0,0574) | 0,249 | −0,048 (`vocabulaire_courant`) | non |
| **bge, 100, `pool=50`** (isolante) | −0,0328 | (−0,0984 ; 0,0328) | 0,098 | −0,065 (`vocabulaire_courant`) | non |
| bge, 100, `pool=100` | −0,0328 | (−0,0984 ; 0,0328) | 0,098 | −0,065 (`vocabulaire_courant`) | non |
| bge, 200 | 0,0000 | (−0,0820 ; 0,0820) | 0,423 | 0,0 | non |

### Lecture : la largeur n'achète RIEN en recall@10, pour aucun des deux modèles

Le résultat est net et il est répété deux fois, une fois par modèle : **mmarco donne 0,713 sur 25 candidats et 0,713 sur 200 ; bge donne 0,738 sur 25 et 0,738 sur 200.** Non seulement les agrégats coïncident, mais la ventilation par catégorie aussi (`vocabulaire_courant` 0,435 pour mmarco aux deux largeurs, 0,484 pour bge aux deux largeurs). Multiplier par 8 le nombre de candidats soumis au cross-encoder — donc par 5,8 la latence pour bge — ne fait entrer aucune citation gold supplémentaire dans le top-10.

**Ce que la largeur change quand même : le haut du classement, et seulement avec le modèle fort.** `bge` sur 200 candidats gagne +0,050 de recall@5 (0,680 → 0,730) et +0,033 de MRR (0,642 → 0,675) à recall@10 constant : le pool large permet au modèle fort de promouvoir dans le top-5 des golds qui étaient déjà entre les rangs 6 et 10. C'est un vrai gain de précision de tête, le premier que ce jalon obtient — mais il ne franchit pas le critère d'adoption, qui porte sur le recall@10 (fixé avant les mesures, et non révisé après coup pour accommoder un résultat). Il est noté ici pour le jalon 4, parce qu'un générateur RAG est alimenté par ~5 passages, pas 10 : c'est le recall@5 et le MRR qui gouverneront la qualité des réponses.

**`n_rerank` est bien le paramètre responsable, et non `pool` — mesuré, après avoir été soulevé en revue.** La première version de cette grille ne contenait aucune configuration isolante : toutes les configs « larges » déplaçaient `n_rerank` **et** `pool` ensemble, alors que le § Ablation E a mesuré `pool=100/200/400` à −0,033 de recall@10. Un effet nul du couple pouvait donc masquer un `n_rerank` positif compensé par un `pool` négatif. La configuration ajoutée (`bge`, `n_rerank=100`, `pool=50` — le vivier reste à sa valeur neutre, et à `pool=50` la fusion expose déjà jusqu'à ~100 candidats distincts, donc `n_rerank=100` y soumet strictement plus que 25) tranche : elle donne **exactement les mêmes chiffres** que `pool=100` — recall@5, recall@10, MRR, ventilation par catégorie et bootstrap complet, à la quatrième décimale. `pool` ne contribue rien à ce niveau ; les −0,033 sont entièrement imputables à `n_rerank`. Le rejet est donc établi **pour `n_rerank` lui-même**, pas pour un couple confondu.

**Non-monotonie à ne pas surinterpréter** : `bge` sur 100 candidats (0,705) fait moins bien que `bge` sur 25 (0,738) ET que `bge` sur 200 (0,738). Une largeur intermédiaire pire que ses deux voisines n'a aucun mécanisme plausible : sur n=61, un écart de 0,033 vaut deux questions. La conclusion honnête est que **tous les écarts de recall@10 entre configurations `bge` sont dans le bruit** ; le seul signal qui survit au bootstrap est le choix du MODÈLE (mmarco perd 0,0246 avec p_amelioration = 0,000, donc une dégradation robuste, cohérente avec le rejet de mmarco au jalon 2.5).

### Décision

**`n_rerank` : REJETÉ — reste à sa valeur neutre par défaut (`25`).** Aucune configuration ne franchit le critère d'adoption ; la meilleure (`bge`, 200) est strictement neutre en recall@10 (p_amelioration = 0,423) pour 5,8× la latence. Le paramètre reste exposé, avec son défaut neutre, parce que le gain de recall@5 mérite d'être re-mesuré sur un split plus large au jalon 4.

**L'arbitrage modèle/largeur est tranché dans le sens inverse de l'hypothèse.** Le jalon était parti de : « le reranker lourd coûte 4,7 s/candidat, donc inutilisable sur 200 ; le modèle léger rejeté au jalon 2.5 redevient le seul candidat crédible dès qu'on élargit ». Les deux moitiés de ce raisonnement sont fausses. Le coût était une estimation CPU jamais vérifiée (le modèle lourd tient 200 candidats en 10,6 s/question sur GPU), et surtout **la largeur n'apporte rien**, ce qui retire au modèle léger sa seule raison d'être ici. Modèle fort sur pool étroit gagne, et c'était déjà la configuration du jalon 2.5.

### Le déficit couverture/classement n'est PAS un déficit de candidats

C'est le résultat de fond, et il ferme une hypothèse : sur `vocabulaire_courant`, le pool à `pool=200` contient une citation gold dans **83,9 %** des cas, et le meilleur reranking mesuré sur ce même pool en restitue **48,4 %** dans le top-10. Trente-cinq points d'écart subsistent alors que le cross-encoder a vu *tous* les candidats du pool. Le goulot n'est donc ni la fenêtre de récupération (ablation E), ni la capacité du reranker à traiter du volume (ablation F) : c'est la capacité d'un modèle à **reconnaître** qu'un article du PCG répond à une question posée en langage courant. Aucun réordonnancement de candidats ne crée cette reconnaissance.

Deux voies restent, et elles sont d'une autre nature : agir sur la REQUÊTE pour qu'elle parle le vocabulaire du corpus (ablation G, réécriture par LLM), ou agir sur la REPRÉSENTATION pour que la proximité sémantique cesse de dépendre du vocabulaire (jalon 4 : embeddings plus forts, ou adaptés au domaine comptable français).

### Réserves

- `n=61`, et les écarts discutés ici valent 2 à 4 questions. Le gain de recall@5 de `bge`+200 (+0,050, soit ~3 questions) n'a PAS été soumis au bootstrap : le protocole du jalon fixe le critère sur le recall@10, et je ne substitue pas après coup la métrique qui arrange le résultat. À re-mesurer proprement, sur un split plus large, avant toute adoption.
- Les latences sont des latences GPU fp32 sur cette machine (cf. § Conditions exactes). Sur CPU, le facteur ~70 rend `bge`+200 inutilisable en interactif (~12 min/question) : la conclusion « le modèle fort tient 200 candidats » est conditionnée à la présence d'une carte.
- Le témoin re-mesuré reproduit la référence question par question, ce qui valide la réutilisation du JSON du jalon 2.5 comme base de bootstrap. Il ne valide pas les autres chiffres de ce jalon-là.
- La combinaison `n_rerank` élargi + `dedup_termes` ou `df_max` n'est pas mesurée : mêmes règles qu'en T2, aucun de ces leviers n'étant adopté séparément.

## Ablation G — réécriture de la question par un LLM (`rewriter`)

Nouveaux paramètres `Searcher(..., rewriter=None, mode_reecriture="remplace")`. Le module `src/accounting_rag/rewrite.py` traduit une question posée en langage courant vers le vocabulaire technique du Plan comptable général via l'API Claude, puis la requête réécrite alimente les canaux lexical et dense. `_route(query)` continue de lire la question **originale** : une référence d'article explicite ne doit jamais dépendre d'une reformulation par LLM.

C'est le seul levier de ce jalon capable de **créer** un lien de vocabulaire qui n'existe pas. Les trois précédents (`df_max`, `pool`, `n_rerank`) ne savent que réordonner ce que le vocabulaire commun a déjà trouvé — et les § ci-dessus montrent qu'ils échouent tous les trois.

### Méthode

Référence = `hybrid` neutre (recall@10 = 0,672), **pas** `hybrid+rerank` : la réécriture agit sur les canaux, et la mesurer sous un reranker mélangerait deux effets. Deux modes mesurés :

- **`remplace`** : les canaux ne reçoivent que la réécriture ;
- **`etend`** : les canaux reçoivent `question + " " + réécriture`, conservant les tokens originaux.

Modèle : `claude-sonnet-5`, `thinking` explicitement désactivé, un appel par question distincte, cache JSON committé (`docs/mesures/jalon3/reecritures.json`), garde-fou dur à 200 appels par exécution. Coût, tel qu'enregistré dans les JSON : **12 appels** lors de cette campagne (`G_dev.json`, champ `cout` — 49 réécritures étaient déjà en cache après la première tentative interrompue), **61 réécritures dev** au total, **90** en comptant le split gelé appelé à la clôture. Quelques centimes en tout, et toutes les re-mesures sont ensuite gratuites et rejouent les mêmes réécritures à l'identique.

### Résultats

| config | recall@5 | recall@10 | MRR | `reference_directe` | `regle` | `vocabulaire_courant` | latence/question |
|---|---|---|---|---|---|---|---|
| `hybrid` neutre, sans réécriture (référence) | 0,639 | 0,672 | 0,565 | 1,000 | 0,935 | 0,403 | 0,14 s |
| réécriture, mode `remplace` | 0,697 | 0,803 | 0,601 | 1,000 | **1,000** | 0,613 | 0,25 s |
| réécriture, mode `etend` | **0,779** | **0,852** | **0,695** | 1,000 | **1,000** | **0,710** | 0,35 s |

**Correction des latences de ce tableau (revue finale de branche).** La campagne avait publié 0,91 s pour `remplace` et 0,23 s pour `etend`, ce qui suggérait que `remplace` était ~4× plus lent. C'était un artefact d'imputation : `remplace` s'exécute en premier dans la grille et a absorbé les 12 appels API non encore cachés, soit ≈41 s répartis sur 61 questions — exactement l'écart observé. Re-mesurées cache chaud, zéro appel API, les deux modes donnent 0,25 s et 0,35 s : **c'est `etend` qui est le plus lent**, et c'est cohérent avec sa construction (voir ci-dessous). Les recall et MRR du tableau ne sont pas affectés : ils ne dépendent pas du temps.

Bootstrap apparié (10 000 tirages, seed 42) sur le recall@10 par question :

| config | delta | IC95 | p_amelioration | pire perte catégorie | adopté ? |
|---|---|---|---|---|---|
| réécriture, `remplace` | +0,1311 | (0,0164 ; 0,2541) | **0,9835** | 0,0 (aucune) | **oui** |
| réécriture, `etend` | +0,1803 | (0,0738 ; 0,2951) | **0,9996** | 0,0 (aucune) | **oui** |

Bootstrap par catégorie pour `etend` : `vocabulaire_courant` +0,3065 (p = 0,9979), `regle` +0,0652 (p = 0,8758), `reference_directe` inchangé (déjà à 1,0 — le routeur regex). Le gain est donc concentré exactement là où le jalon 2 avait chiffré le fossé, et il est significatif **dans cette catégorie prise seule**.

### Décision

**`etend` ADOPTÉ.** Les deux modes franchissent le critère (`p_amelioration ≥ 0,95`, aucune catégorie ne perd plus de 0,05) ; `etend` domine `remplace` sur les trois métriques et sur les trois catégories. Mécanisme : conserver les tokens de la question évite de perdre un terme discriminant que la réécriture omettrait, tout en ajoutant le vocabulaire PCG manquant.

**`etend` gagne en payant, pas en économisant** — et l'affirmation inverse publiée dans une version antérieure de cette section (« `remplace` est ~4× plus lent, la réécriture seule produisant un `MATCH` plus large ») était fausse dans les deux moitiés. Mesuré sur les 61 questions dev, cache chaud : `etend` soumet **3 534 termes** de `MATCH` au total contre **2 109** pour `remplace` (facteur 1,68 — inévitable, `etend` est la concaténation des deux), et coûte **0,35 s/question contre 0,25 s**. Le mode adopté est donc le plus large et le plus lent des deux ; son avantage est de rappel, pas de coût. Cette erreur est le troisième cas, dans ce seul jalon, d'un mécanisme plausible publié avant le contrôle qui le départageait.

### Combiné au reranking : les deux leviers ne se recouvrent pas

| config | recall@5 | recall@10 | MRR | `vocabulaire_courant` | latence/question |
|---|---|---|---|---|---|
| `bge-reranker-v2-m3`, `n_rerank=25` (config adoptée au jalon 2.5) | 0,680 | 0,738 | 0,642 | 0,484 | 1,84 s |
| réécriture `etend` + `bge-reranker-v2-m3`, `n_rerank=25` | 0,762 | **0,877** | 0,714 | **0,774** | 1,75 s |

Bootstrap contre la config du jalon 2.5 : delta **+0,1393**, IC95 (0,0328 ; 0,2459), **p_amelioration = 0,9945**, pire perte catégorie −0,0217 (`regle`, sous la garde de −0,05) → **ADOPTÉ**. Les deux leviers s'additionnent presque intégralement (+0,180 seul, +0,139 par-dessus le reranking) : l'un répare le **vocabulaire de la requête**, l'autre le **classement des candidats**. Ce sont bien deux problèmes distincts, ce que le § Ablation F avait établi par la négative.

### Le sort des questions données hors d'atteinte

Le § Diagnostic fondateur avait classé quatre questions en quatre profils. Recall@10 par question, avant et après :

| question | profil au diagnostic | référence | `remplace` | `etend` |
|---|---|---|---|---|
| q021 | à portée d'un pool élargi (rang lexical 154/1585) | 0,0 | **1,0** | **1,0** |
| q026 | hors fenêtre de peu (46/1659) | 0,0 | **1,0** | **1,0** |
| q060 | **seule muette côté lexical** (1453/1659, 88 %) | 0,0 | **1,0** | **1,0** |
| q023 | quasi-immédiat (rang lexical **2**/1653) | 0,0 | 0,0 | 0,0 |

⚠️ Ces quatre colonnes sont mesurées en mode `hybrid` **sans reranking**, puisque l'ablation G se mesure contre la référence `hybrid` neutre. **q023 réussit (1,0) dans toutes les configurations rerankées** — les six de l'ablation F, et la configuration livrée du jalon 3 (`cloture_dev.json`, champ `C_reecriture_rerank_jalon3`). Une version antérieure de cette section affirmait qu'elle « échoue dans toutes les configurations » : c'était faux, et contredit par le § Clôture du même document, qui nomme les quatre questions réellement résistantes (q057, q063, q082, q089).

Le résultat le plus contre-intuitif du jalon est dans ce tableau : **la question que le diagnostic donnait comme la plus désespérée est réparée, et celle qu'il donnait comme la plus facile résiste.** q060, dont le gold était au 88ᵉ percentile de son classement lexical, est retrouvée dès que la question parle le vocabulaire du corpus. Le « fossé lexical » n'était pas un plafond du système : c'était un plafond du vocabulaire de la question.

### Anatomie de q023 : la fusion RRF évince le meilleur candidat du corpus

q023 mérite son autopsie, parce qu'elle isole un défaut que ce jalon n'a pas traité. Son gold (`pcg-214-22`) est le **2ᵉ meilleur candidat lexical sur 1 653** — présent dans le pool bm25, absent du canal dense. Contributions RRF mesurées (`k=60`) :

| candidat | rang bm25 | rang dense | contribution RRF | sort |
|---|---|---|---|---|
| `pcg-214-22` (**gold**) | **2** | absent | 0,01613 + 0 = **0,01613** | rang 11 après fusion — sort du top-10 |
| `pcg-na-236` | 5 | 6 | 0,01538 + 0,01515 = **0,03054** | 1ᵉʳ |
| `pcg-214-25` | 11 | 17 | 0,01408 + 0,01299 = **0,02707** | 2ᵉ |
| `pcg-1121-1` | 23 | 19 | 0,01205 + 0,01266 = **0,02471** | 3ᵉ |

Ces chiffres sont persistés dans `docs/mesures/jalon3/sondes.json` (champ `anatomie_q023`, produit par `scripts/sondes_jalon3.py`) — comme tout chiffre publié de ce rapport.

Le meilleur candidat lexical du corpus entier perd contre un candidat 5ᵉ et 6ᵉ, parce que **la somme RRF récompense le consensus, pas l'excellence**. Sur ce pool de 81 candidats, 73 ne sont présents que dans un seul canal et 8 dans les deux : ces 8 monopolisent le haut du classement. Le gold manque le top-10 d'**une place**.

C'est la mécanique exacte du découplage couverture/classement décrit au § Ablation E, désormais mesurée sur un cas nommé et non plus en agrégat.

**Mais le reranking rattrape ce cas, et c'est ce qui borne la portée du défaut.** Le gold sort au rang 11 de la fusion — donc à l'intérieur des `n_rerank=25` candidats soumis au cross-encoder, qui le remonte dans le top-10 : q023 vaut 1,0 dans toutes les configurations rerankées, y compris celle livrée. Le défaut de la règle de fusion est donc réel et mesuré, mais **compensé aujourd'hui par la fenêtre du reranker**. Ce qui en fait quand même un levier pour la suite, et pour une raison précise : la compensation ne tient que tant que le gold évincé reste dans les 25 premiers de la fusion. Sur un corpus d'un ordre de grandeur plus grand — le jalon 4 ajoute le BOFiP — le nombre de candidats consensuels croît, l'éviction pousse les golds mono-canal plus loin dans le classement fusionné, et rien ne garantit qu'ils resteront à portée du reranker. Pistes pour ce moment-là : fusion non purement additive sur les rangs (maximum au lieu de somme, normalisation des scores, bonus explicite au rang 1 d'un canal).

### Contrôle d'intégrité du benchmark

La réécriture crée un risque qu'aucune ablation précédente ne portait : le modèle **connaît** le Plan comptable général, et s'il citait de lui-même le numéro de l'article attendu, ce numéro entrerait dans la requête envoyée aux canaux. Le gain mesuré ne mesurerait alors plus le retrieval mais la mémoire du modèle.

Précision sur le canal de fuite, car une version antérieure de cette section le décrivait mal : le routeur de référence exacte **n'est pas** ce canal — `search()` appelle `_route()` sur la question ORIGINALE, jamais sur la réécriture, donc un numéro inventé ne peut pas déclencher le routage. Le canal réel est la **correspondance lexicale et dense sur le token numérique lui-même** : « 214-13 » dans la requête matche le texte de l'article 214-13 dans `chunks_norm`. C'est ce que l'audit ci-dessous couvre. Deux verrous :

1. **Structurel** : le rewriter ne reçoit que le texte de la question — jamais les citations, jamais le corpus, jamais les résultats (test dédié, `tests/test_rewrite.py`).
2. **Empirique et scripté** : `scripts/audit_reecritures.py` audite les 61 réécritures du cache committé et classe chaque numéro d'article trouvé en *recopié depuis la question* / *inventé hors gold* / *fuite*. Résultat : **un seul** numéro sur 61 réécritures (`123-16`, le code de commerce, recopié depuis la question de q003), **zéro inventé, zéro fuite**.

Ce contrôle est **falsifiable** : `tests/test_audit_reecritures.py` injecte une réécriture qui cite le gold sans qu'il figure dans la question et vérifie que l'audit la signale. Sans ce test, le « contrôle OK » serait une affirmation invérifiable — précisément le travers documenté en T2, où un contrôle de reproduction avait validé le bug qu'il devait attraper.

### Réserves

- **Dépendance externe et non déterministe.** Le retrieval dépend désormais d'une API payante. Le cache JSON committé rend les mesures publiées reproductibles à l'identique, mais une **question nouvelle** appelle le modèle, et rien ne garantit qu'un appel futur produira la même réécriture (pas de température fixable sur la série 5 : les paramètres d'échantillonnage sont refusés). Les chiffres de ce rapport sont reproductibles ; le comportement sur une question inédite est stochastique.
- **`n=61`**, et le gain repose largement sur `vocabulaire_courant` (31 questions). Le split de test gelé est le seul juge (§ Clôture).
- **Le mode `remplace` n'est pas retenu mais reste mesuré** : sur un corpus où la réécriture serait moins fiable, conserver la question (`etend`) est le choix robuste.
- **Le reranker reçoit la question ORIGINALE, pas la réécriture** (lecture littérale du brief T5). La combinaison mesurée ci-dessus vaut donc pour cette convention ; soumettre la réécriture au cross-encoder est une variante non mesurée.
- **Coût par question en production** : un appel LLM s'ajoute à chaque requête non cachée. Ce jalon ne mesure pas la latence bout-en-bout d'un déploiement, seulement celle du retrieval.

## Clôture du jalon 3 — dev final et référence gelée

Campagne exécutée par `scripts/cloture_jalon3.py`, dans un seul processus (embedder, reranker et rewriter partagés, ids `par_question` alignés par construction), après **deux contrôles de fraîcheur** passés avant tout engagement sur le split gelé : `hybrid` neutre sur dev redonne exactement `0,672` et la config adoptée au jalon 2.5 redonne exactement `0,738`. Un écart aurait signifié que l'index ou le code avait bougé depuis les mesures publiées, et le script s'arrête dans ce cas.

Trois configurations, qui résument l'histoire du projet :

- **A** — `hybrid` neutre : la baseline du jalon 2.5 ;
- **B** — `hybrid+rerank`, `bge-reranker-v2-m3`, `n_rerank=25` : la config **adoptée au jalon 2.5** ;
- **C** — réécriture `etend` + B : la config **adoptée au jalon 3**.

| split | n | A `hybrid` | B (jalon 2.5) | **C (jalon 3)** | delta C−B | IC95 | p(C>B) | delta C−A | p(C>A) |
|---|---|---|---|---|---|---|---|---|---|
| dev | 61 | 0,672 | 0,738 | **0,877** | +0,1393 | (0,0328 ; 0,2459) | 0,9945 | +0,2049 | 0,9997 |
| **test (gelé)** | 29 | 0,690 | 0,759 | **0,966** | +0,2069 | (0,0690 ; 0,3793) | 0,9984 | +0,2759 | 1,0000 |

Ventilation par catégorie, B → C :

| split | `reference_directe` | `regle` | `vocabulaire_courant` |
|---|---|---|---|
| dev | 1,000 → 1,000 | 1,000 → 0,978 | 0,484 → **0,774** (p = 0,9976) |
| test | 1,000 → 1,000 | 1,000 → 1,000 | 0,500 → **0,929** (p = 0,9994) |

Sur le split gelé, une seule question sur 29 reste imparfaite (`q028`), et **aucune catégorie ne régresse**.

### Lecture honnête : le test est meilleur que le dev, et ce n'est pas une bonne nouvelle en soi

`0,966` sur test contre `0,877` sur dev. Deux choses à en dire, dans cet ordre :

1. **L'effet réplique, franchement.** Le split `test` n'a servi à choisir aucun paramètre — ni seuil, ni mode de réécriture, ni modèle, ni largeur de pool. Il a été gelé le 16 août 2026 et exécuté deux fois en tout : une fois à la clôture du jalon 2.5, une fois ici. Le gain n'est donc pas un sur-ajustement au dev : `p(C>B) = 0,9984` sur des données jamais vues, avec un delta plus grand que sur dev.
2. **Mais `n=29`, et un split de 29 questions se trompe largement.** L'IC95 sur test est (0,069 ; 0,379) — presque deux fois plus large que celui du dev. `0,966` signifie « 28 questions sur 29 » : une seule question de plus en échec ramènerait le chiffre à 0,93. **Le chiffre à citer est celui du dev (0,877)**, pas celui du test : le test confirme la direction, il ne raffine pas l'estimation. La composition des deux splits est comparable (dev 11 / 38 / 51 %, test 10 / 41 / 48 % par catégorie), donc l'écart s'explique par la taille d'échantillon, pas par un split plus facile.

### Comptabilité exacte de la réécriture sur dev

| effet | n | questions |
|---|---|---|
| **améliorées** par la réécriture | **12** | q021, q022, q026, q056, q059, q060, q065, q068, q070, q071, q079, q086 |
| **dégradées** par la réécriture | **3** | q008, q025, q080 |
| résistantes aux deux configs | 4 | q057, q063, q082, q089 |

Net : **+9 questions en compte, +8,5 points de recall** (soit le delta de 0,1393 = 8,5/61). Le mot « améliorées » est délibéré : le compte inclut deux cas partiels — q022 passe de 0,5 à 1,0 et q086 de 0,0 à **0,5** (une question à deux citations dont une seule est retrouvée). Sur les 12, dix sont des réparations complètes. **La réécriture n'est pas gratuite** : elle casse 3 questions qui fonctionnaient, et la catégorie `regle` perd 0,0217 sur dev (une question à deux citations qui n'en garde qu'une). C'est sous la garde de −0,05 du protocole, donc l'adoption tient, mais le mécanisme est réel : réécrire une question qui parlait déjà le bon vocabulaire peut diluer un terme discriminant. Un déploiement soucieux du pire cas pourrait conditionner la réécriture à un signal de recouvrement faible plutôt que l'appliquer systématiquement — piste non mesurée.

### Ce que le jalon 3 laisse en l'état

- **`df_max` REJETÉ** (§ Ablation D) — dégradation monotone.
- **`pool` et `dedup_termes` REJETÉS** (§ Ablation E) — mais le rejet a produit le résultat central du jalon : le découplage couverture/classement.
- **`n_rerank` REJETÉ** (§ Ablation F) — la largeur du pool soumis au cross-encoder n'achète rien en recall@10, pour aucun des deux modèles testés.
- **Réécriture par LLM ADOPTÉE** (§ Ablation G), mode `etend`, `claude-sonnet-5`, `thinking` désactivé.

Les quatre paramètres rejetés restent exposés sur `Searcher` à leur valeur neutre : ils sont mesurés, documentés et re-mesurables, pas supprimés.

### Configuration livrée

```python
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

searcher = Searcher(
    "data/corpus.db",
    reranker=Reranker(),                              # bge-reranker-v2-m3 (jalon 2.5)
    rewriter=Rewriter(cache_path="data/reecritures-cache.json"),   # cache d'EXÉCUTION
    mode_reecriture="etend",                          # jalon 3 (défaut du code)
)
resultats = searcher.search("j'ai payé un logiciel, je fais quoi ?", k=10, mode="hybrid+rerank")
```

Reproduction complète de cette section :

```sh
uv run python scripts/cloture_jalon3.py        # ATTENTION : exécute le split gelé
uv run python scripts/audit_reecritures.py     # contrôle d'intégrité du benchmark
```

`docs/mesures/jalon3/cloture_dev.json` et `cloture_test.json` sont versionnés : tout chiffre de cette section est recalculable sans relancer la campagne.

### Réserves de clôture — à lire avant de citer 0,877 ou 0,966

1. **Le benchmark ne peut pas voir le manque de corpus, et c'est la réserve la plus importante du jalon.** Les 90 questions ont toutes été écrites depuis le PCG : leurs citations attendues existent dans le corpus **par construction**. Une question fiscale (« est-ce déductible ? ») n'a pas de gold, n'est donc pas dans le benchmark, et ne pèse pas dans le score. Le `0,877` mesure la capacité à retrouver ce qui est présent — sur une distribution de questions qui exclut exactement ce qui est absent. Le corpus se limite aux **Livres I à V du règlement ANC 2014-03** : ni consolidation (règlement ANC 2020-01), ni fusions, ni NEP d'audit, ni BOFiP. C'est le périmètre du jalon 4 (BOFiP BIC/IS) puis du jalon 5 (consolidation, fusions, NEP).
2. **Le retrieval dépend désormais d'une API externe et non déterministe.** Le cache committé rend ces chiffres reproductibles à l'identique, mais une question inédite déclenche un appel, et les paramètres d'échantillonnage ne sont pas réglables sur la série 5 : rien ne garantit qu'un appel futur produise la même réécriture. Reproductibilité des mesures publiées ≠ déterminisme en production.
3. **Toutes les latences sont des latences GPU fp32 sur une machine chargée.** Sur CPU, le facteur ~70 documenté ci-dessus rend la config livrée inutilisable en interactif. Le device fait partie des conditions de mesure, pas des détails.
4. **`n=61` sur dev, `n=29` sur test.** Les écarts discutés dans ce rapport valent souvent 2 à 4 questions. Le benchmark doit grandir (150–300 questions, prévu au design) avant que des deltas inférieurs à ~0,05 soient interprétables.
5. **Aucune mesure de génération.** Ce jalon, comme les précédents, mesure exclusivement le **retrieval**. La justesse des réponses, le taux de citations hallucinées et le taux de confusion fiscal/comptable — la métrique signature annoncée au design — ne sont pas encore mesurés.
6. **Le split gelé a été exécuté deux fois en tout** (clôture du jalon 2.5, clôture du jalon 3). Il reste gelé : aucun réglage n'en est dérivé. À la troisième ou quatrième clôture, cette garantie s'usera statistiquement — prévoir un second split de validation avant le jalon 5.
