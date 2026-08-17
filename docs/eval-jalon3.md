# Jalon 3 — Recouvrement lexical nul : diagnostic fondateur, ablation D (T1) et ablation E (T2)

Campagne exécutée le **16 août 2026** sur la branche `jalon-3-recouvrement-nul`, à l'issue de la tâche T1 (filtrage des tokens peu discriminants de la requête lexicale, `df_max`, plus les deux outils de mesure versionnés du jalon : `scripts/diagnostic_rangs.py`, `scripts/ablations_jalon3.py`) puis de la tâche T2 (largeur du pool de candidats avant fusion, `pool`, et déduplication des termes du `MATCH`, `dedup_termes`). Objectif T1 : reproduire par un outil versionné le diagnostic fondateur du jalon (deux sondes jetables du contrôleur, exécutées avant la rédaction du plan) puis mesurer par bootstrap apparié si le filtrage par fréquence documentaire (`df_max`) répare, seul, une partie du recouvrement lexical nul observé sur `vocabulaire_courant`. Objectif T2 : mesurer si élargir le pool de candidats avant fusion RRF (ou dédupliquer les termes du `MATCH`) répare une partie de ce recouvrement nul, et — livrable central de T2 — établir la **couverture du pool** (part des questions dont au moins une citation gold est présente dans le pool avant fusion) à chaque largeur, la métrique qui conditionne l'ablation F (reranking).

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
- **q021** : rang lexical **154**/1585 — hors de la fenêtre par défaut, mais à portée d'un pool élargi (`pool≥154`, donc `pool=200` suffit) : confirmé à la mesure T2 (couvert dès `pool=200`, absent à 50 et 100).
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
   - À pool=50 : le record réglementaire `pcg-122-3` (le gold) est retrouvé UNIQUEMENT par bm25 (rang 0, absent du top-50 dense) → score RRF = `1/61 ≈ 0,01639`, 3e position du top-10 fusionné. Un chunk de commentaire du même article (`pcg-122-3-c1`, qui matche aussi la citation gold) est présent côté dense (rang 21, hors bm25) mais son score RRF solo (`1/82 ≈ 0,01220`) est trop faible pour entrer dans le top-10 — sans effet sur le résultat.
   - À pool=100 : le score RRF de `pcg-122-3` reste `0,01639` (aucun changement de ses rangs individuels) et celui de `pcg-122-3-c1` reste `0,01220` (toujours sous le seuil, vérifié) — mais deux nouveaux candidats, absents à pool=50, entrent maintenant dans LES DEUX canaux à des rangs médiocres (ex. `pcg-na-18` : bm25 rang 70, dense rang 2 → RRF = `1/131 + 1/63 ≈ 0,02351` ; `pcg-na-24` : bm25 rang 48, dense rang 9 → RRF ≈ `0,02346`) — supérieurs à la fois à `pcg-122-3` et à `pcg-122-3-c1`. Le 10e score du top-10 à pool=100 vaut `0,01740` : `pcg-122-3` (0,01639) en sort, `pcg-122-3-c1` (0,01220) n'y était déjà pas. Leur somme de deux contributions MÉDIOCRES (un canal moyen + un canal moyen) dépasse la contribution UNIQUE mais meilleure du gold (un canal excellent, l'autre absent), qui sort du top-10.

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
