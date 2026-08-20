# Correctif du jalon 3 — la règle de fusion récompensait le consensus

> Ce document est écrit en deux temps, et l'ordre est le sujet. Tout ce qui suit
> jusqu'au § « Résultats » a été rédigé et committé **avant** la première mesure :
> le levier, les deux grilles, les deux contextes, le critère d'adoption et le
> livrable central. Ce qui vient après est ce que la mesure a rendu.

## Le défaut, tel que le jalon 3 l'avait laissé

Le rapport du jalon 3 nomme un défaut qu'il ne traite pas, et il le nomme bien :
la somme RRF **récompense le consensus, pas l'excellence**. Son cas est q023.
Son gold `pcg-214-22` est le 2ᵉ meilleur candidat lexical sur 1 653, absent du
canal dense. Il perd contre un candidat classé 5ᵉ et 6ᵉ, parce que deux
contributions médiocres additionnées pèsent plus qu'une contribution excellente
isolée. Sur ce pool de 81 candidats, 73 ne sont présents que dans un seul canal
et 8 dans les deux : ces 8 monopolisent le haut du classement.

Le jalon 3 a fermé le dossier sur un argument exact et insuffisant : le reranker
rattrape le cas. Le gold ressort au rang 11 de la fusion, donc à l'intérieur des
`n_rerank=25` candidats soumis au cross-encoder, qui le remonte — q023 vaut 1,0
dans la configuration livrée. L'argument est exact. Il est insuffisant parce
qu'il décrit un rattrapage **sans jamais mesurer sa marge**.

## Ce qui est mesuré

Un scalaire, exposé sur `Searcher`, neutre à sa valeur par défaut :

```
score = max(contributions) + poids_consensus × (somme − max)
```

À `poids_consensus = 1,0`, c'est la somme RRF historique. À 0, seule
l'excellence dans un canal décide, le consensus ne servant plus qu'à départager.

Un second levier est mesuré dans sa propre grille : `rrf_k`, l'escompte de rang,
figé à 60 depuis le jalon 2. Il est mesuré **bien qu'un calcul le disqualifie
d'avance** — sur les rangs persistés de q023, il faudrait `rrf_k ≤ 1` pour que le
gold repasse devant son vainqueur ; à `rrf_k = 2` il reperd
(`tests/test_fusion.py::test_rrf_k_ne_repare_q023_quaux_valeurs_extremes`).
C'est précisément la raison de le mesurer : la loi 6 du dépôt interdit de publier
un mécanisme avant d'exécuter le contrôle qui le tranche, et trois mécanismes
plausibles du jalon 3 étaient faux pendant que leurs conclusions étaient justes.

### Le livrable central n'est pas le recall

Le recall@10 décide de l'adoption, mais il ne mesure pas ce qui est en jeu. Il
dit que le reranker rattrape aujourd'hui ; il ne dit pas de combien. Le livrable
central est donc la **marge avant éviction** : le rang du gold *dans la fusion*,
avant reranking.

- Tant que le gold évincé reste dans les `n_rerank` premiers, le rattrapage tient.
- Au-delà, il ne tient plus — et le jalon suivant multiplie la taille du corpus,
  donc le nombre de candidats consensuels, donc la profondeur de l'éviction.

C'est l'analogue exact de la couverture du pool, vrai livrable de l'ablation E du
jalon 3, dont le recall seul ne disait rien. Une question dont le gold est
**routé** (référence d'article explicite) n'est pas exposée au défaut et est
comptée à part, jamais comme un succès de la fusion. Un gold **absent de la
fusion** compte au-delà de tous les seuils : il est hors de portée du reranker au
même titre qu'un gold au rang 300, et l'omettre ferait sous-estimer précisément
le risque que cette mesure existe pour rendre visible.

## Conditions

| | |
|---|---|
| Corpus et index | inchangés, aucune reconstruction (`data/corpus.db`, Livres I à V du règlement ANC 2014-03) |
| Split | `dev`, **93 questions** (le périmètre du jalon 3 en comptait 61 ; l'extension date du jalon 4) |
| Split gelé | **non touché** — une exécution unique n'aurait lieu qu'en cas d'adoption sur dev |
| Grille `consensus` | `poids_consensus ∈ {1,0 (référence) ; 0,5 ; 0,25 ; 0,10 ; 0,025}` |
| Grille `escompte` | `rrf_k ∈ {60 (référence) ; 20 ; 5 ; 1}` |
| Contexte `hybrid` | fusion nue, sans réécriture ni reranking — le mécanisme y est visible |
| Contexte `livree` | la configuration livrée au jalon 3 : réécriture `etend` + `hybrid+rerank`, `bge-reranker-v2-m3`, `n_rerank=25`, `pool=50` — celle qui décide de l'adoption |
| Critère d'adoption | `p_amelioration ≥ 0,95` sur recall@10 (bootstrap apparié, `n_boot=10000`, `seed=42`) **et** aucune catégorie perdant plus de 0,05 |
| Coût API | **zéro** — réécritures lues en lecture seule dans l'ancrage versionné du jalon 4 |
| Machine | Quadro RTX 3000 Max-Q, pilote 610.57.04, CUDA 13.3 |

Les deux grilles ne sont **jamais combinées**. Une configuration mêlant les deux
leviers ne serait mesurée que si l'un des deux était adopté seul — même règle
qu'à l'ablation E du jalon 3.

## Contrôles exécutés avant toute mesure

**Fraîcheur, deux configurations publiées, sur le périmètre du jalon 3** — lu
dans le JSON qui porte le chiffre, jamais figé en constante (un compte figé
mesurerait la taille du benchmark et non le périmètre audité, défaut corrigé au
jalon 4) :

| configuration | périmètre | publié | obtenu |
|---|---|---|---|
| `hybrid` neutre | 61 questions | 0,672 | **0,672** |
| livrée jalon 3 | 61 questions | 0,877 | **0,877** |

Ces deux contrôles valident aussi une réécriture de code faite pour ce
correctif : `search()` délègue désormais à `avant_rerank()`, qui rend le
classement avant reranking. L'extraction est neutre sur le corpus réel, et pas
seulement en test synthétique.

**Mutations de la règle de fusion** (loi 5 : un contrôle que personne n'a vu
échouer ne prouve rien). Six mutations appliquées, chacune vérifiée fatale, puis
restaurées :

| mutation | attrapée |
|---|---|
| `poids_consensus` ignoré (toujours 1,0) | ✔ |
| `max` remplacé par `min` | ✔ |
| formule ramenée à la somme nue | ✔ |
| poids appliqué à la somme entière | ✔ |
| `rrf_k` non transmis par `search()` | ✔ |
| `poids_consensus` non transmis par `search()` | ✔ |

**Le contrôle de fraîcheur lui-même est falsifiable** :
`tests/test_ablations_fusion.py` le fait échouer de deux façons — un chiffre qui
dérive de 0,001, et une question du périmètre publié disparue du benchmark.

## Un court-circuit prévu au design, retiré après vérification

Le design de ce correctif prévoyait un court-circuit `if poids_consensus == 1.0:`
renvoyant au code historique, au motif que `max + 1,0 × (somme − max)` ne serait
pas bit à bit égal à `somme` en virgule flottante — et l'égalité bit à bit
importe : un écart d'un ULP peut faire basculer un ex aequo, et le contrôle de
fraîcheur à 0,672 se mettrait à mentir par intermittence.

Vérification exhaustive sur tout le domaine que ce code peut atteindre — deux
canaux, rangs 0 à 400, `rrf_k ∈ {60, 20, 5, 1, 0}`, soit 802 000 couples :
**aucune divergence**. L'identité tient parce que la fusion ne porte que deux
canaux, où `somme − max` redonne exactement le second terme.

Le court-circuit a donc été retiré plutôt qu'écrit : aucune mutation n'aurait pu
le mettre en défaut, et une branche qu'aucun test ne peut faire échouer ne
protège rien — même raison que la garde morte retirée de `citations.py` au
jalon 4. La propriété est devenue un test, ce qui la rend falsifiable au lieu de
la rendre supposée. **Elle ne vaut que pour deux canaux** : un troisième devra
ré-exécuter ce test avant de s'y fier.
