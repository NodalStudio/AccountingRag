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

---

## Résultats

### Verdict

**Aucun réglage n'est adopté. Les deux leviers restent à leur valeur neutre**
(`poids_consensus=1.0`, `rrf_k=60`), donc `Searcher()` sans argument reproduit
exactement la configuration livrée au jalon 3.

Et pourtant le levier fonctionne. Les deux affirmations tiennent ensemble, et
c'est tout le résultat de ce correctif.

### Grille — contexte `hybrid` (le mécanisme)

93 questions dev, fusion nue, sans réécriture ni reranking.

| configuration | recall@10 | delta | IC95 | p_amélioration | pire catégorie | rang de q023 |
|---|---|---|---|---|---|---|
| référence (1,0 / 60) | 0,715 | — | — | — | — | **11** |
| `poids_consensus=0,5` | 0,726 | +0,0108 | [0,0 ; 0,0323] | 0,6294 | 0,0 | 10 |
| `poids_consensus=0,25` | 0,737 | +0,0215 | [0,0 ; 0,0538] | 0,8695 | 0,0 | 6 |
| `poids_consensus=0,10` | **0,747** | +0,0323 | [0,0 ; 0,0753] | **0,9537** | 0,0 | 4 |
| `poids_consensus=0,025` | 0,720 | +0,0054 | [−0,043 ; 0,0538] | 0,5341 | −0,0122 | 3 |
| `rrf_k=20` | 0,737 | +0,0215 | [0,0 ; 0,0538] | 0,8695 | 0,0 | 8 |
| `rrf_k=5` | **0,747** | +0,0323 | [0,0 ; 0,0753] | **0,9537** | 0,0 | 4 |
| `rrf_k=1` | 0,737 | +0,0215 | [−0,0215 ; 0,0645] | 0,7751 | 0,0 | 3 |

Le mécanisme est net et ordonné : moins le consensus pèse, plus le recall monte,
jusqu'à +0,0323 — trois questions sur 93, **sans en casser une seule**. Les
questions réparées à `poids_consensus=0,10` sont **q023, q054 et q1032**, et la
liste des perdues est vide. Puis le gain retombe à 0,025 : trop près du maximum
pur, le départage par la somme ne suffit plus, et trois questions se cassent
(q022, q067, q1007) pour trois réparées.

### Grille — contexte `livree` (celle qui décide)

Configuration livrée au jalon 3 : réécriture `etend` + `hybrid+rerank`,
`bge-reranker-v2-m3`, `n_rerank=25`, `pool=50`.

| configuration | recall@10 | delta | IC95 | p_amélioration | pire catégorie |
|---|---|---|---|---|---|
| référence (1,0 / 60) | 0,866 | — | — | — | — |
| `poids_consensus=0,5` | 0,866 | +0,0000 | [0,0 ; 0,0] | 0,0000 | 0,0 |
| `poids_consensus=0,25` | 0,871 | +0,0054 | [0,0 ; 0,0161] | 0,6356 | 0,0 |
| `poids_consensus=0,10` | 0,871 | +0,0054 | [0,0 ; 0,0161] | 0,6356 | 0,0 |
| `poids_consensus=0,025` | 0,866 | +0,0000 | [−0,0161 ; 0,0161] | 0,3441 | 0,0 |
| `rrf_k=20` | 0,871 | +0,0054 | [0,0 ; 0,0161] | 0,6356 | 0,0 |
| `rrf_k=5` | 0,871 | +0,0054 | [0,0 ; 0,0161] | 0,6356 | 0,0 |
| `rrf_k=1` | 0,871 | +0,0054 | [0,0 ; 0,0161] | 0,6356 | 0,0 |

Le meilleur réglage gagne **une demi-question sur 93** (`delta=0,0054`), à
`p=0,6356`. Le critère demande 0,95. Ce n'est pas un échec de peu : c'est un
effet six fois plus petit que dans le contexte nu.

> **Deux `p` à ne pas lire de travers.** `poids_consensus=0,5` affiche
> `p=0,0000` avec un delta nul, et `poids_consensus=0,025` affiche `p=0,3441`
> avec un delta nul lui aussi. Le bootstrap compte les rééchantillons dont la
> moyenne des deltas est **strictement** positive. Le premier cas a tous ses
> deltas par question exactement à zéro — vecteurs identiques, donc aucun
> rééchantillon positif, donc `p=0`. Le second a des gains et des pertes qui
> s'annulent (q008 gagnée, q1022 perdue). `p=0` dit « rien n'a bougé », pas
> « c'est pire ».

### Pourquoi : le reranker rattrape exactement les questions que la fusion réparerait

Le mécanisme n'est pas invoqué, il est nommé. Les trois questions que le levier
répare en fusion nue sont **déjà toutes à 1,0 dans la configuration livrée** :

| question | `hybrid` référence | `hybrid` à 0,10 | livrée référence | livrée à 0,10 |
|---|---|---|---|---|
| q023 | 0,0 | 1,0 | **1,0** | 1,0 |
| q054 | 0,0 | 1,0 | **1,0** | 1,0 |
| q1032 | 0,0 | 1,0 | **1,0** | 1,0 |

Le cross-encoder les remonte déjà. Réparer la fusion ne peut donc rien leur
ajouter — on ne répare pas deux fois la même question.

La seule question qui bouge dans la configuration livrée est **q008**, et elle
bouge d'une demi-réparation (0,5 → 1,0 : deux citations attendues, une seule
retrouvée jusqu'ici). Elle ne bouge PAS en fusion nue, où elle reste à 0,5. Ce
n'est donc pas la fusion qui la répare, c'est le changement de composition des 25
candidats soumis au cross-encoder — un effet indirect, et le seul du contexte
livré.

### La coïncidence de la grille, tranchée

`poids_consensus=0,10` et `rrf_k=5` rendent le même recall, le même delta et le
même `p` à la quatrième décimale ; idem pour `poids_consensus=0,25` et
`rrf_k=20`. Conclure « les deux leviers sont équivalents » aurait été **faux**.

L'anatomie compare les deux vecteurs séparément
(`docs/mesures/jalon3-fix/anatomie_dev.json`) :

- **vecteurs de recall identiques** : `[poids_consensus=0,10 ; rrf_k=5]` et
  `[poids_consensus=0,25 ; rrf_k=20]` — d'où les `p` identiques, le bootstrap ne
  voyant que ces vecteurs ;
- **classements identiques : aucun groupe.** Les deux leviers ne produisent pas
  le même ordre des candidats. Ils font simplement basculer les mêmes questions
  au-dessus du seuil du top-10.

La métrique est trop grossière pour les distinguer ; les rangs ne le sont pas.

### q023, le cas fondateur, et une précision sur mon propre calcul

Le rang de son gold dans la fusion descend de façon monotone quand le consensus
pèse moins : **11 → 10 → 6 → 4 → 3**. La référence redonne exactement **11**, le
rang publié au jalon 3.

Il faut être précis sur ce que cela prouve, parce que le protocole ci-dessus
annonçait autre chose. J'avais calculé, sur les rangs persistés, qu'il faudrait
`poids_consensus < 0,05` pour que le gold repasse devant son vainqueur
`pcg-na-236` (5ᵉ et 6ᵉ). **Ce calcul reste juste** : le gold ne passe devant lui
qu'à 0,025, où il atteint le rang 3. Mais il entre dans le top-10 dès 0,5, parce
qu'il n'a pas besoin de battre `pcg-na-236` — il lui suffit que les nombreux
autres candidats bi-canaux perdent du terrain. **Battre un rival nommé et entrer
dans le top-10 sont deux choses différentes**, et ma prédiction ne portait que
sur la première.

Ma conclusion sur `rrf_k`, elle, était fausse. Le protocole le disait
« disqualifié d'avance » parce qu'il faudrait `rrf_k ≤ 1` pour renverser le duel.
Le duel, oui ; la grille, non : `rrf_k=5` fait exactement aussi bien que le
meilleur poids. J'avais généralisé à une grille un calcul portant sur une
question. C'est précisément pour cela qu'il était mesuré.

### Le livrable central : la marge avant éviction, décomposée

La lecture brute — « la part de golds au-delà du rang 25 ne bouge pas » —
mélangerait deux causes sans rapport. Un gold **absent du pool** n'a pas été
évincé par la fusion : aucune règle ne peut classer un candidat qui n'est pas là.

Décomposition sur les 93 questions dev (identique pour toutes les configurations
d'un contexte, la fusion ne changeant pas la composition du pool) :

| | `hybrid` | livrée |
|---|---|---|
| questions routées (référence d'article explicite, hors défaut) | 15 | 15 |
| questions exposées à la fusion | 78 | 78 |
| gold **absent du pool** (défaut de couverture) | **14** | **4** |
| gold présent dans la fusion | 64 | 74 |
| — dont au-delà du rang 25 (défaut de classement) | **3** | **2** |
| rang médian des golds présents | 1 | 1 |
| rang maximal d'un gold présent | 90 | 54 |

Le chiffre brut de 0,2179 en fusion nue se décompose donc en 14 + 3 sur 78 : il
est à **82 % un défaut de couverture**, que ce correctif ne touche pas. La
réécriture, elle, le divise par plus de trois (14 → 4 golds absents) — c'est le
gain du jalon 3, relu par une autre métrique.

**Ce qu'il faut retenir pour la suite : dans la configuration livrée, deux
questions seulement sur 78 ont leur gold dans la fusion mais hors de la fenêtre
du reranker** — q057 au rang 54 et q1009 au rang 40. Le rattrapage a donc
aujourd'hui une marge large. C'est cette marge, et non le recall, qu'il faudra
re-mesurer quand le corpus grandira : `n_rerank=25` couvre confortablement un
corpus de 1 660 records, et rien ne dit qu'il en couvrira dix fois plus.

### Recoupement avec le jalon 4

Les quatre questions dont le gold est absent du pool en configuration livrée sont
**q080, q089, q1008, q1031**. Deux d'entre elles — **q089 et q1031** — sont
exactement deux des quatre abstentions que le générateur a produites au jalon 4
et que j'y avais jugées bien fondées, en constatant que le gold ne figurait pas
dans les dix passages fournis.

Les deux mesures sont indépendantes : l'une regarde ce que le générateur répond,
l'autre où le gold se situe dans le pool. Elles nomment les mêmes questions. Une
abstention correcte du générateur y est donc bien la conséquence d'un défaut de
retrieval, et non d'une prudence excessive du modèle.

## Réserves

1. **Les latences de cette campagne ne sont comparables à rien.** La
   configuration livrée y met **12,28 s/question** contre 1,87 s publiées au
   jalon 3 — même machine, même carte, deux jours d'écart. Le GPU est resté
   bloqué en état P8 à 300 MHz sur 2100 pendant toute la campagne, drapeaux
   `SW Power Cap` et `SW Thermal Slowdown` actifs en continu à 50 °C et 13,7 W
   sur une enveloppe de 30 W. C'est la loi 7 du dépôt prise en flagrant délit :
   une latence n'est pas une propriété du système mais du triplet (machine,
   device, charge). Aucun recall n'en dépend. La fusion nue est à
   0,035 s/question dans les mêmes conditions.
2. **`n=93`, et les écarts discutés valent une à trois questions.** Le delta de
   0,0054 du contexte livré est une demi-question. Aucune conclusion de ce
   rapport ne repose sur un écart d'une seule question.
3. **Le split gelé n'a pas été exécuté**, conformément au protocole : une
   exécution unique n'était prévue qu'en cas d'adoption sur dev. Aucun réglage
   n'ayant été adopté, `benchmark/test.jsonl` reste à deux exécutions, celles des
   jalons 2.5 et 3.
4. **Les deux grilles ne sont pas combinées.** Aucun levier n'ayant été adopté
   seul, mesurer leur combinaison n'apporterait rien — même règle qu'à
   l'ablation E du jalon 3.
5. **`poids_consensus=0,10` franchit le seuil dans le contexte `hybrid`
   (p=0,9537), et cela n'est pas une adoption.** Le contexte qui décide était
   fixé avant mesure, et la fusion nue n'est exécutée par personne : ni la démo,
   ni les campagnes, ni le jalon 4. Substituer après coup le contexte qui arrange
   le résultat serait la même faute que substituer la métrique qui l'arrange.
6. **La grille est fine près du neutre et grossière près de zéro.** Entre 0,10 et
   0,025 le recall passe par un maximum non localisé ; il pourrait exister un
   réglage meilleur que 0,10 dans cet intervalle. Le chercher sur dev
   reviendrait à régler un paramètre sur le split de mesure, et le gain visé —
   quelques dixièmes de question dans le contexte qui décide — ne le justifie pas.
7. **Ce correctif ne touche pas la couverture du pool**, qui est la cause de
   82 % des golds hors de portée en fusion nue. C'est le domaine du corpus et de
   la réécriture, pas de la règle de fusion.

## Ce que ce correctif laisse au jalon suivant

Le levier est mesuré, exposé, neutre, et documenté. Il est prêt à être
ré-évalué le jour où l'hypothèse qui le rend inutile cesse de tenir — c'est-à-dire
le jour où le reranker ne rattrapera plus tout seul. La mesure à refaire est la
marge avant éviction, pas le recall : le recall ne dira qu'après coup que la
compensation a cédé.

Les deux questions déjà hors fenêtre (q057 au rang 54, q1009 au rang 40) sont les
témoins à surveiller.
