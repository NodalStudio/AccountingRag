# Second correctif du jalon 3 — la réécriture qui cassait trois questions

> Le protocole de ce correctif — levier, grille, contextes, critère, et une
> **prédiction** — a été committé dans `scripts/ablations_reecriture.py` avant la
> première mesure. « Fixé d'avance » se vérifie donc dans git, pas dans cette prose.

## Le défaut, et pourquoi la piste du jalon 3 ne pouvait pas le réparer

Le jalon 3 a adopté la réécriture de requête sur un gain net mesuré — douze
questions réparées, trois cassées — et a laissé les trois en dette avec une
piste : « conditionner la réécriture à un signal de recouvrement faible plutôt
que l'appliquer systématiquement ».

**Cette piste ne correspond pas au mécanisme**, et c'est la première chose que ce
correctif a établie, avant d'écrire une ligne de code. q080 est la question la
plus familière du benchmark — « j'ai payé plus cher les actions d'une boîte […]
et je viens de l'avaler complètement » — donc n'importe quelle porte fondée sur le
recouvrement déclencherait la réécriture précisément là où elle casse.

### Trois questions, trois mécanismes

La phrase « la réécriture dégrade trois questions » en masquait trois. Rangs
mesurés sur `data/corpus.db`, `pool=400` :

| question | canal | sans réécriture | avec réécriture (`etend`) | conséquence |
|---|---|---|---|---|
| **q080** | bm25 | 7 | **84** | le gold sort du pool de 50 — hors de portée de tout reranker |
| **q025** | bm25 | 1 | 13 | gold au rang 22 de la fusion : le reranker le voit et ne le retient pas |
| **q008** | bm25 | 1 | 1 | rang inchangé ; sa dégradation relève de l'éviction par la fusion |

Le canal dense est hors de cause : il ne trouve **jamais** le gold de q080,
réécriture ou non (rang `None` jusqu'à 400). Le dégât est entièrement lexical.

q008 est le recoupement entre les deux dettes du jalon 3 : c'est exactement la
question que le levier de fusion réparait dans la configuration livrée.

### Une hypothèse séduisante et fausse, écartée par contrôle

La réécriture de q080 parle **consolidation** : ses trois premiers termes sont
« écart d'acquisition », « goodwill », « regroupement d'entreprises ». Or ces
trois termes figurent dans **zéro** record du corpus, tandis que le gold
`pcg-745-3` s'intitule « Traitement du mali pour les opérations évaluées à la
valeur comptable » et que le corpus porte « mali de fusion » (10 records) et
« mali technique » (18). Le modèle a donné la réponse des comptes consolidés
(règlement ANC 2020-01, hors corpus) là où la question est une fusion (Titre VII
du règlement ANC 2014-03, dans le corpus).

C'est une vraie confusion comptable, et ce n'est **pas** le mécanisme de l'échec :
le canal dense, seul à pouvoir souffrir d'un vocabulaire absent de l'index, ne
trouvait déjà pas ce gold. Un terme de fréquence documentaire nulle ne pèse rien
en BM25. La conclusion tentante aurait été juste sur les faits et fausse sur la
cause.

## Le levier

`poids_question` : le nombre de répétitions des tokens de la question originale
devant la réécriture, dans la requête **lexicale**. Neutre à **1**, soit le mode
`etend` du jalon 3 exactement.

Il exploite une propriété déjà mesurée au jalon 3 : `bm25()` de FTS5 pondère la
**multiplicité** des termes de l'expression `MATCH`, et non le seul ensemble des
lignes sélectionnées. C'est la seule pondération par terme que FTS5 offre — d'où
un entier plutôt qu'un flottant.

**Canal lexical seulement**, par contrainte de méthode et non par commodité :
répéter la question dans le texte soumis au canal dense déplacerait aussi
l'embedding, donc mesurerait deux variables sous un seul nom (loi 2). Deux tests
d'espionnage échouent si quelqu'un réunifie les deux requêtes.

Sonde préalable — rang bm25 du gold selon le nombre de répétitions :

| question | n=0 | **n=1** | n=2 | n=3 | n=5 | n=8 | question seule |
|---|---|---|---|---|---|---|---|
| q080 | 280 | **84** | 26 | 12 | 5 | 6 | 7 |
| q025 | 14 | **13** | 10 | 6 | 3 | 2 | 1 |
| q021 *(réparée par la réécriture)* | 7 | **4** | 4 | 7 | 18 | 51 | 154 |
| q060 *(réparée)* | 3 | **11** | 19 | 42 | 159 | None | None |
| q070 *(réparée)* | 10 | **10** | 11 | 13 | 31 | 46 | 136 |

L'arbitrage est réel et monotone : ce qui répare q080 détruit q060. Grille fixée
d'avance sur ce domaine : **{1 (référence), 2, 3, 5, 8}**.

## Conditions

| | |
|---|---|
| Corpus et index | inchangés, aucune reconstruction |
| Split | `dev`, 93 questions ; split gelé **non touché** |
| Contexte `reecriture` | réécriture `etend` **sans reranking** — le contexte de l'ablation G du jalon 3, où le mécanisme est visible. Le levier serait inerte dans `hybrid` nu, faute de réécriture à pondérer |
| Contexte `livree` | la configuration livrée au jalon 3 — celle qui décide |
| Critère d'adoption | `p_amelioration ≥ 0,95` sur recall@10 (bootstrap apparié, `n_boot=10000`, `seed=42`) **et** aucune catégorie perdant plus de 0,05 |
| Coût API | **zéro** — réécritures lues en lecture seule dans l'ancrage du jalon 4 |

**Contrôles de fraîcheur, avant toute mesure** — chiffres lus dans le JSON qui les
porte, jamais figés en constante :

| configuration | périmètre | publié | obtenu |
|---|---|---|---|
| ablation G, réécriture `etend` | 61 questions | 0,852 | **0,852** |
| livrée jalon 3 | 61 questions | 0,877 | **0,877** |

Six mutations du levier ont été appliquées, chacune vérifiée fatale, puis
restaurées : levier ignoré, appliqué aussi au canal dense, appliqué au dense
seulement, décalé d'une répétition, et les deux gardes de validation retirées.

---

## Résultats

### Verdict

**Aucun réglage n'est adopté. Le levier reste à sa valeur neutre** (`poids_question=1`).

Et c'est le refus le plus coûteux de ce dépôt à ce jour : `poids_question=3`
produit **0,898 de recall@10 sur dev, le meilleur chiffre jamais mesuré par ce
projet**, sans qu'aucune catégorie ne perde quoi que ce soit, sans coût de
latence, et en ramenant à **zéro** le nombre de golds hors de portée du reranker.

`p_amelioration = 0,8809`. Le critère demande 0,95. Il a été fixé avant la mesure.

### Grille — contexte `reecriture` (le mécanisme)

93 questions dev, réécriture `etend`, sans reranking.

| configuration | recall@10 | delta | p_amélioration | pire catégorie | golds hors du pool | au-delà de 25 |
|---|---|---|---|---|---|---|
| référence (n=1) | 0,855 | — | — | — | 4 | 2 |
| `poids_question=2` | 0,866 | +0,0108 | 0,6288 | 0,0 | 4 | 1 |
| `poids_question=3` | 0,855 | +0,0000 | 0,3488 | 0,0 | 4 | **0** |
| `poids_question=5` | 0,844 | −0,0108 | 0,2828 | −0,0244 | 4 | 1 |
| `poids_question=8` | 0,839 | −0,0161 | 0,2515 | −0,0366 | 5 | 0 |

Seul, le levier ne vaut presque rien : +0,0108 au mieux, soit une question sur 93,
puis dégradation franche. Les questions qu'il fait basculer ne sont même pas
celles qu'il visait — à n=3 il répare q027 et casse q060.

### Grille — contexte `livree` (celle qui décide)

| configuration | recall@10 | delta | IC95 | p_amélioration | pire catégorie | au-delà de 25 |
|---|---|---|---|---|---|---|
| référence (n=1) | 0,866 | — | — | — | — | 2 |
| `poids_question=2` | 0,887 | +0,0215 | [0,0 ; 0,0538] | 0,8620 | 0,0 | 1 |
| **`poids_question=3`** | **0,898** | **+0,0323** | [−0,0108 ; 0,086] | **0,8809** | 0,0 | **0** |
| `poids_question=5` | 0,887 | +0,0215 | [−0,0323 ; 0,0753] | 0,7577 | 0,0 | 1 |
| `poids_question=8` | 0,876 | +0,0108 | [−0,0484 ; 0,0699] | 0,6119 | −0,0122 | 0 |

L'intervalle de confiance du meilleur réglage **contient zéro**. C'est la forme
statistique du refus : trois questions de gain sur 93, avec une dispersion qui
n'exclut pas l'absence d'effet.

### Le reranker amplifie ce levier, au lieu de l'absorber

Le premier correctif du jalon 3 avait montré que le reranker **absorbe**
intégralement une amélioration de la règle de fusion, parce qu'il rattrapait déjà
les questions concernées. Le protocole de ce correctif-ci prédisait le contraire,
pour une raison structurelle : la fusion réordonne l'intérieur du pool, ce levier
change sa **composition**, et un candidat absent du pool est hors de portée de
tout reranker.

La prédiction tient, et le contrôle est net. Les quatre questions que le levier
répare dans la configuration livrée à n=3 ne sont **pas** réparées par le levier
seul :

| question | mécanisme, réf | mécanisme, n=3 | livrée, réf | livrée, n=3 |
|---|---|---|---|---|
| q025 | 0,0 | **0,0** | 0,0 | **1,0** |
| q080 | 0,0 | **0,0** | 0,0 | **1,0** |
| q1009 | 0,0 | **0,0** | 0,0 | **1,0** |
| q1028 | 1,0 | 1,0 | 0,0 | **1,0** |

Le levier n'en répare aucune. Il amène leur gold assez près pour que le
cross-encoder finisse le travail. C'est une collaboration entre deux briques, et
elle explique pourquoi l'effet est **trois fois plus grand** dans la configuration
livrée (+0,0323) que dans le contexte mécanisme (+0,0108) — l'inverse exact du
correctif précédent.

### Le livrable central : la marge avant éviction

C'est ici que le levier convainc le plus, et c'est ici que le critère
d'adoption ne regarde pas.

Sur les 93 questions dev, 15 sont routées et 78 exposées. Configuration livrée :

| | référence | `poids_question=3` |
|---|---|---|
| golds absents du pool | 4 | 4 |
| golds présents mais au-delà de `n_rerank=25` | **2** | **0** |
| rang maximal d'un gold présent | **54** | **22** |
| part au-delà de 10 | 0,1538 | — |

À n=3, **tout gold présent dans la fusion est dans la fenêtre du reranker.** Les
deux témoins que le premier correctif avait désignés à surveiller — q057 au rang
54 et q1009 au rang 40 — n'y sont plus.

**Mais le compte de golds hors du pool ne bouge pas, et le lire seul serait une
faute.** Il reste à 4 parce que la composition change entièrement :

| | golds hors du pool |
|---|---|
| référence | q080, q089, q1008, q1031 |
| `poids_question=3` | **q057**, q089, q1008, q1031 |

q080 rentre — c'est le but du levier, atteint — et **q057 sort**. Une métrique
publiée sans sa composition aurait dit « rien n'a changé » au sujet exact de ce
correctif. C'est le même piège que la part au-delà de 25 du premier correctif,
sous une autre forme : un agrégat stable peut recouvrir une substitution complète.

### Ce que le refus coûte, mesuré

Pour que le refus soit lisible, voici ce qui est laissé sur la table à n=3 dans la
configuration livrée :

| | référence | `poids_question=3` |
|---|---|---|
| recall@10 | 0,866 | **0,898** |
| recall@5 | 0,785 | 0,806 |
| MRR | 0,744 | 0,755 |
| catégorie `regle` | 0,959 | 0,986 |
| catégorie `vocabulaire_courant` | 0,732 | 0,780 |
| catégorie `reference_directe` | 1,0 | 1,0 |
| latence | 1,56 s/question | 1,59 s/question |
| questions gagnées / perdues | — | **+4 / −1** |

Les quatre gagnées sont q025, q080, q1009, q1028 ; la perdue est q065. Deux des
trois questions que la réécriture cassait sont donc réparées, et la troisième
(q008) relève de l'autre dette.

Rien de tout cela n'autorise l'adoption. Le critère porte sur `p_amelioration` au
recall@10, il a été fixé avant la mesure, et substituer après coup la métrique qui
arrange le résultat est précisément ce que ce dépôt s'interdit — d'autant plus
quand le résultat est flatteur.

## Réserves

1. **`n=93` limite la puissance statistique, et c'est probablement ce qui décide
   ici.** Un delta de +0,0323 vaut trois questions ; `p=0,8809` avec un IC95 qui
   contient zéro est le comportement attendu d'un effet réel mais petit sur un
   échantillon de cette taille. Ce levier mérite d'être re-soumis au même critère
   quand `dev` aura grandi — sans toucher à sa grille, et sans le mesurer à
   nouveau entre-temps pour « voir ».
2. **Je n'ai pas cherché n=4.** Le recall passe par un maximum entre 3 et 5, et un
   réglage intermédiaire pourrait franchir 0,95. Le chercher maintenant serait
   régler un paramètre sur le split de mesure pour atteindre un seuil — la forme
   la plus directe du sur-ajustement que ce dépôt interdit.
3. **La grille est entière par nécessité, pas par choix.** FTS5 ne pondère les
   termes que par multiplicité ; il n'existe pas de `poids_question=1,5`.
4. **Le split gelé n'a pas été exécuté**, le protocole ne le prévoyant qu'en cas
   d'adoption. `benchmark/test.jsonl` reste à deux exécutions.
5. **Le levier ne corrige pas la réécriture, il la contrepèse.** La réécriture de
   q080 reste fausse pour ce corpus — elle répond consolidation à une question de
   fusion. Corriger cela demanderait de changer le prompt du rewriter, donc
   d'invalider le cache versionné de 93 réécritures et de repayer une campagne.
   C'est un autre travail, avec son propre coût et son propre protocole.
6. **q008 n'est pas traitée ici** et ne peut pas l'être : son rang lexical ne bouge
   pas. Elle relève de l'éviction par la fusion, mesurée et non adoptée dans
   l'autre correctif.
7. **Les latences de ce rapport ne sont pas comparables à celles du premier
   correctif** (12,28 s/question), mesurées pendant que le GPU était bloqué à
   300 MHz sur 2100. Elles le sont en revanche à celles du jalon 3 (1,87 s/question
   sur 61 questions), ce qui est cohérent avec les 1,56 s mesurées ici.

## Ce que les deux correctifs laissent au jalon suivant

Les deux dettes du jalon 3 sont maintenant **mesurées, exposées à valeur neutre, et
non adoptées** — un résultat négatif chacune, pour deux raisons opposées. La
fusion échoue parce que le reranker fait déjà son travail ; la réécriture échoue
de peu, parce que l'échantillon est trop petit pour trancher un effet réel.

Le levier qui mérite d'être repris en premier au jalon suivant est
`poids_question=3`, et la mesure à refaire est la même, sur un `dev` plus grand.
