# Jalon 4 — mesurer la génération : design

Date : 17 août 2026. Décidé après la clôture du jalon 3 (recall@10 0,877 sur dev,
0,966 sur le split gelé).

## 1. Le problème, en trois faits mesurés

Trois jalons ont amélioré le *retrieval*. Aucun n'a mesuré ce que le système
**répond**. Trois constats du jalon 3 rendent le corpus prématuré et l'instrument
de mesure urgent :

1. **Le benchmark sature.** 0,966 sur le split gelé, c'est 28 questions sur 29. Un
   instrument qui rend 28/29 ne peut plus mesurer un progrès, et à n=61 sur dev un
   delta inférieur à ~0,05 n'est pas interprétable (`docs/eval-jalon3.md`,
   réserve 4).
2. **Le benchmark ne peut pas voir ce qui manque au corpus.** Ses 90 questions ont
   toutes été écrites depuis le PCG, donc leurs citations attendues existent *par
   construction*. Une question fiscale n'a pas de gold, n'est pas dans le
   benchmark, et ne pèse pas dans le score (réserve 1).
3. **Aucune mesure de génération.** La justesse des réponses, le taux de citations
   hallucinées et le taux de confusion fiscal↔comptable — la métrique signature
   annoncée au design v1 — ne sont pas mesurés (réserve 5).

Corollaire qui fixe l'ordre des jalons : **sans cet instrument, l'ingestion du
BOFiP serait invérifiable.** Aucune question fiscale n'ayant de gold, le score ne
bougerait pas et on aurait ajouté une capacité qu'on ne sait pas juger.

## 2. Objectif

Rendre mesurable ce que le système répond, et pas seulement ce qu'il retrouve.

**Dans le périmètre :** un générateur minimal contraint, la vérification
programmatique des citations, un LLM-juge *calibré*, la famille de questions
d'abstention, et l'extension du benchmark sur ce que le corpus couvre déjà.

**Hors périmètre, explicitement :** toute nouvelle source de corpus (BOFiP, CGI,
consolidation, NEP — jalons suivants), tout changement du retrieval adopté au
jalon 3, toute reconstruction d'index.

## 3. Quatre briques, ordonnées pour être mesurables séparément

### 3.1 Le générateur, minimal et contraint

`src/accounting_rag/generate.py`. Entrée : une question et les passages remontés
par `Searcher`. Sortie : **exactement deux formes possibles**, jamais autre chose.

- une réponse dont chaque affirmation porte une citation au format `pcg-214-13` ;
- une **abstention explicite**, quand les passages ne permettent pas de répondre.

Modèle configurable par `ACCRAG_GEN_MODEL`, comme `ACCRAG_RERANKER` et
`ACCRAG_REWRITE_MODEL` le sont déjà. Défaut : `claude-opus-5`.

Contraintes reprises du jalon 3 : import paresseux d'`anthropic`, cache disque
JSON des réponses (les re-mesures et les revues deviennent gratuites et
reproductibles), garde-fou dur sur le nombre d'appels, comptage des tokens, et
**`ecrire_cache=False` pour tout cache versionné**.

Le générateur ne reçoit que la question et les passages — jamais les citations
attendues, jamais le corpus entier. Test structurel obligatoire, comme pour
`Rewriter`.

### 3.2 La vérification programmatique des citations — avant le juge

Aucun LLM : du SQL et de la comparaison de chaînes. Pour chaque citation émise :

- **le record existe-t-il** dans `data/corpus.db` ?
- **le passage invoqué figure-t-il** réellement dans le texte de ce record ?

Trois métriques en tombent, toutes automatiques et non falsifiables :

| métrique | définition |
|---|---|
| taux de citations inexistantes | citations dont le `record_id` n'est pas dans le corpus |
| taux de citations non portantes | record existant, mais le passage cité n'y figure pas |
| taux de réponses sans citation | réponses non abstentionnistes ne citant rien |

**Cette brique passe avant le LLM-juge, et c'est un changement d'ordre assumé.**
Elle coûte une journée, ne dépend d'aucun jugement, et attrape la faute la plus
grave qu'un RAG comptable puisse commettre — citer un article qui n'existe pas. Le
juge mesure la *qualité* ; celle-ci mesure l'**honnêteté**, et l'honnêteté se
mesure d'abord.

### 3.3 Le LLM-juge, et sa calibration, qui n'est pas optionnelle

C'est la brique risquée, et le risque a un nom : **un juge non calibré produit des
chiffres qui ont l'air rigoureux sans l'être.** Le jalon 3 a puni trois fois
exactement ce travers — une explication plausible publiée avant le contrôle qui la
départage. Donc la contrainte est posée avant le code :

**Le juge est lui-même mesuré avant de servir.**

- Un **jeu de calibration** de 30 réponses, notées à la main, couvrant délibérément
  les cas limites : réponse juste mais mal citée, réponse fausse bien citée,
  abstention correcte, abstention excessive, réponse partielle.
- On mesure l'**accord juge/humain** sur ce jeu. La métrique et le seuil
  d'acceptation sont fixés *avant* la mesure, comme le critère d'adoption des
  ablations.
- **Si l'accord est sous le seuil, le juge ne publie aucun chiffre.** Il est
  re-prompté ou abandonné, et l'échec est documenté comme un résultat négatif.
- Le barème par étape sur les cas pratiques DSCG ne s'applique **qu'après** ce
  contrôle. Le design v1 qualifie lui-même ce barème de « morceau méthodologique
  le plus délicat ».

Le jeu de calibration est versionné sous `docs/mesures/jalon4/`, avec les notes
humaines, pour que l'accord soit recalculable par quiconque.

### 3.4 La famille d'abstention

Des questions dont la bonne réponse est « ce n'est pas dans le corpus » ou « ça
relève du fiscal, pas du comptable ». Notation binaire : abstention correcte
contre réponse inventée.

**Aucune des sources de questions ne la produit**, par construction — elles
génèrent toutes des questions dont la réponse est dans le corpus. Cette famille
s'écrit donc à la main, question par question. C'est la plus petite en volume et la
plus critique pour la sûreté : le design v1 pose qu'« un agent comptable qui
invente est pire qu'inutile ».

Elle n'a de sens qu'une fois le générateur là — d'où sa position après 3.1.

## 4. L'extension du benchmark

Cible du design v1 : 150–300 questions. État : 90.

### 4.1 Le critère qui ordonne les sources : la fuite

Le bon critère n'est pas le coût de goldage mais l'existence d'une **relation de
fuite entre la question et le corpus** :

| source | fuite possible ? | coût de goldage |
|---|---|---|
| **liasse 2058-A** | non — c'est un formulaire fiscal, pas un document du corpus | faible, la liste des lignes est finie |
| **sujets DCG/DSCG** | non — écrits par un jury, sans lien avec le corpus | élevé, golds à établir article par article |
| rescrits BOFiP | **oui** — la question est *dans* le document indexé | nul, mais exige un held-out |

Ordre retenu : **2058-A d'abord**, épreuves ensuite, rescrits au jalon corpus.

### 4.2 Ce qui est goldable dès maintenant, sans nouveau corpus

- **Le dossier fusions du DSCG UE4.** Le Titre VII est dans le corpus (81 records).
  Golds déjà repérés : prime de fusion → 744-1, 744-2, 751-2, 751-3, 752-5 ; mali
  de fusion → 745-3, 745-4, 751-4 ; « mali technique » apparaît dans 18 records.
- **DCG UE9 et UE10**, comptabilité et comptabilité approfondie, couvertes par les
  Livres I à V.
- **Les divergences 2058-A, à moitié.** Une question comme « la provision pour
  indemnités de fin de carrière est-elle déductible ? » a deux réponses : la
  comptable (PCG, disponible) et la fiscale (CGI 39-1-5° et sa doctrine, absente).
  On écrit la question, on golde la moitié comptable, et on **marque la moitié
  fiscale comme à compléter**. La métrique signature se prépare ainsi sans
  attendre le corpus, et le gain se lira immédiatement le jour où il arrive.

Les corrigés DCG/DSCG fournis par l'utilisateur servent de **contre-vérification
privée uniquement** : on rédige nos golds depuis le PCG, puis on confronte. Ils ne
sont pas redistribuables et rien n'en dérive dans le dépôt.

### 4.3 Le second split de validation

Le split `test` a été exécuté deux fois (clôtures 2.5 et 3). Il reste gelé au sens
strict — aucun réglage n'en a été dérivé — mais **cette garantie s'use à chaque
clôture**. Le jalon 4 crée un second split de validation, gelé et jamais exécuté
avant sa propre clôture, pour que le premier puisse être retiré du service.

## 5. Ce que le jalon ne fera pas, et pourquoi

- **Pas de nouveau corpus.** Mesurer d'abord, ingérer ensuite.
- **Pas de retouche du retrieval adopté.** Deux dettes techniques sont connues et
  restent ouvertes, documentées comme telles : la fusion RRF récompense le
  consensus plutôt que l'excellence (masquée aujourd'hui par la fenêtre du
  reranker, elle ne survivra pas à un corpus plus grand), et la réécriture dégrade
  3 questions sur 61, ce qui suggère de la conditionner à un signal de faible
  recouvrement. Aucune des deux n'est mesurée ici.
- **Pas de comparaison entre effectifs de benchmark différents.** Le benchmark
  passe de 90 à 150–300 questions : les chiffres du jalon 3 ne sont **pas**
  comparables à ceux du jalon 4, et la clôture doit le dire.

## 6. Risques

| risque | mitigation |
|---|---|
| Le juge produit des chiffres crédibles et faux | Calibration obligatoire avant publication, seuil fixé d'avance, échec documenté comme résultat négatif |
| Le barème par étape est ingérable sur les cas pratiques | Il n'est abordé qu'après la calibration du juge sur des réponses courtes ; s'il ne passe pas, il devient un résultat négatif publié |
| Le générateur cite un article qui existe mais ne dit pas ce qu'on lui prête | C'est précisément la métrique « citation non portante » de 3.2, mesurée avant tout jugement |
| L'extension du benchmark introduit un biais | Les golds sont établis depuis le corpus, jamais depuis les corrigés ; les questions d'examen ne peuvent pas fuiter |
| Le coût API devient opaque | Cache disque versionné et comptage des tokens, comme au jalon 3 |

## 7. Critères de clôture

Le jalon est clos quand, et seulement quand :

1. Le générateur existe, avec son test structurel d'intégrité (il ne voit que la
   question et les passages).
2. Les trois métriques de citation sont mesurées sur le benchmark, avec leur JSON
   versionné et leur script.
3. Le juge est calibré, son accord juge/humain publié, et **soit** il franchit le
   seuil et sert, **soit** il ne publie rien et l'échec est documenté.
4. La famille d'abstention existe et son taux est mesuré.
5. Le benchmark atteint au moins 150 questions, et le second split de validation
   est gelé.
6. Chaque chiffre publié est recalculable depuis `docs/mesures/jalon4/`.
