# Jalon 4 — mesurer la génération : citations, abstention, juge calibré, benchmark étendu

Campagne exécutée les **18 et 19 août 2026** sur la branche `feat/jalon4-generation` (le split gelé le 19, après le redémarrage qui a récupéré le GPU — cf. § panne). Trois jalons
avaient amélioré le *retrieval* sans jamais mesurer ce que le système **répond**. Ce jalon
construit l'instrument : un générateur contraint, la vérification programmatique des
citations, un LLM-juge dont l'accord avec des notes humaines est mesuré avant qu'il ne
publie quoi que ce soit, la famille de questions d'abstention, et l'extension du benchmark
de 90 à 150 questions.

Aucune nouvelle source de corpus, aucune reconstruction d'index, aucun changement du
retrieval adopté au jalon 3.

## Ce que le jalon établit, en un tableau

| mesure | dev (93 q., 598 citations) | validation (28 q., 175 citations, gelé) | abstention (30 q., 6 citations) |
|---|---|---|---|
| **citations inexistantes** | **0,0** | **0,0** | 0,0 |
| citations non portantes | 0,0067 | **0,0** | 0,0 |
| identifiants sans version | 0,1388 | 0,1429 | 0,0 |
| réponses sans citation | 0,0 | 0,0 | 0,0 |
| correspondance brute | 0,99 | **1,0** | 1,0 |
| abstention | 0,043 | 0,0 | — |
| abstention correcte | — | — | **0,9667** |
| fabrications | — | — | **0** |
| citations par réponse | médiane 6, max 15 | médiane 5,5, max 13 | 6 |

Verdicts détaillés — dev : 511 `ok`, 83 `version_omise`, 4 `extrait_trop_court`, **0
`record_inexistant`**, 0 `extrait_absent`. Validation : 150 `ok`, 25 `version_omise`, et
rien d'autre.

**Aucun article inventé sur les 779 citations des trois splits.** Sur le split gelé, la
correspondance brute vaut 1,0 : la normalisation n'y fait strictement aucun travail.

Coût : 13 appels de génération sur dev (80 réponses déjà en cache, 80 531 tokens d'entrée,
18 754 de sortie), 28 sur le split gelé (196 492 / 40 118), 30 pour le juge (34 112 /
8 539).

Juge : **kappa pondéré 0,9854** contre 30 notes humaines, pour un seuil de **0,60** fixé
avant toute mesure. Le juge sert.

## Conditions exactes

| Paramètre | Valeur |
|---|---|
| Corpus | `data/corpus.db`, 1 660 `records` — **inchangé** depuis le jalon 2.5 |
| Retrieval | configuration livrée au jalon 3, sans y toucher : réécriture `etend` + `hybrid+rerank`, `bge-reranker-v2-m3`, `n_rerank=25`, `pool=50`, `k=10` |
| Modèle de génération | `claude-opus-5`, sortie structurée (`output_config.format`, schéma JSON), `max_tokens=8000` |
| Modèle du juge | `claude-opus-5`, sortie structurée, `max_tokens=4000` |
| Modèle de réécriture | `claude-sonnet-5` (défaut de `Rewriter`, inchangé) |
| Benchmark | 150 questions goldées (93 dev, 29 test non exécuté ici, 28 validation) + 30 questions d'abstention |
| Machine | Linux 6.19.8-arch1-3-surface, 8 cœurs, GPU Quadro RTX 3000 Max-Q 6 Go |

**Les chiffres de ce rapport ne sont pas comparables à ceux du jalon 3.** Le benchmark est
passé de 90 à 150 questions et la métrique a changé de nature : le jalon 3 mesurait le
*recall*, celui-ci mesure ce que le système répond.

### Le contrôle de fraîcheur a bloqué la clôture, et il avait raison

`scripts/eval_generation.py` refuse de dépenser un appel d'API si le retrieval neutre ne
rend plus la valeur publiée au jalon 3 (`recall@10 = 0,672` sur dev). L'extension du
benchmark a porté dev de 61 à 93 questions, et le contrôle a rendu **0,715** au lieu de
0,672 : il a bloqué.

Déplacer la constante aurait détruit le contrôle. Le périmètre est désormais **lu dans le
JSON qui porte le chiffre publié** (`docs/mesures/jalon3/cloture_dev.json`, clés de
`par_question`), et non déduit du contenu courant de `benchmark/dev.jsonl`. Le contrôle
reste donc vrai à travers toute extension future, et il bloque en plus si une question du
périmètre publié disparaît — auquel cas le chiffre de référence cesserait d'être
vérifiable. La valeur sur le dev étendu (0,715) est imprimée comme chiffre informatif,
jamais comme seuil.

## Brique 1 — le générateur contraint

`src/accounting_rag/generate.py`. Deux formes de sortie possibles et pas une troisième :
une réponse dont chaque affirmation porte une citation, ou une abstention explicite. La
contrainte est appliquée par sortie structurée, pas par prompt : un schéma JSON garantit
que la sortie est analysable, ce qui est la condition pour que la vérification des
citations existe.

Le générateur ne reçoit que la question et les passages — jamais les citations attendues,
jamais les notes du benchmark, jamais sa catégorie. Un test structurel le vérifie sur le
prompt envoyé, un autre sur la source du script de campagne (qui coûte de l'argent et ne
peut pas tourner en test).

### `max_tokens=2000` tronquait les réponses, et la moitié dangereuse n'était pas le plantage

Le premier appel réel de la sonde verbatim est mort sur un `JSONDecodeError` opaque. Cause
mesurée sur q001 : 6 200 tokens d'entrée, 2 586 tokens de sortie, 6 276 caractères de JSON,
et le thinking adaptatif — actif par défaut sur les modèles 5 — partage le budget
`max_tokens` avec le texte. `max_tokens` est passé à 8 000.

Le plantage n'était pas le vrai risque. Une troncature tombant sur une accolade fermante
s'analyse proprement, perd ses dernières citations, et entre dans un cache **committé** que
toutes les campagnes suivantes rejouent. `stop_reason` est donc contrôlé **avant**
`json.loads`, et les deux échecs lèvent en nommant le modèle, la question et le
`stop_reason`.

## Brique 2 — la vérification programmatique des citations

`src/accounting_rag/citations.py`. Aucun LLM, aucun réseau : du SQL et de la comparaison de
chaînes. Elle passe volontairement avant le juge, parce que le juge mesure la *qualité* et
celle-ci mesure l'**honnêteté**.

### La contrainte « extrait VERBATIM » était un risque de conception, pas un fait

Le schéma exige un extrait « recopié caractère pour caractère ». Ce n'était pas acquis :
**79,6 % des records portent une apostrophe courbe et 77,5 % un retour à la ligne** hérité
du PDF. Un modèle incapable de les reproduire aurait fait mesurer mon défaut de conception
au lieu de l'honnêteté du système.

Mesuré avant de coder la comparaison (`scripts/sonde_verbatim_jalon4.py`, 12 questions dev
sous la configuration adoptée) : **77 citations, 77 correspondances exactes sans aucune
normalisation**. 42 reproduisent une apostrophe courbe, 9 un retour à la ligne littéral au
milieu d'une énumération, 2 la ligature `œ`. Longueur médiane 200 caractères, maximum 524.

La sonde garde ses quatre niveaux de normalisation exposés (`brut`, `espaces`,
`typographie`, `casse_accents`) avec un test par niveau plus le cas de l'extrait inventé :
sans cela, un classement qui renverrait toujours `brut` produirait le même 100 % sur des
extraits fabriqués.

### 15,64 % de citations hallucinées, dont aucune ne l'était

La première campagne a affiché **15,64 % de citations inexistantes** — la métrique la plus
grave que ce projet publie. Aucune n'était inventée. Les 69 concernées nommaient le bon
article avec un extrait verbatim et avaient seulement perdu le suffixe de version
`@2026-01-01`. Aucune n'était ambiguë. **Le vrai taux d'hallucination sur dev est zéro.**

Mon contrôle confondait « article inventé » et « identifiant abrégé ». Les deux restent des
défauts — 39 articles du corpus portent plusieurs versions, donc omettre la version perd
réellement de la traçabilité — mais ce sont deux défauts différents, et un rapport qui les
additionne ne mesure plus rien. `verifier_citation` rend désormais `version_omise`, ou
`version_ambigue` quand l'article a plusieurs versions et que la citation ne dit pas
laquelle. Un test échoue si les deux taux sont additionnés.

**Ce que ce défaut enseigne** : j'avais énoncé le chiffre de 15,64 % dans mon propre
raisonnement avant d'en chercher le mécanisme, et le mécanisme l'a annulé. C'est la
quatrième occurrence dans ce dépôt du travers « une explication avant le contrôle qui la
départage » — sauf que cette fois le contrôle a précédé la publication.

### L'unique citation non portante tenait à un caractère

Après décomposition des verdicts, il restait une seule citation « non portante » sur les 422 de la **première** campagne, celle qui portait sur les 61 questions du dev d'avant extension.
Elle divergeait du corpus d'**un caractère** : `l'actif` contre `l’actif`, dans un extrait
qui écrivait correctement `l’écart d’acquisition` deux mots plus loin.

Une apostrophe droite ne fait pas dire autre chose à un article. Classer cela « non
portant » était une erreur de catégorie : la métrique existe pour attraper une citation qui
prête à un texte un propos qu'il ne tient pas. La normalisation replie donc les variantes
d'apostrophe — **sur preuve du mécanisme, pas par précaution**.

Ce n'est pas un critère déplacé après coup : le critère d'adoption du dépôt porte sur le
recall, et il s'agit ici de corriger un instrument qui produit un faux positif dont le
mécanisme exact est nommé. Et le repli ne cache rien : `taux_correspondance_brute` publie
en parallèle le taux de correspondance **sans aucune normalisation**, et l'écart entre les
deux taux *est* cette citation.

### Deux tolérances rendues visibles, et une ligne morte retirée

`correspond_brut` applique le même refus que `verifier_citation` sur les extraits plus
courts que `EXTRAIT_MINIMUM`, et résout le suffixe de version de la même façon. Sans cela,
l'écart entre les deux taux mélangerait la normalisation, l'omission de version et le seuil
de longueur, et ne dirait plus ce qu'il est là pour dire.

Un taux dont le dénominateur est nul vaut `None`, jamais `0.0`. Sur un split où le système
s'abstient partout, « taux de citations inexistantes : 0,0 » se lirait comme un sans-faute
alors que le taux n'est pas défini.

Un garde `if "@" in record_id` a été écrit puis **retiré** : aucune mutation ne pouvait le
faire échouer, parce qu'ajouter `@%` à une chaîne contenant déjà `@` ne correspond à aucun
identifiant du corpus. Une ligne qu'aucun test ne peut mettre en défaut ne protège rien.

### `EXTRAIT_MINIMUM = 30` reste à 30

Les citations non portantes qui subsistent sont toutes des fragments du plan de comptes plus
courts que le seuil : `6226 Honoraires`, `Production immobilisée 72`,
`213, 214 215 218 231, 238`. C'est une limite de l'instrument, pas de la réponse.

Le seuil a été fixé avant la mesure et le déplacer maintenant serait exactement substituer
un critère après avoir vu le résultat. Il est publié comme limite et vaut un changement
mesuré à part.

## Brique 3 — la famille d'abstention

`benchmark/abstention.jsonl`, 30 questions sans aucun gold, dix par raison :
`fiscal_pas_comptable`, `hors_corpus`, `hors_perimetre`.

**Taux d'abstention correcte : 0,9667 (29 sur 30). Fabrications : 0.**

### Les questions faciles ne servent à rien

Une question dont aucun mot ne figure au corpus n'est jamais remontée par le retrieval, et
s'abstenir ne coûte alors rien. Les questions utiles sont celles où **le corpus cite sa
source fiscale sans la contenir** :

| piège | record remonté | ce qu'il dit | ce qu'il ne dit pas |
|---|---|---|---|
| conditions de déduction d'une provision pour grosses réparations | `pcg-na-25` | « déductibles si elles remplissent les conditions énoncées à l'article 39 1 5° du CGI » | les conditions |
| périmètre d'intégration fiscale | `pcg-515-2-c1` | cite « Article 223-A du CGI », traite la comptabilisation de l'économie d'impôt | qui entre dans le périmètre |
| seuil du contrôle exclusif | `pcg-741-2` | renvoie à « l'article 211-3 du règlement ANC N° 2020-01 » | le pourcentage |
| barème de l'indemnité de licenciement | `pcg-322-10-c5`, `pcg-324-1-c67` | la provision et son fait générateur, longuement | le barème, qui relève du code du travail |

Chaque question porte dans `notes` la **preuve** de l'absence de réponse : le record que le
retrieval remontera et ce qui lui manque. Un test l'exige — sans elle, rien ne distingue
« le corpus ne répond pas » de « je n'ai pas cherché ». Les 22 records nommés et les 21
comptes affirmés ont été vérifiés en SQL avant la mesure : aucun absent, aucun faux.

### Deux questions retirées ou resserrées par l'inspection des passages

C'était le point de l'inspection (`scripts/inspecter_abstention.py`, aucune métrique — le
recall n'a aucun sens sur un split sans gold, et `evalrag.evaluate` y lèverait une division
par zéro).

- **Le corpus répondait à qa009.** `pcg-745-7` dit que le mali technique « est amorti ou
  rapporté au résultat selon les mêmes règles que les actifs sous-jacents », et
  `pcg-745-7-c1` s'intitule « IR 3 : Amortissement du mali technique ». La question a été
  resserrée sur la seule déductibilité : le piège reste (le retrieval remontera
  l'amortissement comptable à une question fiscale) et la moitié répondable disparaît.
- **qa008 était à moitié répondable.** `pcg-191-1-c7` déroule un cas concret de livraison à
  soi-même. Remplacée par le délai du droit de reprise, dont la seule occurrence dans le
  corpus est « Fonds associatifs sans droit de reprise », sans rapport.

Une question à moitié répondable ne peut pas départager une abstention correcte d'une
chance.

### L'unique non-abstention n'est pas une fabrication, et la métrique ne l'appelle plus ainsi

qa014 (retraitement d'une location-financement en consolidation) a mis `abstention` à
`false`. Elle n'a rien inventé : elle ouvre sur « les passages fournis ne détaillent pas
les écritures de retraitement, qui relèvent du règlement ANC n° 2020-01 », cite six
extraits **tous verbatim et existants**, et conclut sur ce qui manque.

Seul le drapeau était faux. C'est un défaut réel — un appelant qui s'y fie présenterait une
réponse à une question sans réponse — mais l'appeler « réponse inventée » serait une
accusation que la mesure ne soutient pas. Le champ `n_fabrications` est donc publié à côté
du taux d'abstention, et il vaut **0**.

qa014 est **conservée**, alors que c'est la plus faible des 30 : c'est l'item le plus
informatif de la famille, puisqu'il expose la limite du drapeau binaire.

### Les quatre abstentions de dev, et ce qu'elles disent de mon benchmark

Le split dev goldé porte 4 abstentions sur 93 (`taux_abstention` 0,043). Une abstention sur
une question goldée compte comme un échec — sauf si le gold était absent des dix passages
montrés, auquel cas le retrieval a échoué et la génération s'est bien comportée. La clé de
cache du générateur porte la liste exacte des passages, ce qui permet de trancher sans rien
réexécuter :

| question | gold | gold parmi les 10 passages ? | lecture |
|---|---|---|---|
| q089 | `pcg-745-7` | non | abstention **fondée** — échec du retrieval |
| q1028 | `pcg-324-1` | non | abstention **fondée** — échec du retrieval |
| q1031 | `pcg-322-1` | non | abstention **fondée** — échec du retrieval |
| q1032 | `pcg-214-22` | **oui** | à examiner |

Sur q1032, le gold était présent et le système s'est abstenu quand même. Sa raison :
« les passages fournis ne traitent que du traitement comptable de la dépréciation des
stocks ». Or l'énoncé est « j'ai déprécié un stock de marchandises qui ne part plus :
est-ce que le fisc l'accepte comme charge ? » — une question **purement fiscale**.

**C'est un défaut de ma conception du benchmark, pas du système.** Trois des quatre
abstentions portent sur des questions de divergence 2058-A, et parmi les huit questions de
cette famille, deux (q1031, q1032) sont formulées de façon purement fiscale alors que leur
gold est la moitié comptable. Pour celles-là, s'abstenir est le comportement juste et c'est
mon attente de gold qui est fausse. Les six autres demandent explicitement les deux volets ;
sur l'une d'elles (q1028) le retrieval n'a pas remonté le gold, donc aucune n'a produit
d'abstention excessive imputable au générateur.

Bilan honnête : **une seule abstention des 93 est discutable, et elle est discutable à cause
de l'énoncé que j'ai écrit.** Je ne récris pas les deux questions maintenant — modifier un
énoncé après avoir vu le résultat est exactement ce que ce dépôt s'interdit. La correction
(reformuler pour demander les deux volets, ou créditer une abstention qui nomme
correctement le manque fiscal) appartient au jalon suivant, qui apportera le corpus fiscal
et rendra la question entièrement répondable.

## Brique 4 — le LLM-juge et sa calibration

`src/accounting_rag/judge.py`, `scripts/calibrer_juge.py`,
`docs/mesures/jalon4/calibration_juge.json`.

**Seuil fixé avant toute mesure : `kappa_pondere >= 0,60`**, lu dans le JSON de calibration
lui-même pour qu'il ne puisse pas être déplacé après avoir vu le résultat.

| mesure | valeur |
|---|---|
| n | 30 |
| accord exact | 0,9667 |
| écart moyen absolu | 0,0333 |
| **kappa pondéré** | **0,9854** |
| kappa sur les 18 cas de campagne seuls | 1,0 (accord exact 1,0) |

Écart moyen par cas limite : `juste_bien_citee` 0,0 ; `juste_mal_citee` 0,0 ;
`fausse_bien_citee` 0,1667 ; `abstention_correcte` 0,0 ; `abstention_excessive` 0,0.

### Ce que le juge voit, et pourquoi ce n'est pas le gold

La loi 9 du dépôt interdit à un juge de voir les citations attendues, le corpus et les
résultats du retrieval. Appliquée littéralement, elle rend le juge incapable de noter la
justesse : il ne lui reste que la cohérence interne.

La résolution est celle d'un jury d'examen : le juge reçoit un **barème**, pas un
corrigé-citations. Le barème est une liste de critères écrite à la main depuis le PCG ; il
porte ce qu'une réponse juste doit contenir sans dire quels `record_id` la portent. La
justesse des citations est déjà mesurée sans LLM par la brique 2. Ce que la loi 9 protège
est la circularité — un modèle recevant la réponse qu'il doit trouver — et cela vise le
réécriveur et le générateur, pas un correcteur. La frontière a un test structurel.

### Deux cas limites n'existaient pas dans la campagne

« Fausse mais bien citée » n'a **aucune instance réelle** : la campagne n'a produit aucune
citation hallucinée, ni sur les 422 de la première campagne ni sur les 779 des trois splits publiés. « Abstention excessive » non plus : l'unique abstention de dev alors mesurée
est fondée, son gold étant absent des dix passages montrés.

Les douze cas correspondants sont donc **fabriqués** à partir de réponses réelles, en
inversant l'affirmation décisive ou en substituant une abstention, avec les citations
verbatim d'origine conservées. C'est le rôle d'un jeu de calibration : éprouver la
discrimination du juge, pas échantillonner le comportement du système — on calibre une
balance avec des masses connues, pas avec des colis au hasard.

Le champ `origine` sépare `campagne` de `perturbation`, et l'accord est publié des deux
façons. Sur les 18 cas de campagne seuls, l'accord est exact et le kappa vaut 1,0 : le seul
désaccord porte sur un cas fabriqué.

### Le seul désaccord va contre la référence humaine

Sur une réponse crédit-bail perturbée, j'avais compté un critère comme acquis parce que le
bien finit bien à l'actif après la levée d'option. Le juge l'a refusé, en lisant que dire
que le bien y est « déjà » **nie** son entrée à cette date. Le juge a raison.

**La note humaine reste telle qu'écrite.** Elle a été fixée avant la mesure, et la
récrire après coup viderait la calibration de son sens.

### Deux défauts du harnais, au-delà du plan

- Le contrôle de cohérence de la note n'avait aucun test isolant sa première clause, parce
  que le cas de test violait déjà la seconde. Une note dans les bornes avec un dénominateur
  faux la couvre désormais. Relevé par l'implémenteur, pas par moi.
- La calibration indexait les notes humaines par le seul `question_id`, alors que six
  énoncés apparaissent deux fois avec des réponses différentes : le kappa serait
  silencieusement tombé de 30 cas à 24. La clé inclut désormais le cas limite.

## L'extension du benchmark

De 90 à **150 questions goldées**, plus 30 questions d'abstention. Motif : à 0,966 sur le
split gelé, l'instrument rendait 28 réponses justes sur 29 et ne pouvait plus mesurer un
progrès.

### Le critère qui ordonne les sources est la fuite, pas le coût de goldage

| source | fuite possible ? | retenue |
|---|---|---|
| dossier fusions du DSCG UE4 (Titre VII, 117 records dont 81 articles numérotés) | non — écrit par un jury | **32 questions** |
| thèmes DCG UE9/UE10 absents du benchmark | non | **20 questions** |
| divergences de la liasse 2058-A | non | **8 questions** |
| rescrits BOFiP | **oui** — la question est *dans* le document indexé | reportée au jalon corpus |

Tous les golds ont été vérifiés en SQL contre `data/corpus.db` avant rédaction.

**Un chiffre faux attrapé en relisant.** J'ai d'abord écrit « Titre VII, 281 records » dans
ce rapport. 281 est la somme du Titre VII du Livre II (117) et du Titre VIII du Livre III
(164) : ma requête employait `chemin LIKE 'Livre II%Titre VII%'`, où `'Livre II%'` matche
aussi « Livre III » et `'%Titre VII%'` matche aussi « Titre VIII ». Deux chevauchements dans
un même motif. Le périmètre fusions réel est de **117 records, dont 81 portent un numéro
d'article** et 36 sont des tableaux ou des exemples sans article. Aucun gold n'était touché —
ils avaient été vérifiés un par un — mais le chiffre de contexte était faux, et c'est
exactement la catégorie d'erreur que ce dépôt existe pour attraper. Les corrigés
DCG/DSCG fournis servent de contre-vérification privée uniquement : les golds sont rédigés
depuis le PCG, puis confrontés. Rien n'en dérive dans le dépôt.

### La moitié fiscale est marquée, jamais inventée

Les 8 questions de divergence 2058-A portent `gold_fiscal: "a_completer"`. Elles ont deux
réponses : la comptable, dans le corpus et goldée normalement, et la fiscale, absente. La
métrique signature du projet — la confusion fiscal ↔ comptable — se prépare ainsi sans
attendre le corpus fiscal, et le gain se lira le jour où il arrive. Un test refuse qu'un
`gold_fiscal` devienne une vraie citation.

### Le second split de validation

`benchmark/validation.jsonl`, 28 questions, **gelé le 18 août 2026** et exécuté une seule
fois, à la clôture de ce jalon. Il existe parce que la garantie de gel de `test.jsonl` s'use
à chaque clôture : ce split a été exécuté deux fois (jalons 2.5 et 3).

`test.jsonl` n'est **pas** modifié. Sa garantie est dépensée et ce jalon le remplace ; y
ajouter des questions serait du travail sans valeur de mesure.

### Un garde-fou cassé par l'extension, alors que l'audit était juste

`tests/test_audit_reecritures.py` vérifiait que l'audit d'intégrité couvre dev **et** test
en comparant à un effectif codé en dur (90). L'extension l'a fait échouer alors que l'audit
fonctionnait : dev + test valent désormais 122 questions pour 90 réécritures en cache,
l'ancrage du jalon 3 ne couvrant que le benchmark du jalon 3.

Un effectif figé mesurait la taille du benchmark, pas le périmètre audité. Le garde vérifie
maintenant que les deux splits sont dans le périmètre et qu'aucune entrée de cache n'est
silencieusement sautée — le défaut réel pour lequel il avait été écrit.

## Reproductibilité

Chaque chiffre de ce rapport est recalculable depuis `docs/mesures/jalon4/` :

| artefact | contenu |
|---|---|
| `generation_dev.json` | métriques dev, verdict par citation, coût |
| `generation_validation.json` | idem sur le split gelé, une seule exécution |
| `generation_abstention.json` | taux d'abstention correcte, non-abstentions nommées |
| `calibration_juge.json` | 30 notes humaines, notes du juge, accord, écart par cas limite |
| `sonde_verbatim.json` | les 77 citations de la sonde et leur niveau de normalisation |
| `passages_abstention.json` | les passages remontés pour les 30 questions d'abstention |
| `reponses_{split}.json` | cache des réponses — ancrage qui rend la re-mesure gratuite |
| `reecritures_{split}.json` | cache des réécritures du jalon 4 |

`scripts/controle_chiffres_jalon4.py` recalcule chaque agrégat depuis les données brutes du
même fichier **et** vérifie que chaque chiffre publié ici apparaît littéralement. Il a sept
tests, un par mode d'échec : un agrégat qui ne suit plus ses verdicts, un chiffre corrigé
dans le JSON mais pas dans le rapport, un effectif absent, un artefact manquant, un seuil de
calibration déplacé, un kappa qui ne suit plus ses notes.

Le cache de réécriture du jalon 4 est **amorcé à l'identique** depuis l'ancrage du jalon 3
pour les 61 questions communes : leur retrieval est donc inchangé. L'ancrage du jalon 3
reste en lecture seule, et un test refuse tout `Rewriter` qui le pointerait en écriture.

## Réserves

1. **Effectif et comparabilité.** Le benchmark a changé de taille (90 → 150) et la métrique
   de nature. Aucun chiffre de ce rapport ne se compare à un chiffre du jalon 3.
2. **Le kappa de 0,9854 ne doit pas être sur-lu.** Il porte sur 30 cas, dont 18 seulement
   proviennent de la campagne, et j'ai écrit à la fois les barèmes et les notes de
   référence : l'accord mesure donc en partie la clarté de mes propres barèmes. Ce que la
   calibration établit est que le juge ne produit pas des chiffres crédibles et faux, ce
   qu'elle était faite pour vérifier.
3. **Le barème par étape sur cas pratiques DSCG n'est pas abordé.** Le design le qualifie
   lui-même de morceau le plus délicat ; il ne vient qu'après cette calibration.
4. **Verbosité du générateur.** Médiane de 6 citations par réponse sur dev (jusqu'à 15) et
   5,5 sur le split gelé (jusqu'à 13), là où le gold en porte souvent une. Le « taux de réponses sans citation » est donc trivialement
   nul et la précision diluée. Le chiffre est publié (`citations_par_reponse`) pour que le
   biais soit lisible dans les mesures et non affirmé en prose.
5. **Non-déterminisme de l'API.** Le thinking adaptatif est actif et la température par
   défaut n'est pas nulle : deux exécutions sans cache ne rendraient pas exactement les
   mêmes réponses. Le cache versionné est l'ancrage de reproductibilité, comme au jalon 3.
6. **Le coût publié est par exécution, jamais cumulé.** Une valeur nulle avec un cache déjà
   plein est un rejeu gratuit, pas une campagne gratuite ; les champs
   `reponses_deja_en_cache_avant` et `rejeu_depuis_le_cache` permettent de le distinguer.
7. **Deux dettes techniques du jalon 3 restent ouvertes et non mesurées** : la fusion RRF
   récompense le consensus plutôt que l'excellence (masquée aujourd'hui par la fenêtre du
   reranker, elle ne survivra pas à un corpus plus grand), et la réécriture dégrade 3
   questions sur 61.
8. **Panne de GPU pendant la campagne de clôture** — voir la section suivante.

## La panne de GPU, et le contrôle qui rend la reprise légitime

Le dGPU (Quadro RTX 3000 Max-Q) est tombé du bus à la question 77 sur 93 :
`torch.AcceleratorError: CUDA error: unspecified launch failure`, puis `nvidia-smi` cesse de
voir tout périphérique alors que le module et `/dev/nvidia0` sont présents et que la carte
répond en PCI. Recharger le stack de modules n'a pas récupéré la carte. C'est une panne
d'infrastructure, pas un résultat de mesure.

La campagne a été reprise **sur CPU** (~115 s par question mesurés ici contre ~1,5 s sur
GPU — ordres de grandeur, pas décimales, cf. la leçon du jalon 3).

**Un contrôle a été fixé avant de mesurer**, et il faut dire exactement ce qu'il prouve. La
clé de cache du générateur porte la liste exacte des passages montrés : si le retrieval
produisait un autre classement, la clé changerait et l'appel serait payant. Sur 93 questions
dev dont 80 déjà en cache, `appels_api` devait donc valoir **exactement 13**.

Résultat : **13**. Contrôle passé.

**Ce que ce chiffre prouve, et ce qu'il ne prouve pas.** La machine a finalement été
redémarrée et le GPU récupéré, si bien que la campagne publiée est entièrement sur GPU : les
80 réponses en cache l'avaient été avant la panne, les 13 nouvelles après le redémarrage. Le
contrôle établit donc que **le retrieval reproduit à l'identique, après un redémarrage et
une réinitialisation du pilote, les dix passages de chacune des 80 questions déjà mesurées**
— un contrôle de reproductibilité qui vaut mieux que ce pour quoi il avait été écrit. Il ne
prouve **pas** l'équivalence CPU/GPU : cette dernière ne repose que sur une sonde à 4
questions, et elle n'est plus nécessaire à la validité de la mesure. La reprise CPU (10
questions sur 93 avant le redémarrage) n'a laissé aucune trace dans les chiffres publiés.

## Critères de clôture

| critère | état |
|---|---|
| Le générateur existe, avec son test structurel d'intégrité | ✅ |
| Les métriques de citation sont mesurées, avec JSON versionné et script | ✅ |
| Le juge est calibré, son accord publié, et il franchit le seuil | ✅ kappa 0,9854 ≥ 0,60 |
| La famille d'abstention existe et son taux est mesuré | ✅ 0,9667 |
| Le benchmark atteint 150 questions et le second split est gelé | ✅ |
| Chaque chiffre publié est recalculable depuis `docs/mesures/jalon4/` | ✅ contrôle scripté, 7 tests |
