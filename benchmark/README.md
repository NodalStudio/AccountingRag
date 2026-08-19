# Benchmark gold — 150 questions goldées + 30 questions d'abstention (v3)

Jeu de 90 questions à réponse ancrée dans le corpus (`data/corpus.db`), avec citations vérifiées manuellement. Objectif : disposer d'un signal minimal, indépendant du système de retrieval, pour mesurer si le pipeline (`src/accounting_rag/search.py`) retrouve bien les bons articles. Les citations gold ont été choisies en lisant directement le texte des `records` en base — **pas** en interrogeant le retrieval — pour éviter tout biais de circularité entre juge et système jugé.

Le jeu est passé de 30 à 90 questions au jalon 2.5 (`q031`-`q090`), puis de 90 à 150 au jalon 4 (`q1001`-`q1032` en dev et `v001`-`v028` dans un nouveau split de validation), auxquelles s'ajoutent 30 questions d'abstention sans gold. Motif du jalon 4 : à 0,966 sur le split gelé, l'instrument rendait 28 réponses justes sur 29 et ne pouvait plus mesurer un progrès.

**Les chiffres mesurés sur ce benchmark ne sont pas comparables à ceux du jalon 3** : l'effectif a changé.

## Fichiers

- `dev.jsonl` — 61 questions, utilisables librement pour développer/régler le système (choix d'algorithme, tuning, debug).
- `validation.jsonl` — 28 questions, **second split gelé**, créé au jalon 4 et jamais exécuté avant la clôture de ce jalon. Il existe parce que la garantie de gel de `test.jsonl` s'use à chaque clôture : ce split a été exécuté deux fois (jalons 2.5 et 3), et il faut pouvoir le retirer du service sans perdre toute mesure non biaisée.
- `abstention.jsonl` — 30 questions **sans aucun gold**, dont la bonne réponse est de refuser de répondre. Trois raisons, dix questions chacune : `fiscal_pas_comptable`, `hors_corpus`, `hors_perimetre`. Ces questions ne mesurent pas le retrieval — `recall` n'y a aucun sens, et `evalrag.evaluate` y lèverait une division par zéro. Elles mesurent l'honnêteté de la génération.
- `test.jsonl` — 29 questions, **réservées** : ne jamais s'en servir pour régler le système (paramètres, prompts, seuils…). Elles ne servent qu'à une mesure finale non biaisée. Si une question de `test.jsonl` a été consultée pendant un développement ou une analyse d'erreurs, elle est compromise et doit être remplacée, pas réutilisée.

Le split est **stratifié par catégorie et figé** : une fois tiré, il n'est jamais re-tiré (sous peine de fuite progressive du jeu de test vers le dev). Ajouter des questions ne change pas le split existant, voir « Procédure d'ajout » ci-dessous.

**Le split test v2 (29 questions) est gelé le 2026-08-16 ; il ne sera exécuté qu'une fois, à la clôture du jalon 2.5.**

## Format (une question par ligne JSONL)

```json
{"id": "q001", "question": "...", "categorie": "reference_directe", "citations": ["pcg-214-1"], "notes": "..."}
```

| Champ | Description |
|---|---|
| `id` | Identifiant unique `qNNN`, stable dans le temps (ne pas renuméroter les questions existantes). |
| `question` | Question en français naturel. |
| `categorie` | `reference_directe`, `regle` ou `vocabulaire_courant` (voir ci-dessous). |
| `citations` | Liste de **préfixes** d'identifiants d'articles gold (sans le suffixe `@édition`, `-c` ou `#n`), ex. `pcg-214-1`. Au moins une citation par question. Un préfixe est considéré présent dans le corpus s'il correspond exactement à un `id`, ou à un `id` préfixé suivi de `@`, `-c` ou `#` (occurrences multiples, commentaire ANC rattaché, fragment) — voir `tests/test_benchmark_format.py`. |
| `notes` | Un mot sur la réponse attendue (repère de correction, pas une réponse rédigée). Pour les questions `q031`-`q090`, `notes` mentionne aussi le thème couvert et, pour `vocabulaire_courant`, le registre de langue simulé ; le tag `apostrophe:typo` marque les questions écrites avec l'apostrophe typographique U+2019 (« ’ »), voir « Garde-fou apostrophe » ci-dessous. |

## Catégories et répartition cible

| Catégorie | Effectif | Description |
|---|---|---|
| `reference_directe` | 10 | La question cite explicitement le numéro d'article (« Selon l'article 214-13… »). Cas le plus facile : le système doit au moins retrouver l'article nommé. |
| `regle` | 35 | Question en langage professionnel comptable, sans citer de numéro d'article — la formulation employée par un comptable ou un étudiant DSCG/DCG qui connaît le vocabulaire technique (« provision réglementée », « écarts de conversion », « levée d'option »…). |
| `vocabulaire_courant` | 45 | Question posée comme le ferait un créateur d'entreprise ou un non-comptable : mots familiers, zéro terme PCG exact de l'article visé (« machine », « voiture de société en leasing », « client qui ne paie pas », « argent de côté »…). Teste la capacité du système à faire le pont entre langage courant et vocabulaire normatif. Les questions `q056`-`q090` déclinent quatre registres explicites (étudiant DCG, dirigeant de PME, comptable junior, langage familier) et reprennent, sur de nouveaux thèmes, les trois gabarits de difficulté identifiés comme les plus discriminants au jalon 2 : paraphrase totale sans token commun (ex. `q021`), terme grand public pour un concept technique précis (ex. `q026`), question à deux volets appelant deux citations (ex. `q022`). |

Répartition dev/test par catégorie (stratifiée, ~70 %/30 %) :

| Catégorie | dev | test | total |
|---|---|---|---|
| Catégorie | dev | test | validation | total |
|---|---|---|---|---|
| `reference_directe` | 15 | 3 | 7 | 25 |
| `regle` | 37 | 12 | 12 | 61 |
| `vocabulaire_courant` | 41 | 14 | 9 | 64 |
| **Total** | **93** | **29** | **28** | **150** |

Le tableau à deux colonnes ci-dessus est celui du jalon 2.5, conservé pour l'historique.

## La famille d'abstention, et pourquoi ses questions sont difficiles

Une question dont aucun mot ne figure au corpus n'est jamais remontée par le retrieval, et
s'abstenir ne coûte alors rien. Les questions utiles sont celles où **le corpus cite sa
source fiscale sans la contenir** : `pcg-na-25` renvoie à l'article 39-1-5° du CGI sans en
énoncer les conditions, `pcg-515-2-c1` cite l'article 223-A tout en ne traitant que la
comptabilisation, `pcg-741-2` renvoie la définition du contrôle exclusif à l'article 211-3
du règlement ANC 2020-01. Le retrieval remonte un passage très pertinent, et le système
doit reconnaître qu'il ne répond pas.

Chaque question porte dans `notes` la **preuve** de l'absence de réponse dans le corpus :
le record que le retrieval remontera et ce qui lui manque. Un test l'exige, parce que sans
elle rien ne distingue « le corpus ne répond pas » de « je n'ai pas cherché ».

Deux questions ont été retirées ou resserrées par l'inspection des passages avant la
mesure, ce qui était le point de l'inspection : l'une demandait si le mali technique est
amortissable, ce à quoi `pcg-745-7` répond, et elle a été resserrée sur la seule
déductibilité ; l'autre portait sur la livraison à soi-même, dont `pcg-191-1-c7` déroule un
cas concret.

## Les divergences 2058-A, et leur moitié manquante

Huit questions portent le champ `gold_fiscal: "a_completer"`. Elles ont deux réponses : la
comptable, dans le corpus et goldée normalement, et la fiscale, absente du corpus. La
moitié manquante est **marquée, jamais inventée**. La métrique signature du projet — la
confusion fiscal ↔ comptable — se prépare ainsi sans attendre le corpus fiscal, et le gain
se lira le jour où il arrive.

## Garde-fou apostrophe (anti-régression C1)

Le benchmark v1 (30 questions, jalon 2) était rédigé à 100 % en apostrophe ASCII (`'`) et n'exerçait donc jamais le pipeline avec l'apostrophe typographique Unicode U+2019 (`’`), pourtant fréquente en français bien écrit et à l'origine d'un bug réel de recherche (C1). Sur les 60 questions ajoutées au jalon 2.5, **24 questions** (≥ 20 requis) utilisent l'apostrophe typographique U+2019 dans leur texte et portent le tag `apostrophe:typo` dans `notes`. `tests/test_benchmark_format.py` vérifie qu'au moins 20 questions du jeu complet contiennent le caractère U+2019.

## Portée thématique

Les questions `q001`-`q030` couvrent : amortissement (durée, mode, composants, valeur résiduelle), dépréciation des stocks, crédit-bail, provisions (définition, provisions réglementées, engagements de retraite), fonds commercial (acquis vs généré en interne, amortissement, dépréciation), jetons numériques (émission, définition), écarts de conversion en devises, créances clients douteuses ou irrécouvrables, participation des salariés, société en participation, solutions informatiques (logiciels), subventions d'investissement.

Les questions `q031`-`q090` (jalon 2.5) ajoutent 11 thèmes absents de `q001`-`q030`, plus un approfondissement des écarts de conversion :

- stocks et en-cours (définition, coût d'acquisition/production, cas particuliers de valorisation) ;
- subventions d'exploitation (distinctes des subventions d'investissement déjà couvertes) ;
- changements de méthode comptable (vs changement de réglementation, vs changement d'estimation) ;
- contrats à long terme (méthode à l'achèvement/à l'avancement, VEFA, résultat non déterminable) ;
- frais de développement (conditions de capitalisation, coût, durée d'amortissement) ;
- écarts de conversion (nouveaux aspects : créances/dettes, ajustement de provision) ;
- effets de commerce (encaissement, escompte, mobilisation de créances) ;
- opérations en devises (coût d'entrée en monnaie étrangère, titres et stocks à l'étranger) ;
- indemnités d'assurance (destruction d'immobilisation, étalement interdit) ;
- fusion/apports (boni et mali de fusion, mali technique, apports insuffisants) ;
- engagement de crédit-bail en annexe ;
- production immobilisée (compte 72, immobilisations en cours).

## Historique du re-gel

- **2026-08 (jalon 2)** : constitution initiale, 30 questions (21 dev / 9 test), split figé.
- **2026-08-16 (jalon 2.5)** : ajout de 60 questions (`q031`-`q090`), réparties AVANT toute mesure (re-gel du split) selon le tableau ci-dessus. Les 30 questions `q001`-`q030` et leur répartition dev/test existante ne sont pas modifiées.
- **2026-08-18 (jalon 4)** : ajout de 60 questions goldées et de 30 questions d'abstention. Trois sources, ordonnées par le risque de **fuite** entre la question et le corpus et non par le coût de goldage : le dossier fusions du DSCG UE4 (Titre VII du Livre II, 117 records dont 81 articles numérotés — 32 questions), les thèmes DCG UE9/UE10 absents du benchmark (20 questions), et les divergences de la liasse 2058-A (8 questions à moitié fiscale marquée). Les rescrits BOFiP sont écartés jusqu'au jalon corpus : la question y est *dans* le document indexé, donc la mesure serait circulaire.
  - `test.jsonl` n'est **pas** modifié. Sa garantie de gel est dépensée (deux exécutions) et le jalon 4 le remplace par `validation.jsonl` ; y ajouter des questions serait du travail sans valeur de mesure.
  - `validation.jsonl` est **gelé le 2026-08-18** et n'est exécuté qu'une fois, à la clôture du jalon 4. Sa stratification par catégorie et par source est vérifiée par un test.
  - Les corrigés DCG/DSCG fournis par l'utilisateur servent de **contre-vérification privée uniquement** : les golds sont rédigés depuis le PCG, puis confrontés. Rien n'en dérive dans le dépôt et ils ne sont pas redistribuables.

## Procédure d'ajout de questions

1. Choisir un thème du corpus non encore couvert (ou sous-couvert) et lire le texte réel des articles candidats dans `data/corpus.db` — jamais via le retrieval.
2. Rédiger la question et vérifier que la réponse se trouve effectivement dans le(s) article(s) cité(s).
3. Choisir la catégorie (`reference_directe` / `regle` / `vocabulaire_courant`) selon le registre de langue employé ; pour `vocabulaire_courant`, exclure tout terme PCG exact de l'article visé.
4. Attribuer un nouvel `id` (`qNNN` suivant, sans réutiliser ni renuméroter les ids existants).
5. Décider dev ou test **au moment de l'ajout**, en respectant approximativement les proportions ci-dessus, puis ne plus jamais déplacer la question d'un fichier à l'autre.
6. Ajouter la ligne au fichier JSONL concerné et lancer `uv run pytest tests/test_benchmark_format.py -v`.

## Test de format

`tests/test_benchmark_format.py` valide, pour `dev.jsonl` et `test.jsonl` ensemble :

- présence des champs requis et catégorie autorisée ;
- unicité des `id` sur l'ensemble des deux fichiers ;
- longueur minimale de la question ;
- présence d'au moins une citation par question ;
- existence d'au moins un `record` en base pour chaque citation (préfixe exact, ou suivi de `@`, `-c` ou `#`) ;
- effectif exact par fichier (61 dev / 29 test) et par catégorie (10 / 35 / 45) ;
- au moins 20 questions contenant le caractère U+2019 (garde-fou apostrophe).

```sh
uv run pytest tests/test_benchmark_format.py -v
```

## Licence

Les questions et notes de correction sont une œuvre originale de ce dépôt (code, licence MIT — voir `LICENSE`). Les citations pointent vers du contenu dérivé du Recueil des normes comptables françaises (ANC), distribué sous licence **[Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)** (Etalab) — voir `DATA_LICENSE.md` à la racine du dépôt pour les modalités complètes et l'avertissement sur la fidélité du texte extrait.
