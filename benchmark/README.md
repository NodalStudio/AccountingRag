# Benchmark d'amorçage — 30 questions gold

Jeu de 30 questions à réponse ancrée dans le corpus (`data/corpus.db`), avec citations vérifiées manuellement. Objectif : disposer d'un signal minimal, indépendant du système de retrieval, pour mesurer si le pipeline (`src/accounting_rag/search.py`) retrouve bien les bons articles. Les citations gold ont été choisies en lisant directement le texte des `records` en base — **pas** en interrogeant le retrieval — pour éviter tout biais de circularité entre juge et système jugé.

## Fichiers

- `dev.jsonl` — 21 questions, utilisables librement pour développer/régler le système (choix d'algorithme, tuning, debug).
- `test.jsonl` — 9 questions, **réservées** : ne jamais s'en servir pour régler le système (paramètres, prompts, seuils…). Elles ne servent qu'à une mesure finale non biaisée. Si une question de `test.jsonl` a été consultée pendant un développement, elle est compromise et doit être remplacée, pas réutilisée.

Le split est **stratifié par catégorie et figé** : une fois tiré, il n'est jamais re-tiré (sous peine de fuite progressive du jeu de test vers le dev). Ajouter des questions ne change pas le split existant, voir « Procédure d'ajout » ci-dessous.

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
| `notes` | Un mot sur la réponse attendue (repère de correction, pas une réponse rédigée). |

## Catégories et répartition cible

| Catégorie | Effectif | Description |
|---|---|---|
| `reference_directe` | 5 | La question cite explicitement le numéro d'article (« Selon l'article 214-13… »). Cas le plus facile : le système doit au moins retrouver l'article nommé. |
| `regle` | 15 | Question en langage professionnel comptable, sans citer de numéro d'article — la formulation employée par un comptable ou un étudiant DSCG qui connaît le vocabulaire technique (« provision réglementée », « écarts de conversion », « levée d'option »…). |
| `vocabulaire_courant` | 10 | Question posée comme le ferait un créateur d'entreprise ou un non-comptable : mots familiers, pas de jargon PCG (« machine », « voiture de société en leasing », « client qui ne paie pas », « argent de côté »…). Teste la capacité du système à faire le pont entre langage courant et vocabulaire normatif. |

Répartition dev/test par catégorie (stratifiée, ~70 %/30 %) :

| Catégorie | dev | test |
|---|---|---|
| `reference_directe` | 4 | 1 |
| `regle` | 10 | 5 |
| `vocabulaire_courant` | 7 | 3 |
| **Total** | **21** | **9** |

## Portée thématique

Les questions couvrent des règles concrètes réparties sur plusieurs thèmes du PCG : amortissement (durée, mode, composants, valeur résiduelle), dépréciation des stocks, crédit-bail, provisions (définition, provisions réglementées, engagements de retraite), fonds commercial (acquis vs généré en interne, amortissement, dépréciation), jetons numériques (émission, définition), écarts de conversion en devises, créances clients douteuses ou irrécouvrables, participation des salariés, société en participation, solutions informatiques (logiciels), subventions d'investissement.

## Procédure d'ajout de questions

1. Choisir un thème du corpus non encore couvert (ou sous-couvert) et lire le texte réel des articles candidats dans `data/corpus.db` — jamais via le retrieval.
2. Rédiger la question et vérifier que la réponse se trouve effectivement dans le(s) article(s) cité(s).
3. Choisir la catégorie (`reference_directe` / `regle` / `vocabulaire_courant`) selon le registre de langue employé.
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
- total de 30 questions.

```sh
uv run pytest tests/test_benchmark_format.py -v
```

## Licence

Les questions et notes de correction sont une œuvre originale de ce dépôt (code, licence MIT — voir `LICENSE`). Les citations pointent vers du contenu dérivé du Recueil des normes comptables françaises (ANC), distribué sous licence **[Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)** (Etalab) — voir `DATA_LICENSE.md` à la racine du dépôt pour les modalités complètes et l'avertissement sur la fidélité du texte extrait.
