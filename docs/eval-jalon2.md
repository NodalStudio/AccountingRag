# Jalon 2 — Première campagne d'évaluation du retrieval

Campagne exécutée le **15 août 2026** sur la branche `jalon-2-retrieval`, à l'issue des tâches T1-T7 (analyse lexicale, chunking, embeddings, index sqlite-vec, retrieval hybride, benchmark d'amorçage, harnais d'évaluation). Objectif : mesurer le recall@k et le MRR des quatre modes de retrieval (`bm25`, `dense`, `hybrid`, `hybrid+graph`) sur le split **dev** du benchmark, et documenter trois pistes d'amélioration concrètes pour le jalon 2.5.

## Conditions exactes

| Paramètre | Valeur |
|---|---|
| Corpus | `data/corpus.db`, 1 660 `records` (739 réglementaires, 921 commentaires ANC), 981 `renvois` |
| Index de recherche | 2 160 chunks, table FTS5 `chunks_norm` (texte normalisé : élisions, refs atomiques, synonymes, stemming FR), table vectorielle `chunks_vec` (sqlite-vec 0.1.9) |
| Modèle d'embeddings | `intfloat/multilingual-e5-small` (défaut de `Embedder`, 384 dimensions), préfixes `query:` / `passage:` |
| Fusion hybride | RRF (`k=60`) sur BM25 + dense ; `hybrid+graph` ajoute une expansion 1-hop via `renvois` (famille `interne`) sur les 5 premiers résultats, score pondéré ×0,5 |
| Benchmark | `benchmark/dev.jsonl` — 21 questions (4 `reference_directe`, 10 `regle`, 7 `vocabulaire_courant`) |
| Commande | `uv run python scripts/run_eval.py --mode all --split dev` |
| Machine | Linux 6.19.8-arch1-3-surface, x86_64, 8 threads, CPU uniquement (pas de GPU exploité — `torch` signale l'absence de driver CUDA compatible, sans conséquence : e5-small tourne en CPU) |
| Environnement | Python 3.13.14, `sentence-transformers` 5.7.0, `torch` 2.13.0+cu130 (CPU), `sqlite-vec` 0.1.9 — versions résolues par `uv.lock` |

**Contrôle de non-régression** : `bm25`/dev donne recall@5=0,833, recall@10=0,857, MRR=0,735 — identique aux chiffres de contrôle connus avant la campagne. Pas de blocage.

## Résultats — split dev (21 questions)

| mode | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| bm25 | 0,833 | 0,857 | 0,735 | 21 |
| dense | 0,571 | 0,738 | 0,518 | 21 |
| hybrid | 0,81 | 0,857 | 0,764 | 21 |
| hybrid+graph | 0,81 | 0,857 | 0,764 | 21 |

### Ventilation par catégorie (recall@10)

| mode | reference_directe (n=4) | regle (n=10) | vocabulaire_courant (n=7) |
|---|---|---|---|
| bm25 | 1,0 | 0,95 | 0,643 |
| dense | 1,0 | 1,0 | 0,214 |
| hybrid | 1,0 | 0,95 | 0,643 |
| hybrid+graph | 1,0 | 0,95 | 0,643 |

Sortie brute de `scripts/run_eval.py --mode all --split dev` :

```
| mode | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| bm25 | 0.833 | 0.857 | 0.735 | 21 |
|   ↳ reference_directe | | 1.0 | | 4 |
|   ↳ regle | | 0.95 | | 10 |
|   ↳ vocabulaire_courant | | 0.643 | | 7 |
| dense | 0.571 | 0.738 | 0.518 | 21 |
|   ↳ reference_directe | | 1.0 | | 4 |
|   ↳ regle | | 1.0 | | 10 |
|   ↳ vocabulaire_courant | | 0.214 | | 7 |
| hybrid | 0.81 | 0.857 | 0.764 | 21 |
|   ↳ reference_directe | | 1.0 | | 4 |
|   ↳ regle | | 0.95 | | 10 |
|   ↳ vocabulaire_courant | | 0.643 | | 7 |
| hybrid+graph | 0.81 | 0.857 | 0.764 | 21 |
|   ↳ reference_directe | | 1.0 | | 4 |
|   ↳ regle | | 0.95 | | 10 |
|   ↳ vocabulaire_courant | | 0.643 | | 7 |
```

### Lecture

- **`bm25` seul est déjà fort** sur `reference_directe` et `regle` (vocabulaire technique partagé avec les articles), mais s'effondre sur `vocabulaire_courant` (0,643) — le fossé lexical entre langage courant et jargon PCG, déjà pressenti pendant T4-T5, se confirme et se chiffre.
- **`dense` seul est pire que `bm25` globalement** (recall@5 0,571 contre 0,833) mais **meilleur sur `regle`** (1,0) : les embeddings e5-small captent bien la paraphrase professionnelle. Il est en revanche le plus faible mode sur `vocabulaire_courant` (0,214, pire que bm25) — la reformulation « grand public » s'éloigne trop du texte réglementaire pour que l'espace vectoriel généraliste fasse le pont.
- **`hybrid` (RRF bm25+dense) récupère l'essentiel de la force de bm25** (recall@10 identique 0,857) tout en améliorant le MRR (0,764 contre 0,735) : quand bm25 et dense sont tous deux corrects, le bon résultat monte en tête plus souvent.
- **`hybrid+graph` n'apporte aucun gain mesurable sur ce split** : recall et MRR strictement identiques à `hybrid`. L'expansion par renvois profite aux questions déjà résolues par le routeur/BM25 (elle enrichit le top-5 des `reference_directe`, cf. q001/q003 en analyse d'erreurs) mais ne comble aucun manque supplémentaire sur ces 21 questions — les articles gold manquants ne sont pas atteignables en 1-hop depuis les résultats déjà trouvés.
- **Le routeur regex (`_route`) fonctionne exactement comme annoncé** : sur les 4 questions `reference_directe`, l'article cité est toujours renvoyé en position 1 avec `source: "route"`.

## Durées (mesurées séparément, mêmes conditions)

| Étape | Durée |
|---|---|
| Connexion SQLite + chargement `sqlite-vec` | 0,56 s |
| Chargement du modèle `multilingual-e5-small` (premier accès, lazy) | 53,6 s |
| Évaluation `bm25` (21 questions) | 1,72 s (≈ 82 ms/question) |
| Évaluation `dense` (21 questions) | 5,44 s (≈ 259 ms/question) |
| Évaluation `hybrid` (21 questions) | 7,56 s (≈ 360 ms/question) |
| Évaluation `hybrid+graph` (21 questions) | 9,03 s (≈ 430 ms/question) |
| **`--mode all --split dev` de bout en bout** (mesure indépendante, CLI complet) | **87,85 s** au total (dont le chargement du modèle représente l'essentiel du coût fixe) |

Le chargement du modèle e5-small domine largement le temps total ; une fois chargé, chaque mode reste de l'ordre de la seconde à quelques secondes pour 21 questions — compatible avec un retrieval quotidien gratuit sur poste local (T7/spec section 5).

## Analyse d'erreurs — 3 questions dev au recall le plus bas (mode `hybrid+graph`)

Sélection : les 3 questions dev de plus faible recall@10 en mode `hybrid+graph` (ex-aequo à recall@10 nul départagés par MRR le plus faible). Aucune correction n'est implémentée ici — ce diagnostic nourrit le jalon 2.5.

### q021 — recall@5 = 0, recall@10 = 0

> « J'ai acheté une machine pour mon atelier, comment je répartis son coût sur les années où je vais m'en servir ? »

- **Citation attendue** : `pcg-214-13` (amortissement linéaire à défaut de mode mieux adapté).
- **Top-5 obtenu** :
  1. `pcg-131-3-c1@2026-01-01` — Livre V > Titre I logement social > Chapitre 3
  2. `pcg-na-236@2026-01-01` — Livre II > Titre VI > Chapitre III > Section 7 (certificats de valeur garantie)
  3. `pcg-na-280@2026-01-01` — même section, avis non repris
  4. `pcg-na-42@2026-01-01` — Livre I > Titre II > Chapitre II > Section 3 (désendettement de fait)
  5. `pcg-na-149@2026-01-01` — Livre II > Titre VI > Section 8 (instruments financiers à terme)
- **Diagnostic** : fossé lexical total — la question ne contient aucun terme (« amortir », « amortissement », « durée d'utilisation ») partagé avec le texte de l'article 214-13, et aucune entrée de synonymes ne relie « répartir le coût sur les années » à « amortissement » ; le dense (e5-small généraliste) ne comble pas non plus ce pont sémantique, d'où un top-5 entièrement hors-sujet.

### q026 — recall@5 = 0, recall@10 = 0

> « J'ai créé ma clientèle et ma réputation moi-même depuis le début, est-ce que je peux lui donner une valeur à l'actif de mon bilan ? »

- **Citation attendue** : `pcg-212-3` (fonds commercial créé en interne — non immobilisable).
- **Top-5 obtenu** :
  1. `pcg-na-57@2026-01-01` — Livre I > Titre III Le passif > Chapitre II
  2. `pcg-na-318@2026-01-01` — Livre III > Titre VIII > Chapitre IV (comptes intermédiaires)
  3. `pcg-836-1@2026-01-01#3` — Livre III > Titre VIII > Chapitre III > Section 6 (engagements hors bilan)
  4. `pcg-211-5-c1@2026-01-01` — Livre I > Titre II > Chapitre I > Section 1 (définitions actifs incorporels)
  5. `pcg-743-1@2026-01-01#2` — Livre II > Titre VII (fusions)
- **Diagnostic** : synonyme manquant — « clientèle » et « réputation » ne sont reliés ni lexicalement ni par la table de synonymes au terme technique « fonds commercial » ; à titre de comparaison, q009 qui emploie explicitement « fonds commercial constitué progressivement » obtient un recall parfait en rang 1. Le résultat n°4 (`211-5-c1`, définitions des actifs incorporels) est le plus proche thématiquement mais reste un article voisin, pas l'article gold.

### q022 — recall@5 = 0,5, recall@10 = 0,5

> « J'ai un client qui ne paie pas depuis des mois, et un autre dont je suis sûr qu'il ne paiera plus jamais : est-ce que je les traite pareil dans ma compta ? »

- **Citations attendues** : `pcg-1214-41` (reclassement en compte 416, client douteux) **et** `pcg-1221-65` (perte certaine et définitive, compte 654).
- **Top-5 obtenu** :
  1. `pcg-1121-1@2026-01-01` — Livre IV > Titre XI (plan de comptes, chapeau)
  2. `pcg-1214-41@2026-01-01` — Livre IV > Titre XII > Section 4 (comptes de tiers) — **citation trouvée**
  3. `pcg-na-318@2026-01-01` — Livre III > Titre VIII > Chapitre IV
  4. `pcg-324-1-c23@2026-01-01` — Livre I > Titre III > Chapitre II > Section 4 (pensions, retraites)
  5. `pcg-na-230@2026-01-01` — Livre II > Titre VI > Chapitre III > Section 6
- **Diagnostic** : question à deux volets, retrieval qui n'en couvre qu'un — le volet « client douteux » (`1214-41`) est trouvé en rang 2, mais le second volet « perte certaine et définitive/irrécouvrable » (`1221-65`) n'apparaît dans aucun des 10 premiers résultats, probablement parce que le vocabulaire courant « il ne paiera plus jamais » ne recoupe pas les termes de l'article (perte, créance irrécouvrable) plus fortement que d'autres chunks du plan de comptes. Symptomatique des questions multi-citations en `vocabulaire_courant` : chaque sous-question doit indépendamment franchir le fossé lexical.

### Pistes qui se dégagent (à instruire en jalon 2.5, sans implémentation ici)

1. Le fossé lexical `vocabulaire_courant` est le principal goulot (recall@10 0,643 en bm25/hybrid contre 0,95-1,0 pour les autres catégories) — candidat naturel pour un reranker ou un enrichissement ciblé de synonymes, mais rappel du ruling T1 : n'ajouter une entrée de synonymes qu'après un échec mesuré, jamais par intuition.
2. `hybrid+graph` n'aide pas sur ce split dev (0 gain observé) — l'expansion 1-hop ne compense pas un échec initial de bm25/dense, elle ne fait qu'enrichir un top-5 déjà correct. À revérifier sur un split plus large avant de conclure définitivement.
3. Les questions à citations multiples (`vocabulaire_courant` notamment) exposent une limite du recall global : une seule des N citations trouvée suffit à un score partiel qui masque un échec complet sur l'autre volet — pas un défaut de l'outil de mesure, mais un signal que ces questions méritent un examen qualitatif séparé, pas seulement l'agrégat.

## Référence gelée jalon 2 — split test (9 questions, exécuté une seule fois)

Conformément au protocole du benchmark (`benchmark/README.md`) : le split test est **gelé**, on ne le consulte qu'une fois par jalon, à la toute fin, après verrouillage du système, et jamais pour du tuning. Résultat exécuté une unique fois pour cette campagne, à titre de référence non biaisée — **non utilisé** dans l'analyse d'erreurs ci-dessus.

```
| mode | recall@5 | recall@10 | MRR | n |
|---|---|---|---|---|
| bm25 | 0.889 | 0.889 | 0.75 | 9 |
|   ↳ reference_directe | | 1.0 | | 1 |
|   ↳ regle | | 1.0 | | 5 |
|   ↳ vocabulaire_courant | | 0.667 | | 3 |
| dense | 0.556 | 0.667 | 0.485 | 9 |
|   ↳ reference_directe | | 1.0 | | 1 |
|   ↳ regle | | 0.8 | | 5 |
|   ↳ vocabulaire_courant | | 0.333 | | 3 |
| hybrid | 0.778 | 0.889 | 0.631 | 9 |
|   ↳ reference_directe | | 1.0 | | 1 |
|   ↳ regle | | 1.0 | | 5 |
|   ↳ vocabulaire_courant | | 0.667 | | 3 |
| hybrid+graph | 0.778 | 0.889 | 0.631 | 9 |
|   ↳ reference_directe | | 1.0 | | 1 |
|   ↳ regle | | 1.0 | | 5 |
|   ↳ vocabulaire_courant | | 0.667 | | 3 |
```

Tendance cohérente avec le dev (bm25/hybrid autour de 0,89 de recall@10, dense plus faible, `vocabulaire_courant` la catégorie la plus dure) — mais n=9 seulement, à ne pas surinterpréter statistiquement. Ce tableau ne doit **pas** servir à ajuster le système avant le prochain jalon.

## Reproduction exacte

```sh
# Depuis la racine du dépôt, avec uv installé
uv run python scripts/build_corpus.py     # si data/corpus.db n'existe pas encore
uv run python scripts/build_index.py      # construit chunks + FTS + vecteurs (~14 min CPU, télécharge e5-small au premier lancement)
uv run python scripts/run_eval.py --mode all --split dev
uv run pytest tests/test_evalrag.py tests/test_benchmark_format.py -v
```

Le split `test` ne doit être relancé qu'en fin de jalon suivant, jamais pendant le développement.

## Réserves

- n=21 (dev) et n=9 (test) : échantillons volontairement petits (benchmark d'amorçage T6), les pourcentages par catégorie reposent sur 4 à 10 questions — un écart d'une question déplace le recall d'une catégorie de plusieurs points. Pas de test statistique de significativité à ce stade.
- Les durées de chargement du modèle (~53 s) varient selon l'état du cache Hugging Face local et la charge CPU de la machine ; elles ne sont indicatives que dans les conditions mesurées ici.
- `hybrid+graph` égal à `hybrid` sur ce split ne prouve pas l'inutilité de l'expansion par renvois en général — seulement qu'elle n'a comblé aucun manque sur ces 21 questions précises.
