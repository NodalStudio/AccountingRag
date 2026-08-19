# AccountingRAG

Agent open source qui vise à savoir faire la comptabilité française d'aujourd'hui, adossé à un corpus structuré de sources exclusivement publiques. Étoile polaire du projet : composer sur de vrais sujets du DSCG UE4 (« Comptabilité et audit »).

Spécification complète : [`docs/superpowers/specs/2026-08-14-accountingrag-design.md`](docs/superpowers/specs/2026-08-14-accountingrag-design.md). Journal de bord (décisions, erreurs, découvertes, matière à article de blog) : [`JOURNAL.md`](JOURNAL.md).

## Avertissement

Ce projet est une **expérimentation de recherche**. Le corpus est produit par un parseur automatisé et le futur agent produira des synthèses assistées par LLM — **ceci n'est pas une doctrine comptable** et ne remplace pas l'avis d'un expert-comptable. En cas de doute ou d'usage professionnel, seuls les textes originaux font foi : [anc.gouv.fr](https://www.anc.gouv.fr/) et [Légifrance](https://www.legifrance.gouv.fr/).

## État du projet

Le jalon 1 livre un **parseur typographique déterministe** du Recueil des normes comptables françaises 2026 (Autorité des normes comptables), qui transforme le PDF source en un dataset SQLite structuré, sans passer par un LLM (le PDF a une typographie discriminante : taille et graisse de police distinguent réglementaire, commentaires et titres de section — voir la spec, section 3).

Chiffres du corpus produit (mesurés sur le build courant) :

- **1 660 enregistrements** : 739 réglementaires, 921 commentaires infra-réglementaires ANC
- **604 articles** distincts
- **981 renvois** en graphe : 711 internes (PCG), 173 externes légaux (autres codes), 97 historiques (CRC/Avis)
- Index de recherche plein texte **FTS5**
- **203 anomalies** cataloguées en 5 catégories, documentées dans [`docs/rapport-build.md`](docs/rapport-build.md)
- Validé par échantillonnage sur 15 pages du document source, voir [`docs/validation-echantillon.md`](docs/validation-echantillon.md)

## Démarrage rapide

```sh
uv run python scripts/download_data.py
uv run python scripts/build_corpus.py
```

La première commande télécharge le Recueil PCG 2026 depuis anc.gouv.fr dans `data/raw/`. La seconde exécute le pipeline complet (extraction → classification → parsing → contrôle d'intégrité) et produit `data/corpus.db` ainsi que le rapport d'anomalies `docs/rapport-build.md`.

Deux exemples de requêtes sur le corpus obtenu :

```sh
# Un article donné
sqlite3 data/corpus.db "SELECT id, chemin, texte FROM records WHERE article = '214-1';"

# Recherche plein texte (FTS5, correspondance exacte des tokens — pas de stemming
# sur records_fts ; le stemming FR n'est appliqué qu'à l'index jalon 2, chunks_norm)
sqlite3 data/corpus.db "SELECT r.id, r.chemin FROM records_fts f JOIN records r ON r.rowid = f.rowid WHERE f.texte MATCH 'amortissements';"
```

## Jalon 2 — retrieval hybride et première évaluation

Le jalon 2 ajoute, au-dessus du corpus SQLite du jalon 1, une chaîne d'analyse lexicale (élisions, références atomiques, synonymes, stemming FR), un index de recherche (chunking + FTS5 normalisé + embeddings vectoriels sqlite-vec) et un retrieval hybride (routeur de références, BM25, dense, fusion RRF, expansion par renvois 1-hop).

### Démarrage rapide — index et recherche

```sh
uv run python scripts/build_index.py
```

Ajoute l'index de recherche (chunks, FTS5 normalisé, vecteurs) à `data/corpus.db` (~14 min sur CPU ; télécharge le modèle d'embeddings `intfloat/multilingual-e5-small` au premier lancement, ~470 Mo mesurés dans le cache Hugging Face local).

L'encodeur dense est configurable via la variable d'environnement `ACCRAG_EMB_MODEL` (point d'extension dans `src/accounting_rag/embed.py`) : positionnez-la avec l'identifiant Hugging Face d'un autre modèle `sentence-transformers` pour remplacer le défaut `intfloat/multilingual-e5-small` (les préfixes `query:`/`passage:` ne sont ajoutés automatiquement que si `"e5"` figure dans le nom du modèle).

Exemple de recherche en Python :

```python
from pathlib import Path
from accounting_rag.search import Searcher

searcher = Searcher(Path("data/corpus.db"))
resultats = searcher.search("comment amortir un logiciel acheté ?", k=5, mode="hybrid+graph")
```

`mode` accepte `bm25`, `dense`, `hybrid` ou `hybrid+graph`.

### Résultats — campagne dev (21 questions), post-correction apostrophes/stemming du 15 août 2026

| mode | recall@5 | recall@10 | MRR |
|---|---|---|---|
| bm25 | 0,833 | 0,833 | 0,754 |
| dense | 0,571 | 0,738 | 0,518 |
| hybrid | 0,81 | 0,81 | 0,763 |
| hybrid+graph | 0,81 | 0,81 | 0,763 |

Chiffres mesurés après la correction de la table d'apostrophes typographiques (U+2019) et de l'ordre stem→fold dans `normalize.py` (revue finale du jalon 2, index reconstruit) — voir l'avant/après complet dans `docs/eval-jalon2.md`. Le fossé lexical entre langage courant et jargon PCG reste le principal goulot (recall@10 de 0,571 sur la catégorie `vocabulaire_courant` en bm25, contre 0,95-1,0 sur les questions qui citent un article ou emploient le vocabulaire professionnel). Détail complet (ventilation par catégorie, durées, analyse d'erreurs, conditions de reproduction) : [`docs/eval-jalon2.md`](docs/eval-jalon2.md). Méthodologie et format du benchmark de ce jalon (30 questions, remplacé depuis par le benchmark v2 ci-dessous) : voir l'historique dans [`benchmark/README.md`](benchmark/README.md).

## Jalon 2.5 — benchmark étendu et reranker cross-encoder

Le jalon 2.5 étend le benchmark à 90 questions (61 `dev` / 29 `test`, split stratifié et gelé) et mesure trois ablations par bootstrap apparié (`paired_bootstrap`, `n_boot=10000`, `seed=42`) : pondération par champ chemin/type (**rejetée**), reranker cross-encoder (**adoptée**), synonymes pilotés par les échecs mesurés (**rejetés**). Méthode, chiffres complets et réserves : [`docs/eval-jalon25.md`](docs/eval-jalon25.md) ; matériau d'analyse des échecs dev : [`docs/echecs-dev-jalon25.md`](docs/echecs-dev-jalon25.md) ; format et historique du benchmark v2 : [`benchmark/README.md`](benchmark/README.md).

**Configuration finale livrée** : `Searcher.search(mode="hybrid+rerank")`, reranker `BAAI/bge-reranker-v2-m3` (défaut de code), paramètres `Searcher` neutres (`poids_chemin=1.0`, `boost_commentaire=1.0`), synonymes d'origine (9 entrées, jalon 2). Le reranker est configurable via la variable d'environnement `ACCRAG_RERANKER` (identifiant Hugging Face d'un autre modèle `sentence-transformers` de type `CrossEncoder`) :

```python
resultats = searcher.search("comment amortir un logiciel acheté ?", k=5, mode="hybrid+rerank")
```

`mode` accepte désormais `bm25`, `dense`, `hybrid`, `hybrid+graph` ou `hybrid+rerank`.

**Le coût de `hybrid+rerank` dépend entièrement du device** — et le chiffre publié ici au jalon 2.5 était une latence **CPU**, corrigée au jalon 3 : ~1,5 s/question sur GPU contre ~150 s/question sur CPU — deux ordres de grandeur, à code identique (le facteur exact varie de ×66 à ×106 selon la charge : ne pas le citer avec des décimales) (contrôle direct dans [`docs/eval-jalon3.md`](docs/eval-jalon3.md), section « Le reranking du jalon 2.5 était mesuré sur CPU »). Conséquence pratique : `hybrid+rerank` est **utilisable en interactif dès qu'une carte est disponible**, et reste réservé au batch sans GPU (deux à trois heures pour une campagne de 61 questions). Il n'est **pas** inclus dans `scripts/run_eval.py --mode all` — précisément pour ne pas imposer ces heures de calcul à un contributeur sans GPU qui découvre le dépôt — et s'invoque explicitement avec `--mode hybrid+rerank`.

### Résultats — benchmark v2 (61 questions dev / 29 questions test gelé)

| split | config | recall@5 | recall@10 | MRR | latence/question |
|---|---|---|---|---|---|
| dev (n=61) | hybrid (baseline, usage interactif) | 0,639 | 0,672 | 0,565 | 0,22 s |
| dev (n=61) | hybrid+rerank (config finale) | 0,680 | 0,738 | 0,642 | 129,5 s |
| test (n=29, **référence gelée**) | hybrid (baseline) | 0,621 | 0,690 | 0,470 | 0,20 s |
| test (n=29, **référence gelée**) | hybrid+rerank (config finale) | 0,707 | 0,759 | 0,626 | 120,6 s |

Le split `test` a été gelé le 16 août 2026 et n'a jamais servi à régler quoi que ce soit ; à la date de cette section il avait été exécuté **une seule fois**, à la clôture du jalon 2.5 (une seconde exécution a eu lieu à la clôture du jalon 3 — voir la réserve 6 de `docs/eval-jalon3.md`) — voir `benchmark/README.md`. Le gain de recall@10 du reranker réplique sur test (delta=0,069, comparable au delta=0,0656 mesuré sur dev), mais le test statistique global (bootstrap apparié) y est moins net (`p_amelioration`=0,877 contre 0,952 sur dev) : lecture privilégiée, un effet de taille d'échantillon (n=29 vs n=61, puissance statistique réduite) plutôt qu'un sur-ajustement du reranker au dev (il n'a reçu aucun réglage de seuil ou d'hyperparamètre dérivé du dev) — détail complet et réserves dans [`docs/eval-jalon25.md`](docs/eval-jalon25.md), section « Clôture ».

## Jalon 3 — réécriture de requête par LLM

Le jalon 3 attaque le fossé lexical chiffré au jalon 2 : les questions posées en langage courant dont **aucun mot** n'apparaît dans l'article qui y répond. Quatre ablations mesurées par bootstrap apparié, une variable à la fois, critère d'adoption fixé avant les mesures (`p_amelioration ≥ 0,95` et aucune catégorie perdant plus de 0,05 de recall@10) :

| ablation | levier | décision |
|---|---|---|
| D | `df_max` — écarter les tokens peu discriminants de la requête | **rejetée** (dégradation monotone) |
| E | `pool` — élargir la fenêtre de candidats avant fusion | **rejetée** (mais livre le résultat central, ci-dessous) |
| F | `n_rerank` — élargir le pool soumis au cross-encoder | **rejetée** (la largeur n'achète rien) |
| G | **réécriture de la question par un LLM** vers le vocabulaire du PCG | **ADOPTÉE** |

**Résultat central des rejets** : la couverture du pool de candidats monte à 0,918 (0,839 sur `vocabulaire_courant`) quand le recall@10 après fusion RRF plafonne à 0,672. Le système *trouvait* la bonne réponse bien plus souvent qu'il ne la *restituait* — et ni un pool plus large, ni un cross-encoder voyant tous les candidats ne comblent cet écart. Le déficit n'était pas de rappel mais de reconnaissance sémantique.

**Ce qui marche** : traduire la question. `Rewriter` (Claude, `claude-sonnet-5`) réécrit « j'ai des marchandises en stock qui ont perdu de la valeur, je fais quoi ? » en vocabulaire PCG, et les deux canaux reçoivent `question + réécriture` (mode `etend`, qui domine le mode `remplace`). Les réécritures sont mises en cache dans un JSON versionné : les mesures publiées sont reproductibles à l'identique et gratuitement.

### Résultats — benchmark v2, configuration finale du jalon 3

| split | config | recall@5 | recall@10 | MRR | `vocabulaire_courant` |
|---|---|---|---|---|---|
| dev (n=61) | `hybrid` (baseline jalon 2.5) | 0,639 | 0,672 | 0,565 | 0,403 |
| dev (n=61) | `hybrid+rerank` (config jalon 2.5) | 0,680 | 0,738 | 0,642 | 0,484 |
| dev (n=61) | **réécriture `etend` + `hybrid+rerank`** | 0,762 | **0,877** | 0,714 | **0,774** |
| test (n=29, **gelé**) | `hybrid+rerank` (config jalon 2.5) | 0,707 | 0,759 | 0,626 | 0,500 |
| test (n=29, **gelé**) | **réécriture `etend` + `hybrid+rerank`** | 0,879 | **0,966** | 0,753 | **0,929** |

Bootstrap apparié contre la configuration du jalon 2.5 : dev +0,1393 (`p = 0,9945`), test gelé +0,2069 (`p = 0,9984`). **Le chiffre à retenir est celui du dev (0,877)** : le split gelé (29 questions) confirme la direction avec un intervalle de confiance presque deux fois plus large, il ne raffine pas l'estimation. Sur dev, la réécriture améliore 12 questions (dont dix réparations complètes et deux demi-réparations), en dégrade 3, et 4 résistent aux deux configurations — soit +9 questions en compte pour +8,5 points de recall — l'écart vient des deux demi-réparations.

Trois des quatre questions que le diagnostic fondateur donnait hors d'atteinte passent de 0 à 1 — dont celle dont le gold était au 88ᵉ percentile de son classement lexical. La quatrième (`q023`) résiste **en mode `hybrid` sans reranking**, et son autopsie livre une piste pour la suite : son gold est le **2ᵉ meilleur candidat lexical sur 1 653** mais absent du canal dense, et la somme RRF le fait perdre contre un candidat 5ᵉ et 6ᵉ — la fusion récompense le consensus, pas l'excellence. Le gold ressort au rang 11, donc à portée des 25 candidats soumis au cross-encoder, qui le rattrape : q023 réussit dans la configuration livrée. Le défaut de fusion est réel mais aujourd'hui compensé par la fenêtre du reranker — une compensation qui n'est pas garantie sur un corpus plus grand.

Méthode, chiffres complets, autopsies et réserves : [`docs/eval-jalon3.md`](docs/eval-jalon3.md).

**Réserve à lire avant de citer ces chiffres** : les 90 questions du benchmark ont toutes été écrites depuis le PCG, donc leurs citations attendues existent dans le corpus par construction. Le corpus se limite aux Livres I à V du règlement ANC 2014-03 — ni consolidation, ni NEP d'audit, ni BOFiP. (Cette réserve disait aussi « ni fusions » : **c'était faux**, et le jalon 4 l'a corrigé en allant y chercher 32 questions. Le Titre VII du Livre II, « Comptabilisation et évaluation des opérations de fusions et assimilées », compte 117 records dont 81 articles numérotés.) Ces scores mesurent la capacité à retrouver ce qui est présent, sur une distribution de questions qui exclut ce qui est absent.

### Configuration livrée par le jalon 3

```python
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

searcher = Searcher(
    "data/corpus.db",
    reranker=Reranker(),                                   # bge-reranker-v2-m3 (jalon 2.5)
    rewriter=Rewriter(cache_path="data/reecritures-cache.json"),   # cache d'EXÉCUTION
    mode_reecriture="etend",                               # jalon 3
)
resultats = searcher.search("j'ai payé un logiciel, je fais quoi ?", k=10, mode="hybrid+rerank")
```

La réécriture appelle l'API Claude : copier `.env.example` vers `.env` et y renseigner `ANTHROPIC_API_KEY` (`.env` est ignoré par git). Sans clé, `Searcher(rewriter=None)` reproduit exactement le comportement du jalon 2.5.

⚠️ **Le cache passé à `Rewriter` est un cache d'écriture** : `reecrire()` réécrit tout le fichier à chaque question absente. Utilisez un cache d'exécution (`data/reecritures-cache.json`, ignoré par git) et **jamais** `docs/mesures/jalon3/reecritures.json` — ce dernier est le cache **de mesure**, versionné, qui garantit la reproductibilité à l'identique des 0,877 et 0,966 publiés ci-dessus.

Reproduire la configuration livrée (les réécritures de `dev` sont déjà en cache, donc gratuites) :

```sh
# Copier le cache DE MESURE vers un cache d'exécution : la commande ne peut alors plus
# réécrire l'artefact versionné, quelles que soient les questions ajoutées au benchmark.
cp docs/mesures/jalon3/reecritures.json data/reecritures-cache.json
uv run python scripts/run_eval.py --mode hybrid+rerank --reecriture etend --split dev
uv run python scripts/cloture_jalon3.py    # campagne complète, ATTENTION : exécute le split gelé
```

Reproduire (dev, campagne rapide sans le reranker) :

```sh
uv run python scripts/run_eval.py --mode all --split dev
```

Reproduire la configuration du jalon 2.5 (~1,5 s/question sur GPU contre ~150 s/question sur CPU — cf. ci-dessus ; cette commande ne reproduit PAS la configuration du jalon 3, qui ajoute la réécriture) :

```sh
uv run python scripts/run_eval.py --mode hybrid+rerank --split dev
```

## Jalon 4 — mesurer la génération

Trois jalons avaient amélioré le *retrieval* sans jamais mesurer ce que le système **répond**. Le jalon 4 construit l'instrument : un générateur contraint par sortie structurée, la vérification programmatique des citations (aucun LLM), un LLM-juge dont l'accord avec 30 notes humaines est mesuré *avant* qu'il ne publie quoi que ce soit, une famille de questions d'abstention, et l'extension du benchmark de 90 à 150 questions.

### Résultats — benchmark v3

| split | citations inexistantes | non portantes | identifiants sans version | correspondance brute | abstention |
|---|---|---|---|---|---|
| dev (n=93, 598 citations) | **0,0** | 0,0067 | 0,1388 | 0,99 | 0,043 |
| validation (n=28, 175 citations, **gelé**) | **0,0** | **0,0** | 0,1429 | **1,0** | 0,0 |
| abstention (n=30) | 0,0 | 0,0 | 0,0 | 1,0 | **0,9667** correcte |

**Aucun article inventé sur les 779 citations des trois splits.** Sur le split de validation gelé, la correspondance brute vaut 1,0 : la normalisation de comparaison n'y fait strictement aucun travail. La famille d'abstention rend 29 refus corrects sur 30, avec **zéro fabrication**.

Juge : **kappa pondéré 0,9854** contre 30 notes humaines notées à la main, pour un seuil de **0,60** fixé avant toute mesure et lu dans le JSON de calibration pour qu'il ne puisse pas être déplacé après coup.

**Réserve à lire avant de citer ces chiffres** : ils ne sont **pas comparables** à ceux du jalon 3 — le benchmark est passé de 90 à 150 questions et la métrique a changé de nature. Le générateur cite en médiane 6 articles par réponse là où le gold en porte souvent un, ce qui rend le « taux de réponses sans citation » trivialement nul et dilue la précision. Et la première campagne a affiché **15,64 % de citations hallucinées dont aucune ne l'était** : elles nommaient le bon article avec un extrait verbatim et avaient perdu le suffixe de version. Le contrôle confondait « article inventé » et « identifiant abrégé ».

Méthode, chiffres complets, défauts trouvés et huit réserves : [`docs/eval-jalon4.md`](docs/eval-jalon4.md). Chaque chiffre est recalculable depuis `docs/mesures/jalon4/` par `scripts/controle_chiffres_jalon4.py`, qui vérifie aussi que chaque valeur publiée dans le rapport apparaît littéralement — il a sept tests, un par mode d'échec.

### Le benchmark v3

150 questions goldées (93 `dev`, 29 `test` — retiré du service, 28 `validation` **gelé le 18 août 2026**) et 30 questions d'abstention sans gold. Les sources sont ordonnées par le risque de **fuite** entre la question et le corpus, pas par le coût de goldage : dossier fusions du DSCG UE4, thèmes DCG UE9/UE10, divergences de la liasse 2058-A. Les rescrits BOFiP sont écartés jusqu'au jalon corpus — la question y est *dans* le document indexé, donc la mesure serait circulaire.

Huit questions portent `gold_fiscal: "a_completer"` : leur moitié comptable est goldée, leur moitié fiscale est **marquée, jamais inventée**. La métrique signature du projet — la confusion fiscal ↔ comptable — se prépare ainsi sans attendre le corpus fiscal.

Format, portée thématique et historique des gels : [`benchmark/README.md`](benchmark/README.md).

## Schéma

### Table `records`

Un enregistrement par article réglementaire ou par commentaire ANC.

| Champ | Description |
|---|---|
| `id` | Identifiant stable, ex. `pcg-214-1@2026-01-01` — suffixé `#n` en cas de collision d'identifiant : fragments multiples d'un même article (alinéas non contigus, sans titre de section intercalé) ou numérotation réutilisée par une annexe sectorielle (ex. secteur du logement social) |
| `article` | Numéro d'article PCG (`214-1`), ou `null` hors article (avant-propos, annexes non numérotées) |
| `chemin` | Position hiérarchique, ex. `Livre II > Titre I > Chapitre IV > Section 1` |
| `type` | `reglementaire` ou `commentaire_ANC` |
| `nature` | Domaine du contenu (`comptable`) |
| `opposable` | Toujours `false` sur ce corpus, par conception : rien n'est opposable dans les commentaires infra-réglementaires de l'ANC ; le champ est prévu pour le BOFiP (doctrine fiscale opposable), en phase 2 |
| `valide_du` / `valide_au` | Fenêtre de validité temporelle (édition 2026 uniquement en v1 : `valide_au` toujours nul) |
| `source_citation` | Citation normative extraite du titre du commentaire quand elle existe (Avis CNC/CU, règlement CRC cité en tête — mesuré : ~12 % des commentaires ANC), sinon le titre du commentaire lui-même |
| `page_debut` / `page_fin` | Bornes de pages dans le PDF source |

### Table `renvois`

Graphe des références croisées extraites du texte.

| Champ | Description |
|---|---|
| `source_id` | Référence vers `records.id` |
| `cible` | Article ou texte cible du renvoi |
| `famille` | `interne` (PCG), `externe_legal` (autre code), ou `historique` |

## Limitations connues

- **Strate typographique 9,0 non capturée** : citations de textes de rang supérieur (Code de commerce, lois), notes de bas de page et cellules de tableaux de modèles (~2 037 lignes sur 120 pages) sont actuellement classées comme bruit et absentes du corpus. Différé en v1.1 — voir [`docs/validation-echantillon.md`](docs/validation-echantillon.md).
- Certains identifiants sont suffixés `#n` (54 cas) : le plus souvent des fragments réglementaires multiples pour un même article déjà ouvert (alinéas non contigus), ou une numérotation réutilisée par une annexe sectorielle qui reprend partiellement celle du PCG (ex. secteur du logement social).
- **45 renvois pendants résiduels** (cibles non trouvées dans le corpus, essentiellement vers le plan de comptes en tableau, hors périmètre du parseur v1).
- Seule l'édition 2026 du Recueil est couverte : pas d'historique des versions antérieures en v1 (le schéma prévoit les champs temporels pour une extension future).
- **Jalon 2.5 (retrieval)** : reranker cross-encoder adopté (`hybrid+rerank`, ~1,5 s/question sur GPU contre ~150 s/question sur CPU, deux ordres de grandeur — batch/hors ligne **sans GPU** seulement, cf. la correction du jalon 3 ci-dessus) ; la pondération par champ (`chemin`/`type`) reste implémentée (`poids_chemin`, `boost_commentaire` sur `Searcher`) mais **non activée** (mesurée par bootstrap et rejetée, paramètres neutres par défaut) ; un lot de synonymes piloté par les échecs mesurés dev a également été rejeté (aucun effet mesurable, `p_amelioration=0`). Cause racine identifiée pour ce dernier rejet : la fenêtre `limit=50` de `_bm25()` (50 chunks, soit au plus 50 records après agrégation — nombre de candidats bm25 retenus avant la fusion RRF) est trop étroite pour que les gains de rang mesurés (jusqu'à ~1200 rangs sur le cas contrôlé) atteignent le top-50 effectivement exploité — goulot identifié, non résolu ici (hors périmètre "dictionnaire de synonymes"), matière du **jalon 3**. Restent également hors périmètre du jalon 2.5, matière du jalon 3 : la **réécriture de requête** (vocabulaire courant → vocabulaire PCG, prévue par la spec §4) et des **embeddings métier** spécialisés (conditionnés à un plafonnement démontré du dense généraliste — spec §8, Phase 3). Détail des trois ablations (méthode, chiffres, décisions) : [`docs/eval-jalon25.md`](docs/eval-jalon25.md).

## Feuille de route

- **v1.5** : normes d'exercice professionnel (NEP) ; premier benchmark DCG/DSCG.
- **Phase 2** : BOFiP (doctrine fiscale), LEGI (Code de commerce, CGI) ; IFRS adoptées par l'UE en fetch-at-build depuis EUR-Lex (jamais redistribuées dans ce dépôt, pour des raisons de licence).
- **Phase 3 (conditionnelle)** : embeddings spécialisés pour le domaine comptable français, si le benchmark montre un plafonnement du retrieval dense généraliste.

Détails dans la spécification, section 8.

## Licences

- **Code** : [MIT](LICENSE), Benoît Mayer.
- **Contenu extrait de l'ANC** : [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) (Etalab), avec attribution « Autorité des normes comptables ». Voir [`DATA_LICENSE.md`](DATA_LICENSE.md) — le PDF source publié par l'ANC reste la référence faisant foi.
