# AccountingRAG

Agent open source qui vise à savoir faire la comptabilité française d'aujourd'hui, adossé à un corpus structuré de sources exclusivement publiques. Étoile polaire du projet : composer sur de vrais sujets du DSCG UE4 (« Comptabilité et audit »).

Spécification complète : [`docs/superpowers/specs/2026-08-14-accountingrag-design.md`](docs/superpowers/specs/2026-08-14-accountingrag-design.md). Journal de bord (décisions, erreurs, découvertes, matière à article de blog) : [`JOURNAL.md`](JOURNAL.md).

## Avertissement

Ce projet est une **expérimentation de recherche**. Le corpus est produit par un parseur automatisé et le futur agent produira des synthèses assistées par LLM — **ceci n'est pas une doctrine comptable** et ne remplace pas l'avis d'un expert-comptable. En cas de doute ou d'usage professionnel, seuls les textes originaux font foi : [anc.gouv.fr](https://www.anc.gouv.fr/) et [Légifrance](https://www.legifrance.gouv.fr/).

## État du projet — jalon 1 livré

Le jalon 1 livre un **parseur typographique déterministe** du Recueil des normes comptables françaises 2026 (Autorité des normes comptables), qui transforme le PDF source en un dataset SQLite structuré, sans passer par un LLM (le PDF a une typographie discriminante : taille et graisse de police distinguent réglementaire, commentaires et titres de section — voir la spec, section 3).

Chiffres du corpus produit (mesurés sur le build courant) :

- **1 660 enregistrements** : 739 réglementaires, 921 commentaires infra-réglementaires ANC
- **602 articles** distincts
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

# Recherche plein texte (FTS5, correspondance exacte des tokens — pas de stemming en v1)
sqlite3 data/corpus.db "SELECT r.id, r.chemin FROM records_fts f JOIN records r ON r.rowid = f.rowid WHERE f.texte MATCH 'amortissements';"
```

## Jalon 2 — retrieval hybride et première évaluation

Le jalon 2 ajoute, au-dessus du corpus SQLite du jalon 1, une chaîne d'analyse lexicale (élisions, références atomiques, synonymes, stemming FR), un index de recherche (chunking + FTS5 normalisé + embeddings vectoriels sqlite-vec) et un retrieval hybride (routeur de références, BM25, dense, fusion RRF, expansion par renvois 1-hop).

### Démarrage rapide — index et recherche

```sh
uv run python scripts/build_index.py
```

Ajoute l'index de recherche (chunks, FTS5 normalisé, vecteurs) à `data/corpus.db` (~14 min sur CPU ; télécharge le modèle d'embeddings `intfloat/multilingual-e5-small` au premier lancement, ~130 Mo).

Exemple de recherche en Python :

```python
from pathlib import Path
from accounting_rag.search import Searcher

searcher = Searcher(Path("data/corpus.db"))
resultats = searcher.search("comment amortir un logiciel acheté ?", k=5, mode="hybrid+graph")
```

`mode` accepte `bm25`, `dense`, `hybrid` ou `hybrid+graph`.

### Résultats — première campagne (dev, 21 questions, 15 août 2026)

| mode | recall@5 | recall@10 | MRR |
|---|---|---|---|
| bm25 | 0,833 | 0,857 | 0,735 |
| dense | 0,571 | 0,738 | 0,518 |
| hybrid | 0,81 | 0,857 | 0,764 |
| hybrid+graph | 0,81 | 0,857 | 0,764 |

Le fossé lexical entre langage courant et jargon PCG reste le principal goulot (recall@10 de 0,643 sur la catégorie `vocabulaire_courant`, contre 0,95-1,0 sur les questions qui citent un article ou emploient le vocabulaire professionnel). Détail complet (ventilation par catégorie, durées, analyse d'erreurs, conditions de reproduction) : [`docs/eval-jalon2.md`](docs/eval-jalon2.md). Méthodologie et format du benchmark : [`benchmark/README.md`](benchmark/README.md).

Reproduire la campagne :

```sh
uv run python scripts/run_eval.py --mode all --split dev
```

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

## Feuille de route

- **v1.5** : normes d'exercice professionnel (NEP) ; premier benchmark DCG/DSCG.
- **Phase 2** : BOFiP (doctrine fiscale), LEGI (Code de commerce, CGI) ; IFRS adoptées par l'UE en fetch-at-build depuis EUR-Lex (jamais redistribuées dans ce dépôt, pour des raisons de licence).
- **Phase 3 (conditionnelle)** : embeddings spécialisés pour le domaine comptable français, si le benchmark montre un plafonnement du retrieval dense généraliste.

Détails dans la spécification, section 8.

## Licences

- **Code** : [MIT](LICENSE), Benoît Mayer.
- **Contenu extrait de l'ANC** : [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) (Etalab), avec attribution « Autorité des normes comptables ». Voir [`DATA_LICENSE.md`](DATA_LICENSE.md) — le PDF source publié par l'ANC reste la référence faisant foi.
