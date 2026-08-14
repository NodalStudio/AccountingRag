# AccountingRAG — Design v1

Date : 2026-08-14 · Statut : validé en session de brainstorming (voir `JOURNAL.md`)

## 1. Objectif

Construire un **agent open source qui sait faire la comptabilité française d'aujourd'hui**, adossé à un corpus structuré de sources exclusivement publiques et à un benchmark public. Étoile polaire : **composer sur de vrais sujets du DSCG UE4 (« Comptabilité et audit »)** — cadrage médiatique : « faire passer le DSCG à une IA, en candidat libre ».

Les livrables de valeur, dans l'ordre : (1) le **corpus structuré** (première structuration publique du texte normatif comptable français), (2) le **benchmark** (« DCG/DSCG-Bench », premier du genre), (3) le **RAG/agent** comme démonstration au-dessus.

### Non-objectifs

- Pas d'assistant pour experts-comptables en exercice (responsabilité, concurrence GenIA-L intenable — leur fossé : la doctrine propriétaire).
- Pas de couverture européenne multi-pays (jugée non viable — voir journal).
- Pas d'interface de chat en v1 : dataset + benchmark + pipeline d'abord.
- Pas d'historique des versions en v1 (l'agent fait « la compta d'aujourd'hui ») ; le schéma prévoit les champs temporels.

### Public cible

Développeurs, étudiants DCG/DSCG, chercheurs, créateurs d'entreprise. Avertissement explicite dans le README : synthèse par LLM, pas de valeur de doctrine.

## 2. Corpus

| Source | Contenu | Format d'origine | Distribution |
|---|---|---|---|
| Recueil des normes françaises (ANC) | PCG (règlement 2014-03) + fusions (Titre VII) + consolidation (règlement 2020-01) + **commentaires infra-réglementaires** | PDF texte natif (typographie discriminante) | Redistribué |
| BOFiP, séries BIC/IS (phase 2) | Doctrine fiscale opposable (L.80 A) | JSON/CSV + API data.economie.gouv.fr, versionnage natif | Redistribué (Licence Ouverte) |
| NEP (audit) (v1.5) | Normes d'exercice professionnel CAC, homologuées par arrêté | Légifrance | Redistribué |
| CGI / Code de commerce, extraits (phase 2) | Socle légal | Dumps XML LEGI (DILA), versionnage natif | Redistribué |
| IFRS adoptées par l'UE (phase 2+) | Règlement CE 1126/2008 et amendements | EUR-Lex | **Fetch-at-build uniquement** (clause « reproduction EEE seulement » ; identifiants CELEX + script ; jamais dans le repo) |

### Schéma cible (un enregistrement par article / commentaire / paragraphe)

```json
{
  "id": "pcg-214-1@2026-01-01",
  "article": "214-1",
  "chemin": "Livre II > Titre I > Chapitre IV > Section 1",
  "texte": "...",
  "type": "reglementaire | commentaire_ANC | bofip | nep | legi | ifrs",
  "nature": "comptable | fiscale | audit",
  "opposable": false,
  "valide_du": "2026-01-01", "valide_au": null,
  "source_citation": "Avis CU n° 2006-C du 4 octobre 2006",
  "renvois": [{"cible": "pcg-322-2", "famille": "interne|externe_legal|historique"}]
}
```

`nature` et `opposable` sont les champs les plus importants du projet : un index qui mélange PCG et BOFiP sans les tagger répond « non déductible » à qui demandait « amortissable ».

## 3. Parseur (déterministe, LLM aux marges)

Constat empirique (sondage PyMuPDF sur le Recueil 2026, 662 p.) : la typographie est discriminante — réglementaire = Tahoma 10,0 ; commentaires = Tahoma 9,5 (titres gras 9,5 **citant leur source**) ; sections = gras 10,6/12,0 ; en-têtes/pieds = 8,5/9,0.

Étapes : classification des blocs par signature (taille, graisse, x) → hiérarchie par double source (titres de sections × numérotation des articles, vérifiées l'une contre l'autre = test d'intégrité) → rattachement positionnel des commentaires + extraction regex de leur provenance → extraction des renvois (3 familles : internes PCG, externes légaux, historiques) → nettoyage (césures, exposants, sauts de page) → **rapport d'anomalies** ; seules les pages en anomalie passent par un LLM/revue humaine.

## 4. Indexation et retrieval

**Store : SQLite tout-en-un** (tables corpus + graphe de renvois + plan de comptes + FTS5 + sqlite-vec). Zéro serveur, `git clone && make build` ; le fichier .db est aussi le format de distribution du dataset. Couche d'accès isolée pour migration Postgres si le corpus décuple.

**Chaîne d'analyse lexicale domaine** (appliquée au build ET à la requête — c'est elle qui fait la qualité, pas le moteur) :
1. tokenisation préservant les références comme tokens atomiques (`214-1`, `L. 313-7`, `39-1-5°`, `BOI-BIC-AMT-10-20`) ;
2. élision + lemmatisation françaises (spaCy fr) sur colonne dédiée ;
3. dictionnaire de synonymes métier (fonds de commerce↔fonds commercial, IFC↔indemnités de fin de carrière…), enrichi via les échecs du benchmark ;
4. pondération par champ (en-tête/chemin > corps ; réglementaire > commentaire).

**Retrieval** : routeur regex (référence d'article → lookup direct) → hybride BM25 (FTS5 normalisé) + dense (bge-m3) → reranker cross-encoder → expansion 1-hop du graphe de renvois → small-to-big (embed à l'article, retourner la section) → filtres `nature`/dates. Réécriture de requête (vocabulaire courant → vocabulaire PCG) en amont.

**Génération** : citations obligatoires, vérification programmatique post-hoc (l'article cité existe et contient le passage), longueur d'extraits bornée (IFRS notamment).

## 5. Benchmark

Deux étages :
1. **DCG UE9/UE10** — questions courtes, notation quasi automatique : banc de test de développement.
2. **DSCG UE4** — cas pratiques complets (dossiers fusion / consolidation / audit), en conditions d'examen (sujet + annexes) : l'épreuve d'apparat. Sujets publics (ministère) ; **réponses gold rédigées par nous** (les corrigés privés ne sont pas réutilisables), avec citations d'articles attendues. Filtrage « réponse encore valable en 2026 ».

Catégories : question de règle / divergence fiscalo-comptable (métrique signature : taux de confusion fiscal↔comptable) / **écritures comptables** (quel compte, quelle écriture — ce qui distingue « agent comptable » de « chatbot juridique ») / vocabulaire courant / référence directe d'article. Complément : questions dérivées des divergences 2058-A et des commentaires ANC.

**Split dev/test figé dès la création** (~30 % réservés, jamais utilisés pour régler le système).

### Évaluation, deux étages

- **Retrieval** (quotidien, gratuit) : recall@k, MRR, nDCG sur les citations gold.
- **Bout en bout** (hebdomadaire, ~25 $ en batch) : LLM-juge (Opus 5) avec barème ; taux de confusion fiscal/comptable ; exactitude des écritures ; citations hallucinées (vérif. programmatique) ; contrôle humain par échantillon.
- **Baselines** : LLM frontière nu / RAG naïf / système complet — le tableau comparatif est le résultat-titre.

### Protocole d'ablation (dont phase 3 embeddings)

Une seule variable à la fois, tout le reste constant. Mesure en deux points : composant seul (dense seul pour les embeddings) ET bout de chaîne (le gain doit survivre au reranker). Significativité : bootstrap + comparaison appariée par question. Lecture par catégorie (un embedding métier doit gagner sur « vocabulaire courant », pas sur « référence directe »). Garde-fou anti-fuite : les données synthétiques d'entraînement ne doivent jamais paraphraser le test.

## 6. Modèles et exécution

- Juge + baseline frontière : Claude Opus 5. Génération RAG : Sonnet 5. Étapes mécaniques (validation de pages, extraction, mise en forme) : **sous-agents Haiku 4.5 / Sonnet** ; conception et vérifications : modèle principal.
- API Batch pour toute charge non interactive (−50 %) ; prompt caching (contenu stable en tête).
- Budget total estimé < 200–300 $ ; parsing ≈ 0 $ (pas d'OCR, texte natif).

## 7. Publication

- GitHub public dès le premier parseur fonctionnel. Code : MIT. Dataset : Licence Ouverte 2.0. Hugging Face : **différé** à la stabilisation du corpus.
- `JOURNAL.md` tenu à chaque session (matière pour article de blog).

## 8. Phases

1. **v1** : parseur Recueil NF 2026 + rapport d'anomalies → dataset SQLite → 30 questions d'échantillon (validation du format avec l'utilisateur) → chaîne d'analyse + retrieval → baselines + première campagne.
2. **v1.5** : NEP ; benchmark étendu (150–300 q.) ; campagne DSCG UE4 complète.
3. **Phase 2** : BOFiP BIC/IS, extraits CGI/Code de commerce, IFRS fetch-at-build ; divergences fiscalo-comptables actives.
4. **Phase 3 (conditionnelle au plafonnement mesuré)** : embeddings spécialisés « AccountingFR-embed » (paires synthétiques + entraînement contrastif sur base bge-m3, licence libre — recette Fin-E5 transposée) ; SPLADE éventuel.

## 9. Risques identifiés

- Notation des cas pratiques UE4 (calculs multi-étapes, annexes) : le morceau méthodologique le plus délicat — barèmes par étape + contrôle humain.
- Statut des sujets d'annales : sujets publics, corrigés privés — ne réutiliser que les énoncés, rédiger nos gold.
- Gabarit typographique des documents ANC annexes potentiellement différent → le rapport d'anomalies est le filet.
- Maintenance annuelle (nouveau Recueil chaque janvier) : le parseur rejouable est la réponse.
