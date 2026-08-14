# Journal de bord — AccountingRAG

> Notes brutes du projet, tenues au fil de l'eau. Matière première pour un article de blog.
> Convention : chaque session datée, avec ce qu'on a décidé, ce qu'on a tenté, ce qui a marché ou non.

---

## Session du 14 août 2026 — De l'idée au plan

### L'idée de départ (et sa démolition)

Idée initiale : un RAG comptable open source couvrant les règles de tous les pays d'Europe.

Verdict après analyse critique (« be harsh ») : **non viable à cette échelle**. Les raisons :

1. « Règles comptables » = ~30 corpus (GAAP locaux + IFRS + fiscal) en 24 langues, en mouvement permanent. Un corpus périmé de 6 mois est dangereux (les réponses comptables dépendent des dates).
2. Le contenu qui répond vraiment aux questions (doctrine, commentaires — Mémento Lefebvre, Beck'scher Kommentar) est propriétaire. Le droit brut est public mais insuffisant.
3. Le RAG est une commodité en 2026 ; le fossé défensif, c'est le corpus — et les incumbents (Wolters Kluwer « Expert AI », Lefebvre Dalloz « GenIA-L », 45 000+ professionnels) le possèdent.
4. Le RAG naïf (chunk-and-embed) échoue sur la législation : structure hiérarchique, dates d'effet, amendements, renvois.
5. Personne ne maintiendra bénévolement le volet maltais ou slovène.

**Pivot retenu : France uniquement, PCG + règlements ANC.** Et surtout : le livrable de valeur n'est pas un chatbot mais un **dataset structuré + un benchmark** (« PCG-Bench »). Le RAG devient la démo au-dessus.

### État de l'art vérifié

- **Commercial : la course est finie.** GenIA-L (Lefebvre Dalloz) domine, adossé à son contenu éditorial propriétaire mis à jour quotidiennement.
- **Open source : la niche est vide.** Ce qui existe : le plan de comptes en JSON (arrhes/PCG, coulba-compta/PCG, data.gouv.fr) — mais personne n'a structuré le texte normatif du règlement 2014-03. Les corpus juridiques français existants (AgentPublic/legi, louisbrulenaudet/legalkit sur Hugging Face) couvrent tout **sauf** les normes comptables : les règlements ANC vivent hors de LEGI.
- Aucun benchmark de Q&A comptable français n'existe.

### Ce que Lefebvre apporte qu'on n'aura pas (et ce qu'on a qu'ils n'ouvriront jamais)

- Eux : la doctrine (~90 % de la valeur), l'articulation fiscal/comptable/social, la responsabilité éditoriale, la fraîcheur industrialisée. Tout est payant ET non redistribuable (droit d'auteur + droit sui generis des bases de données).
- Nous : un corpus **redistribuable**, les **commentaires infra-réglementaires de l'ANC** (mini-doctrine officielle publique que personne n'a structurée), et un **benchmark public** — qu'aucun éditeur ne publiera jamais (aucun intérêt à être mesuré).
- Principe juridique clé : tout ce qui est *opposable* est public (Légifrance, BOFiP sous Licence Ouverte). La doctrine privée n'est pas du droit — c'est du confort professionnel.

### Positionnement assumé

Cible : développeurs, étudiants (DCG/DSCG), chercheurs, créateurs d'entreprise — **pas** les experts-comptables en exercice (liability, concurrence intenable). Couverture estimée des sources publiques : ~80 % des questions courantes ; la synthèse multi-sources par LLM reste le facteur limitant sur les questions pointues.

### Architecture technique retenue

Séquence : **corpus → benchmark → RAG** (pas l'inverse — sans benchmark on règle le retrieval à l'aveugle).

Techniques spécifiques aux textes juridiques :
1. Chunking structurel au niveau article, chemin hiérarchique préfixé (« Livre II > Titre I > ... »).
2. Routeur regex pour les références directes d'articles (avant tout embedding).
3. Métadonnées temporelles (valide_du/valide_au) — les réponses comptables dépendent des dates.
4. Graphe de renvois, expansion 1-hop au retrieval.
5. Hybride BM25 + dense (bge-m3) + reranker ; small-to-big.
6. Réécriture de requête (vocabulaire courant → vocabulaire PCG).
7. Plan de comptes = données structurées (lookup), pas embeddings.
8. Citations obligatoires + vérification programmatique (anti-hallucination).
9. **Champs critiques du schéma : `nature` (comptable|fiscale) et `opposable`** — un RAG qui mélange BOFiP et PCG sans tag répond « non déductible » à qui demandait « amortissable ».

Schéma cible (un enregistrement par article/commentaire) :
```json
{
  "id": "pcg-214-1@2026-01-01",
  "article": "214-1",
  "chemin": "Livre II > Titre I > Chapitre IV > Section 1",
  "texte": "...",
  "type": "reglementaire | commentaire_ANC | bofip | avis_cnc",
  "nature": "comptable | fiscale",
  "opposable": false,
  "valide_du": "2026-01-01", "valide_au": null,
  "renvois": ["pcg-322-2", "bofip-BIC-AMT-..."]
}
```

### Le benchmark

Sources de questions : divergences fiscalo-comptables (chaque ligne du 2058-A = une question à double réponse), annales DCG/DSCG, commentaires ANC retournés en Q&A. Cible : 150–300 questions vérifiées à la main.

Métriques à deux étages :
- **Retrieval seul** (recall@k sur articles gold) : automatique, gratuit, quotidien.
- **Bout en bout** (LLM-juge) : justesse, **taux de confusion fiscal/comptable** (métrique signature), exactitude temporelle, citations hallucinées.

Baselines : LLM frontière nu / RAG naïf / système complet. Le tableau comparatif = résultat-titre du README.

Exemples de divergences (mine d'or de questions) : provision IFC (obligatoire en compta, jamais déductible — art. 39-1-5° CGI), fonds commercial (amortissable PME 10 ans / non déductible sauf fenêtre 2022-2025), amortissements dérogatoires, participation (décalage N/N+1), écarts de conversion (art. 38-4), régime mère-fille.

### Modèles et coûts

- Baseline frontière + juge : Claude Opus 5 ($5/$25 par MTok). Génération RAG : Sonnet 5 ($3/$15, intro $2/$10 jusqu'au 31/08/2026). Tâches mécaniques : Haiku 4.5 ($1/$5).
- Campagne d'éval complète (300 q × 3 systèmes, génération + juge) ≈ 40–50 $, **÷2 via l'API Batch**, encore réduit par le prompt caching (grille du juge en tête de prompt).
- Développement quotidien : 0 $ (recall@k sans LLM, bge-m3 local).
- Budget total projet estimé < 200–300 $. Le coût dominant = le temps humain sur le parsing.

### Découverte clé : pas d'OCR du tout

Test empirique (Recueil 2026 téléchargé, 662 pages, 6,9 Mo) : **PDF texte natif**, extraction parfaite via pdftotext, accents compris.

Mieux : sondage typographique avec PyMuPDF (script `inspect_fonts.py`) →
- Texte réglementaire : Tahoma **10,0** (en-têtes d'article : gras 10,0, « Art. N »)
- Commentaires infra-réglementaires : Tahoma **9,5** (titres : gras 9,5, citant leur source — « Avis CU n° 2006-C du 4 octobre 2006 »)
- Sections : gras 10,6 / 12,0 ; en-têtes/pieds de page : 8,5 / 9,0.

**Un demi-point de taille de police sépare le droit de la doctrine.** Le parseur peut être déterministe, vérifiable, rejouable ; le LLM ne sert qu'en filet de sécurité sur les anomalies. Coût de parsing ≈ 0 $.

Vérification croisée intégrée : la numérotation des articles (212-5 = Livre 2, Titre 1, Chap. 2) recoupe les titres de sections → test d'intégrité gratuit.

(Pour mémoire, le paysage OCR si un jour on croise des scans : Gemini Flash-Lite ~0,33 $/1000 pages, Mistral OCR ~2-5 $/1000 pages, Docling/Marker/olmOCR gratuits en local.)

### Traitement des autres sources publiques

- **BOFiP** : datasets JSON/CSV + API sur data.economie.gouv.fr (contenu intégral, « en vigueur », rescrits). Chunk au **paragraphe numéroté** (§ = unité de citation), versionnage natif (opposabilité L.80 A = version en vigueur au moment des faits), filtrer aux séries BIC/IS, renvois BOI-xxx par regex.
- **LEGI (CGI, Code de commerce)** : dumps XML DILA / API PISTE — versionnage temporel natif (chaque version d'article a ses dates de début/fin), rien à reconstruire. AgentPublic/legi existe déjà en version embeddée.
- Réponses ministérielles, jurisprudence : phases ultérieures.

### Décisions d'organisation

- Sous-agents moins chers (Haiku/Sonnet) pour les étapes mécaniques ; le modèle principal garde la conception et la vérification.
- Ce journal est tenu à chaque session (demande explicite : matière pour un article de blog).

### Décisions prises (suite de session — questions/réponses)

- **Reformulation fondatrice de l'objectif** (par l'utilisateur) : « un agent qui sait faire la compta aujourd'hui » — pas un assistant de recherche historique. Conséquence : édition 2026 seule en v1, pas de diff historique (champs temporels prévus au schéma quand même). Autre conséquence : le benchmark inclut des questions d'*écritures comptables* (passer une écriture ≠ répondre à un quiz).
- **Étoile polaire : DSCG UE4 « Comptabilité et audit »** — l'épreuve reine. Cadrage blog : « faire passer le DSCG à une IA en candidat libre » (clin d'œil à l'inscription SIEC). Conséquence corpus : Recueil des normes françaises **complet** (PCG + fusions Titre VII + consolidation ANC 2020-01 — même source, même gabarit, même parseur) au lieu du PCG seul ; NEP (audit, publiques sur Légifrance) en v1.5 ; IFRS plus tard. Benchmark à deux étages : DCG UE9/UE10 (banc de développement, notation facile) / DSCG UE4 (cas pratiques complets, l'épreuve d'apparat).
- **IFRS et EUR-Lex — la subtilité juridique** (creusée suite à question utilisateur) : le texte des IFRS adoptées par l'UE est intégralement publié sur EUR-Lex (règlement CE 1126/2008)... mais avec la clause « Reproduction autorisée dans l'EEE, tous droits réservés en dehors » (copyright IFRS Foundation, contenu tiers dans le droit UE). Incompatible avec une redistribution mondiale sous licence libre. **Parade : « ships scripts, not data »** — identifiants CELEX + script fetch-at-build ; l'index local de chaque déploiement contient les IFRS, le repo n'en contient pas une ligne. Pour le RAG : zéro différence (construire un index = TDM, art. L.122-5-3 CPI ; citer un court passage = courte citation). Pépite blog : « pourquoi mon dataset contient tout le droit comptable français mais pas une ligne d'IFRS — et pourquoi ça ne change rien pour l'agent ».
- **Publication** : GitHub public dès maintenant (MIT pour le code, Licence Ouverte 2.0 pour le dataset) ; **Hugging Face différé** à la stabilisation du corpus (éviter de versionner un dataset qui bouge tous les jours).
- **Benchmark** : annales DCG/DSCG comme source principale (choix utilisateur : « mon modèle passe le DSCG » comme métrique-titre). Sujets publics, corrigés privés → on rédige nos réponses gold nous-mêmes, filtrées « encore valables en 2026 ».

### Indexation — discussion technique (question utilisateur : « il n'y a pas plus adapté ? »)

- Choix du store : **SQLite tout-en-un** (FTS5 + sqlite-vec + tables corpus/renvois/plan de comptes). Zéro serveur, reproductible (`git clone && make build`), et le .db EST le format de distribution du dataset. Ligne de blog : « tout le droit comptable français dans un fichier SQLite de 50 Mo ». Alternatives écartées : Postgres (meilleur stemming français natif mais friction d'installation), Elastic/Qdrant (overkill).
- Clarification importante (question « les outils de texte sont-ils plus ou moins efficaces selon le domaine ? ») : **les moteurs FTS se valent (BM25 partout) ; c'est la chaîne d'analyse qui fait le domaine.** Quatre adaptations décisives pour le juridique français : (1) tokenisation préservant les références (`214-1`, `L. 313-7`, `BOI-BIC-AMT-10-20` en tokens atomiques — un tokenizer par défaut les pulvérise), (2) élision + lemmatisation françaises (spaCy au build), (3) dictionnaire de synonymes métier (fonds de commerce↔fonds commercial, IFC↔indemnités de fin de carrière), (4) pondération par champ. FTS5 n'a rien de tout ça nativement → on fait tout au build sur colonne normalisée, même normalisation à la requête.
- Confusion levée sur les « tokenizers » (l'utilisateur avait trouvé FinanceMTEB/Fin-e5-tokenizer sur HF) : le tokenizer *neuronal* d'un modèle d'embeddings (sous-mots → ids) n'a rien à voir avec le tokenizer *lexical* d'un index FTS. Mais la trouvaille pointe une idée retenue : **Fin-E5 prouve que les embeddings adaptés au domaine battent les généralistes** (e5-mistral-7b fine-tuné finance, 1er de FinMTEB — mais anglais + licence CC-BY-NC, inutilisable pour nous). → **Phase 3 conditionnelle : « AccountingFR-embed »**, notre propre embedding comptable français (paires synthétiques depuis le corpus + entraînement contrastif sur bge-m3), si le benchmark montre que le dense plafonne.

### Protocole de mesure des améliorations (embeddings, et toute variante du pipeline)

1. Instrument : recall@k / MRR / nDCG sur les citations gold (automatique, gratuit).
2. Ablation stricte : une variable à la fois ; mesure en deux points — composant seul (dense seul) ET bout de chaîne (le gain doit survivre au reranker, qui rattrape souvent un retrieval médiocre).
3. Anti-fuite : split dev/test figé (~30 % jamais touchés) ; les paires synthétiques d'entraînement ne doivent pas paraphraser le test.
4. Significativité : bootstrap + comparaison appariée par question (gagne/perd/égalise).
5. Lecture par catégorie : le gain d'un embedding métier doit se concentrer sur « vocabulaire courant » ; « référence directe » (routée regex) ne doit pas bouger.

### « Il faudra plus que le DSCG pour juger tout ça » (remarque utilisateur, actée)

Trois angles morts du DSCG comme instrument : pas de citations gold (donc pas de recall@k), couverture limitée aux chapitres fétiches d'examen, français académique (la robustesse au langage courant est invisible). → Benchmark restructuré en **cinq familles** : DSCG UE4 (apparat) / DCG UE9-UE10 (banc de développement) / questions ciblées à citations gold couvrant le corpus (instrument retrieval) / questions « vie réelle » en langage familier (fossé lexical) / **questions-pièges et hors-périmètre mesurant l'abstention correcte** — la famille qu'on oublie toujours et qui fait la crédibilité (un agent comptable qui invente est pire qu'inutile). Spec section 5 amendée.

### Spec écrite

Design complet validé et rédigé dans `docs/superpowers/specs/2026-08-14-accountingrag-design.md`. Prochaine étape : revue de la spec par l'utilisateur, puis plan d'implémentation détaillé (skill writing-plans), exécution avec sous-agents économiques.

---
## Session du 14-15 août 2026 — Exécution du jalon 1 : la saga de l'extraction

Exécution du plan en mode sous-agents (implémenteurs Haiku/Sonnet, relecteurs Sonnet, contrôleur qui n'écrit jamais le code lui-même). T1/T2 (scaffolding, modèle de données) : sans histoire, revue propre du premier coup.

**T3/T4 (extraction PDF + classification typographique) : quatre rounds de correction — et une leçon.** Le code « évident » d'extraction PyMuPDF a caché quatre causes racines de corruption de texte, toutes invisibles sur les pages de test initiales, toutes débusquées par des relecteurs qui vérifiaient sur les 662 pages réelles :

1. **La fusion naïve des spans perd des espaces** : « propres(1) », et 469 lignes où la puce « • » se collait au mot suivant (« •le bénéfice »), cassant leur classification en cascade.
2. **Le premier correctif (espace par défaut) a cassé pire** : les titres en petites capitales sont composés en deux spans (initiale pleine taille + reste à ~80 %) → « C HAPITRE », « B ilan », 114 lignes sur 38 pages. Et « CO2 » devenait « CO 2 ». Leçon : les heuristiques textuelles (seuils de ratio, « finit par un chiffre ») sont des rustines ; le signal fiable est GÉOMÉTRIQUE (l'écart bbox réel entre spans).
3. **Le critère géométrique a révélé un troisième piège** : en texte justifié, le PDF matérialise souvent l'espace inter-mots comme un span dédié — que le code filtrait avant le calcul d'écart. Résultat : « comptesannuels », « cetévénement ». Correctif : les spans blancs sont des séparateurs, les spans invisibles (<2pt, artefacts de croisement Word « 58B », 87 occurrences, tous du motif \d+[BF]) sont supprimés, la puce « o » en police Courier est un glyphe de liste.
4. **Et le séparateur forcé a cassé le texte pivoté à 90°** (bbox horizontaux sans signification, spans blancs de largeur nulle) : « d' affaires ». Correctif final : un span blanc ne sépare que si sa largeur propre dépasse 0,15× la taille de police ; espaces multiples normalisés.

Découverte au passage, en recensant les 34 740 lignes : **plus de lignes de commentaires ANC (12 863) que de texte réglementaire (9 342)** — la doctrine publique de l'ANC est la moitié la plus grosse du Recueil, ce qui valide l'intuition fondatrice du projet. Et les titres de haut niveau (Livre, Chapitre, Partie) sont NON GRAS dans le document réel — le sondage initial (pages 30-80) ne les avait jamais croisés : 100 % des lignes inclassables étaient exactement ces titres. Taux final d'inclassables : 0/34 740.

Morale pour le blog : « le parsing de PDF est résolu » est vrai à 97,7 % — et les 2,3 % restants (échantillon réel : 909 lignes corrigées) sont invisibles tant qu'on ne diffe pas l'extraction complète entre deux versions du code. Le protocole qui a tout attrapé : des relecteurs frais, l'accès au document réel, et l'exigence de preuves d'exécution plutôt que d'affirmations.
