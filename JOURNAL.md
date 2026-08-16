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
### Suite de l'exécution — deux hypothèses tombées, deux découvertes

**L'hypothèse « 1er chiffre d'article = numéro de Livre » était fausse.** Le contrôle d'intégrité a hurlé sur 59 % du corpus (1 097 anomalies) — pas parce que le document était incohérent, mais parce que notre plan avait inventé une règle jamais vérifiée. Calibration empirique (30 articles échantillonnés) : le 1er chiffre correspond au **Titre** (9/9), pas au Livre (1/9). Après recalibrage : 56 anomalies résiduelles, toutes légitimes (annexes à numérotation propre). Leçon : un contrôle d'intégrité qui explose peut avoir raison sur le fond et tort sur la référence — calibrer sur les données, jamais sur l'intuition.

**La forme longue « Article » cachait les règles modernes.** Enquête sur 135 renvois pendants concentrés sur des préfixes entiers (619-x : zéro article dans le corpus) → les articles existent dans le PDF, mais leurs en-têtes sont écrits « Article 619-1 » en toutes lettres : le texte originel de 2014 utilise « Art. », les amendements récents de l'ANC la forme longue. Résultat du correctif : **+48 articles réels retrouvés** (554 → 602), dont toute la section cryptoactifs (« Jetons émis et détenus ») et — surprise — une annexe sectorielle entière invisible jusque-là : le règlement ANC 2015-04 « Secteur du logement social » (30 pages) qui réutilise la numérotation du PCG. Renvois pendants : 135 → 45. Leçon pour le blog : dans un corpus juridique consolidé, les conventions typographiques DÉRIVENT dans le temps — le parseur calibré sur les pages anciennes rate silencieusement les règles les plus récentes, précisément celles qu'un « agent qui fait la compta d'aujourd'hui » ne peut pas se permettre de rater.

**État du corpus après T10 + ruling 23** : `corpus.db` 3,8 Mo — 1 868 enregistrements (869 réglementaires, 999 commentaires ANC), 602 articles distincts, 950+ renvois en graphe, FTS5 fonctionnel, 253 anomalies toutes catégorisées et documentées. Le fichier SQLite promis (« tout le droit comptable français dans un fichier ») existe.
### Jalon 1 livré — bilan d'exécution

**Livré** : `corpus.db` (3,7 Mo) — 1 660 enregistrements (739 réglementaires, 921 commentaires ANC), 602 articles distincts, 981 renvois en graphe, FTS5, 203 anomalies cataloguées en 5 catégories ; README honnête (limitations mesurées), licences MIT + Licence Ouverte 2.0, validation par échantillonnage sur 15 pages.

**La validation par échantillonnage a payé une dernière fois** : découverte d'une troisième strate typographique (taille 9,0) — extraits de lois citées, notes de bas de page, cellules de tableaux de modèles — confondue avec les en-têtes/pieds de page et donc non capturée : 2 037 lignes sur 120 pages, mesurées et différées en connaissance de cause (les lois citées arriveront plus propres via LEGI en phase 2). Leçon : sans échantillonnage adversarial sur le document réel, cette absence aurait été invisible — le corpus « semblait » complet.

**Stats de la méthode (pour le blog)** : 12 tâches, ~10 sous-agents implémenteurs/relecteurs distincts, 4+2+3+1 rounds de correction sur les tâches critiques, 24 rulings de contrôleur consignés, 65+ tests, zéro ligne de code écrite par le contrôleur. Les relecteurs qui vérifient sur le document réel (pas seulement le diff) ont trouvé l'essentiel des vrais bugs ; les implémenteurs économiques (Haiku) ont bien transcrit mais sur-affirmé (« aucune régression » non vérifié) — le protocole preuve-d'exécution-obligatoire a compensé.
**Publié le 15 août 2026** : https://github.com/bemayer/AccountingRag (branche main). Le jalon 1 est en ligne — corpus reconstructible, parseur testé, limitations documentées. Prochain plan : jalon 2, chaîne d'analyse lexicale + retrieval hybride + premières questions de benchmark.

---
## Session du 15 août 2026 — Jalon 2 : analyse lexicale et retrieval hybride

**T1-T3 (normalize / chunks / embed)** : transcription propre par Haiku (88 tests), mais la revue Sonnet a attrapé une **erreur métier dans mon propre plan** : la table de synonymes assimilait « amortissement dégressif » à « amortissement dérogatoire » — un mode de calcul n'est pas une provision réglementée, et cette entrée aurait silencieusement biaisé le retrieval vers les mauvais articles. Ruling : la table SYNONYMES ne grandira plus par intuition, seulement par échecs mesurés au benchmark. Une entrée fausse est pire qu'une entrée absente (leçon générique pour tout RAG à couche de synonymes).

**T4-T5 (index sqlite-vec + retrieval hybride)** : l'index réel tient dans le même fichier SQLite que le corpus — 2 160 chunks, FTS5 normalisé et table vectorielle vec0 à rowids alignés, construit en 14 min sur CPU avec e5-small. La crainte documentée dans le plan (« l'API sqlite-vec varie selon les versions ») ne s'est pas matérialisée : insertion par struct.pack float32 et KNN `MATCH ? AND k = ?` fonctionnent tels quels en 0.1.9. Premiers signaux qualitatifs : le routeur regex répond exactement sur « que dit l'article 214-1 ? » et l'expansion par graphe remonte précisément les renvois cités (214-15, 214-17) ; à l'inverse, la requête informelle « comment comptabiliser un logiciel acheté ? » place le meilleur passage en 3e position — le fossé lexical annoncé, désormais observable, que le benchmark va chiffrer.

**Le relecteur a trouvé des bugs dans le plan, pas dans le code.** Les deux findings « importants » de la revue T4-T5 étaient des défauts de mon code de référence, reproduits fidèlement par l'implémenteur : une regex de routage des références lettrées (L./R./D.) définie mais jamais câblée (code mort — différée à l'ingestion LEGI, le corpus PCG n'a aucun article lettré), et une prose de brief promettant une table FTS5 « content-less » que le code de référence ne créait pas. Leçon pour le blog : dans un pipeline plan → implémenteur → relecteur, l'implémenteur fidèle PROPAGE les bugs du plan — c'est le relecteur qui compare prose, code et corpus réel qui les intercepte. Le contrôleur qui écrit les plans doit être relu comme n'importe qui.

**Anecdote d'orchestration** : mon guetteur d'arrière-plan censé attendre la fin du build d'index (`until ! pgrep -f "scripts/build_index.py"`) ne se terminait jamais — pgrep -f matchait sa propre ligne de commande. Le chien de garde se surveillait lui-même.

### Le bug à trois octets que 98 tests ne voyaient pas

La revue finale de branche (modèle fort, mandat « publiable ? ») a trouvé ce qu'aucune des huit revues de tâche n'avait vu : la table de conversion des apostrophes de `normalize.py` était un no-op. Deux de ses trois clés étaient LE MÊME caractère — l'apostrophe ASCII 0x27 écrite deux fois — parce que visuellement, `'` et `’` sont presque indistinguables dans un éditeur. Les apostrophes typographiques U+2019/U+2018, seules cibles utiles de la table, n'y figuraient pas.

Dégâts mesurés : 11 204 occurrences de U+2019 dans le corpus (1 322 records sur 1 660 — le Recueil ANC est composé en apostrophes typographiques), suppression d'élisions jamais déclenchée, index FTS pollué de milliers de tokens orphelins (« l », « d », « qu »…), et surtout : la même question tapée avec l'apostrophe d'un clavier de téléphone français donnait un top-5 différent de sa jumelle ASCII. Les 30 questions du benchmark étant écrites à 100 % en ASCII, les chiffres publiés ne testaient jamais le chemin défaillant. Le relecteur a aussi débusqué l'ordre fold/stem inversé (Snowball recevait du texte dé-accentué, cassant la fusion des familles généré/génération) — confirmé par deux morceaux de code morts que l'inversion rendait inutiles.

Trois leçons : (1) un invariant « même fonction au build et à la requête » ne suffit pas — il faut tester l'équivalence des CONVENTIONS D'ENTRÉE (`normalize("l’exercice") == normalize("l'exercice")` est désormais un test) ; (2) les caractères homoglyphes dans le code source échappent à la relecture humaine ET aux revues de diff — seul un hexdump tranche ; (3) après correction et rebuild : recall@10 bm25 en légère BAISSE (0,857 → 0,833, MRR en hausse 0,735 → 0,754) — le bug supprimait aussi des matches accidentels. Le mode dense, qui ignore la normalisation, est resté strictement inchangé : le témoin parfait que l'effet mesuré vient bien du canal corrigé. Et n=21 est trop petit pour trancher finement ces ±1 question : agrandir le benchmark passe devant tout nouvel ajustement lexical.

### Jalon 2 livré — bilan chiffré

Tableau final (dev, 21 questions, après correctifs) : bm25 recall@5 0,833 / recall@10 0,833 / MRR 0,754 ; dense 0,571 / 0,738 / 0,518 ; hybrid et hybrid+graph 0,81 / 0,81 / 0,763. Ventilation : les trois modes lexicaux sont à 1,0 sur les références directes (le routeur regex) et ~0,95 sur les règles ; tout s'effondre sur le vocabulaire courant (0,643 lexical, 0,214 dense pur) — le fossé lexical annoncé au design est maintenant un chiffre, et c'est LA cible du jalon 2.5 (mesuré, pas supposé : les 3 pires questions dev sont toutes des reformulations grand public dont aucun token normalisé n'apparaît dans l'article gold). Découverte annexe : l'expansion par graphe de renvois n'apporte rien sur ce split — les articles manquants ne sont pas à 1 hop des résultats trouvés.

Méthode : 8 tâches, 5 implémenteurs + 6 relecteurs (haiku pour la transcription, sonnet pour l'intégration et les revues, opus pour la porte finale), 4 fix rounds, 11 rulings consignés, 107 tests. Les revues de tâche ont surtout attrapé des défauts DE PLAN (synonyme comptablement faux, regex morte héritée du code de référence) ; la revue finale de branche a attrapé le bug de données que personne ne pouvait voir dans un diff. Coût de re-publication : un rebuild de 14 minutes et une re-campagne de 90 secondes — c'est le prix d'avoir tout scripté.

---
## Session du 16 août 2026 — Jalon 2.5 : trois leviers, un seul survivant

Le jalon 2.5 attaque le fossé lexical chiffré au jalon 2, avec la discipline promise : benchmark d'abord (30 → 90 questions, split re-gelé AVANT toute mesure, 24 questions en apostrophes typographiques pour ne plus jamais être aveugle au bug C1), puis trois ablations, une variable à la fois, chacune jugée au bootstrap apparié (10 000 tirages, seed fixe, adoption si p ≥ 0,95 sans régression de catégorie).

**Le sur-échantillonnage de la catégorie faible a fait son travail.** La baseline sur le nouveau dev (61 questions, dont 31 en vocabulaire courant) tombe à 0,672 de recall@10 en hybrid — contre 0,81 sur l'ancien dev, non par régression mais parce que l'instrument mesure enfin là où ça fait mal.

**Ablation A (pondération par champ) : rejetée, avec une leçon de mécanique.** Doubler le poids du chemin hiérarchique dans BM25 change l'ordre des résultats bm25 sur 45 des 61 questions… et ne change STRICTEMENT RIEN au recall@10 (delta exactement nul). Explication : la fusion RRF est basée sur les rangs, et les permutations touchent des rangs bas ou des documents non-gold. Première intuition du relecteur (« l'ordre ne change pas ») elle-même fausse — il a fallu un contrôle indépendant pour l'établir. Pénaliser les commentaires ANC : dégrade. Rejeté aussi.

**Ablation B (reranker cross-encoder) : le seul survivant.** Le modèle léger pressenti (mmarco-MiniLM, ~10 s/question) améliore (0,672 → 0,713) mais rate le seuil (p = 0,858) — sans le protocole, on l'aurait adopté sur sa bonne mine. Le lourd (bge-reranker-v2-m3, 2,2 Go) passe : 0,672 → 0,738, p = 0,952, aucune catégorie ne régresse. Prix : ~117 secondes par question sur CPU (×585 la baseline). Décision d'architecture en cascade : le mode reste HORS de la campagne par défaut (2 h de calcul surprise pour un nouveau contributeur, non merci) et le chargement affiche un avertissement taille/latence.

**Ablation C (synonymes pilotés par les échecs) : rejet le plus instructif du jalon.** Sur les 21 échecs du dev, la plupart n'offrent AUCUNE paire terme-courant → terme-PCG légitime (la règle du jalon 2 : jamais rapprocher deux concepts comptables distincts). Le lot prudent de 3 entrées mesuré : delta exactement nul, résultats identiques bit à bit. Autopsie scriptée : les synonymes ATTEIGNENT la requête normalisée et font gagner ~1 200 rangs au document gold (1453 → 210 sur 1659)… qui reste à 160 rangs de la fenêtre limit=50 que BM25 transmet à la fusion. Le fossé restant n'est pas un problème de vocabulaire : c'est un problème de discriminance (IDF) et de fenêtre de candidats. Voilà le goulot du jalon 3, identifié par la mesure et non par l'intuition.

**Note de méthode** : les trois « défauts de traçabilité » attrapés par les revues de ce jalon (explication mécanique fausse mais conclusion juste ; chiffre-clé non reproductible à ±15 % ; scores par question non persistés) relèvent tous du même antipattern — affirmer « vérifié » sans artefact reproductible. Nouvelle règle du projet : toute mesure persiste ses données brutes, tout chiffre cité a son script.
