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

Tableau final (dev, 21 questions, après correctifs) : bm25 recall@5 0,833 / recall@10 0,833 / MRR 0,754 ; dense 0,571 / 0,738 / 0,518 ; hybrid et hybrid+graph 0,81 / 0,81 / 0,763. Ventilation : les trois modes lexicaux sont à 1,0 sur les références directes (le routeur regex) et ~0,95 sur les règles ; tout s'effondre sur le vocabulaire courant (0,571 lexical, 0,214 dense pur) — le fossé lexical annoncé au design est maintenant un chiffre, et c'est LA cible du jalon 2.5 (mesuré, pas supposé : les 3 pires questions dev sont toutes des reformulations grand public dont aucun token normalisé n'apparaît dans l'article gold). Découverte annexe : l'expansion par graphe de renvois n'apporte rien sur ce split — les articles manquants ne sont pas à 1 hop des résultats trouvés.

Méthode : 8 tâches, 5 implémenteurs + 6 relecteurs (haiku pour la transcription, sonnet pour l'intégration et les revues, opus pour la porte finale), 4 fix rounds, 11 rulings consignés, 107 tests. Les revues de tâche ont surtout attrapé des défauts DE PLAN (synonyme comptablement faux, regex morte héritée du code de référence) ; la revue finale de branche a attrapé le bug de données que personne ne pouvait voir dans un diff. Coût de re-publication : un rebuild de 14 minutes et une re-campagne de 90 secondes — c'est le prix d'avoir tout scripté.

---
## Session du 16 août 2026 — Jalon 2.5 : trois leviers, un seul survivant

Le jalon 2.5 attaque le fossé lexical chiffré au jalon 2, avec la discipline promise : benchmark d'abord (30 → 90 questions, split re-gelé AVANT toute mesure, 24 questions en apostrophes typographiques pour ne plus jamais être aveugle au bug C1), puis trois ablations, une variable à la fois, chacune jugée au bootstrap apparié (10 000 tirages, seed fixe, adoption si p ≥ 0,95 sans régression de catégorie).

**Le sur-échantillonnage de la catégorie faible a fait son travail.** La baseline sur le nouveau dev (61 questions, dont 31 en vocabulaire courant) tombe à 0,672 de recall@10 en hybrid — non comparable à l'ancien chiffre : le benchmark v2 sur-échantillonne délibérément la catégorie faible (31 des 61 questions).

**Ablation A (pondération par champ) : rejetée, avec une leçon de mécanique.** Doubler le poids du chemin hiérarchique dans BM25 change l'ordre des résultats bm25 sur 45 des 61 questions… et ne change STRICTEMENT RIEN au recall@10 (delta exactement nul). Explication : la fusion RRF est basée sur les rangs, et les permutations touchent des rangs bas ou des documents non-gold. Première intuition du relecteur (« l'ordre ne change pas ») elle-même fausse — il a fallu un contrôle indépendant pour l'établir. Pénaliser les commentaires ANC : dégrade. Rejeté aussi.

**Ablation B (reranker cross-encoder) : le seul survivant.** Le modèle léger pressenti (mmarco-MiniLM, ~10 s/question) améliore (0,672 → 0,713) mais rate le seuil (p_amélioration = 0,858) — sans le protocole, on l'aurait adopté sur sa bonne mine. Le lourd (bge-reranker-v2-m3, 2,2 Go) passe : 0,672 → 0,738, p_amélioration = 0,952, aucune catégorie ne régresse. Prix : ~117 secondes par question sur CPU (×585 la baseline). Décision d'architecture en cascade : le mode reste HORS de la campagne par défaut (2 h de calcul surprise pour un nouveau contributeur, non merci) et le chargement affiche un avertissement taille/latence.

**Ablation C (synonymes pilotés par les échecs) : rejet le plus instructif du jalon.** Sur les 21 échecs du dev, la plupart n'offrent AUCUNE paire terme-courant → terme-PCG légitime (la règle du jalon 2 : jamais rapprocher deux concepts comptables distincts). Le lot prudent de 3 entrées mesuré : delta exactement nul, résultats identiques bit à bit. Autopsie scriptée : les synonymes ATTEIGNENT la requête normalisée et font gagner ~1 200 rangs au document gold (1453 → 210 sur 1659)… qui reste à 160 rangs de la fenêtre limit=50 que BM25 transmet à la fusion. Le fossé restant n'est pas un problème de vocabulaire : c'est un problème de discriminance (IDF) et de fenêtre de candidats. Voilà le goulot du jalon 3, identifié par la mesure et non par l'intuition.

**Note de méthode** : les trois « défauts de traçabilité » attrapés par les revues de ce jalon (explication mécanique fausse mais conclusion juste ; chiffre-clé non reproductible à ±15 % ; scores par question non persistés) relèvent tous du même antipattern — affirmer « vérifié » sans artefact reproductible. Nouvelle règle du projet : toute mesure persiste ses données brutes, tout chiffre cité a son script.

---
## Session du 16 août 2026 — Jalon 3 : sonder avant de planifier

Le jalon 2.5 s'était conclu sur un coupable désigné : « le goulot, c'est la fenêtre de 50 candidats de BM25 ». J'allais écrire un plan entier autour de cette hypothèse. Avant, j'ai passé dix minutes à la sonder. Elle était **partiellement fausse**, et ce que la sonde a trouvé à la place a réécrit le plan.

**Deux régimes d'échec, et ma sonde qui se trompe.** Premier jet de la sonde : le gold est « toujours dernier » de sa liste — 154/154, 46/46, 1430/1430. Conclusion spectaculaire, et fausse : ma sonde retournait dès qu'elle trouvait le gold, si bien que le total affiché valait mécaniquement le rang. Le relecteur de la tâche 1 a buté sur un écart d'un chiffre (1453 au lieu de 1430) ; en creusant, j'ai trouvé mon propre bug de reporting. Les vrais chiffres racontent une autre histoire, plus utile : q026 est à 46 sur 1659 (3 % de tête), q021 à 154 sur 1585 (10 %), q060 à 1453 sur 1659 (88 %). **Il y a deux régimes d'échec, pas un.** q021 et q026 sont des quasi-succès qui ratent la fenêtre de 50 candidats de peu — élargir le pool devrait les rattraper. q060 est un échec de fond, où le canal lexical est réellement muet. Une seule ablation ne pouvait pas traiter les deux, et mon « toujours dernier » les confondait.

**Filtrer les mots trop fréquents nettoie beaucoup, mais ne crée rien.** En écartant les tokens présents dans plus de 2 % des chunks, le pool de la question q060 tombe de 1659 à 63 candidats, et le gold de q026 remonte du rang 46 au rang 14 — il entre enfin dans la fenêtre que voit le reranker. Mais q021 et q060 *disparaissent* du pool : quand il n'y a aucun mot rare partagé, filtrer ne peut pas inventer un lien. Et la conjonction des mots rares (`AND` au lieu de `OR`) donne zéro candidat dans les trois cas — voie morte, éliminée avant d'avoir coûté une tâche.

**Le canal dense, lui, voit quelque chose.** Il place ces mêmes golds aux rangs 178, 248, 257 et 528 sur 1660. Hors de la fenêtre de 50 — mais parfaitement atteignable avec un pool de 200 à 400. D'où le vrai arbitrage du jalon, que je n'aurais pas su formuler sans la sonde : **modèle de reranking faible sur beaucoup de candidats, ou modèle fort sur peu ?** Le reranker lourd adopté au jalon 2.5 coûte 4,7 s par candidat — inutilisable sur 200 (15 min/question). Le modèle léger *rejeté* au jalon 2.5, lui, tient 200 candidats en 80 secondes. Le perdant d'hier redevient le seul candidat crédible dès qu'on change une autre variable : voilà pourquoi on garde les résultats négatifs.

Leçon de méthode : une sonde de dix minutes avant d'écrire un plan de quatre tâches. Le plan qui en sort ne teste plus mes intuitions, il teste ce que les données désignent — et il commence par un tableau de rangs que n'importe qui peut régénérer.

### Le fichier .env qui n'était pas ignoré

L'utilisateur a déposé une clé API dans un `.env` en demandant de bien l'exclure de git. Vérification : `.env` n'était couvert par **aucune** règle du `.gitignore` — il apparaissait en fichier non suivi, à un `git add -A` de se retrouver dans un commit, sur un dépôt public, avec plusieurs sous-agents en train de committer sur la branche. L'historique était heureusement vierge (zéro occurrence de `sk-ant-` dans les fichiers suivis comme dans tout l'historique poussé), donc rien à révoquer.

Ce qui mérite d'être noté pour le blog, c'est l'ordre des opérations : la demande était « ajoute la clé au gitignore », le réflexe correct est « vérifie d'abord si elle a déjà fuité, ensuite protège ». Un `.gitignore` corrigé sur un secret déjà commité donne une fausse impression de sécurité — le fichier disparaît du statut, et la clé reste dans l'historique pour toujours.

### Ce que la clé débloque, et à quel prix

Une ablation supplémentaire : la réécriture de requête par LLM, qui traduit « comment je répartis le coût d'une machine » en « amortissement immobilisation corporelle plan d'amortissement ». C'est le seul levier restant pour les questions à recouvrement nul. Trois garde-fous inscrits au plan : le réécriveur ne voit **que** la question (jamais les citations attendues — sinon on ferait fuiter la réponse dans la requête et la mesure ne vaudrait plus rien), les réécritures sont mises en cache dans un JSON commité (premier run à quelques centimes, tous les suivants gratuits et reproductibles à l'identique), et un garde-fou dur à 200 appels par exécution.

### Le contrôle de reproduction qui validait le bug

Suite de l'affaire de la sonde. Une fois mon bug de reporting compris, j'avais demandé à la tâche 1 d'en faire un outil versionné — `scripts/diagnostic_rangs.py` — pour que le tableau de rangs du plan soit régénérable par n'importe qui. Mon étape de vérification disait : « l'outil doit reproduire les chiffres de la sonde ». Elle a été satisfaite. Parfaitement.

En relisant l'outil deux tâches plus tard, j'ai trouvé sa docstring : « convention conservée à l'identique pour reproduire ces chiffres bit à bit (contrôle de l'outil) ». L'implémenteur avait **repéré** le bug de ma sonde — et l'avait délibérément reproduit, parce que mon contrôle exigeait la coïncidence des chiffres. Le contrôle censé valider l'outil a validé l'artefact.

C'est la loi de Goodhart appliquée à la vérification : dès qu'un contrôle devient la cible, il cesse de mesurer ce qu'il croyait mesurer. Un contrôle de reproduction n'a de valeur que si la reproduction est **indépendante** — recalculée depuis les données, pas alignée sur un chiffre de référence dont personne ne réinterroge la provenance. Corrigé à la source, avec un test de non-régression construit pour qu'un total égal au rang soit impossible par construction (`rang=3`, `total=5`).

Effet collatéral utile : le relecteur de la tâche 2 a ensuite signalé un « écart » entre le JSON commité et mes chiffres — en prenant le JSON buggé pour référence. Son finding était **inversé** : il aurait corrigé les bons chiffres vers les mauvais. Vérifié par SQL direct avant de trancher. Deuxième leçon : un relecteur qui ne peut pas recalculer prend l'artefact le plus récent pour la vérité.

### Le résultat central : couverture et classement se découplent

Les deux premières ablations du jalon sont des **rejets**, et le second a produit le résultat le plus utile de la session.

**Ablation D — filtrer les tokens peu discriminants (df_max) : rejetée.** Dégradation monotone, plus le filtre se resserre : 0,672 → 0,664 (p=0,264) → 0,656 (p=0,000) → 0,590 (p=0,003), et le vocabulaire courant en souffre à chaque tour de vis. Le mécanisme est instructif : le gain isolé sur les questions à token rare partagé (q026, qui remonte du rang 46 au rang 14) est effacé par celles dont le gold quitte le pool filtré. **Être retrouvé par des mots vides vaut mieux que ne pas être retrouvé du tout**, parce que la fusion RRF laisse au canal dense une chance de repêcher un candidat que le lexical a mal classé. Filtrer supprime cette chance.

**Ablation E — élargir le pool de candidats : rejetée aussi, et voilà pourquoi c'est intéressant.** J'attendais que les quasi-succès (q021 à 10 % de tête, q026 à 3 %) rentrent dans la fenêtre. Ils rentrent. La couverture du pool avant fusion — au moins une citation attendue présente dans l'union bm25 ∪ dense — monte de 0,820 à 0,918 en global, et de 0,645 à **0,839** sur la catégorie vocabulaire courant. Et le recall@10 après fusion **baisse** : 0,672 → 0,639.

Le système trouve davantage et restitue moins. Vérification mécanique du paradoxe : sur 610 entrées de top-10 fusionné à pool=200, **zéro** a son meilleur rang par canal au-delà de 100 — ce qui explique l'égalité stricte des mesures à pool 100, 200 et 400. Le diff par question montre le reste : des candidats médiocres mais présents dans **les deux** canaux évincent, par simple addition des scores RRF, des golds excellents dans **un seul** canal. Élargir le pool ne fait pas entrer les bons candidats dans le top-10, il fait entrer plus de concurrents consensuels.

Conséquence pour la suite du jalon : **le déficit n'est plus un problème de rappel, c'est un problème de classement.** Le pool contient déjà la réponse dans 84 % des cas de vocabulaire courant contre 40 % de recall@10 restitué. Cet écart — couverture atteignable moins recall effectif — est exactement la marge qu'un reranker peut aller chercher, et il est maintenant chiffré au lieu d'être espéré. Avec trois réserves que je m'interdis d'oublier en communiquant le chiffre : n=61 (le delta entre pool 50 et 100 tient sur 4 questions), le coût du reranking sur jusqu'à 2×pool candidats n'est pas encore mesuré, et un plafond d'**atteignabilité** n'est pas un recall promis.

**Note sur la valeur des rejets.** Trois leviers mesurés dans ce jalon, trois rejets — et le jalon n'est pas perdu pour autant : c'est le troisième rejet qui a produit la métrique qui oriente la suite. Un protocole d'adoption strict (bootstrap apparié, p ≥ 0,95, aucune catégorie perdant plus de 0,05) transforme les échecs en information. Sans lui, j'aurais adopté deux des trois leviers sur leur point estimé — dont l'un (la déduplication des termes de la requête FTS5, +0,017 de recall@10) que le bootstrap donne à p=0,63, soit un pur bruit d'échantillonnage.

### Deux ordres de grandeur : le chiffre qui a orienté tout un jalon était une latence CPU

Le jalon 3 était bâti sur un arbitrage que je croyais imposé par le matériel : « le reranker lourd coûte 4,7 s par candidat, donc 15 minutes par question sur 200 candidats — inutilisable. Le modèle léger *rejeté* au jalon 2.5 redevient donc le seul candidat crédible dès qu'on élargit le pool. » J'avais même trouvé ça élégant : le perdant d'hier réhabilité par un changement de variable.

Les deux moitiés du raisonnement étaient fausses.

**La première mesure a démenti l'estimation d'un facteur 50.** Le modèle léger sur 200 candidats : 1,5 s/question, pas 80. J'ai alors calibré le modèle lourd sur 100 candidats — de l'ordre de 5 s/question, soit environ cinq minutes pour tout le split de développement, là où j'annonçais 8 heures. La configuration décisive du jalon, modèle fort sur pool large, avait donc été écartée du plan **sur une estimation que personne n'avait mesurée**. Coût de la vérification : trois minutes.

**Puis la cause de l'écart.** Ma première explication tenait debout : machine partagée, charge moyenne 16 sur 8 cœurs, 1,28 million de pages envoyées en swap. J'ai écrit cette explication dans le rapport. Ensuite j'ai fait le contrôle qui la départageait — même configuration, une question, le modèle déplacé explicitement d'un device à l'autre :

| device | latence par question (25 candidats) |
|---|---|
| `cuda:0` | ordre de 1,5 s |
| `cpu` | ordre de 150 s |

Deux ordres de grandeur. Et les 129,5 s/question publiées au jalon 2.5 tombent en plein dans le voisinage du CPU.

Post-scriptum honnête, et il a fallu trois tours pour le rédiger correctement : j'ai d'abord publié « facteur 71 » sur un tirage unique. La revue a refait la sonde et obtenu 66. En la versionnant sur trois questions avec un tour de chauffe j'ai obtenu 90, et j'ai cru avoir corrigé le problème en publiant une fourchette — sauf que cette fourchette était la dispersion entre les trois questions d'une même exécution, pas l'incertitude de la mesure. La vérification suivante a obtenu 106, entièrement hors de ma fourchette. Quatre tirages : 66, 78, 90, 106. **J'ai corrigé une fausse précision par une autre fausse précision, deux fois de suite**, avant d'accepter que la seule affirmation défendable soit « deux ordres de grandeur ». Ce n'était pas le swap : **la campagne du jalon 2.5 exécutait le cross-encoder sur CPU**, sur une machine qui avait déjà une carte — même lockfile, même torch CUDA, vérifié. Ce qui l'en a privée reste inconnu, et je l'annonce comme hypothèse plutôt que de conclure : plusieurs sous-agents se disputant 6 Go de VRAM, probablement.

Trois leçons, et la troisième est la plus désagréable.

La première est technique : **une latence n'est pas une propriété du système mesuré**, c'est une propriété du triplet (machine, device, charge). Un chiffre en secondes publié sans son device ne vaut rien, et ce projet a mis un jalon et demi à l'apprendre. Le device et le dtype sont désormais dans le tableau des conditions de chaque rapport.

La deuxième est de conception : la décision du jalon 2.5 de garder le reranking hors de la campagne par défaut — « 2 h de calcul surprise pour un nouveau contributeur » — reste **juste pour qui n'a pas de GPU** (deux à trois heures sur 61 questions — la campagne du jalon 2.5 en avait pris 2 h 09), et devient absurde dès qu'une carte est là (2 minutes). Le défaut ne doit pas dépendre du mode, il doit dépendre du matériel. Une bonne décision peut reposer sur un chiffre faux, et le découvrir ne l'invalide pas automatiquement — ça déplace juste la condition qui la justifie.

La troisième est celle que je préférerais ne pas écrire : **j'ai publié mon hypothèse avant de faire le contrôle qui la testait**, alors que ce jalon tout entier documente ce travers depuis sa sonde fondatrice. Le swap était plausible, mesurable, et faux. Trois minutes de contrôle séparaient l'explication crédible de l'explication vraie. Écrire « hypothèse retenue » dans un rapport ne protège de rien quand le test tenait dans une boucle sur deux devices.

### L'ablation F, ou pourquoi plus de candidats ne sert à rien

Résultat propre, et il se répète une fois par modèle : le modèle léger donne 0,713 de recall@10 sur 25 candidats et 0,713 sur 200 ; le modèle lourd donne 0,738 sur 25 et 0,738 sur 200. Pas seulement les agrégats — la ventilation par catégorie est identique aux deux largeurs. Multiplier par huit le nombre de candidats soumis au cross-encoder ne fait entrer **aucune** citation attendue de plus dans le top-10.

Ça ferme l'hypothèse qui portait le jalon depuis l'ablation E. Le pool contenait la bonne réponse dans 83,9 % des cas de vocabulaire courant contre 40 % restitués, et je lisais cet écart comme une marge à récupérer par un meilleur classement. Un cross-encoder à qui l'on donne *tout* le pool en restitue 48,4 %. Trente-cinq points d'écart subsistent alors que le modèle a vu tous les candidats : le goulot n'est ni la fenêtre de récupération, ni le volume traité, c'est la capacité à **reconnaître** qu'un article du Plan comptable répond à une question posée en langage courant. Aucun réordonnancement ne crée cette reconnaissance.

Une nuance que je garde pour la suite, parce qu'elle contredit ma propre conclusion sur un point précis : le modèle lourd sur 200 candidats gagne +0,050 de recall@5 et +0,033 de MRR, à recall@10 rigoureusement constant. Le pool large lui permet de faire remonter dans le top-5 des documents qui étaient déjà entre les rangs 6 et 10. C'est un vrai gain de précision de tête — et il n'est pas adoptable, parce que le critère du jalon porte sur le recall@10, fixé avant les mesures. J'ai été tenté de basculer la métrique d'adoption vers le recall@5, où mon résultat devenait positif. C'est exactement ce que le protocole existe pour empêcher. Le gain est noté, avec sa réserve, et sera re-mesuré proprement au jalon 4 — d'autant qu'un générateur RAG est alimenté par cinq passages, pas dix : c'est le recall@5 qui gouvernera la qualité des réponses, pas la métrique sur laquelle j'ai bâti mon protocole.

### Le levier qui gagne est celui que je n'avais pas prévu de tester

Bilan du jalon 3, en une phrase : **trois ablations pour mieux classer ce que le vocabulaire commun trouvait déjà, toutes rejetées ; une quatrième pour changer le vocabulaire, adoptée à 0,877.**

Le plan d'origine avait quatre tâches et l'ablation G n'en faisait pas partie. Elle est née d'une remarque de l'utilisateur au milieu du jalon — « clef api de quoi ? » — après qu'il ait déposé une clé Anthropic dans un `.env`. J'avais écrit le plan autour de l'idée que le fossé lexical était un problème de *fenêtre* : le bon document était trouvé mais mal classé, il fallait élargir et rereranker. Trois mesures plus tard, les trois leviers de classement étaient rejetés et le déficit intact.

**Ce que les rejets ont appris, et qui valait le jalon à lui seul.** L'ablation E a établi le découplage : la couverture du pool monte à 0,918 pendant que le recall@10 après fusion baisse. L'ablation F a fermé l'hypothèse qui restait — un cross-encoder à qui l'on donne *tous* les candidats du pool en restitue 48 %, contre 84 % de couverture disponible. Le déficit n'était donc ni une fenêtre trop étroite, ni un reranker trop myope : c'était l'incapacité à *reconnaître* qu'un article du PCG répond à une question posée en langage courant. Aucun réordonnancement ne crée cette reconnaissance.

**Et la réécriture répare exactement ce que le diagnostic donnait pour perdu.** q060 — celle dont le gold était au 88ᵉ percentile de son classement lexical, la seule vraiment muette côté lexical, celle que ma sonde présentait comme le cas désespéré — passe de 0 à 1. Le « fossé lexical » n'était pas un plafond du système. C'était un plafond du vocabulaire de la question.

**Le chiffre que je refuse d'arrondir en ma faveur.** Sur le split gelé, exécuté une seule fois, jamais utilisé pour choisir quoi que ce soit : 0,966, contre 0,759 pour la configuration du jalon 2.5. C'est plus haut que le dev (0,877), ce qui prouve que l'effet réplique — mais je cite le dev, parce que 0,966 signifie « 28 sur 29 » et qu'une question de plus en échec ramènerait à 0,93. Un intervalle de confiance deux fois plus large ne raffine pas une estimation, il la confirme au mieux. Et sur dev, la comptabilité exacte est : **12 questions améliorées — dix réparations complètes, deux demi-réparations —, 3 dégradées, 4 résistantes.** La réécriture casse des questions qui marchaient. Réécrire une question qui parlait déjà le bon vocabulaire peut diluer un terme discriminant — c'est sous la garde du protocole, donc l'adoption tient, mais ça figure dans le rapport avec les noms des trois questions.

**La question qui résiste au retrieval nu livre une piste pour la suite.** q023 échoue dans les trois configurations mesurées de l'ablation G — qui sont toutes en mode `hybrid` **sans reranking**, puisque c'est la référence de cette ablation. Elle réussit dès qu'on rerankee, y compris dans la configuration livrée : j'ai d'abord écrit qu'elle « échoue dans toutes les configurations », ce que les JSON de mon propre dépôt contredisaient. Son autopsie reste le plus joli résultat du jalon. Son gold est le **2ᵉ meilleur candidat lexical sur 1 653** — et il est absent du canal dense. Contributions RRF mesurées : 0,01613 pour le gold, sur un seul canal ; 0,03054 pour un candidat 5ᵉ en lexical et 6ᵉ en dense. Le meilleur candidat du corpus entier finit au rang 11 et manque le top-10 d'une place. Sur ce pool de 81 candidats, 73 n'existent que dans un canal, et les 8 bicanaux monopolisent la tête. **La somme RRF récompense le consensus, pas l'excellence.**

Et le rang 11 est précisément ce qui sauve ce cas : il tient dans les 25 candidats que reçoit le cross-encoder, qui remonte le gold. Le défaut de fusion est donc réel, mesuré, et aujourd'hui masqué par la fenêtre du reranker — un masquage qui ne survivra pas mécaniquement à un corpus dix fois plus gros, où l'éviction poussera les golds mono-canal plus loin que 25.

### Trois fois le même piège en un jalon : publier l'explication avant de faire le contrôle

Le jalon a produit trois corrections de chiffres publiés, et les trois viennent de la même faute.

**La latence du reranking.** Le jalon 2.5 publiait 129,5 s/question pour le cross-encoder, ce qui avait motivé une décision de conception (le mode reste hors de la campagne par défaut, « 2 h de calcul surprise pour un contributeur »). Re-mesurée à l'identique : 1,71 s/question. Deux ordres de grandeur, code inchangé depuis son commit d'origine. J'ai d'abord écrit dans le rapport que c'était la pression mémoire — machine chargée, 1,28 million de pages en swap, charge 16 sur 8 cœurs. Plausible, mesurable, faux. Le contrôle qui départageait tenait dans une boucle sur deux devices : ~2 s sur `cuda:0`, ~150 s sur `cpu` (la sonde versionnée, `docs/mesures/jalon3/sondes.json`, publie une fourchette — un tirage unique varie de 25 % sur cette machine). La machine avait une carte depuis le début et le jalon 2.5 tournait sur CPU. Corollaire de conception : la décision du jalon 2.5 reste *juste pour qui n'a pas de GPU* et devient absurde dès qu'une carte est là. Une bonne décision peut reposer sur un chiffre faux ; le découvrir déplace la condition qui la justifie, il ne l'invalide pas.

**Le coût du reranking sur pool large.** Le plan annonçait 8 heures pour le modèle lourd sur 100 candidats et rangeait donc la configuration décisive du jalon derrière une option désactivée par défaut. Calibration réelle : 4,4 s/question à `pool` neutre, 5,5 s à `pool=100`, soit environ cinq minutes pour le split entier. J'avais écarté la config qui répondait à la question du jalon sur une estimation que personne n'avait mesurée — dans un jalon dont le premier enseignement était « sonder avant de planifier ».

Leçon commune, et elle est désagréable parce qu'elle est exactement celle que ce jalon documentait déjà : **écrire « hypothèse retenue » dans un rapport ne protège de rien quand le test tient en trois minutes.** J'ai eu raison sur les conclusions et tort sur les mécanismes, trois fois, faute d'avoir dépensé ces trois minutes avant de rédiger.

### Le contrôle d'intégrité que j'ai rendu falsifiable

La réécriture par LLM crée un risque qu'aucune ablation précédente ne portait : si le modèle citait de lui-même le numéro d'article attendu, ce numéro matcherait lexicalement le texte de l'article dans l'index, et le gain mesuré ne mesurerait plus le retrieval mais la mémoire du modèle. (J'avais d'abord désigné le routeur de référence exacte comme canal de fuite — impossible, puisqu'il lit toujours la question originale ; la revue finale l'a relevé.) Le plan prévoyait « inspecter 5 réécritures à la main ».

Cinq sur quatre-vingt-dix, à l'œil, sur un contrôle dont dépend la validité de tout le jalon. J'ai scripté l'audit à la place. Deux fois, d'ailleurs : ma première version ne chargeait que le split dev et sautait en silence les 29 réécritures du split gelé — celles qui produisent le chiffre que je mets en avant — tout en affichant fièrement « 90 auditées ». La revue finale l'a attrapée. Corrigé : sur les 90 réécritures, deux numéros apparaissent — `123-16` (le code de commerce, recopié depuis la question qui le contenait) et `2004-01` (un règlement ANC cité de lui-même par le modèle dans le split gelé, hors gold, donc du bruit) — et zéro égal au gold. Puis j'ai ajouté le test qui manquait : **une réécriture synthétique qui cite le gold sans qu'il figure dans la question, et l'assertion que l'audit la signale.** Sans ce test, « contrôle OK » aurait été une affirmation invérifiable — et ce jalon avait déjà produit un contrôle de reproduction qui validait le bug qu'il devait attraper. Un contrôle qu'on n'a jamais vu échouer ne prouve rien.

### Le trou que le benchmark ne peut pas voir

Fin de jalon, l'utilisateur demande pourquoi le BOFiP n'est pas dans le corpus — « c'est pourtant la base en compta non ? ». Réponse factuelle : il est au design, en phase 2, et les champs `nature` et `opposable` du schéma n'existent que pour lui. Correction de vocabulaire : le BOFiP est la base de la *fiscalité*, pas de la comptabilité — mais l'intuition est juste, parce que le CGI impose de suivre les définitions du PCG sauf incompatibilité fiscale, et que toute la liasse 2058-A existe pour réconcilier les écarts.

Ce que la question a surtout révélé, c'est une réserve que je n'avais pas formulée : **le benchmark ne peut pas voir ce qui manque au corpus.** Ses 90 questions ont toutes été écrites depuis le PCG, donc leurs citations attendues existent par construction. « Est-ce déductible ? » n'a pas de gold, n'est pas dans le benchmark, et ne pèse pas dans le 0,877. Le score mesure la capacité à retrouver ce qui est présent, sur une distribution de questions qui exclut précisément ce qui est absent. Un benchmark construit depuis son propre corpus ne peut pas mesurer les limites de ce corpus — c'est structurel, et aucune ablation ne l'aurait montré.

Le corpus, vérifié à cette occasion : Livres I à V du règlement ANC 2014-03. Ni consolidation (règlement 2020-01), ni fusions, ni normes d'exercice professionnel — alors que l'étoile polaire déclarée est le DSCG UE4, « Comptabilité et audit ». Jalon 4 : BOFiP. Jalon 5 : consolidation, fusions, NEP. Et une contrainte technique à porter dès le plan du jalon 4 : le corpus change d'ordre de grandeur, la discriminance des termes avec lui, et l'éviction RRF que q023 documente s'aggrave mécaniquement avec le nombre de candidats consensuels. Le jalon 4 devra re-mesurer le retrieval, pas hériter de ces chiffres.

## Jalon 4 — mesurer ce que le système répond

Trois jalons à améliorer le retrieval, et pas une ligne mesurant la réponse. Le benchmark saturait — 0,966 sur le split gelé, c'est 28 sur 29, un instrument qui ne peut plus voir un progrès — et la métrique signature annoncée au design v1, le taux de confusion fiscal ↔ comptable, n'existait que sur le papier. Le jalon 4 ne cherche donc pas à faire mieux : il construit de quoi savoir.

Résultat central, et il tient en une ligne : **aucun article inventé sur les 779 citations des trois splits.** Sur le split de validation gelé, 175 citations, zéro inexistante, zéro non portante, et la correspondance brute vaut 1,0 — la normalisation de comparaison n'y fait strictement aucun travail. La famille d'abstention rend 29 refus corrects sur 30 avec zéro fabrication. Le juge franchit son seuil à 0,9854 pour 0,60 exigés.

### Le chiffre qui a failli être une catastrophe de communication

La première campagne a affiché **15,64 % de citations inexistantes**. C'est la métrique la plus grave que ce projet publie : un RAG comptable qui cite un article qui n'existe pas est pire qu'inutile. J'avais déjà le chiffre dans mon raisonnement, prêt à devenir la ligne d'ouverture du rapport.

Aucune n'était inventée. Les 69 concernées nommaient le bon article avec un extrait verbatim et avaient seulement perdu le suffixe `@2026-01-01`. Mon contrôle confondait « article inventé » et « identifiant abrégé ». Le vrai taux d'hallucination est zéro.

Les deux fautes restent des fautes — 39 articles du corpus portent plusieurs versions, donc omettre la version perd réellement de la traçabilité — mais ce sont deux fautes différentes, et un rapport qui les additionne ne mesure plus rien. Un test échoue maintenant si on les additionne.

C'est la quatrième occurrence dans ce dépôt du travers que le jalon 3 documentait déjà : **énoncer une explication avant le contrôle qui la départage.** La différence, cette fois, est que le contrôle a précédé la publication. La loi 6 du `CLAUDE.md` a servi exactement à ce pour quoi elle a été écrite.

### Il restait une citation non portante, et elle tenait à un caractère

Après décomposition des verdicts, une seule des 422 citations de la première campagne échouait — celle qui portait sur les 61 questions du dev d'avant extension. Elle divergeait du corpus d'une apostrophe : `l'actif` contre `l’actif`, dans un extrait qui écrivait correctement `l’écart d’acquisition` deux mots plus loin.

Une apostrophe droite ne fait pas dire autre chose à un article. Classer cela « non portant » était une erreur de catégorie — la métrique existe pour attraper une citation qui prête à un texte un propos qu'il ne tient pas. La normalisation replie donc les variantes d'apostrophe, **sur preuve du mécanisme**. Et le repli ne cache rien, parce que le taux de correspondance brute est publié à côté : l'écart entre les deux chiffres *est* cette citation.

Ce compteur-là avait été ajouté avant de savoir qu'il servirait. Ma première décision, après la sonde qui montrait 77 extraits verbatim sur 77, avait été « on compare brut, la normalisation ne sert à rien ». Raisonnement incomplet : le 77/77 prouve que la normalisation ne fait aucun travail sur des réponses honnêtes, donc qu'elle ne masque rien — il ne prouve pas que brut soit préférable. Ce qui départage est le sens de l'erreur. Comparer brut risque un faux « non portant », c'est-à-dire publier un défaut que le système n'a pas.

### Trois contrôles qui rassuraient sans rien vérifier

Le jalon en a produit trois, et c'est le motif le plus instructif de la semaine.

Un garde `if "@" in record_id` écrit pour empêcher qu'un identifiant versionné faux soit rattrapé. Aucune mutation ne pouvait le faire échouer : ajouter `@%` à une chaîne contenant déjà `@` ne correspond à aucun identifiant du corpus. Ligne morte, retirée.

Un garde-fou du jalon 3 qui vérifiait que l'audit d'intégrité couvre les deux splits — en comparant à un effectif codé en dur, 90. L'extension du benchmark l'a cassé alors que l'audit fonctionnait parfaitement. Un effectif figé mesurait la taille du benchmark, pas le périmètre audité.

Et le pire, parce qu'il était le mien et neuf : mon contrôle des chiffres publiés lisait la clé `detail` là où `citations.metriques` rend `details`. **Ma fixture de test portait la même faute**, donc les sept tests passaient sur un schéma qui n'existe pas, et le contrôle plantait sur les artefacts réels dès la première exécution. Il y a maintenant un test qui compare la fixture à la vraie sortie de la fonction. Une fixture qui invente son schéma ne teste rien.

### Ce que la loi 9 interdisait, et pourquoi il a fallu la lire de près

« Un rewriter, generator ou juge ne voit que le texte de la question — jamais les citations gold, le corpus, ou les résultats du retrieval. » Appliquée telle quelle à un juge, cette règle le rend incapable de noter la justesse : il ne lui reste que la cohérence interne.

La résolution est celle d'un jury d'examen. Le juge reçoit un **barème**, pas un corrigé-citations : une liste de critères écrite à la main depuis le PCG, qui porte ce qu'une réponse juste doit contenir sans dire quels `record_id` la portent. La justesse des citations est déjà mesurée sans LLM par la brique précédente. Ce que la loi 9 protège est la circularité — un modèle recevant la réponse qu'il doit trouver — et cela vise le réécriveur et le générateur, pas un correcteur.

### Le juge m'a corrigé, et j'ai laissé ma note fausse

Deux des cinq cas limites du jeu de calibration n'avaient aucune instance réelle : la campagne n'a produit aucune citation hallucinée, et l'unique abstention de dev était fondée. Les douze cas correspondants sont donc fabriqués à partir de réponses réelles, en inversant l'affirmation décisive et en gardant les citations verbatim. On calibre une balance avec des masses connues, pas avec des colis au hasard — mais le champ `origine` sépare les cas fabriqués des cas de campagne, et l'accord est publié des deux façons. Sur les 18 cas réels, il est exact.

Le seul désaccord des 30 va contre moi. Sur une réponse crédit-bail perturbée, j'avais compté un critère comme acquis parce que le bien finit bien à l'actif après la levée d'option. Le juge l'a refusé : dire que le bien y est « déjà » **nie** son entrée à cette date. Il a raison.

La note humaine reste telle qu'écrite. Elle a été fixée avant la mesure, et la récrire après coup viderait la calibration de son sens — c'est la même discipline que le seuil, qui est lu dans le JSON précisément pour ne pas pouvoir être déplacé une fois le résultat connu.

### Les questions d'abstention faciles ne servent à rien

Une question dont aucun mot ne figure au corpus n'est jamais remontée par le retrieval, et s'abstenir ne coûte alors rien. Les questions utiles sont celles où **le corpus cite sa source fiscale sans la contenir** : `pcg-na-25` renvoie à l'article 39-1-5° du CGI sans en énoncer les conditions, `pcg-741-2` renvoie la définition du contrôle exclusif au règlement ANC 2020-01 sans en donner le seuil, et le corpus traite longuement la provision pour licenciement sans jamais donner le barème du code du travail.

L'inspection des passages avant la mesure a retiré deux questions sur trente. L'une demandait si le mali technique est amortissable — `pcg-745-7` y répond mot pour mot. L'autre portait sur la livraison à soi-même, dont un commentaire déroule un cas concret. Une question à moitié répondable ne peut pas départager une abstention correcte d'une chance.

### La seule non-abstention n'était pas une fabrication, et je l'ai dit dans le code

Sur les 30 questions d'abstention, une seule réponse a mis le drapeau à `false`. Elle n'a rien inventé : elle ouvre sur « les passages fournis ne détaillent pas les écritures de retraitement, qui relèvent du règlement ANC n° 2020-01 », cite six extraits tous verbatim et existants, et conclut sur ce qui manque.

Seul le drapeau était faux, et c'est un défaut réel — un appelant qui s'y fie présenterait une réponse à une question sans réponse. Mais l'appeler « réponse inventée », comme le faisait ma première version de la métrique, est une accusation que la mesure ne soutient pas. Le champ s'appelle maintenant `non_abstentions`, et `n_fabrications` est publié à côté. Il vaut zéro.

### Le benchmark a corrigé une affirmation du README

Le jalon 3 publiait une réserve : « le corpus se limite aux Livres I à V — ni consolidation, ni fusions, ni NEP d'audit, ni BOFiP ». Le « ni fusions » était faux. Le Titre VII du Livre II, « Comptabilisation et évaluation des opérations de fusions et assimilées », est là depuis le début : 117 records, dont 81 portent un numéro d'article. C'est de là que viennent 32 des 60 nouvelles questions, y compris tout le dossier fusions du DSCG UE4 — prime de fusion, mali technique, période intercalaire, opérations réciproques.

Et j'ai failli publier 281 à la place de 117. Ma requête écrivait `chemin LIKE 'Livre II%Titre VII%'`, où `'Livre II%'` matche aussi « Livre III » et `'%Titre VII%'` matche aussi « Titre VIII » : deux chevauchements dans un même motif, et 281 = 117 + 164. Aucun gold n'était touché — ils avaient été vérifiés un par un — mais un chiffre de contexte faux dans un rapport reste un chiffre faux.

### Le contrôle de fraîcheur a bloqué ma propre clôture

`eval_generation.py` refuse de dépenser un appel d'API si le retrieval neutre ne rend plus la valeur publiée au jalon 3. L'extension du benchmark a porté dev de 61 à 93 questions, le contrôle a rendu 0,715 au lieu de 0,672, et il a bloqué.

La tentation était évidente : déplacer la constante. Cela aurait détruit le contrôle. Le périmètre est désormais **lu dans le JSON qui porte le chiffre publié**, pas déduit du contenu courant du benchmark — le contrôle reste donc vrai à travers toute extension future, et il bloque en plus si une question du périmètre publié disparaît, auquel cas le chiffre de référence cesserait d'être vérifiable.

### Le GPU est tombé du bus au milieu de la clôture

À la question 77 sur 93 : `CUDA error: unspecified launch failure`, puis `nvidia-smi` ne voit plus aucun périphérique alors que le module est chargé, `/dev/nvidia0` existe et la carte répond en PCI. Décrochage classique d'un dGPU Max-Q sous Linux. Recharger le stack de modules n'a rien récupéré ; seul un redémarrage l'a fait.

La partie payante avait survécu : 80 réponses dans un cache versionné sur disque. J'ai repris sur CPU en attendant — 115 s par question contre 1,5 s, mesurés, ordres de grandeur et pas décimales — et **fixé le contrôle avant de mesurer** : la clé de cache du générateur porte la liste exacte des passages montrés, donc si le retrieval produisait un autre classement, la clé changerait et l'appel serait payant. Sur 93 questions dont 80 en cache, `appels_api` devait valoir exactement 13.

Il a valu 13. Mais il faut dire ce que ce chiffre prouve : la machine a finalement redémarré, la campagne publiée est entièrement sur GPU, et le contrôle établit donc que **le retrieval reproduit à l'identique, après un redémarrage et une réinitialisation du pilote, les dix passages de chacune des 80 questions déjà mesurées.** C'est un meilleur contrôle que celui pour lequel il avait été écrit. Il ne prouve pas l'équivalence CPU/GPU, qui ne repose que sur une sonde à quatre questions et n'est plus nécessaire à la validité de la mesure.

### Ce que mes propres questions ont révélé de mes attentes

Quatre abstentions sur les 93 questions goldées de dev. Trois sont fondées : le gold était absent des dix passages montrés, donc le retrieval a échoué et la génération s'est bien comportée. La clé de cache permet de le trancher sans rien réexécuter.

La quatrième avait son gold parmi les passages. Sauf que l'énoncé est « j'ai déprécié un stock de marchandises qui ne part plus : est-ce que le fisc l'accepte comme charge ? » — une question **purement fiscale**, dont j'avais goldé la moitié comptable. S'abstenir est le comportement juste ; c'est mon attente qui est fausse.

Parmi les huit questions de divergence 2058-A, six demandent explicitement les deux volets et deux sont formulées de façon purement fiscale. Bilan honnête : une seule abstention des 93 est discutable, et elle est discutable à cause de l'énoncé que j'ai écrit. Je ne récris pas les questions maintenant — modifier un énoncé après avoir vu le résultat est exactement ce que ce dépôt s'interdit. La correction appartient au jalon suivant, qui apportera le corpus fiscal et rendra la question entièrement répondable.

### Ce que le jalon laisse ouvert

Le générateur cite en médiane 6 articles par réponse, jusqu'à 15, là où le gold en porte souvent un. Le « taux de réponses sans citation » est donc trivialement nul et la précision diluée. Le chiffre est publié pour que le biais soit lisible dans les mesures et non affirmé en prose — mais il n'est pas traité.

Le seuil de 30 caractères sous lequel un extrait est refusé pénalise les citations de tableaux : les quatre non-portantes qui subsistent sur dev sont toutes des lignes du plan de comptes (`6226 Honoraires`, `Production immobilisée 72`). Le seuil a été fixé avant la mesure et le déplacer maintenant serait substituer un critère après avoir vu le résultat. Il reste à 30, publié comme limite, et vaut un changement mesuré à part.

Et les deux dettes du jalon 3 sont toujours là, intactes : la fusion RRF récompense le consensus plutôt que l'excellence, masquée par la fenêtre du reranker et condamnée à réapparaître sur un corpus plus grand ; la réécriture dégrade trois questions sur soixante-et-une. Le jalon 4 ne les a pas touchées — il a construit de quoi les juger.

## Correctif du jalon 3 — réparer un défaut réel pour découvrir qu'on ne le voit plus

Le jalon 4 s'était terminé sur une phrase que je viens de prendre au mot : « les
deux dettes du jalon 3 sont toujours là, intactes ». J'ai ouvert la première.

Le défaut est propre : la somme RRF récompense le consensus, pas l'excellence. Sur
q023, le 2ᵉ meilleur candidat lexical du corpus entier — absent du canal dense —
perd contre un candidat 5ᵉ et 6ᵉ, parce que deux contributions médiocres
additionnées pèsent plus qu'une excellente isolée. Le jalon 3 l'avait fermé sur un
argument exact : le reranker rattrape le cas. Exact et insuffisant, parce qu'il
décrivait un rattrapage sans jamais mesurer sa marge.

### Le paramètre a une valeur neutre, et c'est tout le design

`score = max + poids_consensus × (somme − max)`. À 1,0 c'est la somme RRF
historique, à 0 seule l'excellence compte. Un scalaire, une grille, un défaut qui
reproduit la baseline publiée. C'est la loi 8 du dépôt appliquée avant même de
savoir si le levier sert à quelque chose — le rejet est le cas nominal, pas
l'exception.

### Trois fois où vérifier a coûté moins cher que supposer

**Le court-circuit qui n'a jamais existé.** J'avais prévu une branche spéciale à
`poids_consensus = 1,0`, au motif que `max + 1,0 × (somme − max)` ne serait pas
bit à bit égal à `somme` — et l'égalité bit à bit compte, un ULP d'écart peut
faire basculer un ex aequo et rendre le contrôle de fraîcheur menteur par
intermittence. Vérification exhaustive sur les 802 000 couples que le code peut
atteindre : aucune divergence, parce que la fusion ne porte que deux canaux. J'ai
retiré la branche au lieu de l'écrire. Une ligne qu'aucune mutation ne peut mettre
en défaut ne protège rien — même leçon que la garde morte de `citations.py` au
jalon 4, mais apprise avant de l'écrire cette fois.

**Mon propre test se contredisait.** Sa docstring affirmait qu'un gold absent de
la fusion compte au-delà de tous les seuils ; le code ne le comptait pas. C'est le
code qui avait tort : un gold introuvable est aussi hors de portée du reranker
qu'un gold au rang 300, et l'omettre faisait sous-estimer précisément le risque
que la métrique existe pour rendre visible.

**Le contrôle qui criait au loup.** À la première exécution : 73 chiffres publiés
sur 74 retrouvés dans le rapport. Le manquant, `−0,0122`, y figurait — avec le
signe moins typographique là où mon contrôle cherchait un tiret ASCII. Un
faux négatif de ce genre, répété, apprend à lire « 73 sur 74 » comme un succès, et
un contrôle qu'on apprend à ignorer ne vaut pas mieux qu'un contrôle absent.

### Le résultat, et pourquoi il tient en deux phrases contradictoires

**Le levier fonctionne. Il n'est pas adopté.**

En fusion nue, il répare trois questions sur 93 sans en casser une seule
(0,715 → 0,747, `p = 0,9537`). Le rang du gold de q023 descend de 11 à 4. Dans la
configuration livrée, il gagne une demi-question, `p = 0,6356`, très loin du seuil
de 0,95 fixé d'avance.

La raison ne s'invoque pas, elle se nomme. Les trois questions que la fusion
réparerait — q023, q054, q1032 — **valent déjà 1,0 dans la configuration livrée**.
Le cross-encoder les remonte tout seul. On ne répare pas deux fois la même
question.

### Ce que la coïncidence a failli me faire écrire

`poids_consensus = 0,10` et `rrf_k = 5` rendaient le même recall, le même delta et
le même `p` à la quatrième décimale. La conclusion tentante — « les deux leviers
sont équivalents » — aurait été fausse. Un `p` identique impose des deltas par
question identiques ; il n'impose rien sur les classements. Comparés séparément,
les vecteurs de recall sont identiques et les classements ne le sont **jamais**.
La métrique est trop grossière pour distinguer les deux leviers ; les rangs ne le
sont pas.

J'avais aussi écrit, dans le protocole committé avant la mesure, que `rrf_k`
serait « disqualifié d'avance » parce qu'il faudrait `rrf_k ≤ 1` pour renverser le
duel de q023. Le calcul était juste sur le duel et faux sur la grille : `rrf_k=5`
fait exactement aussi bien que le meilleur poids. J'avais généralisé à une grille
un calcul portant sur une question. C'est exactement pour ça que je l'avais mis
dans la grille au lieu de le publier comme mécanisme.

Symétriquement, ma prédiction sur `poids_consensus` méritait d'être relue plutôt
que déclarée confirmée : q023 ne passe devant son vainqueur nommé qu'à 0,025,
comme calculé, mais elle entre dans le top-10 dès 0,5 — parce qu'elle n'a pas
besoin de le battre, il suffit que les autres candidats bi-canaux perdent du
terrain. Battre un rival et entrer dans le top-10 sont deux choses différentes.

### Le chiffre qui compte n'est pas celui qui décide

La marge avant éviction, décomposée, dit ce que le recall cache. Sur 93 questions,
15 sont routées et 78 exposées à la fusion. En configuration livrée : 4 golds
absents du pool — un défaut de couverture, qu'aucune règle de fusion ne peut
corriger — et **2 seulement présents mais au-delà de la fenêtre du reranker**.

En fusion nue, le chiffre brut de 0,2179 se décompose en 14 + 3 sur 78 : il est à
82 % un défaut de couverture. La réécriture, elle, divise ce défaut par plus de
trois. Une métrique publiée sans sa décomposition aurait fait porter à la fusion
la responsabilité d'un problème de corpus.

Et un recoupement que je n'attendais pas : deux des quatre golds absents du pool,
q089 et q1031, sont exactement deux des quatre abstentions que le générateur avait
produites au jalon 4 et que j'y avais jugées bien fondées. Deux mesures
indépendantes — ce que le modèle répond, où le gold se situe — nomment les mêmes
questions. L'abstention correcte du générateur y est bien la conséquence d'un
défaut de retrieval, pas d'une prudence excessive.

### La loi 7 prise en flagrant délit

La configuration livrée a mis 12,28 s par question. Le jalon 3 en publiait 1,87.
Même machine, même carte, deux jours d'écart. Le GPU est resté bloqué en P8 à
300 MHz sur 2100 pendant les deux heures et demie de campagne, drapeaux
`SW Power Cap` et `SW Thermal Slowdown` actifs en continu — à 50 °C et 13,7 W sur
une enveloppe de 30 W. Aucun recall n'en dépend ; aucune latence de ce rapport
n'est comparable à celle d'un autre. « Une latence n'est pas une propriété du
système » n'est pas une précaution de style.

### Ce que je n'ai pas fait

Je n'ai pas exécuté le split gelé : le protocole ne le prévoyait qu'en cas
d'adoption. Je n'ai pas combiné les deux grilles, aucun levier n'ayant été adopté
seul. Je n'ai pas cherché le maximum entre 0,10 et 0,025, où le recall passe par
un sommet non localisé — le chercher sur dev reviendrait à régler un paramètre sur
le split de mesure, pour quelques dixièmes de question dans le contexte qui décide.

Et je n'ai pas déplacé le contexte d'adoption. `poids_consensus = 0,10` franchit
0,95 en fusion nue. La fusion nue n'est exécutée par personne — ni la démo, ni les
campagnes, ni le jalon 4. Substituer après coup le contexte qui arrange le
résultat serait la même faute que substituer la métrique qui l'arrange, et elle
serait plus difficile à repérer.

Reste donc la seconde dette du jalon 3, intacte : la réécriture dégrade trois
questions sur soixante-et-une.

## Seconde dette du jalon 3 — le meilleur chiffre du projet, et je ne le livre pas

La première dette s'était close sur un résultat propre : le levier marchait, le
reranker le rendait inutile, rejet sans regret. La seconde s'est close sur
l'inverse exact, et c'est beaucoup plus inconfortable.

### La piste laissée par le jalon 3 était fausse, et je l'ai su avant d'écrire du code

Le rapport proposait de conditionner la réécriture à un signal de recouvrement
faible. J'ai regardé quelles questions cassaient avant de concevoir quoi que ce
soit — q008, q025, q080 — et la piste s'est effondrée en une ligne : q080 est la
question la plus familière du benchmark (« j'ai avalé une boîte »), donc toute
porte fondée sur le recouvrement se déclencherait précisément là où la réécriture
casse.

C'est la première fois dans ce projet qu'une piste laissée par un jalon précédent
se révèle inapplicable, et le seul moyen de le voir était de regarder les
questions au lieu de la phrase qui les résumait. « La réécriture dégrade trois
questions » cachait trois mécanismes différents : un gold qui sort du pool, un
gold que le reranker voit et rejette, et un troisième qui ne bouge dans aucun
canal parce qu'il relève de l'autre dette. Les deux dettes du jalon 3 se
recoupaient sur une question, et personne ne l'avait remarqué.

### Une hypothèse séduisante, vraie sur les faits, fausse sur la cause

La réécriture de q080 parle consolidation : « écart d'acquisition », « goodwill »,
« regroupement d'entreprises ». Ces trois termes figurent dans **zéro** record du
corpus, alors que le gold s'intitule « Traitement du mali de fusion ». Le modèle a
donné la réponse des comptes consolidés à une question de fusion — une vraie
confusion comptable, exactement le genre que ce projet existe pour attraper.

Et ce n'est pas la cause de l'échec. Le canal dense, seul à pouvoir souffrir d'un
vocabulaire absent de l'index, ne trouvait déjà pas ce gold. Un terme de fréquence
documentaire nulle ne pèse rien en BM25. J'aurais publié un mécanisme juste sur
les faits et faux sur la causalité, et il aurait été impossible à démentir en
lisant le rapport.

### Le refus

`poids_question=3` donne **0,898 de recall@10 sur dev** — le meilleur chiffre que
ce projet ait jamais mesuré. Il améliore les deux catégories non triviales, ne
coûte rien en latence, répare deux des trois questions que la réécriture cassait,
et ramène à zéro le nombre de golds présents dans la fusion mais hors de portée du
reranker.

`p_amelioration = 0,8809`. Le critère est 0,95. Il a été fixé avant la mesure.
L'IC95 contient zéro.

Je n'adopte pas. Et je n'ai pas mesuré `poids_question=4`, alors que le recall
passe par un maximum entre 3 et 5 et qu'un réglage intermédiaire franchirait
peut-être le seuil. Chercher ce réglage-là, maintenant, serait régler un paramètre
sur le split de mesure pour atteindre un seuil — la forme la plus directe du
sur-ajustement, et la plus facile à se justifier à soi-même parce que le chiffre
est beau.

C'est la première fois que la discipline de ce dépôt me coûte un résultat que
j'avais envie de garder. Le rapport publie donc explicitement ce que le refus
laisse sur la table, ligne par ligne, pour que le prochain jalon puisse le
reprendre sans avoir à me croire sur parole.

### Le reranker collabore, au lieu d'absorber

Le premier correctif avait établi que le reranker absorbe intégralement une
amélioration de la fusion. J'avais enregistré, avant de mesurer, la prédiction
inverse pour ce levier-ci : la fusion réordonne l'intérieur du pool, celui-ci
change sa composition, et un candidat absent du pool est hors de portée de tout
reranker.

La prédiction tient, et le contrôle est plus net que je n'espérais : les quatre
questions réparées dans la configuration livrée ne sont réparées par le levier
seul dans **aucun** cas. Il amène leur gold assez près pour que le cross-encoder
finisse le travail. L'effet est trois fois plus grand avec reranking que sans —
l'inverse exact du correctif précédent, pour une raison structurelle et non par
accident de mesure.

### L'agrégat stable qui recouvrait une substitution complète

Le nombre de golds hors du pool ne bouge pas : 4 avant, 4 après. Lu seul, il dit
que rien n'a changé au sujet même du correctif. En réalité q080 rentre — c'est le
but atteint — et q057 sort. La composition change entièrement sous un compte
identique.

C'est la deuxième fois en deux correctifs qu'un agrégat stable cache un mouvement
complet, après la part de golds au-delà du rang 25 du premier. Je commence à
penser que publier un compte sans sa composition est, dans ce projet, une faute
par défaut plutôt qu'une omission occasionnelle.

### Trois erreurs de méthode, dont deux dans mes propres outils

Mon `pkill -f pytest` tuait le shell qui lançait pytest, sa propre ligne de
commande contenant « pytest ». Les modifications de tests n'avaient jamais été
appliquées et j'ai cru qu'elles l'étaient.

Après avoir extrait la machinerie de mesure dans le paquet — pour ne pas la
recopier d'un correctif à l'autre — un test patchait encore l'espace de noms du
script. Il lançait donc une **vraie campagne** sous un test unitaire. J'avais
identifié ce risque en concevant l'extraction ; l'avoir prévu ne l'a pas empêché.

Et rien n'empêchait `--contexte hybrid` d'écraser un ancrage publié par un
artefact amputé, qui se lit exactement comme un artefact complet. Le trou est
apparu parce que j'avais besoin de cette commande pour contrôler l'extraction. La
garde qui le refuse existe maintenant, et l'extraction elle-même est prouvée
neutre : huit configurations rejouées, zéro écart, vecteurs de rangs compris.

### Où en sont les deux dettes

Mesurées, exposées à valeur neutre, non adoptées — un résultat négatif chacune,
pour deux raisons opposées. La fusion échoue parce que le reranker fait déjà son
travail. La réécriture échoue de peu, parce que 93 questions ne suffisent pas à
trancher un effet de trois questions.

La seconde mérite d'être re-soumise au même critère quand `dev` aura grandi. Sans
toucher à sa grille, et sans la remesurer entre-temps pour voir.

### Toutes les directions essayées, y compris celles qui n'ont jamais atteint le code

Les sections ci-dessus racontent ce qui a été mesuré. Voici ce qui a été envisagé
puis écarté, avec la raison et le moment où la décision a été prise — parce qu'une
piste abandonnée pour une bonne raison est une information, et qu'un rapport qui
ne montre que le chemin retenu laisse croire qu'il était évident.

**Quelle dette attaquer en premier.** Le jalon 3 en laissait deux. J'ai choisi la
fusion et écarté la réécriture, au motif que la piste du jalon 3 pour cette
dernière — conditionner la réécriture à un signal de recouvrement — introduisait
un seuil qu'il faudrait calibrer sur le split de mesure, pour un net déjà positif.
La raison était bonne. Elle n'était pas la bonne : quand j'y suis revenu, la piste
s'est révélée inapplicable pour un motif entièrement différent, que seule la
lecture des questions cassées pouvait révéler.

**Trois règles de fusion, dont deux jamais mesurées.** Le jalon 3 suggérait
« maximum au lieu de somme, normalisation des scores, bonus explicite au rang 1 ».

- *Maximum pur* : écarté comme paramètre discret. Deux documents au même rang dans
  deux canaux différents obtiennent exactement le même score, donc le maximum
  produit des ex aequo en masse, départagés par l'ordre d'insertion — arbitraire et
  invisible. Récupéré autrement : `poids_consensus → 0` est le maximum comme
  **limite continue**, où la somme résiduelle sert de départage.
- *Normalisation des scores par canal* (min-max puis somme) : écartée **sans
  mesure**, et c'est une réserve assumée. BM25 et distance cosinus ne vivent pas sur
  la même échelle ; une normalisation min-max dépend de la composition du pool,
  donc bouger la règle bougerait aussi implicitement le pool — deux variables.
- *Bonus au rang 1* : couvert par la grille, c'est un cas particulier de
  `poids_consensus` faible.

**Trois hypothèses successives sur la cause de la casse par réécriture.**

1. *Dérive de l'embedding.* Ma première, et elle était séduisante : la réécriture
   de q080 emmène la requête vers le vocabulaire de la consolidation, donc le
   vecteur dérive. **Fausse.** Le canal dense ne trouve jamais ce gold, réécriture
   ou non, jusqu'au rang 400.
2. *Filtrer les termes de fréquence documentaire nulle* — un `df_min`, exactement
   l'inverse du `df_max` mesuré et rejeté au jalon 3. Écarté **avant de coder** :
   un terme absent de l'index ne matche rien dans FTS5, il ne peut donc pas diluer
   BM25. Le levier n'aurait rien fait, et l'aurait fait de façon convaincante.
3. *Dilution lexicale.* Confirmée par sonde, table des rangs à l'appui. C'est la
   seule des trois qui ait été mesurée, et la seule vraie.

**Corriger le prompt du rewriter plutôt que le contrepeser.** La réécriture de
q080 est objectivement fausse pour ce corpus — elle répond consolidation à une
question de fusion. La corriger à la source invaliderait le cache versionné de 93
réécritures, donc une campagne payante à refaire et un protocole neuf. Écarté pour
ce correctif, consigné en réserve.

**Trois façons de partager la machinerie de mesure.** Dupliquer les fonctions dans
un second script (rejeté : c'est la reconstitution parallèle qui dérive, l'argument
que j'avais moi-même employé pour extraire `avant_rerank`). Renommer le script de
fusion en script générique (rejeté : il a produit un artefact publié, et le rapport
doit pouvoir nommer le script qui l'a produit). Extraire dans le paquet — retenu,
avec un contrôle de reproduction qui rejoue la grille et exige zéro écart.

**Chercher `poids_question=4`.** Le recall passe par un maximum entre 3 et 5 et un
réglage intermédiaire franchirait peut-être le seuil d'adoption. Écarté, et c'est
le seul écart de cette liste qui m'a coûté quelque chose : chercher ce réglage-là
serait régler un paramètre sur le split de mesure pour atteindre un seuil.

**Un test qui passait chez moi et échouait sur le runner.** Le contrôle du
contexte `reecriture` construisait un `Searcher` sur `data/corpus.db`, gitignoré
donc absent en CI. Toutes mes exécutions locales l'avaient. Récrit sur base
synthétique, puis vérifié en masquant le corpus pour reproduire les conditions du
runner — 297 tests, 7 ignorés, comme sur la CI.
