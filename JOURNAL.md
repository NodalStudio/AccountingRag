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
