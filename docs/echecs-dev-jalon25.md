Sortie brute de `uv run python scripts/analyse_echecs.py --split dev --mode hybrid`, exécutée sur `data/corpus.db` **avant** tout changement de `SYNONYMES` (état de la branche à l'issue de T4, jalon 2.5). C'est le matériau brut du choix des synonymes de l'Ablation C (T5) — voir `docs/eval-jalon25.md`, section « Ablation C ».

**Config** (ruling J25-7) : mode `hybrid` (RRF bm25+dense, paramètres neutres `poids_chemin=1.0`/`boost_commentaire=1.0`) — pas `hybrid+rerank` : le canal synonymes n'agit que sur la jambe bm25 (`normalize()` est appliqué avant `bm25()`, pas avant l'encodage dense), et `hybrid` tourne en ~1 min contre ~2h pour `hybrid+rerank` sur cette machine (cf. `docs/eval-jalon25.md`, Ablation B).

**Résultat de la mesure (post-hoc)** : le lot de 3 entrées tiré de cette analyse (`aides`→`subventions`, `la/une boite`→`l'/une entite`) a été mesuré par bootstrap apparié et **REJETÉ globalement** (`p_amelioration=0,000`, aucun effet sur aucune des 61 questions) — voir `docs/eval-jalon25.md`, section « Ablation C », pour le détail complet et la cause racine identifiée. `SYNONYMES` a été restauré à son état d'avant T5 ; **les 21 échecs listés ci-dessous restent donc, sans exception, l'état final du split dev à l'issue de cette tâche.**

---

# Analyse des échecs — split `dev`, mode `hybrid` (2026-08-16)

Recall@10 global : 0.672 ; MRR : 0.565 ; n=61.

Questions en échec (recall@10 < 1) : 21/61.

- `reference_directe` : recall@10=1.0 (n=7)
- `regle` : recall@10=0.935 (n=23)
- `vocabulaire_courant` : recall@10=0.403 (n=31)

---

## q008 (regle) — recall@10=0.500

**Question** : En quoi une provision réglementée se distingue-t-elle d'une provision au sens général du terme ?

**Citations gold** : ['pcg-313-1', 'pcg-321-5']

**Top-10 obtenu (hybrid)** :

1. `pcg-313-1-c1@2026-01-01` (source=fusion, score=0.0325) **← gold**
2. `pcg-313-1@2026-01-01` (source=fusion, score=0.0318) **← gold**
3. `pcg-1211-14@2026-01-01` (source=fusion, score=0.0306)
4. `pcg-214-8@2026-01-01` (source=fusion, score=0.0306)
5. `pcg-500-2-c2@2026-01-01` (source=fusion, score=0.03)
6. `pcg-401-1-c3@2026-01-01` (source=fusion, score=0.0294)
7. `pcg-1121-1@2026-01-01` (source=fusion, score=0.0264)
8. `pcg-324-1-c8@2026-01-01` (source=fusion, score=0.0258)
9. `pcg-na-69@2026-01-01` (source=fusion, score=0.0246)
10. `pcg-na-64@2026-01-01` (source=fusion, score=0.0243)

**Records gold** :

- `pcg-313-1-c1@2026-01-01` (article 313-1) : IR 4 : exemples de provisions réglementées Ont le caractère de provisions réglementées les provisions : - pour hausse des prix ; - pour risques afférents aux crédits à moyen terme résultant d'opérations faites à l'étranger ; - autorisées spécialement pour certaines professions, par exemple pour la r…
- `pcg-313-1@2026-01-01` (article 313-1) : Les provisions réglementées sont des provisions constituées en application de textes particuliers de niveau supérieur. Elles ne correspondent pas à la définition d'une provision telle que définie à l'article 321-5. Elles sont comptabilisées suivant un mécanisme analogue à celui des provisions propre…
- `pcg-321-5-c1@2026-01-01` (article 321-5) : Informations en annexe – Se reporter à l'art. 832-13 et à l'art. 832-14
- `pcg-321-5-c2@2026-01-01` (article 321-5) : Provisions pour risques et charges – Avis CNC n° 00-01 du 20 avril 2000 relatif aux passifs Les provisions pour risques et charges ont un caractère éventuel au titre de leur montant ou de leur échéance mais correspondent à une obligation probable ou certaine à la date de clôture.
- `pcg-321-5@2026-01-01` (article 321-5) : Une provision est un passif dont l'échéance ou le montant n'est pas fixé de façon précise.

**Tokens question absents du(des) record(s) gold** : `distingue-t-el`, `general`, `quoi`, `sen`

**Tokens gold absents de la question** : `00-01`, `20`, `2000`, `214-8`, `321-5`, `4`, `832-13`, `832-14`, `afferent`, `amort`, `analogu`, `annex`, `appliqu`, `art`, `articl`, `autoris`, `avis`, `avril`, `caracter`, `celui`, `certain`, `charg`, `clotur`, `cnc`, `comptabilis`, `condit`, `constitu`, `constituent`, `correspondent`, `cre`, `credit`, `dat`, `defin`, `definit`, `derogatoir`, `dit`, `dont`, `echeanc`, `egal`, `etrang` *(+40 tronqués)*

---

## q021 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai acheté une machine pour mon atelier, comment je répartis son coût sur les années où je vais m'en servir ?

**Citations gold** : ['pcg-214-13']

**Top-10 obtenu (hybrid)** :

1. `pcg-131-3-c1@2026-01-01` (source=fusion, score=0.0263)
2. `pcg-na-236@2026-01-01` (source=fusion, score=0.023)
3. `pcg-na-280@2026-01-01` (source=fusion, score=0.0164)
4. `pcg-na-42@2026-01-01` (source=fusion, score=0.0164)
5. `pcg-na-149@2026-01-01` (source=fusion, score=0.0161)
6. `pcg-836-1@2026-01-01#3` (source=fusion, score=0.0161)
7. `pcg-324-1-c16@2026-01-01` (source=fusion, score=0.0159)
8. `pcg-836-4@2026-01-01#2` (source=fusion, score=0.0159)
9. `pcg-324-1-c49@2026-01-01` (source=fusion, score=0.0156)
10. `pcg-na-41@2026-01-01` (source=fusion, score=0.0156)

**Records gold** :

- `pcg-214-13-c1@2026-01-01` (article 214-13) : Durées d'amortissement – Note de présentation du règlement ANC n° 2015-06 du 23 novembre 2015 modifiant le plan comptable général Dans les comptes individuels, pour les actifs à durée d'utilisation limitée, les durées résultant des usages professionnels peuvent être retenues si elles ne sont pas con…
- `pcg-214-13-c2@2026-01-01` (article 214-13) : Lien avec les comptes consolidés – Avis CU n° 2005-D du 1er juin 2005 afférent aux modalités d'application des règlements n° 2002-10 relatif à l'amortissement et la dépréciation des actifs et n° 2004-06 relatif à la définition, la comptabilisation et l'évaluation des actifs L'amortissement d'une imm…
- `pcg-214-13-c3@2026-01-01` (article 214-13) : Règles d'amortissement – Avis CU n° 2006-C du 4 octobre 2006 afférant à l'interprétation des dispositions de l'avis CNC n° 2004-15 du 23 juin 2004 relatif à la définition, la comptabilisation et l'évaluation des actifs, excluant dans les comptes individuels, les contrats de location au sens d'IAS 17…
- `pcg-214-13@2026-01-01` (article 214-13) : L'amortissement d'un actif est la répartition systématique de son montant amortissable en fonction de son utilisation. L'amortissement est déterminé par le plan d'amortissement établi en fonction de la durée et du mode d'amortissement propres à chaque actif amortissable, tels qu'ils sont déterminés …

**Tokens question absents du(des) record(s) gold** : `achet`, `anne`, `ateli`, `cout`, `machin`, `serv`, `vais`

**Tokens gold absents de la question** : `-financ`, `1`, `15`, `17`, `1980`, `1er`, `2`, `2000`, `2002-10`, `2004`, `2004-06`, `2004-15`, `2005`, `2005-d`, `2006`, `2006-c`, `2015`, `2015-06`, `214-13`, `214-8`, `23`, `30`, `33`, `39`, `39-1-2`, `4`, `4ème`, `80-531`, `99-07`, `achat`, `acquis`, `acquisit`, `actif`, `activ`, `adapt`, `administr`, `admis`, `affect`, `affer`, `afferent` *(+240 tronqués)*

---

## q022 (vocabulaire_courant) — recall@10=0.500

**Question** : J'ai un client qui ne paie pas depuis des mois, et un autre dont je suis sûr qu'il ne paiera plus jamais : est-ce que je les traite pareil dans ma compta ?

**Citations gold** : ['pcg-1214-41', 'pcg-1221-65']

**Top-10 obtenu (hybrid)** :

1. `pcg-1121-1@2026-01-01` (source=fusion, score=0.0305)
2. `pcg-1214-41@2026-01-01` (source=fusion, score=0.0284) **← gold**
3. `pcg-na-318@2026-01-01` (source=fusion, score=0.0273)
4. `pcg-324-1-c23@2026-01-01` (source=fusion, score=0.0229)
5. `pcg-na-230@2026-01-01` (source=fusion, score=0.0164)
6. `pcg-na-178@2026-01-01` (source=fusion, score=0.0164)
7. `pcg-na-141@2026-01-01` (source=fusion, score=0.0161)
8. `pcg-625-9-c1@2026-01-01` (source=fusion, score=0.0161)
9. `pcg-324-1-c15@2026-01-01` (source=fusion, score=0.0159)
10. `pcg-na-263@2026-01-01` (source=fusion, score=0.0159)

**Records gold** :

- `pcg-1214-41@2026-01-01` (article 1214-41) : 41 : Clients et comptes rattachés Les créances liées à la vente de biens ou services rattachés au cycle d'exploitation de l'entité sont enregistrées au compte 41 « Clients et comptes rattachés ». Le compte 411 « Clients » est débité du montant des factures de ventes de biens ou de prestations de ser…
- `pcg-1221-65@2026-01-01` (article 1221-65) : 65 : Autres charges de gestion courante Les pertes sur créances irrécouvrables qui présentent un caractère habituel eu égard notamment à la nature de l'activité ou au volume des affaires traitées sont enregistrées au débit du compte 654 « Pertes sur créances irrécouvrables ». Le compte 655 « Quote-p…

**Tokens question absents du(des) record(s) gold** : `depuis`, `est-ce`, `jam`, `mois`, `pai`, `pareil`

**Tokens gold absents de la question** : `4`, `41`, `411`, `4117`, `413`, `416`, `418`, `4191`, `4196`, `4457`, `458`, `65`, `654`, `655`, `656`, `657`, `661`, `7`, `70`, `708`, `709`, `absenc`, `accept`, `accord`, `acompt`, `actif`, `activ`, `affair`, `agir`, `ajust`, `amort`, `apparait`, `approvision`, `apres`, `associ`, `avanc`, `avis`, `bancair`, `benefic`, `bien` *(+171 tronqués)*

---

## q023 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai des marchandises en stock qui ont perdu de la valeur et que je ne pense pas vendre au prix que j'espérais, je fais quoi ?

**Citations gold** : ['pcg-214-22']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-236@2026-01-01` (source=fusion, score=0.0305)
2. `pcg-214-25@2026-01-01` (source=fusion, score=0.0271)
3. `pcg-1121-1@2026-01-01` (source=fusion, score=0.0247)
4. `pcg-500-2@2026-01-01` (source=fusion, score=0.0245)
5. `pcg-821-2@2026-01-01` (source=fusion, score=0.0245)
6. `pcg-622-5-c4@2026-01-01` (source=fusion, score=0.0231)
7. `pcg-832-7@2026-01-01` (source=fusion, score=0.0216)
8. `pcg-na-263@2026-01-01` (source=fusion, score=0.021)
9. `pcg-214-23@2026-01-01` (source=fusion, score=0.0164)
10. `pcg-213-31-c2@2026-01-01` (source=fusion, score=0.0164)

**Records gold** :

- `pcg-214-22@2026-01-01` (article 214-22) : A la date de clôture de l'exercice, les stocks et les productions en cours sont évalués selon les règles générales d'évaluation énoncées aux articles 214-1 à 214-6 et 214-16 à 214-19, sous réserve des dispositions prévues aux articles 214-23 et 214-24. A l'inventaire, les stocks et les productions e…

**Tokens question absents du(des) record(s) gold** : `esper`, `fais`, `pens`, `perdu`, `quoi`, `valeur`, `vendr`

**Tokens gold absents de la question** : `214-1`, `214-16`, `214-19`, `214-23`, `214-24`, `214-6`, `actif`, `articl`, `bas`, `categor`, `chaqu`, `clotur`, `compr`, `comprend`, `concern`, `consider`, `cour`, `couvert`, `couvertur`, `dat`, `depreci`, `deux`, `disposit`, `document`, `doivent`, `element`, `engag`, `enonc`, `ensembl`, `estim`, `evalu`, `eventuel`, `exercic`, `ferm`, `financi`, `form`, `futur`, `general`, `global`, `hauteur` *(+35 tronqués)*

---

## q026 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai créé ma clientèle et ma réputation moi-même depuis le début, est-ce que je peux lui donner une valeur à l'actif de mon bilan ?

**Citations gold** : ['pcg-212-3']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-263@2026-01-01` (source=fusion, score=0.0274)
2. `pcg-na-57@2026-01-01` (source=fusion, score=0.0215)
3. `pcg-na-318@2026-01-01` (source=fusion, score=0.0164)
4. `pcg-836-1@2026-01-01#3` (source=fusion, score=0.0164)
5. `pcg-211-5-c1@2026-01-01` (source=fusion, score=0.0161)
6. `pcg-743-1@2026-01-01#2` (source=fusion, score=0.0161)
7. `pcg-214-12@2026-01-01` (source=fusion, score=0.0159)
8. `pcg-625-7@2026-01-01` (source=fusion, score=0.0156)
9. `pcg-214-6-c1@2026-01-01` (source=fusion, score=0.0156)
10. `pcg-na-280@2026-01-01` (source=fusion, score=0.0154)

**Records gold** :

- `pcg-212-3-c1@2026-01-01` (article 212-3) : Immobilisations incorporelles générées en interne – Avis CNC n° 2004-15 du 23 juin 2004 relatif à la définition, la comptabilisation et l'évaluation des actifs - Distinction phase de recherche/phase de développement Pour apprécier si une immobilisation incorporelle générée en interne satisfait aux c…
- `pcg-212-3-c2@2026-01-01` (article 212-3) : Fonds commercial – Note de présentation du règlement ANC n° 2015-06 du 23 novembre 2015 modifiant le plan comptable général Le fonds commercial, notion juridique spécifique en droit comptable français, constitue la partie - « pivot » du fonds de commerce, notion consacrée par le droit commercial fra…
- `pcg-212-3-c3@2026-01-01` (article 212-3) : IR 2 - Frais d'exploration minière assimilés à des frais de développement – Note de présentation du règlement ANC n° 2017-03 du 3 novembre 2017 modifiant le plan comptable général L'article R.123-188 du code de commerce dispose que les frais d'exploration minière assimilés à des frais de développeme…
- `pcg-212-3-c4@2026-01-01` (article 212-3) : IR 4 - Frais d'exploration minière assimilés à des frais de développement– Note de présentation du règlement ANC n° 2017-03 du 3 novembre 2017 modifiant le plan comptable général Le schéma illustratif ci-dessous, basé sur un cycle de prospection pétrolière, fournit une typologie indicative des phase…
- `pcg-212-3@2026-01-01` (article 212-3) : 1. Les frais de développement peuvent être comptabilisés à l'actif s'ils se rapportent à des projets nettement individualisés, ayant de sérieuses chances de réussite technique et de rentabilité commerciale – ou de viabilité économique pour les projets de développement pluriannuels associatifs. Ceci …

**Tokens question absents du(des) record(s) gold** : `debut`, `depuis`, `don`, `est-ce`, `lui`, `moi-mem`, `reput`, `valeur`

**Tokens gold absents de la question** : `1`, `2`, `2004`, `2004-15`, `2015`, `2015-06`, `2017`, `2017-03`, `212-3`, `23`, `3`, `4`, `611-5`, `achalandag`, `achev`, `acquis`, `activ`, `aient`, `altern`, `amelior`, `amont`, `anc`, `appliqu`, `appliquent`, `appreci`, `appropri`, `articl`, `assimil`, `associ`, `attribu`, `aucun`, `autr`, `avantag`, `avis`, `ayant`, `b`, `bail`, `bas`, `bien`, `brevet` *(+259 tronqués)*

---

## q054 (regle) — recall@10=0.000

**Question** : Comment le coût d'une immobilisation créée par les moyens propres de l'entité est-il suivi jusqu'à son achèvement ?

**Citations gold** : ['pcg-1212-23']

**Top-10 obtenu (hybrid)** :

1. `pcg-212-3@2026-01-01` (source=fusion, score=0.0282)
2. `pcg-131-3-c1@2026-01-01` (source=fusion, score=0.0277)
3. `pcg-212-1@2026-01-01` (source=fusion, score=0.0227)
4. `pcg-213-1@2026-01-01` (source=fusion, score=0.0222)
5. `pcg-213-9@2026-01-01` (source=fusion, score=0.0216)
6. `pcg-832-4@2026-01-01` (source=fusion, score=0.0215)
7. `pcg-na-236@2026-01-01` (source=fusion, score=0.0214)
8. `pcg-311-1-c1@2026-01-01` (source=fusion, score=0.0205)
9. `pcg-1222-72@2026-01-01` (source=fusion, score=0.0164)
10. `pcg-na-74@2026-01-01` (source=fusion, score=0.0164)

**Records gold** :

- `pcg-1212-23@2026-01-01` (article 1212-23) : 23 : Immobilisations en cours, avances et acomptes Le compte 23 « Immobilisations en cours, avances et acomptes » a pour objet de faire apparaître la valeur des immobilisations non terminées à la fin de chaque exercice. Du point de vue de leur origine, les immobilisations inscrites aux comptes 231 e…

**Tokens question absents du(des) record(s) gold** : `achev`, `est-il`, `suiv`

**Tokens gold absents de la question** : `20`, `21`, `23`, `231`, `232`, `237`, `238`, `4091`, `72`, `acompt`, `acquisit`, `apparaitr`, `appropri`, `avanc`, `cel`, `celui`, `chaqu`, `ci-dessus`, `command`, `compt`, `comptabilis`, `concern`, `confi`, `corporel`, `cour`, `cred`, `deb`, `debit`, `deux`, `dur`, `effectu`, `enregistr`, `exercic`, `facult`, `fin`, `fournisseur`, `fur`, `group`, `incorporel`, `inscrit` *(+30 tronqués)*

**Note d'analyse (choix des synonymes)** : le diff « tokens question absents du gold » est minuscule (3 tokens : `achev`, `est-il`, `suiv`) — la question partage déjà presque tout son vocabulaire normalisé avec le record gold (« immobilisation créée par les moyens propres de l'entité » est quasiment une citation littérale de `pcg-1212-23`/`pcg-1222-72`). **Ce n'est donc pas un fossé lexical** mais un problème de rang dans la fusion RRF (le canal dense dilue vraisemblablement le score bm25 déjà pertinent) — aucun synonyme ne peut agir ici puisqu'il n'y a rien à rapprocher lexicalement. Classé « échec restant » (matériau jalon 3).

---

## q056 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai des matières premières que je viens d'acheter et que je n'ai pas encore utilisées : comment je calcule la valeur à laquelle je les inscris dans ma compta ?

**Citations gold** : ['pcg-213-31']

**Top-10 obtenu (hybrid)** :

1. `pcg-832-7@2026-01-01` (source=fusion, score=0.0323)
2. `pcg-na-323@2026-01-01` (source=fusion, score=0.0315)
3. `pcg-na-280@2026-01-01` (source=fusion, score=0.0257)
4. `pcg-628-18-c1@2026-01-01` (source=fusion, score=0.0256)
5. `pcg-na-20@2026-01-01` (source=fusion, score=0.0245)
6. `pcg-121-2-c7@2026-01-01` (source=fusion, score=0.0243)
7. `pcg-na-207@2026-01-01` (source=fusion, score=0.024)
8. `pcg-na-24@2026-01-01` (source=fusion, score=0.0225)
9. `pcg-na-149@2026-01-01` (source=fusion, score=0.0164)
10. `pcg-616-16@2026-01-01#2` (source=fusion, score=0.0161)

**Records gold** :

- `pcg-213-31-c1@2026-01-01` (article 213-31) : Avis CNC n° 2004-15 du 23 juin 2004 relatif à la définition, la comptabilisation et l'évaluation des actifs - Charges de stockage Les charges de stockage s'ajoutent aux coûts d'acquisition ou de production lorsque les conditions spécifiques d'exploitation le justifient. Les pertes et gaspillages son…
- `pcg-213-31-c2@2026-01-01` (article 213-31) : - Exemples de coûts exclus du coût des stocks et comptabilisés en charges de l'exercice au cours duquel ils sont encourus : - montants anormaux de déchets de fabrication, de main-d'œuvre ou d'autres coûts de production ; - coûts de stockage, à moins que ces coûts soient nécessaires au processus de p…
- `pcg-213-31@2026-01-01` (article 213-31) : Le coût d'acquisition des stocks est constitué du : - prix d'achat, y compris les droits de douane et autres taxes non récupérables, après déduction des rabais commerciaux, remises, escomptes de règlement et autres éléments similaires ; - ainsi que des frais de transport, de manutention et autres co…

**Tokens question absents du(des) record(s) gold** : `achet`, `calcul`, `compt`, `encor`, `inscris`, `laquel`, `utilis`, `valeur`, `vien`

**Tokens gold absents de la question** : `2004`, `2004-15`, `23`, `achat`, `acquisit`, `actif`, `administr`, `ains`, `ajoutent`, `anormal`, `apres`, `attribu`, `autr`, `avis`, `charg`, `cnc`, `commercial`, `commercialis`, `compr`, `comptabilis`, `condit`, `constitu`, `contribuent`, `cour`, `cout`, `dechet`, `dedi`, `deduct`, `definit`, `direct`, `douan`, `droit`, `duquel`, `element`, `encourus`, `endroit`, `escompt`, `etap`, `etat`, `evalu` *(+43 tronqués)*

---

## q057 (vocabulaire_courant) — recall@10=0.000

**Question** : On me demande de donner une valeur à ce qui est en train d'être fabriqué dans l'atelier mais qui n'est pas encore terminé à la fin de l'année : comment on appelle ça et à quoi ça correspond exactement ?

**Citations gold** : ['pcg-211-7']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-263@2026-01-01` (source=fusion, score=0.0272)
2. `pcg-na-236@2026-01-01` (source=fusion, score=0.0267)
3. `pcg-191-1-c5@2026-01-01` (source=fusion, score=0.0221)
4. `pcg-324-1-c16@2026-01-01` (source=fusion, score=0.0164)
5. `pcg-836-1@2026-01-01#3` (source=fusion, score=0.0164)
6. `pcg-324-1-c26@2026-01-01` (source=fusion, score=0.0161)
7. `pcg-na-323@2026-01-01` (source=fusion, score=0.0159)
8. `pcg-838-1@2026-01-01` (source=fusion, score=0.0159)
9. `pcg-324-1-c24@2026-01-01` (source=fusion, score=0.0156)
10. `pcg-743-1@2026-01-01#2` (source=fusion, score=0.0156)

**Records gold** :

- `pcg-211-7@2026-01-01` (article 211-7) : Un stock est un actif détenu pour être vendu dans le cours normal de l'activité, ou en cours de production pour une telle vente, ou destiné à être consommé dans le processus de production ou de prestation de services, sous forme de matières premières ou de fournitures.

**Tokens question absents du(des) record(s) gold** : `anne`, `appel`, `ateli`, `ca`, `correspond`, `demand`, `don`, `encor`, `exact`, `fabriqu`, `fin`, `mais`, `quoi`, `termin`, `train`, `valeur`

**Tokens gold absents de la question** : `actif`, `activ`, `consomm`, `cour`, `destin`, `detenu`, `form`, `fournitur`, `mati`, `normal`, `premi`, `prestat`, `processus`, `product`, `servic`, `sous`, `stock`, `tel`, `vendu`, `vent`

**Note d'analyse (choix des synonymes)** : « en train d'être fabriqué » ≈ « en cours de production » — une entrée de synonyme aurait été possible, mais elle serait quasi certainement du sur-ajustement à cette formulation précise du benchmark (peu de chance qu'une future question reprenne exactement cette tournure), pas un vrai terme métier réutilisable comme « leasing »/« credit-bail ». **Écarté par prudence.**

---

## q059 (vocabulaire_courant) — recall@10=0.000

**Question** : La région m'a versé de l'argent cette année pour m'aider à faire tourner la boîte, pas pour acheter une machine : comment j'enregistre ça dans mes comptes ?

**Citations gold** : ['pcg-1222-74']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-280@2026-01-01` (source=fusion, score=0.0262)
2. `pcg-324-1-c16@2026-01-01` (source=fusion, score=0.0164)
3. `pcg-616-16@2026-01-01#2` (source=fusion, score=0.0164)
4. `pcg-na-109@2026-01-01` (source=fusion, score=0.0161)
5. `pcg-na-149@2026-01-01` (source=fusion, score=0.0159)
6. `pcg-621-11-c1@2026-01-01` (source=fusion, score=0.0159)
7. `pcg-112-4@2026-01-01` (source=fusion, score=0.0156)
8. `pcg-191-1@2026-01-01#2` (source=fusion, score=0.0156)
9. `pcg-619-14-c1@2026-01-01` (source=fusion, score=0.0154)
10. `pcg-627-1-c4@2026-01-01` (source=fusion, score=0.0154)

**Records gold** :

- `pcg-1222-74@2026-01-01` (article 1222-74) : 74 : Subventions Le compte 741 « Subventions d'exploitation » est crédité du montant des subventions d'exploitation acquises à l'entité par le débit du compte de tiers ou de trésorerie intéressé. Le compte 742 « Subventions d'équilibre » est crédité du montant des subventions d'équilibre acquises à …

**Tokens question absents du(des) record(s) gold** : `achet`, `aid`, `anne`, `argent`, `boit`, `ca`, `machin`, `region`, `tourn`, `vers`

**Tokens gold absents de la question** : `139`, `74`, `741`, `742`, `747`, `acquis`, `cred`, `credit`, `deb`, `entit`, `equilibr`, `exercic`, `exploit`, `inscrit`, `interess`, `invest`, `mont`, `quote-part`, `resultat`, `subvent`, `tier`, `tresorer`, `vir`

**Note d'analyse (choix des synonymes)** : cible du lot — « boîte » (`boit`) → « entité » (`entit`, token gold). La forme verbale « m'aider » (pas le nom « aide ») rend le synonyme « aides → subventions » inopérant ici ; seul l'apport du token `entit` (générique, faible pouvoir discriminant en bm25 — présent dans presque tous les records) est attendu, effet probablement faible.

---

## q060 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai reçu deux aides différentes de l'Etat cette année : l'une pour compenser une grosse perte exceptionnelle, l'autre pour soutenir mon activité courante habituelle. Est-ce que ça va dans la même case de mes comptes ?

**Citations gold** : ['pcg-1222-74']

**Top-10 obtenu (hybrid)** :

1. `pcg-513-5@2026-01-01` (source=fusion, score=0.0272)
2. `pcg-512-1-c4@2026-01-01` (source=fusion, score=0.0262)
3. `pcg-131-1-c1@2026-01-01` (source=fusion, score=0.025)
4. `pcg-na-318@2026-01-01` (source=fusion, score=0.0229)
5. `pcg-1214-44@2026-01-01` (source=fusion, score=0.0216)
6. `pcg-512-1-c6@2026-01-01` (source=fusion, score=0.0164)
7. `pcg-191-1-c24@2026-01-01` (source=fusion, score=0.0164)
8. `pcg-na-225@2026-01-01` (source=fusion, score=0.0161)
9. `pcg-324-1-c62@2026-01-01` (source=fusion, score=0.0159)
10. `pcg-121-2-c5@2026-01-01` (source=fusion, score=0.0159)

**Records gold** :

- `pcg-1222-74@2026-01-01` (article 1222-74) : 74 : Subventions Le compte 741 « Subventions d'exploitation » est crédité du montant des subventions d'exploitation acquises à l'entité par le débit du compte de tiers ou de trésorerie intéressé. Le compte 742 « Subventions d'équilibre » est crédité du montant des subventions d'équilibre acquises à …

**Tokens question absents du(des) record(s) gold** : `activ`, `aid`, `anne`, `autr`, `ca`, `cas`, `compens`, `cour`, `deux`, `different`, `est-ce`, `etat`, `exceptionnel`, `gross`, `habituel`, `mem`, `pert`, `recu`, `souten`, `va`

**Tokens gold absents de la question** : `139`, `74`, `741`, `742`, `747`, `acquis`, `cred`, `credit`, `deb`, `enregistr`, `entit`, `equilibr`, `exercic`, `exploit`, `inscrit`, `interess`, `invest`, `mont`, `quote-part`, `resultat`, `subvent`, `tier`, `tresorer`, `vir`

**Note d'analyse (choix des synonymes)** : cible du lot — « aides » (`aid`) → « subventions » (`subvent`, token gold), présent littéralement (« deux aides différentes de l'Etat »).

---

## q063 (vocabulaire_courant) — recall@10=0.000

**Question** : Je me suis trompé dans le calcul de la durée de vie d'une machine l'an dernier ; si je corrige maintenant, je dois recalculer tout depuis le début ou juste changer à partir de maintenant ?

**Citations gold** : ['pcg-122-4']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-318@2026-01-01` (source=fusion, score=0.0267)
2. `pcg-na-36@2026-01-01` (source=fusion, score=0.0265)
3. `pcg-121-2-c5@2026-01-01` (source=fusion, score=0.0243)
4. `pcg-na-15@2026-01-01` (source=fusion, score=0.024)
5. `pcg-na-18@2026-01-01` (source=fusion, score=0.024)
6. `pcg-324-1-c16@2026-01-01` (source=fusion, score=0.0225)
7. `pcg-na-303@2026-01-01` (source=fusion, score=0.0222)
8. `pcg-622-5-c4@2026-01-01` (source=fusion, score=0.0219)
9. `pcg-na-84@2026-01-01` (source=fusion, score=0.0164)
10. `pcg-na-27@2026-01-01` (source=fusion, score=0.0164)

**Records gold** :

- `pcg-122-4-c1@2026-01-01` (article 122-4) : IR 2 - Précisions sur les estimations comptables En raison des incertitudes inhérentes à la vie des affaires, de nombreux éléments des états financiers ne peuvent être évalués avec précision. L'entité doit alors recourir à des estimations comptables pour appliquer ses méthodes comptables. Ces estima…
- `pcg-122-4-c1@2026-01-01#2` (article 122-4) : IR 1 - Les provisions pour gros entretien ou grandes révisions Conformément à l'article 214-9 du plan comptable général, les dépenses d'entretien faisant l'objet de programmes pluriannuels de gros entretien ou grandes révisions qui ont pour seul objet de vérifier le bon état de fonctionnement des in…
- `pcg-122-4-c2@2026-01-01` (article 122-4) : IR 3 - Les provisions pour gros entretien ou grandes révisions En application des règles de droit commun et compte tenu de la pratique en vigueur dans les organismes de logement social, les conditions suivantes doivent être réunies pour justifier la comptabilisation des provisions pour gros entretie…
- `pcg-122-4-c3@2026-01-01` (article 122-4) : Exemples (cf Annexe 4) : - travaux présumés dans le champ de la provision pour gros entretien : o ravalement de façade, sans amélioration (nettoyage et peinture) ; o peinture et sols des parties communes (halls, cages d'escaliers, parkings…) ; - travaux présumés non provisionnables : o réfection iso…
- `pcg-122-4-c4@2026-01-01` (article 122-4) : IR 3 - Modalités de calcul A la date de clôture, la probabilité de sortie de ressources est directement liée à l'usage passé des éléments du patrimoine objet du programme pluriannuel d'entretien. En conséquence, un passif doit être constaté à hauteur de la quote-part des dépenses futures d'entretien…
- `pcg-122-4@2026-01-01` (article 122-4) : Définition des estimations comptables Les estimations comptables sont le résultat de l'exercice du jugement et de la mise en œuvre d'hypothèses dans l'application d'une méthode comptable.
- `pcg-122-4@2026-01-01#2` (article 122-4) : - Les provisions pour gros entretien ou grandes révisions Les entités comptabilisant des provisions pour gros entretien ou grandes révisions calculent la provision pour chaque immeuble objet de programmes pluriannuels de gros entretien ou grandes révisions.

**Tokens question absents du(des) record(s) gold** : `corrig`, `debut`, `depuis`, `just`, `machin`, `mainten`, `recalcul`, `tromp`

**Tokens gold absents de la question** : `000`, `1`, `10`, `1er`, `2`, `200`, `2003`, `2003-e`, `2005`, `214-9`, `3`, `4`, `4-2`, `5`, `500`, `7`, `8`, `9`, `actif`, `affair`, `afin`, `ailleur`, `ains`, `ajust`, `alor`, `amelior`, `amort`, `analys`, `anne`, `annex`, `appliqu`, `apport`, `approch`, `apres`, `arbre`, `art`, `articl`, `assimil`, `attest`, `au-dela` *(+319 tronqués)*

---

## q065 (vocabulaire_courant) — recall@10=0.000

**Question** : Le chantier va durer plus de deux ans et je ne suis pas sûr du résultat final : je peux quand même noter un bénéfice dans mes comptes chaque année ?

**Citations gold** : ['pcg-622-4']

**Top-10 obtenu (hybrid)** :

1. `pcg-131-3-c1@2026-01-01` (source=fusion, score=0.0311)
2. `pcg-na-318@2026-01-01` (source=fusion, score=0.0311)
3. `pcg-141-3-c1@2026-01-01` (source=fusion, score=0.0292)
4. `pcg-122-4-c4@2026-01-01` (source=fusion, score=0.0252)
5. `pcg-615-11-c1@2026-01-01` (source=fusion, score=0.0247)
6. `pcg-na-329@2026-01-01` (source=fusion, score=0.0201)
7. `pcg-324-1-c16@2026-01-01` (source=fusion, score=0.0164)
8. `pcg-na-39@2026-01-01` (source=fusion, score=0.0161)
9. `pcg-324-1-c18@2026-01-01` (source=fusion, score=0.0159)
10. `pcg-na-249@2026-01-01` (source=fusion, score=0.0159)

**Records gold** :

- `pcg-622-4-c1@2026-01-01` (article 622-4) : Résultat à terminaison non déterminable de façon fiable – Avis CNC n°99-10 du 23 septembre 1999 relatif aux contrats à long terme Lorsque la situation à terminaison la plus probable est une perte, la constatation d'une provision dépend de la capacité ou non à estimer cette dernière de façon raisonna…
- `pcg-622-4@2026-01-01` (article 622-4) : Si l'entité retient la méthode à l'avancement mais n'est pas en mesure d'estimer de façon fiable le résultat à terminaison, aucun profit n'est dégagé. A la date de clôture, le montant inscrit en chiffre d'affaires est limité à celui des charges ayant concouru à l'exécution du contrat.

**Tokens question absents du(des) record(s) gold** : `an`, `anne`, `benefic`, `chanti`, `chaqu`, `compt`, `deux`, `dur`, `final`, `mem`, `not`, `quand`, `va`

**Tokens gold absents de la question** : `1999`, `23`, `99-10`, `additionnel`, `affair`, `affirm`, `annex`, `aucun`, `avanc`, `avis`, `ayant`, `capac`, `celui`, `charg`, `chiffr`, `clotur`, `cnc`, `concouru`, `constat`, `contrat`, `correspond`, `dat`, `degag`, `depend`, `derni`, `determin`, `entit`, `entre`, `estim`, `eventuel`, `execu`, `existent`, `facon`, `faibl`, `fiabl`, `general`, `hypothes`, `incertitud`, `inscrit`, `lieu` *(+29 tronqués)*

---

## q068 (vocabulaire_courant) — recall@10=0.000

**Question** : Mes ingénieurs ont travaillé un an sur un nouveau modèle avant qu'on commence à le vendre : qu'est-ce qui rentre dans le coût de cette création quand je l'inscris dans mes comptes ?

**Citations gold** : ['pcg-213-27']

**Top-10 obtenu (hybrid)** :

1. `pcg-743-1@2026-01-01#2` (source=fusion, score=0.0278)
2. `pcg-121-2-c5@2026-01-01` (source=fusion, score=0.0272)
3. `pcg-212-4-c2@2026-01-01` (source=fusion, score=0.0248)
4. `pcg-131-3-c1@2026-01-01` (source=fusion, score=0.0241)
5. `pcg-na-236@2026-01-01` (source=fusion, score=0.0238)
6. `pcg-na-182@2026-01-01` (source=fusion, score=0.0237)
7. `pcg-na-318@2026-01-01` (source=fusion, score=0.0198)
8. `pcg-na-74@2026-01-01` (source=fusion, score=0.0187)
9. `pcg-324-1-c19@2026-01-01` (source=fusion, score=0.0164)
10. `pcg-836-1@2026-01-01#3` (source=fusion, score=0.0164)

**Records gold** :

- `pcg-213-27-c1@2026-01-01` (article 213-27) : Avis CNC 2004-15 du 23 juin 2004 relatif à la définition, la comptabilisation et l'évaluation des actifs - Coûts attribuables aux coûts de développement Ces coûts incluent, s'il y a lieu : - les coûts au titre des matériaux et services utilisés ou consommés pour générer l'immobilisation incorporelle…
- `pcg-213-27-c2@2026-01-01` (article 213-27) : - Coûts non attribuables aux coûts de développement Sont considérés comme tels : - les coûts de vente, coûts administratifs et autres frais généraux à moins que ces dépenses puissent être directement attribuées à la préparation de l'actif en vue de son utilisation ; - les inefficiences clairement id…
- `pcg-213-27@2026-01-01` (article 213-27) : Le coût d'une immobilisation incorporelle générée en interne, répondant aux conditions de comptabilisation prévues à l'article 212-3/1, comprend toutes les dépenses pouvant lui être directement attribuées et qui sont nécessaires à la création, la production et la préparation de l'actif afin qu'il so…

**Tokens question absents du(des) record(s) gold** : `an`, `commenc`, `compt`, `est-ce`, `ingenieur`, `inscris`, `model`, `nouveau`, `quand`, `rentr`, `travaill`, `vendr`

**Tokens gold absents de la question** : `1`, `2004`, `2004-15`, `212-3`, `23`, `acquis`, `acquisit`, `actif`, `activ`, `administr`, `afin`, `amort`, `anterieur`, `articl`, `atteign`, `attribu`, `autr`, `avis`, `brevet`, `charg`, `clair`, `cnc`, `comm`, `comprend`, `comptabilis`, `condit`, `consider`, `consomm`, `dat`, `definit`, `depens`, `depot`, `developp`, `direct`, `droit`, `encouru`, `engag`, `enregistr`, `evalu`, `fonction` *(+53 tronqués)*

---

## q070 (vocabulaire_courant) — recall@10=0.000

**Question** : J'avais mis de l'argent de côté au cas où le dollar baisserait, et finalement le risque a disparu : je fais quoi de cette réserve maintenant ?

**Citations gold** : ['pcg-420-6']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-263@2026-01-01` (source=fusion, score=0.0287)
2. `pcg-na-70@2026-01-01` (source=fusion, score=0.0284)
3. `pcg-214-25@2026-01-01` (source=fusion, score=0.0272)
4. `pcg-221-7-c2@2026-01-01` (source=fusion, score=0.0211)
5. `pcg-na-213@2026-01-01` (source=fusion, score=0.0205)
6. `pcg-na-162@2026-01-01` (source=fusion, score=0.0164)
7. `pcg-221-7-c1@2026-01-01` (source=fusion, score=0.0161)
8. `pcg-na-236@2026-01-01` (source=fusion, score=0.0161)
9. `pcg-221-7-c5@2026-01-01` (source=fusion, score=0.0159)
10. `pcg-213-9-c1@2026-01-01` (source=fusion, score=0.0159)

**Records gold** :

- `pcg-420-6@2026-01-01` (article 420-6) : Lorsque les circonstances suppriment en tout ou partie le risque de perte, les provisions sont ajustées en conséquence. Il en est ainsi dans les cas suivants : 1. Lorsque l'opération traitée en devises est assortie par l'entité d'une opération symétrique destinées à couvrir les conséquences de la fl…

**Tokens question absents du(des) record(s) gold** : `argent`, `avais`, `baiss`, `cot`, `disparu`, `dollar`, `fais`, `final`, `mainten`, `mis`, `quoi`, `reserv`

**Tokens gold absents de la question** : `1`, `2`, `3`, `628-6`, `628-7`, `ains`, `ajust`, `appel`, `articl`, `assort`, `aucun`, `celle-c`, `chang`, `circonst`, `comm`, `compris`, `comptabl`, `concour`, `concurrent`, `conform`, `consequent`, `consider`, `constat`, `constitu`, `couvert`, `couvertur`, `couvr`, `creanc`, `destin`, `det`, `determin`, `devis`, `doivent`, `don`, `dont`, `dotat`, `echeanc`, `element`, `entit`, `excedent` *(+36 tronqués)*

**Note d'analyse (choix des synonymes)** : « dollar » → « devise étrangère » a été envisagé, puis écarté : « dollar » n'est qu'UN exemple de devise parmi d'autres (yen, livre sterling…) — un mapping systématique d'une devise nommée vers une catégorie générique n'est pas une relation terme-à-terme stable (ce n'est pas une synonymie, mais une appartenance à une catégorie), et l'incohérence de ne couvrir qu'une devise sur toutes celles possibles est un signe de sur-ajustement. **Écarté par prudence.**

---

## q071 (vocabulaire_courant) — recall@10=0.000

**Question** : Quand le cours du dollar bouge et que ça change la valeur de ce qu'un client étranger me doit, c'est tout de suite dans mon résultat de l'année ou pas ?

**Citations gold** : ['pcg-420-5']

**Top-10 obtenu (hybrid)** :

1. `pcg-622-5-c4@2026-01-01` (source=fusion, score=0.0288)
2. `pcg-420-2@2026-01-01` (source=fusion, score=0.0287)
3. `pcg-na-236@2026-01-01` (source=fusion, score=0.0265)
4. `pcg-na-121@2026-01-01` (source=fusion, score=0.0164)
5. `pcg-836-1@2026-01-01#3` (source=fusion, score=0.0164)
6. `pcg-na-143@2026-01-01` (source=fusion, score=0.0161)
7. `pcg-625-9-c1@2026-01-01` (source=fusion, score=0.0161)
8. `pcg-420-4@2026-01-01` (source=fusion, score=0.0159)
9. `pcg-na-230@2026-01-01` (source=fusion, score=0.0156)
10. `pcg-na-146@2026-01-01` (source=fusion, score=0.0154)

**Records gold** :

- `pcg-420-5@2026-01-01` (article 420-5) : Les créances et les dettes en monnaies étrangères sont converties et comptabilisées en monnaie nationale sur la base du dernier cours du change. Lorsque l'application du taux de conversion à la date de clôture de l'exercice a pour effet de modifier les montants en monnaie nationale précédemment comp…

**Tokens question absents du(des) record(s) gold** : `anne`, `boug`, `ca`, `client`, `dollar`, `etrang`, `quand`, `resultat`, `suit`, `tout`, `valeur`

**Tokens gold absents de la question** : `420-6`, `actif`, `appliqu`, `articl`, `attent`, `bas`, `bilan`, `clotur`, `compt`, `comptabilis`, `concurrent`, `constitu`, `convers`, `convert`, `correspond`, `creanc`, `dat`, `derni`, `det`, `different`, `disposit`, `effet`, `entrainent`, `etranger`, `exercic`, `gain`, `inscrit`, `latent`, `lorsqu`, `modifi`, `monnai`, `mont`, `national`, `particuli`, `passif`, `pert`, `precedent`, `provis`, `regularis`, `reserv` *(+5 tronqués)*

---

## q074 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai acheté une machine facturée en dollars : je la mets dans mes comptes à quel taux de change ?

**Citations gold** : ['pcg-420-1']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-318@2026-01-01` (source=fusion, score=0.0315)
2. `pcg-na-131@2026-01-01` (source=fusion, score=0.0294)
3. `pcg-na-18@2026-01-01` (source=fusion, score=0.0293)
4. `pcg-1214-40@2026-01-01` (source=fusion, score=0.0276)
5. `pcg-1214-41@2026-01-01` (source=fusion, score=0.0265)
6. `pcg-420-5@2026-01-01` (source=fusion, score=0.0264)
7. `pcg-214-14-c1@2026-01-01` (source=fusion, score=0.0262)
8. `pcg-na-119@2026-01-01` (source=fusion, score=0.0261)
9. `pcg-na-146@2026-01-01` (source=fusion, score=0.026)
10. `pcg-na-143@2026-01-01` (source=fusion, score=0.0259)

**Records gold** :

- `pcg-420-1-c1@2026-01-01` (article 420-1) : IR3 - Modalités de mise en œuvre - Frais de couverture Question : En cas de couverture globale visant toutes les opérations réalisées par l'entreprise hors de la zone euro, les frais engagés pour mettre en place cette couverture globale doivent-ils être intégrés au coût d'acquisition de l'immobilisa…
- `pcg-420-1@2026-01-01` (article 420-1) : Le coût d'entrée des immobilisations incorporelles et corporelles et stocks exprimé en monnaie étrangère est converti en monnaie nationale au cours du jour de l'opération. En cas d'acquisition d'actif en monnaie étrangère, le taux de conversion utilisé est le taux de change à la date d'entrée. En ca…

**Tokens question absents du(des) record(s) gold** : `achet`, `compt`, `dollar`, `factur`, `machin`, `met`

**Tokens gold absents de la question** : `628-12`, `628-13`, `acquisit`, `actif`, `amort`, `articl`, `attribu`, `bien`, `calcul`, `cas`, `contrair`, `convers`, `convert`, `corporel`, `correspond`, `cour`, `cout`, `couvertur`, `dat`, `dedi`, `depreci`, `determin`, `direct`, `doivent-il`, `effet`, `egal`, `element`, `engag`, `entre`, `entrepris`, `envisage`, `etait`, `etranger`, `euro`, `exprim`, `frais`, `global`, `hor`, `immobilis`, `incorporel` *(+27 tronqués)*

---

## q079 (vocabulaire_courant) — recall@10=0.000

**Question** : Ma société en absorbe une autre, et la différence entre ce que je reçois et ce que valaient mes actions dans cette société était positive : comment j'appelle et j'enregistre ce gain ?

**Citations gold** : ['pcg-745-2']

**Top-10 obtenu (hybrid)** :

1. `pcg-213-3-c2@2026-01-01` (source=fusion, score=0.0307)
2. `pcg-141-2@2026-01-01` (source=fusion, score=0.0294)
3. `pcg-1211-10@2026-01-01` (source=fusion, score=0.0277)
4. `pcg-625-12-c1@2026-01-01` (source=fusion, score=0.0277)
5. `pcg-141-3-c2@2026-01-01` (source=fusion, score=0.022)
6. `pcg-324-1-c16@2026-01-01` (source=fusion, score=0.0215)
7. `pcg-na-127@2026-01-01` (source=fusion, score=0.0212)
8. `pcg-627-1-c4@2026-01-01` (source=fusion, score=0.0164)
9. `pcg-221-7-c5@2026-01-01` (source=fusion, score=0.0161)
10. `pcg-na-162@2026-01-01` (source=fusion, score=0.0161)

**Records gold** :

- `pcg-745-2@2026-01-01` (article 745-2) : Le boni représente l'écart positif entre l'actif net positif reçu par l'entité absorbante, après harmonisation des méthodes comptables telle que défini à l'article 744-3, à hauteur de sa participation détenue dans l'entité absorbée, et la valeur comptable de cette participation. Le boni est comptabi…

**Tokens question absents du(des) record(s) gold** : `action`, `appel`, `autr`, `different`, `enregistr`, `etait`, `gain`, `recois`, `societ`, `val`

**Tokens gold absents de la question** : `744-3`, `accumul`, `acquisit`, `actif`, `apres`, `articl`, `bon`, `capital`, `comptabilis`, `comptabl`, `defin`, `depuis`, `detenu`, `determin`, `distribu`, `ecart`, `entit`, `fiabl`, `financi`, `harmonis`, `hauteur`, `mani`, `method`, `mont`, `net`, `non`, `particip`, `peuvent`, `propr`, `quote-part`, `recu`, `represent`, `residuel`, `resultat`, `tel`, `valeur`

**Note d'analyse (choix des synonymes)** : « boni de fusion » est bien un terme métier précis, mais il n'existe pas de synonyme courant unique et stable pour lui (contrairement à « leasing » = « crédit-bail ») — la question décrit un scénario complet (différence positive entre prix payé et valeur reçue lors d'une absorption), pas un terme à rapprocher. **Écarté par prudence** (risque de mapper une clause entière plutôt qu'un terme).

---

## q080 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai payé plus cher les actions d'une boîte que ce qu'elle valait vraiment sur le papier, et je viens de l'avaler complètement : cet écart en trop, ça devient quoi dans mes comptes ?

**Citations gold** : ['pcg-745-3']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-236@2026-01-01` (source=fusion, score=0.0313)
2. `pcg-na-263@2026-01-01` (source=fusion, score=0.031)
3. `pcg-na-318@2026-01-01` (source=fusion, score=0.0231)
4. `pcg-1121-1@2026-01-01` (source=fusion, score=0.0224)
5. `pcg-na-141@2026-01-01` (source=fusion, score=0.0164)
6. `pcg-836-1@2026-01-01#3` (source=fusion, score=0.0164)
7. `pcg-838-1@2026-01-01` (source=fusion, score=0.0161)
8. `pcg-616-16-c3@2026-01-01` (source=fusion, score=0.0159)
9. `pcg-214-5@2026-01-01` (source=fusion, score=0.0159)
10. `pcg-625-12@2026-01-01` (source=fusion, score=0.0156)

**Records gold** :

- `pcg-745-3-c1@2026-01-01` (article 745-3) : IR 3 : Ajustements de prix positifs ou négatifs Les ajustements de prix correspondent à des compléments ou des diminutions de prix : - de la participation détenue antérieurement à la fusion, - résultant de l'application de clauses de garantie ou de révisions de prix, - versés ou perçus postérieureme…
- `pcg-745-3-c2@2026-01-01` (article 745-3) : IR 3 : Annulation des actions propres reçues par voie de fusion En cas d'absorption d'une mère par sa fille, la fusion a pour effet de transférer à l'entité absorbante (la fille) ses propres titres qu'elle doit annuler par capitaux propres. L'écart résultant de cette annulation ne peut être assimilé…
- `pcg-745-3-c3@2026-01-01` (article 745-3) : IR 3 : Valeur nette comptable des titres La valeur comptable retenue pour les titres dans le calcul du mali est la valeur comptable défini à l'article 745-3 et s'entend de la valeur comptable nette.
- `pcg-745-3@2026-01-01` (article 745-3) : Le mali de fusion représente l'écart négatif entre l'actif net, positif ou négatif, reçu par l'entité absorbante, après harmonisation des méthodes comptables telle que défini à l'article 744-3, à hauteur de sa participation dans l'entité absorbée et la valeur comptable de cette participation. Le cas…

**Tokens question absents du(des) record(s) gold** : `aval`, `boit`, `ca`, `cher`, `complet`, `compt`, `devient`, `pai`, `papi`, `quoi`, `trop`, `val`, `vien`, `vrai`

**Tokens gold absents de la question** : `3`, `7`, `744-3`, `745-3`, `942-27`, `absorb`, `absorpt`, `actif`, `ajust`, `aline`, `annul`, `anterieur`, `appliqu`, `apres`, `articl`, `assimil`, `aucun`, `calcul`, `capital`, `car`, `cas`, `cel`, `claus`, `compl`, `comptabilis`, `comptabl`, `constat`, `convient`, `correspondent`, `corrig`, `defin`, `derni`, `detenu`, `diminu`, `disposit`, `eche`, `effet`, `entend`, `entit`, `entre` *(+39 tronqués)*

**Note d'analyse (choix des synonymes)** : cible du lot — « boîte » (`boit`) → « entité » (`entit`), même remarque qu'en q059 sur l'apport probablement faible du token générique `entit` en bm25 ; le vrai fossé (« mali de fusion ») reste un échec restant (même raisonnement que q079).

---

## q085 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai un bâtiment que mes propres équipes ont commencé à construire mais qui n'est pas fini à la clôture, et un autre tout juste terminé cette année : je les traite différemment dans mes comptes ?

**Citations gold** : ['pcg-1212-23', 'pcg-1222-72']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-318@2026-01-01` (source=fusion, score=0.0273)
2. `pcg-512-4@2026-01-01` (source=fusion, score=0.0247)
3. `pcg-121-2-c7@2026-01-01` (source=fusion, score=0.0232)
4. `pcg-na-229@2026-01-01` (source=fusion, score=0.0164)
5. `pcg-616-15@2026-01-01#2` (source=fusion, score=0.0164)
6. `pcg-na-219@2026-01-01` (source=fusion, score=0.0161)
7. `pcg-121-2-c5@2026-01-01` (source=fusion, score=0.0161)
8. `pcg-na-217@2026-01-01` (source=fusion, score=0.0159)
9. `pcg-na-245@2026-01-01` (source=fusion, score=0.0159)
10. `pcg-na-220@2026-01-01` (source=fusion, score=0.0156)

**Records gold** :

- `pcg-1212-23@2026-01-01` (article 1212-23) : 23 : Immobilisations en cours, avances et acomptes Le compte 23 « Immobilisations en cours, avances et acomptes » a pour objet de faire apparaître la valeur des immobilisations non terminées à la fin de chaque exercice. Du point de vue de leur origine, les immobilisations inscrites aux comptes 231 e…
- `pcg-1222-72@2026-01-01` (article 1222-72) : 72 : Production immobilisée Le compte 72 « Production immobilisée » enregistre le coût des travaux faits par l'entité pour elle-même. Il est crédité soit par le débit du compte 23 « Immobilisations en cours, avances et acomptes » du coût de production des immobilisations créées par les moyens propre…

**Tokens question absents du(des) record(s) gold** : `anne`, `autr`, `bat`, `clotur`, `commenc`, `construir`, `different`, `equip`, `just`, `tout`, `trait`

**Tokens gold absents de la question** : `20`, `21`, `23`, `231`, `232`, `237`, `238`, `4091`, `72`, `acompt`, `acquisit`, `apparaitr`, `appropri`, `avanc`, `aver`, `cel`, `celui`, `chaqu`, `ci-dessus`, `command`, `comptabilis`, `concern`, `confi`, `corporel`, `cour`, `cout`, `cre`, `cred`, `credit`, `deb`, `debit`, `deux`, `direct`, `dur`, `effectu`, `elle-mem`, `enregistr`, `entit`, `exercic`, `facult` *(+41 tronqués)*

**Note d'analyse (choix des synonymes)** : comme q054, deux des concepts en jeu (immobilisation en cours / production immobilisée) exigent de distinguer « en construction » de « juste terminée » — un savoir comptable, pas un rapprochement lexical. Classé « échec restant ».

---

## q086 (vocabulaire_courant) — recall@10=0.000

**Question** : J'ai reçu deux aides différentes cette année : une pour m'aider à payer mes charges courantes, et une autre pour financer l'achat d'une machine neuve. Est-ce qu'elles se comptabilisent au même endroit dans mes comptes ?

**Citations gold** : ['pcg-1222-74', 'pcg-312-1']

**Top-10 obtenu (hybrid)** :

1. `pcg-131-1-c1@2026-01-01` (source=fusion, score=0.0273)
2. `pcg-na-318@2026-01-01` (source=fusion, score=0.027)
3. `pcg-1221-61@2026-01-01` (source=fusion, score=0.0269)
4. `pcg-621-4@2026-01-01` (source=fusion, score=0.0222)
5. `pcg-615-19-c2@2026-01-01` (source=fusion, score=0.0164)
6. `pcg-324-1-c15@2026-01-01` (source=fusion, score=0.0161)
7. `pcg-121-2-c5@2026-01-01` (source=fusion, score=0.0161)
8. `pcg-512-1-c6@2026-01-01` (source=fusion, score=0.0159)
9. `pcg-213-26@2026-01-01` (source=fusion, score=0.0159)
10. `pcg-213-19@2026-01-01` (source=fusion, score=0.0156)

**Records gold** :

- `pcg-1222-74@2026-01-01` (article 1222-74) : 74 : Subventions Le compte 741 « Subventions d'exploitation » est crédité du montant des subventions d'exploitation acquises à l'entité par le débit du compte de tiers ou de trésorerie intéressé. Le compte 742 « Subventions d'équilibre » est crédité du montant des subventions d'équilibre acquises à …
- `pcg-312-1@2026-01-01` (article 312-1) : Le montant des subventions d'investissement, lorsqu'il est inscrit dans les capitaux propres, est repris au compte de résultat selon les modalités qui suivent : 1. La reprise de la subvention d'investissement qui finance une immobilisation amortissable s'effectue sur la même durée et au même rythme …

**Tokens question absents du(des) record(s) gold** : `achat`, `aid`, `autr`, `charg`, `comptabilisent`, `cour`, `deux`, `different`, `endroit`, `est-ce`, `machin`, `neuv`, `pai`, `recu`

**Tokens gold absents de la question** : `1`, `139`, `2`, `74`, `741`, `742`, `747`, `acquis`, `activ`, `admis`, `allou`, `amort`, `amortiss`, `autor`, `ayant`, `capital`, `chaqu`, `circonst`, `claus`, `condit`, `contrat`, `cre`, `cred`, `credit`, `deb`, `defaut`, `demand`, `derog`, `determin`, `dixiem`, `dur`, `effectu`, `egal`, `engag`, `enregistr`, `entit`, `equilibr`, `etal`, `exempl`, `exercic` *(+37 tronqués)*

**Note d'analyse (choix des synonymes)** : cible du lot — « aides » (`aid`) → « subventions » (`subvent`, token gold), présent littéralement (« deux aides différentes cette année »).

---

## q089 (vocabulaire_courant) — recall@10=0.000

**Question** : Quand la différence entre ce que j'ai payé pour racheter une boîte et sa valeur comptable réelle est collée à un bien précis que j'ai identifié dedans, sur combien de temps j'étale cette différence ?

**Citations gold** : ['pcg-745-7']

**Top-10 obtenu (hybrid)** :

1. `pcg-na-236@2026-01-01` (source=fusion, score=0.0294)
2. `pcg-213-9-c4@2026-01-01` (source=fusion, score=0.0289)
3. `pcg-na-39@2026-01-01` (source=fusion, score=0.0273)
4. `pcg-743-1@2026-01-01` (source=fusion, score=0.027)
5. `pcg-122-4-c1@2026-01-01` (source=fusion, score=0.0263)
6. `pcg-na-18@2026-01-01` (source=fusion, score=0.0263)
7. `pcg-na-72@2026-01-01` (source=fusion, score=0.0259)
8. `pcg-744-1@2026-01-01` (source=fusion, score=0.0257)
9. `pcg-744-1-c2@2026-01-01` (source=fusion, score=0.0255)
10. `pcg-214-18-c1@2026-01-01` (source=fusion, score=0.0251)

**Records gold** :

- `pcg-745-7-c1@2026-01-01` (article 745-7) : IR 3 : Amortissement du mali technique Le mali technique suit les règles d'amortissement de l'actif sous-jacent sur lequel porte la plus-value. Ainsi, la quote-part de mali affectée à un terrain ou à des titres n'est pas amorti mais fait l'objet, le cas échéant, d'une dépréciation conformément à l'a…
- `pcg-745-7@2026-01-01` (article 745-7) : Le mali technique est amorti ou rapporté au résultat selon les mêmes règles et dans les mêmes conditions que les actifs sous-jacents auquel il est affecté.

**Tokens question absents du(des) record(s) gold** : `bien`, `boit`, `coll`, `combien`, `comptabl`, `dedan`, `different`, `entre`, `etal`, `identifi`, `pai`, `prec`, `quand`, `rachet`, `reel`, `temp`, `valeur`

**Tokens gold absents de la question** : `3`, `745-8`, `actif`, `affect`, `ains`, `amort`, `amortiss`, `articl`, `auquel`, `auxquel`, `brevet`, `cas`, `comm`, `commercial`, `commercialis`, `condit`, `conform`, `depreci`, `dur`, `eche`, `fait`, `fond`, `ir`, `lequel`, `mais`, `mal`, `mem`, `non`, `objet`, `outillag`, `part`, `plus-valu`, `port`, `pourr`, `present`, `quote-part`, `rapport`, `regl`, `residuel`, `resultat`, `resultat` *(+7 tronqués)*

**Note d'analyse (choix des synonymes)** : cible du lot — « boîte » (`boit`) → « entité » (`entit`), même remarque qu'en q059/q080.

---
