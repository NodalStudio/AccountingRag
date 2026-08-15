# Validation par échantillonnage — jalon 1

**Méthode** : 15 pages vérifiées par 5 sous-agents indépendants (lecture seule) — 12 pages aléatoires (seed 42 : 45, 52, 109, 124, 134, 162, 248, 270, 301, 452, 578, 624) + 3 pages forcées dans les zones sensibles (243 jetons/forme longue, 570 logement social, 640 annexes sectorielles). Chaque page : texte brut PyMuPDF comparé aux records de `corpus.db` (couverture, attribution d'articles, séparation réglementaire/commentaire, fusion inter-pages).

## Verdicts

| Page | Verdict | Détail |
|---|---|---|
| 45, 134, 162, 248, 452, 578 | ✅ OK | Couverture complète, types corrects |
| 243 (jetons), 570 (logement social), 640 (coopératives) | ✅ OK | Zones sensibles validées : articles 619-x présents, suffixes #n et chemins d'annexes corrects |
| 52, 109, 270*, 301* | ✅ OK après revérification | Écarts signalés = faux positifs : les titres de section vivent dans le champ `chemin` (vérifié : « Sous-section 4 – Coût d'entrée des stocks » et « Chapitre II – Passifs » bien présents dans les chemins), et les numéros d'en-têtes d'articles dans le champ `article` |
| 124 | ⚠️ Manque réel | Note de bas de page substantielle non capturée (strate ≤9,0 → BRUIT) |
| 270 | ⚠️ Manque réel | Encadré citant le Code de commerce (Art. L. 225-177 s.) non capturé — voir limitation ci-dessous |
| 624 | ⚠️ Manque réel | Cellules du tableau « modèle de bilan » (exploitations agricoles) largement absentes |

## Limitation connue documentée (ruling 24)

Le Recueil contient une **troisième strate de contenu en taille 9,0** — extraits de textes de niveau supérieur (Code de commerce, lois) cités pour référence, notes de bas de page, cellules de tableaux de modèles — qui partage sa taille avec les en-têtes/pieds de page et est actuellement classée BRUIT : **2 037 lignes substantielles sur 120 pages** non capturées en v1.

Décision : différé au backlog (v1.1 / phase 2) car (a) les extraits de lois citées arriveront plus proprement via l'ingestion LEGI prévue en phase 2 (source XML native, versionnée), (b) les tableaux du plan de comptes ont leur ingestion structurée dédiée prévue par la spec, (c) la capture des notes de bas de page nécessite une désambiguïsation positionnelle (y en page) qui mérite son propre cycle TDD.

## Bilan

Le corpus v1 restitue fidèlement les **strates réglementaire (10,0) et commentaires ANC (9,5)** — le cœur normatif — sur 15/15 pages échantillonnées, zones sensibles comprises. Les manques identifiés sont circonscrits à la strate 9,0 (contenu de référence), mesurés, et documentés.
