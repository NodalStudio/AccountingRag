# Licence du contenu

Ce dépôt distribue du code (voir `LICENSE`, MIT) et un jeu de données extrait de sources publiques. Ces deux éléments ont des licences distinctes.

## Contenu extrait du Recueil des normes comptables françaises (ANC)

Le contenu structuré dans `data/corpus.db` (une fois le pipeline exécuté), ainsi que tout extrait de texte du Recueil des normes comptables françaises reproduit dans ce dépôt (rapports, exemples, documentation), est dérivé du **Recueil des normes comptables françaises**, publié par l'**Autorité des normes comptables (ANC)**.

Ce contenu est mis à disposition sous licence **[Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)** (Etalab), qui autorise la réutilisation, la redistribution et la modification, y compris à des fins commerciales, sous réserve de la mention de la paternité :

> Source : Autorité des normes comptables (ANC), Recueil des normes comptables françaises.

## Avertissement sur la fidélité du texte

L'extraction opérée par le parseur de ce dépôt (voir `docs/rapport-build.md`) est déterministe mais reste une transformation automatisée d'un document PDF : elle peut comporter des erreurs de segmentation, d'attribution d'article ou des omissions (voir la section « Limitations connues » du `README.md`). **Le PDF source publié par l'ANC (anc.gouv.fr) et les publications au Journal officiel / Légifrance restent la seule référence faisant foi.** En cas de doute ou d'usage professionnel, se reporter systématiquement au texte original.

## Sources futures (hors jalon 1)

Les phases ultérieures du projet prévoient l'intégration de BOFiP et LEGI (Licence Ouverte 2.0 également) et des IFRS adoptées par l'Union européenne (EUR-Lex, clause de reproduction limitée à l'EEE — ces textes ne seront jamais redistribués dans ce dépôt, uniquement récupérés au moment du build). Voir `docs/superpowers/specs/2026-08-14-accountingrag-design.md`, section 2.
