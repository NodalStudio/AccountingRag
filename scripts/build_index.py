"""CLI : ajoute l'index de recherche (chunks + FTS normalisé + vecteurs) à data/corpus.db."""
from pathlib import Path
from accounting_rag.index import build_index

stats = build_index(Path("data/corpus.db"))
print(stats)
