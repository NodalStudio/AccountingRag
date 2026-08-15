#!/usr/bin/env python
"""Pipeline complet : PDF -> corpus.db + rapport d'anomalies."""
from pathlib import Path
from accounting_rag.parse import parse
from accounting_rag.integrity import check, report
from accounting_rag.db import write_db

PDF = Path("data/raw/recueil-pcg-2026.pdf")
DB = Path("data/corpus.db")
RAPPORT = Path("docs/rapport-build.md")

records, anomalies = parse(PDF)
anomalies += check(records)
write_db(records, DB)
RAPPORT.write_text(report(records, anomalies), encoding="utf-8")
print(f"{len(records)} enregistrements -> {DB}")
print(f"{len(anomalies)} anomalies -> {RAPPORT}")
