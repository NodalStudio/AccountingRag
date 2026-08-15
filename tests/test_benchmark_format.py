import json
import sqlite3
from pathlib import Path
import pytest

FILES = [Path("benchmark/dev.jsonl"), Path("benchmark/test.jsonl")]
DB = Path("data/corpus.db")


@pytest.mark.skipif(not DB.exists(), reason="corpus.db absent")
def test_format_et_citations_existantes():
    con = sqlite3.connect(DB)
    total = 0
    ids = set()
    for f in FILES:
        assert f.exists(), f"{f} manquant"
        for line in f.read_text(encoding="utf-8").splitlines():
            q = json.loads(line)
            assert set(q) >= {"id", "question", "categorie", "citations"}
            assert q["categorie"] in {"reference_directe", "regle", "vocabulaire_courant"}
            assert q["id"] not in ids
            ids.add(q["id"])
            assert len(q["question"]) > 15
            assert q["citations"]
            for c in q["citations"]:
                n = con.execute(
                    "SELECT COUNT(*) FROM records WHERE id = ? OR id LIKE ? OR id LIKE ? OR id LIKE ?",
                    (c, c + "@%", c + "-c%", c + "#%"),
                ).fetchone()[0]
                assert n > 0, f"{q['id']}: citation {c} sans record"
            total += 1
    assert total == 30
