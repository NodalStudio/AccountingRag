import sqlite3
from accounting_rag.db import write_db
from accounting_rag.model import Renvoi
from conftest import _rec


def test_roundtrip(tmp_path):
    dest = tmp_path / "corpus.db"
    rec = _rec(
        "pcg-212-5@2026-01-01",
        "212-5",
        texte="Le titulaire d'un contrat de crédit-bail…",
        renvois=[Renvoi("legi-L313-7-comofi", "externe_legal")],
    )
    write_db([rec], dest)
    con = sqlite3.connect(dest)
    assert con.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1
    assert (
        con.execute("SELECT famille FROM renvois").fetchone()[0] == "externe_legal"
    )
    hits = con.execute(
        "SELECT rowid FROM records_fts WHERE records_fts MATCH 'crédit'"
    ).fetchall()
    assert len(hits) == 1
