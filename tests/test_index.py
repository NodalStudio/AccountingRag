import sqlite3
import sqlite_vec
from pathlib import Path
from accounting_rag.db import write_db
from accounting_rag.index import build_index
from conftest import _rec


def _connect_avec_vec(db_path):
    """Connexion d'inspection : sqlite-vec doit être chargé sur CHAQUE connexion qui
    touche une table virtuelle vec0, pas seulement celle utilisée pour l'écrire."""
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


class FakeEmbedder:
    dim = 4

    def encode_passages(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


def test_build_index_cree_les_tables(tmp_path):
    db = tmp_path / "c.db"
    write_db([
        _rec("pcg-1-1@e", "1-1", texte="Un article sur l'amortissement dérogatoire.\n" + "mot " * 500),
        _rec("pcg-1-2@e", "1-2", texte="Le fonds commercial est amorti."),
    ], db)
    stats = build_index(db, embedder=FakeEmbedder())
    con = _connect_avec_vec(db)
    n_chunks = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert stats["chunks"] == n_chunks >= 3  # le record long produit plusieurs chunks
    assert con.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == n_chunks
    # le FTS normalisé matche une flexion différente
    hits = con.execute(
        "SELECT rowid FROM chunks_norm WHERE chunks_norm MATCH 'derogatoir*'"
    ).fetchall()
    assert hits
    # idempotence
    stats2 = build_index(db, embedder=FakeEmbedder())
    assert stats2 == stats
