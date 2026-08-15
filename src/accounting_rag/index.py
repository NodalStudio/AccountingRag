"""Construit chunks + FTS normalisé + vecteurs au-dessus du corpus.db (idempotent)."""
import sqlite3
import struct
from pathlib import Path
import sqlite_vec
from .chunks import make_chunks
from .normalize import normalize


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con


def build_index(db_path: Path, embedder=None) -> dict:
    if embedder is None:
        from .embed import Embedder
        embedder = Embedder()
    con = _connect(db_path)
    con.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_norm;
        DROP TABLE IF EXISTS chunks_vec;
        CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, record_id TEXT NOT NULL, seq INT, texte TEXT);
        CREATE VIRTUAL TABLE chunks_norm USING fts5(texte_norm, chemin_norm);
    """)
    con.execute(f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{embedder.dim}])")
    rows = con.execute("SELECT id, texte, chemin FROM records").fetchall()
    all_chunks: list[tuple[str, str, int, str, str]] = []
    for rid, texte, chemin in rows:
        for chunk_id, seq, piece in make_chunks(rid, texte or ""):
            all_chunks.append((chunk_id, rid, seq, piece, chemin or ""))
    for i, (chunk_id, rid, seq, piece, chemin) in enumerate(all_chunks, start=1):
        con.execute("INSERT INTO chunks(rowid, chunk_id, record_id, seq, texte) VALUES (?,?,?,?,?)",
                    (i, chunk_id, rid, seq, piece))
        con.execute("INSERT INTO chunks_norm(rowid, texte_norm, chemin_norm) VALUES (?,?,?)",
                    (i, normalize(piece), normalize(chemin)))
    BATCH = 64
    for start in range(0, len(all_chunks), BATCH):
        batch = all_chunks[start:start + BATCH]
        vecs = embedder.encode_passages([c[3] for c in batch])
        for offset, vec in enumerate(vecs):
            rowid = start + offset + 1
            con.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
                        (rowid, struct.pack(f"{len(vec)}f", *vec)))
    con.commit()
    stats = {"chunks": len(all_chunks), "vecteurs": len(all_chunks),
             "records_norm": len(rows)}
    con.close()
    return stats
