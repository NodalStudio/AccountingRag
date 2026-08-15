import sqlite3
from pathlib import Path
from .model import Record

_SCHEMA = """
CREATE TABLE records(
  id TEXT PRIMARY KEY, article TEXT, chemin TEXT, texte TEXT,
  type TEXT NOT NULL, nature TEXT NOT NULL, opposable INTEGER NOT NULL,
  valide_du TEXT, valide_au TEXT, source_citation TEXT,
  page_debut INTEGER, page_fin INTEGER
);
CREATE TABLE renvois(
  source_id TEXT NOT NULL REFERENCES records(id),
  cible TEXT NOT NULL, famille TEXT NOT NULL
);
CREATE INDEX idx_renvois_source ON renvois(source_id);
CREATE VIRTUAL TABLE records_fts USING fts5(
  texte, chemin, content='records', content_rowid='rowid'
);
"""


def write_db(records: list[Record], dest: Path) -> None:
    """Écrit les Records dans une base SQLite avec tables records, renvois, et FTS5."""
    dest.unlink(missing_ok=True)
    con = sqlite3.connect(dest)
    con.executescript(_SCHEMA)
    for r in records:
        con.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r.id,
                r.article,
                r.chemin,
                r.texte,
                r.type,
                r.nature,
                int(r.opposable),
                r.valide_du,
                r.valide_au,
                r.source_citation,
                r.page_debut,
                r.page_fin,
            ),
        )
        rowid = con.execute("SELECT rowid FROM records WHERE id=?", (r.id,)).fetchone()[
            0
        ]
        con.execute(
            "INSERT INTO records_fts(rowid, texte, chemin) VALUES (?,?,?)",
            (rowid, r.texte, r.chemin),
        )
        con.executemany(
            "INSERT INTO renvois VALUES (?,?,?)",
            [(r.id, rv.cible, rv.famille) for rv in r.renvois],
        )
    con.commit()
    con.close()
