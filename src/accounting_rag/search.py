"""Retrieval hybride : routeur de références -> BM25 normalisé + dense -> RRF -> renvois."""
import re
import sqlite3
import struct
from pathlib import Path
from .normalize import normalize

_REF_QUERY = re.compile(r"\bart(?:icle)?s?\.?\s*(\d{2,4}-\d+(?:-\d+)*)", re.I)
# Routage des références lettrées (L./R./D., code de commerce) différé à l'ingestion LEGI — aucun article lettré dans le corpus PCG actuel.
_RRF_K = 60
_MODES = {"bm25", "dense", "hybrid", "hybrid+graph"}


class Searcher:
    def __init__(self, db_path: Path, embedder=None,
                 poids_chemin: float = 1.0, boost_commentaire: float = 1.0):
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"corpus introuvable : {db_path} — lancez scripts/download_data.py, "
                "scripts/build_corpus.py puis scripts/build_index.py"
            )
        import sqlite_vec
        self.con = sqlite3.connect(db_path)
        self.con.enable_load_extension(True)
        sqlite_vec.load(self.con)
        self.con.enable_load_extension(False)
        self._embedder = embedder
        # Valeurs neutres par défaut (1.0, 1.0) : comportement jalon 2 inchangé.
        # float() valide l'entrée avant de la lier en paramètre SQL (Ruling J25-2).
        self.poids_chemin = float(poids_chemin)
        self.boost_commentaire = float(boost_commentaire)

    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _record(self, record_id: str, score: float, source: str) -> dict:
        row = self.con.execute(
            "SELECT article, chemin, texte FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        return {"record_id": record_id, "article": row[0], "chemin": row[1],
                "texte": row[2], "score": round(score, 4), "source": source}

    def _route(self, query: str) -> list[dict]:
        nums = _REF_QUERY.findall(query)
        out = []
        for num in nums:
            # type='reglementaire' : le routeur renvoie le texte canonique de l'article, pas les commentaires ANC.
            for (rid,) in self.con.execute(
                "SELECT id FROM records WHERE article = ? AND type = 'reglementaire'", (num,)
            ).fetchall():
                out.append(self._record(rid, 100.0, "route"))
        return out

    def _bm25(self, query: str, limit: int = 50) -> dict[str, float]:
        toks = normalize(query).split()
        if not toks:
            return {}
        match = " OR ".join(f'"{t}"' for t in toks)
        # Poids par colonne liés en paramètres SQL (texte_norm, chemin_norm dans cet ordre) :
        # bm25() aux accepte des paramètres liés sur cette version de SQLite (vérifié empiriquement,
        # cf. Ruling J25-2) — pas de repli par interpolation nécessaire ici.
        rows = self.con.execute(
            "SELECT c.record_id, bm25(chunks_norm, ?, ?) AS b FROM chunks_norm "
            "JOIN chunks c ON c.rowid = chunks_norm.rowid "
            "WHERE chunks_norm MATCH ? ORDER BY b LIMIT ?",
            (1.0, self.poids_chemin, match, limit),
        ).fetchall()
        if not rows:
            return {}
        record_ids = {rid for rid, _ in rows}
        placeholders = ",".join("?" * len(record_ids))
        types = dict(self.con.execute(
            f"SELECT id, type FROM records WHERE id IN ({placeholders})",
            tuple(record_ids),
        ).fetchall())
        scores: dict[str, float] = {}
        for rid, b in rows:
            s = -b  # bm25() de sqlite : plus petit = meilleur
            if types.get(rid) != "reglementaire":
                s *= self.boost_commentaire
            scores[rid] = max(scores.get(rid, -1e9), s)
        return scores

    def _dense(self, query: str, limit: int = 50) -> dict[str, float]:
        vec = self.embedder.encode_query(query)
        rows = self.con.execute(
            "SELECT c.record_id, v.distance FROM chunks_vec v "
            "JOIN chunks c ON c.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND v.k = ?",
            (struct.pack(f"{len(vec)}f", *vec), limit),
        ).fetchall()
        scores: dict[str, float] = {}
        for rid, dist in rows:
            s = -dist
            scores[rid] = max(scores.get(rid, -1e9), s)
        return scores

    @staticmethod
    def _rrf(rankings: list[dict[str, float]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for scores in rankings:
            ordered = sorted(scores, key=scores.get, reverse=True)
            for rank, rid in enumerate(ordered):
                fused[rid] = fused.get(rid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        return fused

    def _expand_graph(self, results: list[dict], k: int) -> list[dict]:
        seen = {r["record_id"] for r in results}
        extra: list[dict] = []
        for r in results[:5]:
            for (cible,) in self.con.execute(
                "SELECT cible FROM renvois WHERE source_id = ? AND famille = 'interne'",
                (r["record_id"],),
            ).fetchall():
                for (rid,) in self.con.execute(
                    "SELECT id FROM records WHERE article = ? AND type='reglementaire'",
                    (cible.removeprefix("pcg-"),),
                ).fetchall():
                    if rid not in seen:
                        seen.add(rid)
                        extra.append(self._record(rid, r["score"] * 0.5, "graph"))
        merged = results + extra
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:k]

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[dict]:
        if mode not in _MODES:
            raise ValueError(f"mode inconnu : {mode}")
        routed = self._route(query)
        routed_ids = {r["record_id"] for r in routed}
        if mode == "bm25":
            scores = self._bm25(query)
        elif mode == "dense":
            scores = self._dense(query)
        else:
            scores = self._rrf([self._bm25(query), self._dense(query)])
        ranked = sorted(scores, key=scores.get, reverse=True)
        source = "fusion" if mode.startswith("hybrid") else mode
        n_restants = max(k - len(routed), 0)
        results = [
            self._record(rid, scores[rid], source)
            for rid in ranked
            if rid not in routed_ids
        ][:n_restants]
        out = routed + results
        if mode == "hybrid+graph":
            out = self._expand_graph(out, k)
        return out[:k]
