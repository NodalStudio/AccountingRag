"""Retrieval hybride : routeur de références -> BM25 normalisé + dense -> RRF -> renvois."""
import re
import sqlite3
import struct
from pathlib import Path
from .normalize import normalize

_REF_QUERY = re.compile(r"\bart(?:icle)?s?\.?\s*(\d{2,4}-\d+(?:-\d+)*)", re.I)
# Routage des références lettrées (L./R./D., code de commerce) différé à l'ingestion LEGI — aucun article lettré dans le corpus PCG actuel.
_RRF_K = 60
_MODES = {"bm25", "dense", "hybrid", "hybrid+graph", "hybrid+rerank"}


class Searcher:
    def __init__(self, db_path: Path, embedder=None,
                 poids_chemin: float = 1.0, boost_commentaire: float = 1.0,
                 reranker=None, df_max: float | None = None,
                 pool: int = 50, dedup_termes: bool = False, n_rerank: int = 25):
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
        self._reranker = reranker
        # Valeurs neutres par défaut (1.0, 1.0) : comportement jalon 2 inchangé.
        # float() valide l'entrée avant de la lier en paramètre SQL (Ruling J25-2).
        self.poids_chemin = float(poids_chemin)
        self.boost_commentaire = float(boost_commentaire)
        # df_max : fraction de n_chunks au-delà de laquelle un token de requête est
        # écarté du MATCH lexical (T1, jalon 3). None = neutre, comportement jalon 2.5
        # inchangé (aucun filtrage). Mesuré par bootstrap apparié, cf. docs/eval-jalon3.md.
        self.df_max = df_max
        self._df_cache: dict[str, int] = {}
        self._n_chunks: int | None = None
        # pool : nombre de lignes récupérées par canal (bm25 et dense) AVANT fusion RRF
        # (T2, jalon 3). 50 = comportement jalon 2.5. `search()` transmet toujours
        # explicitement `self.pool` aux deux canaux (défaut du brief T2 constaté
        # incohérent avec son propre test à l'exécution : un appel sans argument
        # explicite est invérifiable par espionnage d'attribut d'instance, la fonction
        # de substitution n'étant jamais liée comme méthode — cf. docs/eval-jalon3.md).
        # `_bm25`/`_dense` gardent en outre leur propre défaut `limit=None -> self.pool`,
        # pour les appelants directs qui n'en passent pas (ex. scripts/diagnostic_rangs.py).
        self.pool = pool
        # dedup_termes : déduplique les tokens de `_termes_match` avant construction du
        # MATCH lexical (T2, jalon 3). False = comportement actuel (T1) : les tokens
        # répétés dans la question restent répétés dans le MATCH, car bm25() de SQLite
        # FTS5 pondère leur MULTIPLICITÉ, pas seulement l'ensemble des lignes
        # sélectionnées (cf. _termes_match). N'a d'effet que sur le chemin neutre
        # (df_max=None) : le chemin de filtrage actif déduplique déjà, pour raisonner
        # sur des tokens uniques face au seuil — dedup_termes n'y change donc rien
        # (ce cas combiné n'est pas mesuré, df_max ayant été rejeté en T1, mais reste
        # ainsi défini sans incohérence : jamais moins déduplique que le filtrage lui-même).
        self.dedup_termes = dedup_termes
        # n_rerank : nombre de candidats (issus de la fusion, hors résultats routés)
        # soumis au reranker en mode `hybrid+rerank` (T3, jalon 3). 25 = comportement
        # jalon 2.5 inchangé (valeur codée en dur jusqu'ici dans `search()`). Un
        # `n_rerank` supérieur au nombre de candidats disponibles après fusion ne casse
        # rien : la troncature par slice (`[:self.n_rerank]`) renvoie simplement tout ce
        # qui existe (ruling J3-3, brief T3).
        self.n_rerank = n_rerank

    @property
    def n_chunks(self) -> int:
        if self._n_chunks is None:
            self._n_chunks = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return self._n_chunks

    def df(self, token: str) -> int:
        """Fréquence documentaire : nombre de chunks contenant le token (mis en cache).

        `chunks_norm` est indexé sur du texte déjà normalisé (normalize() appliqué à
        l'ingestion, cf. index.py) : un token brut (mot du langage courant, comme dans
        les tests) doit donc être normalisé avant la recherche FTS pour retrouver sa
        forme indexée (ex. "amortissement" -> "amort"). Idempotent sur un token déjà
        normalisé (appel interne depuis `_termes_match`) — vérifié empiriquement,
        normalize() d'un stem déjà réduit retourne le même stem. Le cache est indexé
        par le token tel que reçu (avant normalisation), pas par sa forme normalisée.
        """
        if token not in self._df_cache:
            terme = normalize(token)
            try:
                n = self.con.execute(
                    "SELECT COUNT(*) FROM chunks_norm WHERE chunks_norm MATCH ?", (f'"{terme}"',)
                ).fetchone()[0] if terme else 0
            except sqlite3.OperationalError:
                n = 0  # token rejeté par la syntaxe FTS5
            self._df_cache[token] = n
        return self._df_cache[token]

    def _termes_match(self, query: str) -> list[str]:
        """Tokens retenus pour le MATCH, dans l'ordre.

        Avec df_max, les tokens présents dans plus de df_max × n_chunks chunks sont
        écartés : ce sont les mots fonctionnels par lesquels un article sans aucun mot
        de contenu commun se fait quand même retrouver (et se classe alors dernier).
        Repli : si le filtrage ne laisse rien, on garde tous les tokens.

        df_max=None (chemin neutre) retourne la liste BRUTE (non dédupliquée) par
        défaut : `bm25()` de SQLite FTS5 est sensible à la MULTIPLICITÉ des termes
        dans l'expression MATCH, pas seulement à l'ensemble des lignes qu'elle
        sélectionne (vérifié empiriquement sur data/corpus.db, mode hybrid, dev :
        dédupliquer même à df_max=None fait passer recall@10 de 0,672 à 0,689 —
        49/61 questions dev contiennent au moins un token répété après normalize()).
        Dédupliquer inconditionnellement, comme le code de référence du brief T1,
        romprait donc la garantie explicite « None = comportement jalon 2.5 inchangé »
        sur le corpus réel — défaut constaté à la mesure T1 (step 8), corrigé alors.

        `dedup_termes` (T2, jalon 3) rend ce comportement explicite et mesurable :
        False (défaut) reproduit exactement la liste brute ci-dessus ; True déduplique
        les tokens du chemin neutre (ordre préservé, premières occurrences), au prix de
        cette pondération par multiplicité (mesuré : recall@10=0,689, cf. § Ablation E,
        docs/eval-jalon3.md).

        Sémantique retenue quand df_max ET dedup_termes sont actifs (cas non mesuré,
        df_max ayant été rejeté en T1, mais qui doit rester cohérent) : le chemin de
        filtrage actif déduplique TOUJOURS ses tokens en interne (nécessaire pour
        raisonner sur des tokens UNIQUES face au seuil df(t) <= seuil) — indépendamment
        de `dedup_termes`, qui ne gouverne que le chemin neutre. Le filtrage ne peut
        donc jamais être MOINS déduplique que dedup_termes=True ; il n'y a pas de
        combinaison qui redonnerait un MATCH non dédupliqué en présence de df_max.
        """
        bruts = normalize(query).split()
        if self.df_max is None:
            return list(dict.fromkeys(bruts)) if self.dedup_termes else bruts
        toks = list(dict.fromkeys(bruts))
        seuil = self.df_max * self.n_chunks
        gardes = [t for t in toks if 0 < self.df(t) <= seuil]
        return gardes or toks

    @property
    def embedder(self):
        if self._embedder is None:
            from .embed import Embedder
            self._embedder = Embedder()
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from .rerank import Reranker
            self._reranker = Reranker()
        return self._reranker

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

    def _bm25(self, query: str, limit: int | None = None) -> dict[str, float]:
        limit = self.pool if limit is None else limit
        toks = self._termes_match(query)
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

    def _dense(self, query: str, limit: int | None = None) -> dict[str, float]:
        limit = self.pool if limit is None else limit
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
            scores = self._bm25(query, self.pool)
        elif mode == "dense":
            scores = self._dense(query, self.pool)
        else:
            scores = self._rrf([self._bm25(query, self.pool), self._dense(query, self.pool)])
        ranked = sorted(scores, key=scores.get, reverse=True)
        source = "fusion" if mode.startswith("hybrid") else mode
        n_restants = max(k - len(routed), 0)
        if mode == "hybrid+rerank":
            # Les résultats routés (référence d'article exacte) sont épinglés sans
            # repasser par le reranker. La fusion est récupérée à self.n_rerank
            # candidats (borne le nombre d'appels predict() du cross-encoder, T3
            # jalon 3 — 25 = comportement jalon 2.5 inchangé) puis rerankée, tronquée
            # à n_restants.
            candidates = [
                self._record(rid, scores[rid], source)
                for rid in ranked
                if rid not in routed_ids
            ][:self.n_rerank]
            results = self.reranker.rerank(query, candidates, top_k=n_restants)
        else:
            results = [
                self._record(rid, scores[rid], source)
                for rid in ranked
                if rid not in routed_ids
            ][:n_restants]
        out = routed + results
        if mode == "hybrid+graph":
            out = self._expand_graph(out, k)
        return out[:k]
