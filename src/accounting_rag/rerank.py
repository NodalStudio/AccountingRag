"""Reranker cross-encoder optionnel : rejuge une liste de résultats de retrieval.

Défaut adopté après mesure bootstrap sur benchmark/dev.jsonl (Ablation B, T4,
jalon 2.5, cf. docs/eval-jalon25.md) : BAAI/bge-reranker-v2-m3, seul des deux
modèles testés à franchir le critère d'adoption (p_amelioration>=0,95, aucune
catégorie perdant plus de 0,05 de recall@10) contre le baseline hybrid. Le
modèle initialement pressenti (cross-encoder/mmarco-mMiniLMv2-L12-H384-v1,
plus léger) a été mesuré et REJETÉ (p_amelioration=0,858) ; il reste
utilisable via ACCRAG_RERANKER pour les contextes sensibles à la latence
(~10s/question contre ~117s/question pour bge-reranker-v2-m3 sur cette
machine, cf. docs/eval-jalon25.md, section « Ablation B »)."""
import os
import sys

_DEFAULT = "BAAI/bge-reranker-v2-m3"
_MAX_CHARS = 1000  # borne la latence par appel predict() (paires (query, passage))


class Reranker:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder  # import paresseux (lourd), comme Embedder

        self.model_name = model_name or os.environ.get("ACCRAG_RERANKER", _DEFAULT)
        print(f"[Reranker] chargement de {self.model_name} (~2,2 Go au premier "
              f"téléchargement pour le défaut, latence CPU indicative ~117s/question "
              f"pour 25 candidats — cf. docs/eval-jalon25.md, section « Ablation B »)",
              file=sys.stderr)
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: list[dict], top_k: int) -> list[dict]:
        """Rejuge `results` (dicts portant un champ `texte`) avec le cross-encoder et
        retourne les `top_k` meilleurs par ordre décroissant de `score_rerank`.

        Le champ `score` (score de fusion RRF bm25+dense, posé en amont par la
        recherche hybride) n'est PAS réécrit ici : il reste présent, inchangé, sur
        chaque dict d'entrée. Seul `score_rerank` est ajouté/écrasé. L'ordre de
        sortie suit exclusivement `score_rerank` (tri décroissant) — pas `score` —
        donc un appelant qui trierait par `score` après un appel à `rerank()`
        obtiendrait un ordre différent (et incorrect pour ce mode).
        """
        if top_k <= 0:
            return []
        if not results:
            return []
        pairs = [(query, r["texte"][:_MAX_CHARS]) for r in results]
        scores = self.model.predict(pairs)
        for r, s in zip(results, scores):
            r["score_rerank"] = float(s)
        ranked = sorted(results, key=lambda r: r["score_rerank"], reverse=True)
        return ranked[:top_k]
