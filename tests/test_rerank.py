"""Tests du reranker cross-encoder (Ablation B, T4, jalon 2.5) — aucun téléchargement :
FakeCrossEncoder injecté directement sur l'attribut `model`, jamais via __init__."""
import pytest
from accounting_rag.rerank import Reranker
from accounting_rag.search import Searcher
from test_search import searcher_synthetique, FakeEmbedder  # noqa: F401 (fixture réutilisée)


class FakeCrossEncoder:
    def predict(self, pairs):
        # score = longueur du recouvrement lexical naïf, déterministe
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


class RecordingFakeCrossEncoder:
    """Comme FakeCrossEncoder, mais conserve les paires reçues pour vérifier la troncature."""

    def __init__(self):
        self.pairs_recus = None

    def predict(self, pairs):
        self.pairs_recus = pairs
        return [len(p) for _, p in pairs]


class FakeReranker:
    """Reranker factice pour les tests de search.py : score = longueur du texte
    (déterministe), afin de vérifier que le mode hybrid+rerank rerank bien les
    résultats non routés (et laisse le routé intact)."""

    def rerank(self, query, results, top_k):
        ranked = sorted(results, key=lambda r: len(r["texte"]), reverse=True)
        for r in ranked:
            r["score_rerank"] = len(r["texte"])
        return ranked[:top_k]


def test_rerank_trie_et_tronque():
    r = Reranker.__new__(Reranker)
    r.model = FakeCrossEncoder()
    results = [{"texte": "rien ici", "record_id": "a"},
               {"texte": "amortissement des logiciels", "record_id": "b"}]
    out = r.rerank("amortissement logiciels", results, top_k=1)
    assert [x["record_id"] for x in out] == ["b"]
    assert "score_rerank" in out[0]


def test_rerank_resultats_vides():
    r = Reranker.__new__(Reranker)
    r.model = FakeCrossEncoder()
    assert r.rerank("peu importe", [], top_k=5) == []


def test_rerank_tronque_le_texte_a_1000_caracteres():
    r = Reranker.__new__(Reranker)
    fake = RecordingFakeCrossEncoder()
    r.model = fake
    results = [{"texte": "x" * 5000, "record_id": "a"}]
    r.rerank("requete", results, top_k=1)
    assert fake.pairs_recus is not None
    assert all(len(passage) <= 1000 for _, passage in fake.pairs_recus)


def test_mode_hybrid_rerank_epingle_le_route(searcher_synthetique):
    s = Searcher(searcher_synthetique, embedder=FakeEmbedder(), reranker=FakeReranker())
    hits = s.search("que dit l'article 214-1 ?", mode="hybrid+rerank", k=3)
    assert hits[0]["source"] == "route"
    assert hits[0]["article"] == "214-1"
    assert "score_rerank" not in hits[0]
    # les résultats non routés doivent porter la trace du passage par le reranker
    assert all("score_rerank" in h for h in hits[1:])
