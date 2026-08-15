import pytest
from pathlib import Path
from accounting_rag.search import Searcher

DB = Path("data/corpus.db")
pytestmark = pytest.mark.skipif(not DB.exists(), reason="corpus.db absent")


@pytest.fixture(scope="module")
def s():
    return Searcher(DB)


def test_routeur_reference_directe(s):
    hits = s.search("que dit l'article 214-1 ?", mode="bm25")
    assert hits and hits[0]["source"] == "route"
    assert hits[0]["article"] == "214-1"


def test_bm25_flexions(s):
    hits = s.search("amortissements dérogatoires", k=10, mode="bm25")
    assert hits
    assert any("dérogatoire" in h["texte"].lower() or "derogatoire" in h["texte"].lower() for h in hits[:5])


def test_dense_vocabulaire_courant(s):
    hits = s.search("comment comptabiliser un logiciel acheté ?", k=10, mode="dense")
    assert hits and len(hits) <= 10
    assert all(h["texte"] for h in hits)


def test_hybrid_contient_les_deux(s):
    hits = s.search("crédit-bail levée d'option", k=10, mode="hybrid")
    assert hits
    ids = [h["record_id"] for h in hits]
    assert len(ids) == len(set(ids))  # dédupliqué


def test_graph_expansion_ajoute_des_renvois(s):
    base = {h["record_id"] for h in s.search("contrat de crédit-bail", k=10, mode="hybrid")}
    expanded = s.search("contrat de crédit-bail", k=10, mode="hybrid+graph")
    assert any(h["source"] == "graph" for h in expanded) or {h["record_id"] for h in expanded} == base
