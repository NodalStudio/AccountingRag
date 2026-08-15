import pytest
from accounting_rag.embed import Embedder


@pytest.fixture(scope="module")
def emb():
    return Embedder()


def test_dimensions_et_normalisation(emb):
    vecs = emb.encode_passages(["l'amortissement des immobilisations"])
    assert len(vecs) == 1 and len(vecs[0]) == emb.dim
    norm = sum(x * x for x in vecs[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_similarite_semantique(emb):
    q = emb.encode_query("comment comptabiliser un logiciel acheté ?")
    p_proche = emb.encode_passages(["Les immobilisations incorporelles comprennent les logiciels acquis."])[0]
    p_loin = emb.encode_passages(["Le montant des primes de remboursement d'emprunt est amorti."])[0]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, p_proche) > dot(q, p_loin)
