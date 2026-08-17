"""Tests du reranker cross-encoder (Ablation B, T4, jalon 2.5 ; n_rerank, T3, jalon 3) —
aucun téléchargement : FakeCrossEncoder injecté directement sur l'attribut `model`,
jamais via __init__."""
import pytest
from accounting_rag.db import write_db
from accounting_rag.index import build_index
from accounting_rag.model import Record
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


class CountingFakeCrossEncoder:
    """Comme FakeCrossEncoder, mais compte les appels à predict() — pour vérifier la
    garde top_k<=0 (les routés remplissent déjà k : aucun appel predict ne doit avoir
    lieu, jusqu'à ~2 min de calcul cross-encoder inutile évitées en production)."""

    def __init__(self):
        self.appels = 0

    def predict(self, pairs):
        self.appels += 1
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


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


def test_rerank_top_k_zero_appelle_pas_predict():
    # Cas réel : les routés remplissent déjà k (n_restants=0 dans search()). Sans cette
    # garde, on appellerait predict() sur jusqu'à 25 candidats pour rien — jusqu'à ~2 min
    # de calcul cross-encoder inutile sur bge-reranker-v2-m3.
    r = Reranker.__new__(Reranker)
    fake = CountingFakeCrossEncoder()
    r.model = fake
    results = [{"texte": "amortissement des logiciels", "record_id": "a"},
               {"texte": "un autre texte quelconque", "record_id": "b"}]
    out = r.rerank("amortissement logiciels", results, top_k=0)
    assert out == []
    assert fake.appels == 0


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


# --- n_rerank (T3, jalon 3) : largeur du pool de candidats soumis au reranker en
# mode hybrid+rerank (25 = comportement jalon 2.5, codé en dur jusqu'ici dans search()).


def _rec_generique(i):
    return Record(
        id=f"pcg-{900 + i}-1@2026-01-01", article=f"{900 + i}-1", chemin="Livre X",
        texte=f"Un texte generique numero {i} pour remplir le corpus synthetique.",
        type="reglementaire", nature="comptable", opposable=False,
        valide_du="2026-01-01", valide_au=None, source_citation=None,
        page_debut=1, page_fin=1, renvois=[],
    )


@pytest.fixture
def db_synthetique_pool_large(tmp_path):
    """41 records (40 génériques partageant le terme "generique", matchés par bm25 ET
    dense à pool>=40 ; 1 record routé, article 111-1) — pour exercer `n_rerank`
    au-delà de 25 candidats.

    Défaut constaté à l'écriture des tests (brief T3, step 1) : le brief nomme la
    fixture `db_synthetique` pour `test_n_rerank_elargit_les_candidats_soumis` et
    `test_routes_restent_epingles_avec_pool_elargi`, mais la fixture de ce nom
    (`tests/test_search.py`, 7 records) ne fournit au plus que 6 candidats non
    routés — au-delà de 6, la troncature à `n_rerank=25` et à `n_rerank=200` renvoie
    exactement le même nombre de candidats (6), rendant `test_n_rerank_elargit_les_
    candidats_soumis` incapable de distinguer les deux largeurs (l'assertion `>25`
    échouerait quelle que soit l'implémentation). Fixture dédiée ici, avec assez de
    candidats génériques pour qu'une troncature à 25 diffère effectivement d'une
    troncature à 200 (40 candidats disponibles, aucun routé pour la requête
    "generique") — signalé au rapport de tâche, dans la continuité des défauts de
    brief déjà documentés en T1/T2.
    """
    db = tmp_path / "pool_large.db"
    records = [
        Record(id="pcg-111-1@2026-01-01", article="111-1", chemin="Livre II > Titre I",
               texte="Un immeuble s'amortit sur sa duree d'utilisation prevue.",
               type="reglementaire", nature="comptable", opposable=False,
               valide_du="2026-01-01", valide_au=None, source_citation=None,
               page_debut=1, page_fin=1, renvois=[]),
    ] + [_rec_generique(i) for i in range(40)]
    write_db(records, db)
    build_index(db, embedder=FakeEmbedder())
    return db


@pytest.fixture
def searcher_synthetique_avec_fake_reranker(db_synthetique_pool_large):
    """Searcher (défauts neutres, n_rerank=25) câblé sur un vrai `Reranker` dont le
    `model` est un `RecordingFakeCrossEncoder` — permet de compter les paires
    (query, passage) effectivement soumises à `predict()` par `search()`."""
    fake = RecordingFakeCrossEncoder()
    reranker = Reranker.__new__(Reranker)
    reranker.model = fake
    s = Searcher(db_synthetique_pool_large, embedder=FakeEmbedder(), reranker=reranker)
    return s, fake


def test_n_rerank_neutre_par_defaut(searcher_synthetique_avec_fake_reranker):
    """n_rerank=25 par défaut : mêmes candidats soumis qu'au jalon 2.5."""
    s, fake = searcher_synthetique_avec_fake_reranker
    assert s.n_rerank == 25
    s.search("generique", mode="hybrid+rerank", k=10)
    assert fake.pairs_recus is not None
    assert len(fake.pairs_recus) == 25  # 40 candidats disponibles, tronqués à 25


def test_n_rerank_elargit_les_candidats_soumis(db_synthetique_pool_large):
    """n_rerank=200 soumet tous les candidats disponibles quand le pool le permet."""
    fake = RecordingFakeCrossEncoder()
    reranker = Reranker.__new__(Reranker)
    reranker.model = fake
    s = Searcher(db_synthetique_pool_large, embedder=FakeEmbedder(), reranker=reranker,
                 pool=200, n_rerank=200)
    s.search("generique", mode="hybrid+rerank", k=10)
    assert fake.pairs_recus is not None
    # 41 candidats disponibles (< n_rerank=200) : rien à tronquer, tous soumis. Le
    # record routé (article 111-1) entre aussi dans le pool dense pour CETTE requête
    # ("generique" ne déclenche pas le routeur) : FakeEmbedder donne un vecteur
    # constant à tous les passages, donc les 41 records sont à égalité en distance
    # dense et entrent tous dans le pool à limit=200.
    assert len(fake.pairs_recus) == 41
    assert len(fake.pairs_recus) > 25  # strictement plus qu'à n_rerank=25 (défaut)


def test_routes_restent_epingles_avec_pool_elargi(db_synthetique_pool_large):
    """Élargir le pool ne doit pas déloger un résultat routé de la première place."""
    fake = RecordingFakeCrossEncoder()
    reranker = Reranker.__new__(Reranker)
    reranker.model = fake
    s = Searcher(db_synthetique_pool_large, embedder=FakeEmbedder(), reranker=reranker,
                 pool=200, n_rerank=200)
    hits = s.search("que dit l'article 111-1 ?", mode="hybrid+rerank", k=10)
    assert hits[0]["source"] == "route"
    assert hits[0]["article"] == "111-1"
