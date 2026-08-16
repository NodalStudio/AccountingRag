import pytest
from pathlib import Path
from accounting_rag.db import write_db
from accounting_rag.index import build_index
from accounting_rag.model import Record, Renvoi
from accounting_rag.search import Searcher
from conftest import _rec

DB = Path("data/corpus.db")


class FakeEmbedder:
    """Embedder factice à vecteur constant : suffisant quand seul le canal bm25/route
    est exercé, ou que le classement dense n'a pas besoin d'être discriminant."""

    dim = 4

    def encode_passages(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class VectorEmbedder:
    """Embedder factice pilotable : associe un vecteur choisi à chaque passage selon
    un mot-clé présent dans le texte, pour contrôler précisément et sans ambiguïté
    (aucune égalité de distance) le classement dense dans les tests de fusion/graphe."""

    dim = 4

    def __init__(self, query_vec, passage_vecs: dict[str, list[float]]):
        self._query_vec = list(query_vec)
        self._passage_vecs = passage_vecs

    def encode_query(self, text):
        return list(self._query_vec)

    def encode_passages(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for key, v in self._passage_vecs.items():
                if key in t:
                    vec = v
                    break
            out.append(list(vec))
        return out


class TestSearcherCorpusReel:
    """Tests de fumée sur le corpus réel (data/corpus.db, gitignoré) : sautés sur un
    clone frais sans corpus construit. Les tests exerçant le comportement précis du
    Searcher (routeur, fusion, expansion graphe, invariant apostrophe) tournent sur
    base synthétique ci-dessous, indépendamment de data/corpus.db."""

    pytestmark = pytest.mark.skipif(not DB.exists(), reason="corpus.db absent")

    @pytest.fixture(scope="class")
    @classmethod
    def s(cls):
        return Searcher(DB)

    def test_routeur_reference_directe(self, s):
        hits = s.search("que dit l'article 214-1 ?", mode="bm25")
        assert hits and hits[0]["source"] == "route"
        assert hits[0]["article"] == "214-1"

    def test_bm25_flexions(self, s):
        hits = s.search("amortissements dérogatoires", k=10, mode="bm25")
        assert hits
        assert any("dérogatoire" in h["texte"].lower() or "derogatoire" in h["texte"].lower() for h in hits[:5])

    def test_dense_vocabulaire_courant(self, s):
        hits = s.search("comment comptabiliser un logiciel acheté ?", k=10, mode="dense")
        assert hits and len(hits) <= 10
        assert all(h["texte"] for h in hits)

    def test_hybrid_contient_les_deux(self, s):
        hits = s.search("crédit-bail levée d'option", k=10, mode="hybrid")
        assert hits
        ids = [h["record_id"] for h in hits]
        assert len(ids) == len(set(ids))  # dédupliqué

    def test_graph_expansion_dedoublonnee(self, s):
        # I4 : l'ancienne assertion `any(source=='graph') or expanded == base` était
        # toujours vraie. Le cas où l'expansion DOIT produire un résultat graph est
        # couvert de façon déterministe par test_expansion_graph_* ci-dessous, sur
        # base synthétique. Ici on ne vérifie que les invariants valables quel que
        # soit le contenu réel du corpus.
        expanded = s.search("contrat de crédit-bail", k=10, mode="hybrid+graph")
        ids = [h["record_id"] for h in expanded]
        assert ids
        assert len(ids) == len(set(ids))
        assert all(h["source"] in {"route", "fusion", "graph"} for h in expanded)


# --- Tests sur base synthétique (tmp_path) : tournent sur un clone frais, sans dépendre
# de data/corpus.db. Même patron que tests/test_index.py (write_db + FakeEmbedder + build_index).


def test_mode_inconnu_leve_value_error(tmp_path):
    db = tmp_path / "mode.db"
    write_db([_rec("pcg-1-1@2026-01-01", "1-1", texte="x")], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder())
    with pytest.raises(ValueError):
        s.search("x", mode="bogus")


def test_searcher_leve_filenotfounderror_si_corpus_absent(tmp_path):
    # I2 : sur un clone frais sans data/corpus.db, sqlite3.connect créerait sinon
    # silencieusement une base vide et l'erreur ne surviendrait qu'au premier accès.
    with pytest.raises(FileNotFoundError):
        Searcher(tmp_path / "absent.db")


def test_routeur_score_100_sur_reference_directe(tmp_path):
    db = tmp_path / "route.db"
    write_db([
        _rec("pcg-214-1@2026-01-01", "214-1", texte="Un immeuble s'amortit sur sa duree d'utilisation."),
        _rec("pcg-300-1@2026-01-01", "300-1", texte="Les stocks sont valorises au cout d'achat."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder())
    hits = s.search("que dit l'article 214-1 ?", mode="bm25")
    assert hits[0]["source"] == "route"
    assert hits[0]["article"] == "214-1"
    assert hits[0]["score"] == 100.0


def test_fusion_rrf_renvoie_des_resultats(tmp_path):
    db = tmp_path / "fusion.db"
    write_db([
        _rec("pcg-214-1@2026-01-01", "214-1",
             texte="Le credit-bail est comptabilise chez le locataire selon les regles applicables au contrat."),
        _rec("pcg-300-1@2026-01-01", "300-1", texte="Les stocks sont valorises au cout d'achat, hors ristournes."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder())
    hits = s.search("contrat de credit-bail", mode="hybrid", k=5)
    assert hits
    assert hits[0]["source"] == "fusion"


def test_expansion_graph_ajoute_un_renvoi_hors_top_k_base(tmp_path):
    # Construit un cas où le renvoi cible (article 214-1) est strictement moins bien
    # classé que les deux meilleurs résultats de base (aucune égalité de distance
    # dense, donc aucun classement ambigu) : il est donc absent des résultats de base
    # tronqués à k=2, et son apparition dans hybrid+graph ne peut venir QUE de
    # l'expansion par renvois.
    db = tmp_path / "graph.db"
    write_db([
        _rec("pcg-300-1@2026-01-01", "300-1",
             texte="Le contrat de credit-bail est comptabilise chez le locataire.",
             renvois=[Renvoi("pcg-214-1", "interne")]),
        _rec("pcg-500-1@2026-01-01", "500-1", texte="Les stocks sont valorises au cout d'achat."),
        _rec("pcg-214-1@2026-01-01", "214-1", texte="Un immeuble s'amortit sur sa duree d'utilisation prevue."),
    ], db)
    emb = VectorEmbedder(
        query_vec=[1.0, 0.0, 0.0, 0.0],
        passage_vecs={
            "credit-bail": [1.0, 0.0, 0.0, 0.0],   # distance 0 à la requête (meilleur)
            "stocks": [0.0, 1.0, 0.0, 0.0],        # distance sqrt(2)
            "amortit": [-1.0, 0.0, 0.0, 0.0],       # distance 2 (le plus loin)
        },
    )
    build_index(db, embedder=emb)
    s = Searcher(db, embedder=emb)

    base = s.search("contrat de credit-bail", k=2, mode="hybrid")
    assert {h["record_id"] for h in base} == {"pcg-300-1@2026-01-01", "pcg-500-1@2026-01-01"}

    expanded = s.search("contrat de credit-bail", k=2, mode="hybrid+graph")
    graph_hits = [h for h in expanded if h["source"] == "graph"]
    assert graph_hits, "l'expansion par renvois doit produire strictement un résultat graph ici"
    assert graph_hits[0]["record_id"] == "pcg-214-1@2026-01-01"


def test_invariant_apostrophe_build_query(tmp_path):
    # Invariant central de C1 : la même requête, écrite avec une apostrophe ASCII ou
    # typographique (U+2019), doit retrouver le même document après le fix normalize.py.
    db = tmp_path / "apostrophe.db"
    write_db([
        _rec("pcg-1-1@2026-01-01", "1-1",
             texte="L'exercice comptable correspond a la periode d'imputation."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder())
    hits_ascii = s.search("l'exercice comptable", mode="bm25")
    hits_typo = s.search("l’exercice comptable", mode="bm25")
    assert hits_ascii and hits_typo
    assert {h["record_id"] for h in hits_ascii} == {h["record_id"] for h in hits_typo}


# --- Ablation A (T3, jalon 2.5) : pondération par champ (chemin, type de record).
# Deux nouveaux paramètres neutres par défaut sur Searcher : poids_chemin, boost_commentaire.


def _rec_type(rid, article, type_, chemin="Livre II > Titre I", texte="x"):
    """Comme _rec (conftest) mais avec type explicite (_rec fixe toujours 'reglementaire')."""
    return Record(
        id=rid, article=article, chemin=chemin, texte=texte, type=type_,
        nature="comptable", opposable=False, valide_du="2026-01-01", valide_au=None,
        source_citation=None, page_debut=1, page_fin=1, renvois=[],
    )


@pytest.fixture
def searcher_synthetique(tmp_path):
    """Petite base variée (route + bm25 multi-documents) pour comparer deux instances
    de Searcher à paramètres différents mais censés produire un résultat identique."""
    db = tmp_path / "neutre.db"
    write_db([
        _rec("pcg-214-1@2026-01-01", "214-1",
             texte="Un immeuble s'amortit sur sa duree d'utilisation prevue."),
        _rec("pcg-300-1@2026-01-01", "300-1",
             texte="Les stocks sont valorises au cout d'achat, hors ristournes."),
        _rec("pcg-400-1@2026-01-01", "400-1", chemin="Livre III > Titre II",
             texte="Le contrat de credit-bail est comptabilise chez le locataire selon les regles applicables."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    return db


@pytest.fixture
def db_synthetique(tmp_path):
    """Base combinant deux paires de documents conçues pour isoler chacun des deux
    nouveaux paramètres :
    - pcg-100-1 (terme "wombat" UNIQUEMENT dans le chemin, chemin long) vs pcg-200-1
      (terme "wombat" UNIQUEMENT dans le texte, texte court) — pcg-300/400-1 servent de
      remplissage pour stabiliser les longueurs moyennes des colonnes FTS. Ordre de
      référence vérifié empiriquement à poids neutre (1.0, 1.0) : pcg-200-1 devant
      pcg-100-1 (le texte court l'emporte) ; à poids_chemin=3.0, pcg-100-1 passe devant.
    - pcg-500-1 (type='reglementaire', terme "zabulon" une fois) et pcg-600-1
      (type='commentaire_ANC', terme "zabulon" répété trois fois) : à poids neutre le
      commentaire est un meilleur match bm25 (tf plus élevé) et passe devant le
      réglementaire — condition nécessaire pour que le test puisse distinguer un boost
      réellement appliqué d'un boost ignoré (avec une empreinte bm25 identique entre les
      deux records, un boost ignoré ET un boost appliqué produisent le même ordre par
      hasard, cf. revue). boost_commentaire=0.5 doit inverser cet ordre : le
      réglementaire (score inchangé) doit repasser devant le commentaire (score divisé
      par deux). Vérifié empiriquement (scores bruts) : neutre {600: 1.52e-6, 500:
      1.07e-6} -> commentaire en tête ; boost=0.5 {600: 0.76e-6, 500: 1.07e-6} ->
      réglementaire en tête.
    """
    db = tmp_path / "ablation.db"
    write_db([
        _rec("pcg-100-1@2026-01-01", "100-1",
             chemin="Livre wombat Deuxieme Titre Premier Chapitre Trois Section Deux Sous section Quatre",
             texte="Un texte generique sans importance particuliere du tout ici et la pour remplir un peu plus encore."),
        _rec("pcg-200-1@2026-01-01", "200-1",
             chemin="Livre Deuxieme Titre Un",
             texte="Un texte contenant wombat brievement."),
        _rec("pcg-300-1@2026-01-01", "300-1",
             chemin="Livre Deuxieme Titre Un Chapitre Deux",
             texte="Un texte de remplissage sans aucun rapport avec le sujet traite ici pour equilibrer les longueurs moyennes des colonnes du corpus."),
        _rec("pcg-400-1@2026-01-01", "400-1",
             chemin="Livre Deuxieme Titre Un Chapitre Deux",
             texte="Un texte de remplissage sans aucun rapport avec le sujet traite ici pour equilibrer les longueurs moyennes des colonnes du corpus."),
        _rec_type("pcg-500-1@2026-01-01", "500-1", "reglementaire",
                  chemin="Livre Deuxieme Titre Un", texte="Le traitement de zabulon est precise ici."),
        _rec_type("pcg-600-1@2026-01-01", "600-1", "commentaire_ANC",
                  chemin="Livre Deuxieme Titre Un",
                  texte="Le traitement de zabulon zabulon est precise ici et zabulon encore."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    return db


def test_poids_chemin_neutre_par_defaut(searcher_synthetique):
    # Deux Searcher, poids_chemin=1.0 explicite vs défaut : mêmes résultats bm25 sur
    # trois requêtes distinctes de la fixture (avant l'implémentation, poids_chemin
    # n'existe pas encore -> TypeError sur la construction du second Searcher).
    s_defaut = Searcher(searcher_synthetique, embedder=FakeEmbedder())
    s_neutre = Searcher(searcher_synthetique, embedder=FakeEmbedder(), poids_chemin=1.0)
    for query in ["amortissement immeuble", "stocks cout d'achat", "contrat de credit-bail"]:
        ids_defaut = [h["record_id"] for h in s_defaut.search(query, mode="bm25")]
        ids_neutre = [h["record_id"] for h in s_neutre.search(query, mode="bm25")]
        assert ids_defaut == ids_neutre


def test_poids_chemin_favorise_le_chemin(db_synthetique):
    # pcg-100-1 : terme "wombat" uniquement dans le chemin. pcg-200-1 : uniquement
    # dans le texte. À poids neutre l'ordre de référence place pcg-200-1 en tête ;
    # à poids_chemin=3.0, pcg-100-1 doit passer devant.
    s_neutre = Searcher(db_synthetique, embedder=FakeEmbedder(), poids_chemin=1.0)
    s_favorise = Searcher(db_synthetique, embedder=FakeEmbedder(), poids_chemin=3.0)
    ids_neutre = [h["record_id"] for h in s_neutre.search("wombat", mode="bm25", k=2)]
    ids_favorise = [h["record_id"] for h in s_favorise.search("wombat", mode="bm25", k=2)]
    assert ids_neutre[0] == "pcg-200-1@2026-01-01"
    assert ids_favorise[0] == "pcg-100-1@2026-01-01"


def test_boost_commentaire_penalise(db_synthetique):
    # pcg-600-1 (commentaire_ANC, "zabulon" x3) est un meilleur match bm25 que pcg-500-1
    # (reglementaire, "zabulon" x1) : à poids neutre le commentaire est donc STRICTEMENT
    # en tête. boost_commentaire=0.5 doit inverser l'ordre : le réglementaire (score
    # inchangé) passe devant le commentaire (score divisé par deux). Une empreinte bm25
    # identique entre les deux records ne permettrait pas de distinguer un boost
    # réellement appliqué d'un boost ignoré (tri stable + ordre d'insertion favoriseraient
    # déjà le réglementaire par hasard) — d'où l'asymétrie délibérée de tf ci-dessus.
    s_neutre = Searcher(db_synthetique, embedder=FakeEmbedder(), boost_commentaire=1.0)
    s_penalise = Searcher(db_synthetique, embedder=FakeEmbedder(), boost_commentaire=0.5)
    hits_neutre = s_neutre.search("zabulon", mode="bm25", k=2)
    hits_penalise = s_penalise.search("zabulon", mode="bm25", k=2)
    assert hits_neutre[0]["record_id"] == "pcg-600-1@2026-01-01"
    assert hits_penalise[0]["record_id"] == "pcg-500-1@2026-01-01"
