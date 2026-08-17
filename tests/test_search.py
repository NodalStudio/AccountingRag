import pytest
from pathlib import Path
from accounting_rag.db import write_db
from accounting_rag.index import build_index
from accounting_rag.model import Record, Renvoi
from accounting_rag.normalize import normalize
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


class FakeRewriter:
    """Faux rewriter (méthode `reecrire`) : renvoie une réécriture fixée, sans jamais
    appeler l'API Anthropic. Utilisé pour brancher `Searcher(rewriter=...)` en test."""

    def __init__(self, reecriture: str):
        self._reecriture = reecriture
        self.questions_recues: list[str] = []

    def reecrire(self, question: str) -> str:
        self.questions_recues.append(question)
        return self._reecriture


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
    - pcg-700-1 (ajout jalon 3, T1, `df_max`) : enregistrement isolé, terme
      "amortissement" (df=1, un seul chunk sur les 7). Chemin composé uniquement de
      tokens déjà communs à tous les autres chunks (livr/deuxiem/titr/un) : n'introduit
      aucun nouveau token de chemin, ne recherche ni "wombat" ni "zabulon" — vérifié
      sans effet sur les deux comparaisons ci-dessus (`test_poids_chemin_favorise_le_chemin`,
      `test_boost_commentaire_penalise` passent toujours après cet ajout). Le brief
      suppose "'le' dans tous les chunks" pour tester `df_max` ; en réalité, sur les 6
      chunks d'origine, seuls "livr"/"titr"/"deuxiem"/"un" sont à 100 % — "le" n'est
      que dans 4 chunks sur 6 (300, 400, 500, 600). Après ajout de pcg-700-1 (7 chunks
      au total, "le" absent de ce nouveau chunk), "le" reste à 4/7 ≈ 57 %, toujours
      strictement supérieur au seuil `df_max=0.5` exercé par
      `test_df_max_ecarte_les_tokens_trop_frequents` — suffisant pour ce test sans
      qu'aucun chunk n'ait besoin de contenir "le" à 100 %.
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
        _rec("pcg-700-1@2026-01-01", "700-1",
             chemin="Livre Deuxieme Titre Un",
             texte="Un amortissement neutre est mentionne dans ce chunk supplementaire."),
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


# --- Ablation D (T1, jalon 3) : filtrage des tokens peu discriminants de la requête
# lexicale (df_max), par fréquence documentaire.


def test_df_compte_les_chunks_contenant_le_token(db_synthetique):
    s = Searcher(db_synthetique, embedder=FakeEmbedder())
    # le corpus synthétique contient le terme dans un seul chunk
    assert s.df("amortissement") == 1
    assert s.df("motabsent") == 0


def test_df_est_mis_en_cache(db_synthetique):
    s = Searcher(db_synthetique, embedder=FakeEmbedder())
    s.df("amortissement")
    assert "amortissement" in s._df_cache
    # deuxième appel : servi par le cache (pas de requête), valeur identique
    assert s.df("amortissement") == s._df_cache["amortissement"]


def test_df_max_neutre_par_defaut(db_synthetique):
    """df_max=None doit reproduire exactement le comportement jalon 2.5."""
    a = Searcher(db_synthetique, embedder=FakeEmbedder())
    b = Searcher(db_synthetique, embedder=FakeEmbedder(), df_max=None)
    for requete in ("amortissement", "immobilisation corporelle", "que dit l'article 111-1 ?"):
        assert [r["record_id"] for r in a.search(requete, mode="bm25")] == \
               [r["record_id"] for r in b.search(requete, mode="bm25")]


def test_df_max_ecarte_les_tokens_trop_frequents(db_synthetique):
    """Un token présent dans plus de 50 % des chunks est écarté ; le token rare décide seul.

    Défaut du brief constaté à l'exécution (distinct de la ruling J3-1) : `_termes_match`
    retourne des tokens déjà NORMALISÉS (stemmés) — "amortissement" y apparaît sous sa
    forme stem "amort" (snowball FR), jamais littéralement "amortissement". La version
    verbatim du brief (`assert "amortissement" in termes`) échoue donc systématiquement,
    quelle que soit l'implémentation de `_termes_match`, puisque ce token normalisé
    n'existe nulle part dans sa sortie. Assertion corrigée sur la forme normalisée
    (vérifiée : `normalize("amortissement") == "amort"`) — signalé au rapport de tâche.
    """
    s = Searcher(db_synthetique, embedder=FakeEmbedder(), df_max=0.5)
    termes = s._termes_match("le amortissement")  # 'le' dépasse 50 % des chunks de la fixture
    assert "amort" in termes
    assert "le" not in termes


def test_df_max_repli_si_tous_les_tokens_sont_ecartes(db_synthetique):
    """Si le filtrage vide la requête, on retombe sur tous les tokens (jamais zéro résultat gratuit)."""
    s = Searcher(db_synthetique, embedder=FakeEmbedder(), df_max=0.0)
    termes = s._termes_match("le amortissement")
    assert set(termes) == set(normalize("le amortissement").split())


# --- Ablation E (T2, jalon 3) : largeur du pool de candidats avant fusion (pool),
# plus la déduplication des termes du MATCH (dedup_termes) comme configuration mesurée.


def test_pool_neutre_par_defaut(db_synthetique):
    a = Searcher(db_synthetique, embedder=FakeEmbedder())
    b = Searcher(db_synthetique, embedder=FakeEmbedder(), pool=50)
    for requete in ("amortissement", "immobilisation corporelle"):
        assert [r["record_id"] for r in a.search(requete)] == \
               [r["record_id"] for r in b.search(requete)]


def test_pool_est_transmis_aux_deux_canaux(db_synthetique):
    """Le pool doit piloter la limite lexicale ET le k dense, pas seulement l'un des deux.

    Défaut du brief constaté à l'exécution : la mécanique d'espionnage verbatim
    (`vus.setdefault("bm25", limit) or vrai_bm25(q, limit)`) casse dès que `limit` est
    une valeur entière vraie (tout `pool` non nul) — `setdefault` renvoie alors cette
    valeur, l'opérateur `or` court-circuite, et `vrai_bm25` n'est JAMAIS appelé : la
    valeur renvoyée à `search()` (censée être un dict de scores) devient l'entier
    `limit` lui-même, ce qui casse `_rrf()` (`AttributeError: 'int' object has no
    attribute 'get'`). Le brief anticipait ce risque (« adapter la mécanique
    d'espionnage... si un patron plus simple y est déjà en usage ») : ici, la capture
    de `limit` est séparée du calcul de la vraie valeur de retour, pour ne jamais
    dépendre de la véracité de `limit`.
    """
    s = Searcher(db_synthetique, embedder=FakeEmbedder(), pool=7)
    vus = {}
    vrai_bm25, vrai_dense = s._bm25, s._dense

    def espion_bm25(q, limit=None):
        vus["bm25"] = limit
        return vrai_bm25(q, limit)

    def espion_dense(q, limit=None):
        vus["dense"] = limit
        return vrai_dense(q, limit)

    s._bm25 = espion_bm25
    s._dense = espion_dense
    s.search("amortissement", mode="hybrid")
    assert vus["bm25"] == 7 and vus["dense"] == 7


def test_dedup_termes_neutre_par_defaut(db_synthetique):
    """dedup_termes=False doit reproduire exactement le comportement actuel (non dédupliqué)."""
    a = Searcher(db_synthetique, embedder=FakeEmbedder())
    b = Searcher(db_synthetique, embedder=FakeEmbedder(), dedup_termes=False)
    for requete in ("amortissement", "le amortissement le", "immobilisation corporelle"):
        assert a._termes_match(requete) == b._termes_match(requete)
        assert [r["record_id"] for r in a.search(requete)] == \
               [r["record_id"] for r in b.search(requete)]


def test_dedup_termes_deduplique_le_match(db_synthetique):
    """dedup_termes=True déduplique les tokens du MATCH, sans changer leur ensemble ni l'ordre."""
    s_brut = Searcher(db_synthetique, embedder=FakeEmbedder(), dedup_termes=False)
    s_dedup = Searcher(db_synthetique, embedder=FakeEmbedder(), dedup_termes=True)
    requete = "le amortissement le amortissement"
    termes_bruts = s_brut._termes_match(requete)
    termes_dedup = s_dedup._termes_match(requete)
    assert termes_bruts == ["le", "amort", "le", "amort"]
    assert termes_dedup == ["le", "amort"]


# --- Ablation G (T5, jalon 3) : réécriture de requête par LLM (rewriter). Le rewriter
# ne reçoit que le texte de la question ; il agit sur les canaux lexical et dense
# uniquement, jamais sur le routeur de référence d'article (`_route` lit toujours la
# question originale).


def test_rewriter_none_reproduit_le_comportement_actuel(db_synthetique):
    """rewriter=None (défaut) : search() ne doit jamais tenter d'appeler .reecrire()
    sur None (planterait avec AttributeError) ni changer le résultat existant."""
    s = Searcher(db_synthetique, embedder=FakeEmbedder())
    assert s.rewriter is None
    hits = s.search("amortissement", mode="bm25")
    assert hits and hits[0]["record_id"] == "pcg-700-1@2026-01-01"


def test_rewriter_retrouve_le_document_designe_par_la_reecriture(tmp_path):
    """Question sans AUCUN token commun (après normalize) avec le corpus : le canal
    lexical est muet sans réécriture, et retrouve le bon document une fois réécrit
    vers le vocabulaire du document cible."""
    db = tmp_path / "reecriture.db"
    write_db([
        _rec("pcg-800-1@2026-01-01", "800-1",
             texte="Amortissement immobilisation corporelle plan amortissement "
                    "durée utilisation base amortissable."),
        _rec("pcg-900-1@2026-01-01", "900-1",
             texte="Valorisation stocks prix achat ristournes diverses gestion "
                    "approvisionnement."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    question = "comment je répartis le coût d'une machine sur plusieurs années ?"
    corpus = (
        "Amortissement immobilisation corporelle plan amortissement durée "
        "utilisation base amortissable. Valorisation stocks prix achat "
        "ristournes diverses gestion approvisionnement."
    )
    # Précondition du test : sans réécriture, aucun token de la question ne peut
    # retrouver quoi que ce soit dans ce corpus (sinon le test ne serait pas
    # discriminant — le canal lexical répondrait déjà tout seul).
    assert not (set(normalize(question).split()) & set(normalize(corpus).split()))

    rewriter = FakeRewriter("amortissement immobilisation corporelle")
    s_sans = Searcher(db, embedder=FakeEmbedder())
    s_avec = Searcher(db, embedder=FakeEmbedder(), rewriter=rewriter)

    hits_sans = s_sans.search(question, mode="bm25")
    hits_avec = s_avec.search(question, mode="bm25")

    assert hits_sans == []  # canal lexical muet sans réécriture
    assert hits_avec and hits_avec[0]["record_id"] == "pcg-800-1@2026-01-01"
    assert rewriter.questions_recues == [question]  # le rewriter a bien reçu la question originale


def test_route_reste_epingle_meme_avec_reecriture_absurde(tmp_path):
    """Une référence d'article explicite reste routée en position 0, même si la
    réécriture est absurde et sans rapport : `_route` lit toujours la question
    ORIGINALE, jamais la réécriture."""
    db = tmp_path / "route_reecriture.db"
    write_db([
        _rec("pcg-111-1@2026-01-01", "111-1",
             texte="Un immeuble s'amortit sur sa duree d'utilisation."),
        _rec("pcg-300-1@2026-01-01", "300-1",
             texte="Les stocks sont valorises au cout d'achat."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    rewriter = FakeRewriter("xyzzy plugh grault qwerty")  # absurde, sans rapport avec le corpus
    s = Searcher(db, embedder=FakeEmbedder(), rewriter=rewriter)
    hits = s.search("que dit l'article 111-1 ?", mode="bm25")
    assert hits[0]["source"] == "route"
    assert hits[0]["article"] == "111-1"


def test_mode_reecriture_etend_concatene_question_et_reecriture(tmp_path):
    """mode_reecriture="etend" (défaut="remplace") : la requête envoyée aux canaux est
    `question + " " + réécriture`, pas seulement la réécriture — un token discriminant
    de la question originale, absent de la réécriture, reste donc exploitable."""
    db = tmp_path / "etend.db"
    write_db([
        _rec("pcg-800-1@2026-01-01", "800-1", texte="Valorisation stocks prix achat."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    rewriter = FakeRewriter("xyzzy")  # réécriture sans rapport avec le corpus
    s_remplace = Searcher(db, embedder=FakeEmbedder(), rewriter=rewriter, mode_reecriture="remplace")
    s_etend = Searcher(db, embedder=FakeEmbedder(), rewriter=rewriter, mode_reecriture="etend")
    hits_remplace = s_remplace.search("stocks", mode="bm25")
    hits_etend = s_etend.search("stocks", mode="bm25")
    assert hits_remplace == []  # "xyzzy" seul ne matche rien : le token "stocks" est perdu
    assert hits_etend and hits_etend[0]["record_id"] == "pcg-800-1@2026-01-01"
