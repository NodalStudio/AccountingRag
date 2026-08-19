import sqlite3
import pytest
from accounting_rag.citations import (
    EXTRAIT_MINIMUM, correspond_brut, metriques, normaliser_pour_comparaison,
    verifier_citation)


@pytest.fixture
def con(tmp_path):
    """Base minimale : deux records, dont un au texte à espaces irréguliers."""
    db = tmp_path / "mini.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE records (id TEXT PRIMARY KEY, texte TEXT)")
    c.executemany("INSERT INTO records VALUES (?, ?)", [
        ("pcg-214-13@2026-01-01",
         "Le mode  d'amortissement\nlinéaire est appliqué à défaut de mode mieux adapté."),
        ("pcg-212-3@2026-01-01", "Un actif est un élément identifiable du patrimoine."),
    ])
    c.commit()
    return c


def test_normalisation_replie_les_espaces_et_la_casse():
    assert (normaliser_pour_comparaison("Le mode  d'amortissement\nLINÉAIRE")
            == normaliser_pour_comparaison("le mode d'amortissement linéaire"))


def test_extrait_verbatim_est_ok_malgre_les_espaces(con):
    assert verifier_citation(
        con, "pcg-214-13@2026-01-01",
        "Le mode d'amortissement linéaire est appliqué à défaut") == "ok"


def test_record_inexistant_est_detecte(con):
    assert verifier_citation(
        con, "pcg-999-99@2026-01-01",
        "Le mode d'amortissement linéaire est appliqué à défaut") == "record_inexistant"


def test_extrait_absent_du_record_est_detecte(con):
    """Le record existe mais ne contient pas l'extrait : citation non portante."""
    assert verifier_citation(
        con, "pcg-212-3@2026-01-01",
        "Le mode d'amortissement linéaire est appliqué à défaut") == "extrait_absent"


def test_extrait_trop_court_est_refuse(con):
    """Un extrait de quelques caractères matcherait n'importe quoi : il ne prouve rien."""
    assert len("amortir") < EXTRAIT_MINIMUM
    assert verifier_citation(con, "pcg-214-13@2026-01-01", "amortir") == "extrait_trop_court"


def test_metriques_agrege_les_trois_taux(tmp_path, con):
    db = tmp_path / "mini.db"
    reponses = {
        "q1": {"abstention": False, "reponse": "a", "citations": [
            {"record_id": "pcg-214-13@2026-01-01",
             "extrait": "Le mode d'amortissement linéaire est appliqué à défaut"}]},
        "q2": {"abstention": False, "reponse": "b", "citations": [
            {"record_id": "pcg-999-99@2026-01-01", "extrait": "un extrait inventé de longueur suffisante"}]},
        "q3": {"abstention": False, "reponse": "c", "citations": []},
        "q4": {"abstention": True, "reponse": "le corpus ne le dit pas", "citations": []},
    }
    m = metriques(reponses, db)
    assert m["n"] == 4
    assert m["taux_citations_inexistantes"] == 0.5      # 1 citation sur 2 vérifiables
    assert m["taux_reponses_sans_citation"] == round(1 / 3, 4)    # q3 sur les 3 non-abstentions
    assert m["taux_abstention"] == 0.25
    assert m["par_question"]["q1"] == "ok"


def test_le_taux_de_correspondance_brute_est_publie_a_cote(tmp_path, con):
    """La tolérance de normalisation doit être visible, donc falsifiable.

    q1 cite un extrait dont le texte du corpus porte deux espaces et un retour à la
    ligne : il correspond APRÈS normalisation, pas avant. Le taux brut doit donc valoir
    0.0 et le taux non portant 0.0 — l'écart entre les deux EST la tolérance.
    """
    reponses = {
        "q1": {"abstention": False, "reponse": "a", "citations": [
            {"record_id": "pcg-214-13@2026-01-01",
             "extrait": "Le mode d'amortissement linéaire est appliqué à défaut"}]},
    }
    m = metriques(reponses, tmp_path / "mini.db")
    assert m["taux_citations_non_portantes"] == 0.0
    assert m["taux_correspondance_brute"] == 0.0


def test_citations_par_reponse_est_publie(tmp_path, con):
    """Le générateur cite jusqu'à 10 passages pour un gold qui en porte un : le nombre
    de citations par réponse doit apparaître dans les chiffres, pas dans la prose."""
    extrait = "Le mode d'amortissement linéaire est appliqué à défaut"
    reponses = {
        "q1": {"abstention": False, "reponse": "a", "citations": [
            {"record_id": "pcg-214-13@2026-01-01", "extrait": extrait}]},
        "q2": {"abstention": False, "reponse": "b", "citations": [
            {"record_id": "pcg-214-13@2026-01-01", "extrait": extrait},
            {"record_id": "pcg-214-13@2026-01-01", "extrait": extrait},
            {"record_id": "pcg-214-13@2026-01-01", "extrait": extrait}]},
        "q3": {"abstention": True, "reponse": "rien", "citations": []},
    }
    m = metriques(reponses, tmp_path / "mini.db")
    # Les abstentions ne comptent pas : elles n'ont pas à citer.
    assert m["citations_par_reponse"] == {"min": 1, "median": 2.0, "max": 3, "moyenne": 2.0}


def test_un_extrait_trop_court_ne_compte_pas_comme_correspondance_brute(con):
    """L'écart entre les deux taux doit mesurer la normalisation, et rien d'autre.

    Un extrait de moins de EXTRAIT_MINIMUM caractères présent VERBATIM serait sinon
    compté brut et refusé par le verdict : l'écart mélangerait alors deux causes et ne
    dirait plus ce que la normalisation tolère.
    """
    court = "Le mode  d'amortissement"
    assert len(court) < EXTRAIT_MINIMUM
    assert court in con.execute(
        "SELECT texte FROM records WHERE id = 'pcg-214-13@2026-01-01'").fetchone()[0]
    assert verifier_citation(con, "pcg-214-13@2026-01-01", court) == "extrait_trop_court"
    assert correspond_brut(con, "pcg-214-13@2026-01-01", court) is False


def test_un_taux_sans_denominateur_vaut_none_et_pas_zero(tmp_path, con):
    """Sur un split tout en abstention, « 0,0 citation inexistante » se lirait comme un
    sans-faute alors que le taux n'est pas défini."""
    reponses = {"q1": {"abstention": True, "reponse": "le corpus ne le dit pas",
                       "citations": []}}
    m = metriques(reponses, tmp_path / "mini.db")
    assert m["n_citations"] == 0
    assert m["taux_citations_inexistantes"] is None
    assert m["taux_citations_non_portantes"] is None
    assert m["taux_correspondance_brute"] is None
    # Celui-là reste défini : il se calcule sur les réponses, pas sur les citations.
    assert m["taux_abstention"] == 1.0


@pytest.fixture
def con_versions(tmp_path):
    """Base avec un article à version unique et un article à DEUX versions.

    Le corpus réel porte 39 articles à plusieurs versions : l'ambiguïté n'est pas
    théorique, elle attend le premier corpus historisé.
    """
    db = tmp_path / "versions.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE records (id TEXT PRIMARY KEY, texte TEXT)")
    c.executemany("INSERT INTO records VALUES (?, ?)", [
        ("pcg-1121-1@2026-01-01",
         "Le compte 698 Intégration fiscale enregistre les charges et produits du groupe."),
        ("pcg-212-3@2025-01-01", "Un actif est un élément identifiable du patrimoine."),
        ("pcg-212-3@2026-01-01", "Un actif est un élément identifiable du patrimoine."),
    ])
    c.commit()
    return c


EXTRAIT_1121 = "Le compte 698 Intégration fiscale enregistre les charges"


def test_identifiant_sans_version_nest_pas_une_hallucination(con_versions):
    """La leçon de la première campagne : 66 citations comptées « inexistantes »
    portaient le bon article et un extrait verbatim, seul le @date manquait."""
    assert verifier_citation(con_versions, "pcg-1121-1", EXTRAIT_1121) == "version_omise"


def test_identifiant_sans_version_reste_faux_si_lextrait_ne_porte_pas(con_versions):
    """Omettre la version ne rachète pas un extrait absent du texte."""
    assert verifier_citation(
        con_versions, "pcg-1121-1",
        "un extrait totalement inventé et de longueur suffisante") == "extrait_absent"


def test_article_a_plusieurs_versions_cite_sans_version_est_ambigu(con_versions):
    """La citation ne dit pas laquelle des deux versions elle invoque : pas traçable."""
    assert verifier_citation(
        con_versions, "pcg-212-3",
        "Un actif est un élément identifiable du patrimoine.") == "version_ambigue"


def test_un_article_inexistant_reste_inexistant(con_versions):
    """Le rattrapage de version ne doit pas absoudre une vraie hallucination."""
    assert verifier_citation(
        con_versions, "pcg-999-99", EXTRAIT_1121) == "record_inexistant"


def test_un_identifiant_versionne_faux_nest_pas_rattrape(con_versions):
    """`pcg-999-99@2026-01-01` porte déjà un @ : il ne doit pas passer par le rattrapage,
    sinon un identifiant versionné inventé deviendrait indétectable."""
    assert verifier_citation(
        con_versions, "pcg-999-99@2026-01-01", EXTRAIT_1121) == "record_inexistant"


def test_correspond_brut_resout_aussi_la_version(con_versions):
    """Sinon l'écart entre le taux brut et le taux « ok » mesurerait l'omission de
    version au lieu de la normalisation."""
    assert correspond_brut(con_versions, "pcg-1121-1", EXTRAIT_1121) is True
    assert correspond_brut(con_versions, "pcg-999-99", EXTRAIT_1121) is False


def test_les_deux_fautes_ne_sont_jamais_additionnees(tmp_path, con_versions):
    """Le taux de citations inexistantes et le taux de version omise sont deux chiffres.

    Les additionner est exactement ce qui a fait afficher 15,64 % d'hallucinations là où
    il n'y en avait aucune.
    """
    reponses = {
        "q1": {"abstention": False, "reponse": "a", "citations": [
            {"record_id": "pcg-1121-1", "extrait": EXTRAIT_1121}]},
        "q2": {"abstention": False, "reponse": "b", "citations": [
            {"record_id": "pcg-999-99", "extrait": EXTRAIT_1121}]},
    }
    m = metriques(reponses, tmp_path / "versions.db")
    assert m["taux_citations_inexistantes"] == 0.5
    assert m["taux_citations_version_omise"] == 0.5
    assert m["taux_citations_non_portantes"] == 0.0
    assert m["par_question"] == {"q1": "version_omise", "q2": "record_inexistant"}


# Cas RÉEL, repris tel quel de la première campagne sur dev : l'unique citation classée
# « non portante » sur 422. Le corpus écrit `l’actif` (apostrophe courbe), le modèle a
# écrit `l'actif` (droite) tout en écrivant correctement `l’écart d’acquisition` deux mots
# plus loin. Un caractère d'apostrophe ne fait pas dire autre chose à un article.
_TEXTE_212_1 = ("Ainsi, le fonds commercial acquis, évalué par différence, est inscrit à "
                "l’actif dans les comptes individuels; il en est de même de l’écart "
                "d’acquisition dans les comptes consolidés.")
_EXTRAIT_212_1 = ("le fonds commercial acquis, évalué par différence, est inscrit à "
                  "l'actif dans les comptes individuels; il en est de même de l’écart "
                  "d’acquisition dans les comptes consolidés")


@pytest.fixture
def con_apostrophe(tmp_path):
    db = tmp_path / "apo.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE records (id TEXT PRIMARY KEY, texte TEXT)")
    c.execute("INSERT INTO records VALUES ('pcg-212-1-c1@2026-01-01', ?)", (_TEXTE_212_1,))
    c.commit()
    return c


def test_une_apostrophe_droite_ne_rend_pas_une_citation_non_portante(con_apostrophe):
    assert verifier_citation(
        con_apostrophe, "pcg-212-1-c1@2026-01-01", _EXTRAIT_212_1) == "ok"


def test_le_taux_brut_garde_lapostrophe_visible(con_apostrophe):
    """Le repli ne doit rien cacher : la correspondance BRUTE reste fausse, donc l'écart
    entre les deux taux publie exactement ce que la tolérance a laissé passer."""
    assert correspond_brut(
        con_apostrophe, "pcg-212-1-c1@2026-01-01", _EXTRAIT_212_1) is False


def test_le_repli_dapostrophe_ne_sauve_pas_un_extrait_faux(con_apostrophe):
    """La tolérance porte sur la typographie, pas sur le contenu."""
    assert verifier_citation(
        con_apostrophe, "pcg-212-1-c1@2026-01-01",
        "le fonds commercial acquis est inscrit en charges de l'exercice") == "extrait_absent"
