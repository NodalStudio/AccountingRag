import sqlite3
import pytest
from accounting_rag.citations import (
    EXTRAIT_MINIMUM, metriques, normaliser_pour_comparaison, verifier_citation)


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
