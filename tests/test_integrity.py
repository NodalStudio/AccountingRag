from accounting_rag.integrity import check, report, _roman_to_int
from accounting_rag.model import Renvoi
from conftest import _rec


def test_roman():
    assert _roman_to_int("II") == 2 and _roman_to_int("IX") == 9


def test_incoherence_titre_detectee():
    # Article 312-1 : 1er chiffre = 3, doit être sous Titre III
    # Mais on le met sous Titre II → incohérence
    bad = _rec("pcg-312-1@2026-01-01", "312-1", chemin="Livre II > Titre II")
    assert any("Titre" in a.raison for a in check([bad]))


def test_dangling_renvoi():
    r = _rec(
        "pcg-212-1@2026-01-01",
        "212-1",
        renvois=[Renvoi("pcg-999-99", "interne")],
    )
    assert any("999-99" in a.raison for a in check([r]))


def test_report_contient_compteurs():
    r = _rec("pcg-212-1@2026-01-01", "212-1")
    md = report([r], [])
    assert "1 enregistrement" in md


def test_report_groupement_par_categorie():
    """Test que report() groupe par catégorie, pas par détail d'article."""
    from accounting_rag.parse import Anomalie

    # Créer plusieurs anomalies avec la même catégorie mais des détails différents
    anomalies = [
        Anomalie(1, "r1", "article 111-1 sous Titre 5 : détail", "incohérence de Titre"),
        Anomalie(2, "r2", "article 111-2 sous Titre 5 : détail", "incohérence de Titre"),
        Anomalie(3, "r3", "article 201-3 sous Titre 6 : détail", "incohérence de Titre"),
        Anomalie(4, "r4", "renvoi sans cible : 999-99", "renvoi interne sans cible"),
        Anomalie(5, "r5", "renvoi sans cible : 888-88", "renvoi interne sans cible"),
    ]
    r = _rec("pcg-212-1@2026-01-01", "212-1")
    md = report([r], anomalies)

    # Vérifier qu'il n'y a que 2 rubriques (## ...), pas 4+
    lines = md.split("\n")
    headers = [line for line in lines if line.startswith("##")]
    assert len(headers) == 2, f"Attendu 2 rubriques (incohérence de Titre, renvoi interne sans cible), trouvé {len(headers)}: {headers}"
    assert any("incohérence de Titre" in h for h in headers)
    assert any("renvoi interne sans cible" in h for h in headers)
