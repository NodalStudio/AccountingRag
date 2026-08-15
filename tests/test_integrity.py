from accounting_rag.integrity import check, report, _roman_to_int
from accounting_rag.model import Renvoi
from conftest import _rec


def test_roman():
    assert _roman_to_int("II") == 2 and _roman_to_int("IX") == 9


def test_incoherence_livre_detectee():
    bad = _rec("pcg-312-1@2026-01-01", "312-1", chemin="Livre II > Titre I")
    assert any("Livre" in a.raison for a in check([bad]))


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
