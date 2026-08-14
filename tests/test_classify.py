from accounting_rag.model import Line, Kind
from accounting_rag.classify import classify
from accounting_rag.extract import extract_lines


def L(text, size, bold=False, font="Tahoma", x=99.0):
    return Line(text=text, size=size, bold=bold, font=font, x=x, y=300.0, page=40)


def test_signatures_synthetiques():
    assert classify(L("Art. 212-5", 10.0, bold=True)) == Kind.ARTICLE_HEADER
    assert classify(L("Le titulaire d'un contrat…", 10.0)) == Kind.REGLEMENTAIRE
    assert classify(L("Les immobilisations exploitées…", 9.5)) == Kind.COMMENTAIRE
    assert classify(L("Exclusion des contrats – Avis CU n° 2006-C", 9.5, bold=True)) == Kind.COMMENTAIRE_TITRE
    assert classify(L("Sous-section 2 – Dispositions particulières", 10.0, bold=True)) == Kind.SECTION_HEADER
    assert classify(L("Chapitre IV – Immobilisations", 10.6, bold=True)) == Kind.SECTION_HEADER
    assert classify(L("RECUEIL DES NORMES COMPTABLES FRANÇAISES", 8.5)) == Kind.BRUIT
    assert classify(L("•", 10.0, font="Symbol")) == Kind.PUCE
    assert classify(L("-", 9.5, font="Calibri")) == Kind.PUCE


def test_page40_reelle_sans_inconnu(recueil_path):
    lines = extract_lines(recueil_path, pages=range(39, 41))
    kinds = [classify(l) for l in lines]
    assert Kind.INCONNU not in kinds, [l.text for l, k in zip(lines, kinds) if k == Kind.INCONNU]
