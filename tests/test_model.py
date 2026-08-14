from accounting_rag.model import Record, Renvoi


def test_record_to_dict():
    r = Record(
        id="pcg-212-5@2026-01-01", article="212-5",
        chemin="Livre II > Titre I > Chapitre II", texte="Le titulaire…",
        type="reglementaire", nature="comptable", opposable=False,
        valide_du="2026-01-01", valide_au=None, source_citation=None,
        page_debut=40, page_fin=40,
        renvois=[Renvoi("legi-L313-7-comofi", "externe_legal")],
    )
    d = r.to_dict()
    assert d["id"] == "pcg-212-5@2026-01-01"
    assert d["renvois"] == [{"cible": "legi-L313-7-comofi", "famille": "externe_legal"}]
