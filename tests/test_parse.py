from accounting_rag.parse import parse


def test_pages_40_41(recueil_path):
    records, anomalies = parse(recueil_path)
    by_id = {r.id: r for r in records}

    art = by_id["pcg-212-5@2026-01-01"]
    assert art.type == "reglementaire"
    assert art.texte.startswith("Le titulaire d'un contrat de crédit-bail")
    assert art.page_debut == 40
    assert "Sous-section" not in art.texte           # les titres ne fuient pas dans le texte

    # le commentaire qui suit 212-5 lui est rattaché, avec sa provenance
    c = by_id.get("pcg-212-5-c1@2026-01-01")
    assert c is not None and c.type == "commentaire_ANC"
    assert c.opposable is False
    assert "avis-cu-2006-C" in [r.cible for r in c.renvois]

    # la série d'articles de la sous-section 2 est présente
    for num in ("212-6", "212-7", "212-8", "212-9", "212-10", "212-11"):
        assert f"pcg-{num}@2026-01-01" in by_id

    # le chemin porte la sous-section pour 212-6
    assert "Sous-section 2" in by_id["pcg-212-6@2026-01-01"].chemin


def test_articles_reglementaires_opposables_non(recueil_path):
    records, _ = parse(recueil_path)
    assert all(r.opposable is False for r in records)  # rien d'opposable dans l'ANC (≠ BOFiP)
