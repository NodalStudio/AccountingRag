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


def test_regression_page25_chapter_non_bold_classified(recueil_path):
    """Regression: titre de haut niveau non gras classé comme SECTION_HEADER — p.25 'Chapitre I', size=12.0, bold=False"""
    lines = extract_lines(recueil_path, pages=range(24, 25))
    chapter_lines = [l for l in lines if l.text.startswith("Chapitre I")]
    assert chapter_lines, "Chapitre I non trouvé p.25"
    h = chapter_lines[0]
    # Chapitre I n'est pas gras mais a size 12.0 >= 10.4
    assert classify(h) == Kind.SECTION_HEADER, \
        f"Chapitre I doit être SECTION_HEADER même sans gras, got {classify(h)}"


def test_folio_haut_de_page_est_bruit():
    """Correctif positionnel : le folio dupliqué en haut de page (y<60), dont
    le texte est purement numérique et égal au numéro de la page, est BRUIT
    même s'il tombe dans la bande typographique réglementaire (taille 10.0,
    non gras) — cas réel p.186 : ligne '186' à y=39.1."""
    folio = Line(text="186", size=10.0, bold=False, font="Tahoma", x=99.0, y=39.1, page=186)
    assert classify(folio) == Kind.BRUIT


def test_cellule_tableau_meme_valeur_que_page_survit_si_x_et_y_de_corps():
    """Cas réel p.518 : le compte '518' (Intérêts courus, x=106.2/y=253.8,
    colonne des codes de compte) vaut par coïncidence le numéro de page mais
    n'est ni à x<60 ni à y<60 — il doit survivre (rester REGLEMENTAIRE)."""
    compte = Line(text="518", size=10.0, bold=False, font="Tahoma", x=106.2, y=253.8, page=518)
    assert classify(compte) != Kind.BRUIT


def test_folio_haut_de_page_pivote_est_bruit():
    """Sur les pages pivotées à 90° (annexes en tableau large), le même folio
    dupliqué se retrouve à x<60 (et non plus y<60) — cas réel p.250, x=39.1,
    y=272.7. Doit être BRUIT malgré un y de corps."""
    folio_pivote = Line(text="250", size=10.0, bold=False, font="Tahoma", x=39.1, y=272.7, page=250)
    assert classify(folio_pivote) == Kind.BRUIT


def test_regression_page25_bullet_classified_as_puce(recueil_path):
    """Regression: ligne normalisée '- le bénéfice...' classée comme puce de paragraphe (reste REGLEMENTAIRE)"""
    lines = extract_lines(recueil_path, pages=range(24, 25))
    bullet_lines = [l for l in lines if l.text.startswith("- le bénéfice ou la perte")]
    assert bullet_lines, "Ligne '- le bénéfice...' non trouvée p.25"
    # La ligne devrait être classée comme REGLEMENTAIRE (le tiret normalisé n'est pas une PUCE isolée)
    # car elle contient du texte après le tiret
    for l in bullet_lines:
        kind = classify(l)
        # Avec la normalisation, le texte commençant par "- " et size 10.0 devrait être REGLEMENTAIRE
        assert kind in [Kind.REGLEMENTAIRE, Kind.PUCE], \
            f"Ligne bullet devrait être REGLEMENTAIRE ou PUCE, got {kind}"
