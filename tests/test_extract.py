from accounting_rag.extract import extract_lines


def test_page40_contains_article_header(recueil_path):
    lines = extract_lines(recueil_path, pages=range(39, 40))  # page 40 (0-indexée 39)
    headers = [l for l in lines if l.text.startswith("Art. 212-5")]
    assert len(headers) == 1
    h = headers[0]
    assert h.bold is True
    assert abs(h.size - 10.0) < 0.1
    assert h.page == 40


def test_superscript_merged(recueil_path):
    lines = extract_lines(recueil_path, pages=range(39, 40))
    footer = [l for l in lines if "1er janvier 2026" in l.text.replace(" ", " ")]
    assert footer, "l'exposant 'er' doit être recollé à '1'"


def test_regression_page88_space_before_reference(recueil_path):
    """Regression: espace restauré avant appel de note — p.88 'propres (1)' au lieu de 'propres(1)'"""
    lines = extract_lines(recueil_path, pages=range(87, 88))
    propres_lines = [l for l in lines if "propres" in l.text and "(1)" in l.text]
    assert propres_lines, "Ligne 'propres (1)' non trouvée p.88"
    # Le span "propres " devrait être séparé de "(1)" par un espace
    for l in propres_lines:
        assert "propres (1)" in l.text or "propres (1)" in l.text.replace(" ", " "), \
            f"Espace manquant avant (1) : {l.text}"


def test_regression_page25_bullet_normalization(recueil_path):
    """Regression: puce au début de ligne normalisée en '- ' — p.25 '- le bénéfice ou la perte de l'exercice,'"""
    lines = extract_lines(recueil_path, pages=range(24, 25))
    bullet_lines = [l for l in lines if "le bénéfice ou la perte de l'exercice" in l.text]
    assert bullet_lines, "Ligne avec 'le bénéfice ou la perte' non trouvée p.25"
    # La ligne doit commencer par '- ' et non '•'
    for l in bullet_lines:
        assert l.text.startswith("- "), \
            f"Puce non normalisée en '- ' : {l.text}"


def test_regression_page25_chapter_non_bold(recueil_path):
    """Regression: titre de haut niveau non gras classé comme SECTION_HEADER — p.25 'Chapitre I', size=12.0, bold=False"""
    lines = extract_lines(recueil_path, pages=range(24, 25))
    chapter_lines = [l for l in lines if l.text.startswith("Chapitre I")]
    assert chapter_lines, "Chapitre I non trouvé p.25"
    h = chapter_lines[0]
    assert h.size == 12.0, f"Size incorrect : {h.size}"
    assert h.bold is False, f"Ne doit pas être gras : {h.bold}"


def test_regression_fix2_page5_small_caps_chapitre(recueil_path):
    """Fix Round 2: petites capitales 'CHAPITRE' non fragmentées — p.5 'CHAPITRE I – OBJET DE LA COMPTABILITÉ'"""
    lines = extract_lines(recueil_path, pages=range(4, 5))
    chapitre_lines = [l for l in lines if 'CHAPITRE' in l.text and 'OBJET' in l.text]
    assert chapitre_lines, "CHAPITRE I line non trouvée p.5"
    for l in chapitre_lines:
        # Doit contenir 'CHAPITRE' sans espaces internes (pas 'C HAPITRE')
        assert 'CHAPITRE' in l.text, f"CHAPITRE fragmenté : {l.text}"
        assert 'C HAPITRE' not in l.text, f"Petits espaces parasites détectés : {l.text}"


def test_regression_fix2_page480_co2_intact(recueil_path):
    """Fix Round 2: indices chimiques 'CO2' non fragmentés — p.480 'gestion du risque CO2'"""
    lines = extract_lines(recueil_path, pages=range(479, 480))
    co2_lines = [l for l in lines if 'CO' in l.text and '2' in l.text and 'risque' in l.text]
    assert co2_lines, "Ligne CO2 non trouvée p.480"
    for l in co2_lines:
        # Doit contenir 'CO2' ou au minimum pas d'espace entre CO et 2
        assert 'CO2' in l.text or 'CO 2' not in l.text, f"CO2 fragmenté : {l.text}"


def test_regression_fix2_page313_bilan(recueil_path):
    """Fix Round 2: petites capitales 'Bilan' non fragmentées — p.313 'Bilan'"""
    lines = extract_lines(recueil_path, pages=range(312, 313))
    bilan_lines = [l for l in lines if l.text.startswith('Bilan')]
    assert bilan_lines, "Bilan non trouvé p.313"
    for l in bilan_lines:
        # Doit contenir 'Bilan' sans espace interne
        assert 'Bilan' in l.text, f"Bilan fragmenté : {l.text}"
        assert 'B ilan' not in l.text, f"Espaces parasites : {l.text}"


def test_regression_fix3_page34_comptes_annuels(recueil_path):
    """Fix Round 3: spans blancs conservés — p.34 'des comptes annuels' (pas 'comptesannuels')"""
    lines = extract_lines(recueil_path, pages=range(33, 34))
    comptes_lines = [l for l in lines if 'comptes' in l.text and 'annuels' in l.text]
    assert comptes_lines, "Ligne 'comptes annuels' non trouvée p.34"
    for l in comptes_lines:
        assert 'comptes annuels' in l.text or 'comptes  annuels' in l.text, f"Mots fusionnés : {l.text}"
        assert 'comptesannuels' not in l.text, f"Fusion indésirable : {l.text}"


def test_regression_fix3_page144_cet_evenement(recueil_path):
    """Fix Round 3: spans blancs conservés — p.144 'cet événement' (pas 'cetévénement')"""
    lines = extract_lines(recueil_path, pages=range(143, 144))
    evt_lines = [l for l in lines if 'cet' in l.text and 'événement' in l.text]
    assert evt_lines, "Ligne 'cet événement' non trouvée p.144"
    for l in evt_lines:
        assert 'cetévénement' not in l.text, f"Fusion indésirable : {l.text}"


def test_regression_fix3_page95_materiaux_extraits(recueil_path):
    """Fix Round 3: spans blancs conservés — p.95 'matériaux extraits'"""
    lines = extract_lines(recueil_path, pages=range(94, 95))
    mat_lines = [l for l in lines if 'matériaux' in l.text and 'extraits' in l.text]
    assert mat_lines, "Ligne 'matériaux extraits' non trouvée p.95"
    for l in mat_lines:
        assert 'matériauxextraits' not in l.text, f"Fusion indésirable : {l.text}"


def test_regression_fix4_page461_chiffre_affaires_pivoted_text(recueil_path):
    """Fix Round 4: texte pivoté à 90° — p.461 'Chiffre d'affaires hors taxes'
    (pas 'Chiffre d' affaires') : un span blanc de largeur ~nulle ne doit pas
    forcer d'espace au milieu d'un mot élidé."""
    lines = extract_lines(recueil_path, pages=range(460, 461))
    normalized = [l.text.replace("’", "'") for l in lines]
    matches = [t for t in normalized if "Chiffre d'affaires hors taxes" in t]
    assert matches, [t for t in normalized if "affaires" in t or "Chiffre" in t]
    for t in matches:
        assert "d' affaires" not in t, f"Mot élidé cassé : {t}"


def test_regression_fix4_page184_no_double_space(recueil_path):
    """Fix Round 4: p.184 '3.  Lorsque' → aucun double espace dans la ligne
    (span blanc suivi d'un span à espace de tête)."""
    lines = extract_lines(recueil_path, pages=range(183, 184))
    matches = [l for l in lines if "Lorsque" in l.text and l.text.lstrip().startswith("3.")]
    assert matches, [l.text for l in lines if "Lorsque" in l.text]
    for l in matches:
        assert "  " not in l.text, f"Double espace détecté : {l.text!r}"


def test_regression_fix4_page34_des_comptes_annuels_intact(recueil_path):
    """Fix Round 4 (non-régression): p.34 'des comptes annuels' reste séparé —
    la largeur du span blanc y est ~3pt, bien au-dessus du seuil de 0.15*size,
    donc il doit continuer à agir comme séparateur."""
    lines = extract_lines(recueil_path, pages=range(33, 34))
    comptes_lines = [l for l in lines if 'des comptes annuels' in l.text]
    assert comptes_lines, "Ligne 'des comptes annuels' non trouvée p.34"
