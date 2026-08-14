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
