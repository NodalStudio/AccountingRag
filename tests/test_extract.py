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
