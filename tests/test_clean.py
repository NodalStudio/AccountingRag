from accounting_rag.clean import join_lines


def test_tiret_conserve_articule():
    """Le tiret de fin de ligne est CONSERVÉ (Word ne coupe pas les mots)"""
    assert join_lines(["...articles 214-15 à 214-", "18."]) == "...articles 214-15 à 214-18."


def test_tiret_conserve_micro():
    """Tiret dans un composé est préservé"""
    assert join_lines(["les micro-", "entreprises"]) == "les micro-entreprises"


def test_paragraphes():
    txt = join_lines(["Première phrase.", "Deuxième phrase", "qui continue."])
    assert txt == "Première phrase.\nDeuxième phrase qui continue."


def test_puce_conservee():
    txt = join_lines(["traitement suivant :", "- les frais de constitution ;", "- les frais de fusion."])
    assert txt == "traitement suivant :\n- les frais de constitution ;\n- les frais de fusion."


def test_puce_vide():
    """Puce seule est transformée en préfixe de la ligne suivante"""
    txt = join_lines(["Traitement :", "-", "suite du texte"])
    assert txt == "Traitement :\n- suite du texte"


def test_espace_insecable():
    assert join_lines(["n° 2006-C"]) == "n° 2006-C"


def test_tiret_separateur_p544():
    """F3-bis: Page 544 — tiret séparateur (espace+tiret) conserve l'espace"""
    txt = join_lines(
        ["rémunération en cas de maladie, d'accident ou de maternité, le compte 439 « Organismes sociaux -",
         "Produits à recevoir »"]
    )
    assert "sociaux - Produits" in txt


def test_tiret_separateur_p609():
    """F3-bis: Page 609 — formule avec tiret opérateur"""
    txt = join_lines(["(9) = (1) + (2) -", "(3+4+5+6+7+8)"])
    assert txt == "(9) = (1) + (2) - (3+4+5+6+7+8)"


def test_puces_multiples_p250():
    """F7-bis: Page 250 — séquence de tirets dans tableau (cellules vides)"""
    txt = join_lines(["4 060 000", "-", "-", "-", "11 710 000"])
    assert txt == "4 060 000\n-\n-\n- 11 710 000"


def test_nbsp_normalization():
    """F9-bis: NBSP et NNBSP normalisés en espace"""
    # Test avec U+00A0 (NBSP) et U+202F (NNBSP)
    txt = join_lines(["n° 2014-03 et n° 2016-07"])
    # Vérifier qu'aucun NBSP/NNBSP ne subsiste
    assert " " not in txt
    assert " " not in txt
    assert txt == "n° 2014-03 et n° 2016-07"
