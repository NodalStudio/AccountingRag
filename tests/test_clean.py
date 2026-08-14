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
