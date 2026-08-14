from accounting_rag.clean import join_lines


def test_cesure_recollee():
    assert join_lines(["les immobili-", "sations existantes."]) == "les immobilisations existantes."


def test_paragraphes():
    txt = join_lines(["Première phrase.", "Deuxième phrase", "qui continue."])
    assert txt == "Première phrase.\nDeuxième phrase qui continue."


def test_puce_conservee():
    txt = join_lines(["traitement suivant :", "- les frais de constitution ;", "- les frais de fusion."])
    assert txt == "traitement suivant :\n- les frais de constitution ;\n- les frais de fusion."


def test_espace_insecable():
    assert join_lines(["n° 2006-C"]) == "n° 2006-C"
