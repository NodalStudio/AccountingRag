from accounting_rag.chunks import make_chunks


def test_record_court_un_seul_chunk():
    chunks = make_chunks("pcg-x@e", "Une phrase courte.")
    assert chunks == [("pcg-x@e::0", 0, "Une phrase courte.")]


def test_decoupe_aux_paragraphes():
    texte = "\n".join(f"Paragraphe {i} " + "mot " * 100 for i in range(5))
    chunks = make_chunks("pcg-x@e", texte, max_words=220)
    assert len(chunks) >= 2
    assert all(len(c[2].split()) <= 220 for c in chunks)
    assert [c[1] for c in chunks] == list(range(len(chunks)))


def test_paragraphe_geant_fenetre_glissante():
    texte = "mot " * 1000
    chunks = make_chunks("pcg-x@e", texte.strip(), max_words=220, overlap=40)
    assert len(chunks) >= 5
    # le chevauchement existe : la fin d'un chunk se retrouve au début du suivant
    a, b = chunks[0][2].split(), chunks[1][2].split()
    assert a[-40:] == b[:40]


def test_couverture_totale():
    texte = "\n".join(f"Alinea {i} unique_{i}" for i in range(30))
    chunks = make_chunks("pcg-x@e", texte, max_words=20)
    joined = " ".join(c[2] for c in chunks)
    assert all(f"unique_{i}" in joined for i in range(30))
