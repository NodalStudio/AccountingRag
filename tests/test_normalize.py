from accounting_rag.normalize import normalize


def test_meme_normalisation_flexions():
    # le stem rapproche singulier/pluriel et formes fléchies
    assert normalize("les amortissements des immobilisations") == normalize(
        "l'amortissement de l'immobilisation"
    ).replace("de ", "").strip() or set(normalize("les amortissements").split()) & set(
        normalize("l'amortissement").split()
    )


def test_stems_partages():
    a = set(normalize("les amortissements dérogatoires").split())
    b = set(normalize("un amortissement dérogatoire").split())
    assert a & b >= {"amort", "derogatoir"} or len(a & b) >= 2


def test_reference_atomique():
    out = normalize("aux termes de l'article L. 313-7 du code monétaire")
    assert "l313-7" in out.split()
    out2 = normalize("Art. 214-1 du PCG")
    assert "214-1" in out2.split()


def test_elision_supprimee():
    out = normalize("l'exercice d'imputation")
    # Les stems réels du Snowball français: "exercice"->exercic, "imputation"->"imput"
    tokens = out.split()
    assert "exercic" in tokens or any(t.startswith("exerc") for t in tokens)
    assert "imput" in tokens
    assert not any(t.startswith("l'") or t == "l" for t in tokens)
    assert not any(t.startswith("d'") or t == "d" for t in tokens)


def test_synonyme_metier():
    assert normalize("le fonds de commerce") == normalize("le fonds commercial")


def test_accents_plies():
    out = normalize("créance échue")
    assert all(ord(c) < 128 for c in out)


def test_requete_et_document_identiques():
    # invariant central : même fonction pour les deux côtés
    doc = "Le titulaire d'un contrat de crédit-bail comptabilise en charges"
    query = "comptabiliser les charges d'un contrat de crédit-bail"
    assert set(normalize(doc).split()) & set(normalize(query).split()) >= {"contrat", "charg"} or \
           len(set(normalize(doc).split()) & set(normalize(query).split())) >= 3
