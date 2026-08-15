from accounting_rag.normalize import normalize


def test_meme_normalisation_flexions():
    # le stem rapproche singulier/pluriel et formes fléchies
    a = set(normalize("les amortissements des immobilisations").split())
    b = set(normalize("l'amortissement de l'immobilisation").split())
    assert a & b == {"amort", "de", "immobilis"}


def test_stems_partages():
    a = set(normalize("les amortissements dérogatoires").split())
    b = set(normalize("un amortissement dérogatoire").split())
    assert a & b == {"amort", "derogatoir"}


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
    assert set(normalize(doc).split()) & set(normalize(query).split()) == {
        "comptabilis", "le", "charg", "un", "contrat", "de", "credit-bail",
    }


def test_apostrophe_typographique_equivalente_ascii():
    assert normalize("l’exercice") == normalize("l'exercice")
    assert normalize("d’amortissement") == normalize("d'amortissement")


def test_flexions_accentuees_fusionnent():
    # Le stemmer Snowball français doit recevoir du texte ACCENTUÉ (pas déjà plié) :
    # généré/génération/générer stemment tous en "géner", plié en "gener".
    assert normalize("généré") == normalize("génération") == normalize("générer") == "gener"


def test_amortissement_degressif_et_derogatoire_distincts():
    # F1: Amortissement dégressif (mode de calcul) ≠ Amortissement dérogatoire (provision réglementée)
    # Concepts distincts du plan comptable — ne pas les fusionner (risque de biais retrieval)
    degressif = set(normalize("amortissement dégressif").split())
    derogatoire = set(normalize("amortissement dérogatoire").split())
    assert degressif != derogatoire, "Amortissement dégressif et dérogatoire doivent rester distincts"


def test_stock_options_couverture_droits_francais():
    # F2: Stock-options doit couvrir à la fois options de souscription ET d'achat (droit français)
    stock_opts = set(normalize("stock-options").split())
    souscription = set(normalize("options de souscription d'actions").split())
    achat = set(normalize("options d'achat d'actions").split())
    # Les tokens de stock-options doivent avoir une intersection non vide avec chacun
    assert stock_opts & souscription, "stock-options doit couvrir options de souscription"
    assert stock_opts & achat, "stock-options doit couvrir options d'achat"
