from accounting_rag.refs import extract_renvois


def _cibles(txt):
    return {(r.cible, r.famille) for r in extract_renvois(txt)}


def test_interne_pluriel():
    c = _cibles("définis aux articles 212-1 et 212-2.")
    assert ("pcg-212-1", "interne") in c and ("pcg-212-2", "interne") in c


def test_interne_cf():
    assert ("pcg-1214-48", "interne") in _cibles("Cf. article 1214-48")


def test_interne_plage_a():
    """F4: Articles internes avec plage utilisant 'à'"""
    c = _cibles("définis aux articles 212-1 à 212-4.")
    assert ("pcg-212-1", "interne") in c and ("pcg-212-4", "interne") in c


def test_externe_comofi():
    c = _cibles("à l'article L. 313-7 du Code monétaire et financier")
    assert ("legi-L313-7-comofi", "externe_legal") in c


def test_externe_comofi_bounded():
    """F1: Capture du code limitée à la whitelist (ne s'étend pas au texte suivant)"""
    txt = "du Code monétaire et financier et aux établissements de paiement. Article L. 511-1 du Code monétaire et financier"
    c = _cibles(txt)
    # Doit trouver une référence, pas s'étendre au delà de "Code monétaire et financier"
    assert any(cible.startswith("legi-") and cible.endswith("-comofi") for cible, _ in c)
    # Ne doit pas matcher l'établissement
    assert not any("établissements" in cible or "paiement" in cible for cible, _ in c)


def test_externe_articles_pluriel():
    """F2: Plural 'articles' avec liste de références"""
    txt = "dispositions des articles L. 123-16 et L. 123-16-1 du code de commerce"
    c = _cibles(txt)
    assert ("legi-L123-16-code-de-commerce", "externe_legal") in c
    assert ("legi-L123-16-1-code-de-commerce", "externe_legal") in c


def test_externe_articles_plage():
    """F2+F4: Plural articles avec plage 'à'"""
    txt = "les dispositions des articles L. 236-1 à L. 236-6 du code de commerce"
    c = _cibles(txt)
    assert ("legi-L236-1-code-de-commerce", "externe_legal") in c
    assert ("legi-L236-6-code-de-commerce", "externe_legal") in c


def test_meme_code():
    """F5: 'du même code' réutilise le dernier code vu"""
    txt = "article L. 521-1 du Code monétaire et financier et article L. 521-2 du même code."
    c = _cibles(txt)
    assert ("legi-L521-1-comofi", "externe_legal") in c
    assert ("legi-L521-2-comofi", "externe_legal") in c


def test_crc_ordre_inverse():
    """F6: CRC avec format '04-01' (pas seulement '2004-06')"""
    txt = "règlement CRC n° 04-01 du 4 mai 2004"
    c = _cibles(txt)
    assert ("crc-04-01", "historique") in c


def test_externe_sans_prefixe_code_civil():
    """Ruling 19 (F3): 'article 1844-5 du code civil' est externe, pas interne PCG"""
    c = _cibles("article 1844-5 du code civil")
    assert ("legi-1844-5-code-civil", "externe_legal") in c
    assert ("pcg-1844-5", "interne") not in c


def test_interne_inchange_apres_ruling19():
    """Ruling 19 (F3): les cas internes existants ne doivent pas régresser"""
    c1 = _cibles("définis aux articles 212-1 et 212-2.")
    assert ("pcg-212-1", "interne") in c1 and ("pcg-212-2", "interne") in c1

    c2 = _cibles("Cf. article 1214-48")
    assert ("pcg-1214-48", "interne") in c2


def test_externe_code_rural():
    """F7 (revue finale): code rural et de la pêche maritime whitelisté."""
    c = _cibles("les dispositions de l'article L. 511-4 du code rural et de la pêche maritime")
    assert ("legi-L511-4-code-rural", "externe_legal") in c


def test_externe_code_environnement_apostrophe_typographique():
    """F7: code de l'environnement, avec l'apostrophe typographique '’' du
    PDF source (« code de l’environnement ») — pas seulement l'apostrophe
    droite utilisée dans le code Python."""
    c = _cibles("prévues à l’article L. 229-7 du code de l’environnement")
    assert ("legi-L229-7-code-de-l-environnement", "externe_legal") in c


def test_externe_cpi():
    """F7: code de la propriété intellectuelle -> slug 'cpi'."""
    c = _cibles("mentionnées à l'article L. 112-2 du code de la propriété intellectuelle")
    assert ("legi-L112-2-cpi", "externe_legal") in c


def test_historique_crc_et_avis():
    txt = ("du règlement n° 2004-06 du CRC ; Avis CNC n° 2004-15 du 23 juin 2004 ; "
           "Avis CU n° 2006-C du 4 octobre 2006")
    c = _cibles(txt)
    assert ("crc-2004-06", "historique") in c
    assert ("avis-cnc-2004-15", "historique") in c
    assert ("avis-cu-2006-C", "historique") in c
