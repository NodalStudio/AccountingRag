from accounting_rag.refs import extract_renvois


def _cibles(txt):
    return {(r.cible, r.famille) for r in extract_renvois(txt)}


def test_interne_pluriel():
    c = _cibles("définis aux articles 212-1 et 212-2.")
    assert ("pcg-212-1", "interne") in c and ("pcg-212-2", "interne") in c


def test_interne_cf():
    assert ("pcg-1214-48", "interne") in _cibles("Cf. article 1214-48")


def test_externe_comofi():
    c = _cibles("à l'article L. 313-7 du Code monétaire et financier")
    assert ("legi-L313-7-comofi", "externe_legal") in c


def test_historique_crc_et_avis():
    txt = ("du règlement n° 2004-06 du CRC ; Avis CNC n° 2004-15 du 23 juin 2004 ; "
           "Avis CU n° 2006-C du 4 octobre 2006")
    c = _cibles(txt)
    assert ("crc-2004-06", "historique") in c
    assert ("avis-cnc-2004-15", "historique") in c
    assert ("avis-cu-2006-C", "historique") in c
