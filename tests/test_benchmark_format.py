import json
import sqlite3
from pathlib import Path
import pytest

DEV = Path("benchmark/dev.jsonl")
TEST = Path("benchmark/test.jsonl")
# Second split de validation, gelé au jalon 4 : jamais exécuté avant sa propre clôture.
# Il existe parce que la garantie de gel de `test.jsonl` s'use à chaque clôture — ce split
# a été exécuté deux fois (jalons 2.5 et 3) — et qu'il faut pouvoir le retirer du service.
VALIDATION = Path("benchmark/validation.jsonl")
FILES = [DEV, TEST, VALIDATION]
DB = Path("data/corpus.db")

# Effectifs après l'extension du jalon 4 (90 -> 150 questions). Les chiffres du jalon 3
# ne sont PAS comparables à ceux mesurés sur ce benchmark : l'effectif a changé.
EXPECTED_COUNTS_PAR_FICHIER = {DEV: 93, TEST: 29, VALIDATION: 28}
EXPECTED_COUNTS_PAR_CATEGORIE = {
    "reference_directe": 25,
    "regle": 61,
    "vocabulaire_courant": 64,
}
# Stratification par fichier × catégorie (split gelé au jalon 2.5, cf. benchmark/README.md) :
# protège le gel du split lui-même, pas seulement les effectifs agrégés ci-dessus.
EXPECTED_COUNTS_PAR_FICHIER_ET_CATEGORIE = {
    DEV: {'reference_directe': 15, 'regle': 37, 'vocabulaire_courant': 41},
    TEST: {'reference_directe': 3, 'regle': 12, 'vocabulaire_courant': 14},
    VALIDATION: {'reference_directe': 7, 'regle': 12, 'vocabulaire_courant': 9},
}


@pytest.mark.skipif(not DB.exists(), reason="corpus.db absent")
def test_format_et_citations_existantes():
    con = sqlite3.connect(DB)
    total = 0
    ids = set()
    par_categorie = {}
    par_fichier_et_categorie = {f: {} for f in FILES}
    apostrophe_typo = 0
    for f in FILES:
        assert f.exists(), f"{f} manquant"
        lignes = f.read_text(encoding="utf-8").splitlines()
        assert len(lignes) == EXPECTED_COUNTS_PAR_FICHIER[f], (
            f"{f} : {len(lignes)} questions, {EXPECTED_COUNTS_PAR_FICHIER[f]} attendues"
        )
        for line in lignes:
            q = json.loads(line)
            assert set(q) >= {"id", "question", "categorie", "citations"}
            assert q["categorie"] in {"reference_directe", "regle", "vocabulaire_courant"}
            assert q["id"] not in ids
            ids.add(q["id"])
            assert len(q["question"]) > 15
            assert q["citations"]
            for c in q["citations"]:
                n = con.execute(
                    "SELECT COUNT(*) FROM records WHERE id = ? OR id LIKE ? OR id LIKE ? OR id LIKE ?",
                    (c, c + "@%", c + "-c%", c + "#%"),
                ).fetchone()[0]
                assert n > 0, f"{q['id']}: citation {c} sans record"
            par_categorie[q["categorie"]] = par_categorie.get(q["categorie"], 0) + 1
            par_fichier_et_categorie[f][q["categorie"]] = (
                par_fichier_et_categorie[f].get(q["categorie"], 0) + 1
            )
            if "’" in q["question"]:
                apostrophe_typo += 1
            total += 1
    assert total == 150
    assert par_categorie == EXPECTED_COUNTS_PAR_CATEGORIE
    assert par_fichier_et_categorie == EXPECTED_COUNTS_PAR_FICHIER_ET_CATEGORIE, (
        "stratification dev/test par catégorie divergente du split gelé "
        f"(cf. benchmark/README.md) : {par_fichier_et_categorie}"
    )
    assert apostrophe_typo >= 20, (
        f"seulement {apostrophe_typo} questions avec l'apostrophe typographique U+2019, 20 attendues au minimum"
    )


ABSTENTION = Path("benchmark/abstention.jsonl")
RAISONS_ABSTENTION = {"hors_corpus", "fiscal_pas_comptable", "hors_perimetre"}


def _abstention() -> list[dict]:
    return [json.loads(l) for l in
            ABSTENTION.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_abstention_format():
    """Le split d'abstention n'a de sens que si AUCUNE question n'y a de gold.

    Une seule citation attendue trahirait une question dont le corpus porte la réponse, et
    le taux d'abstention correcte cesserait alors de mesurer une abstention correcte.
    """
    lignes = _abstention()
    assert len(lignes) >= 30, f"{len(lignes)} questions, 30 attendues au minimum"
    assert all(q["citations"] == [] for q in lignes), \
        "une question d'abstention porte un gold"
    assert all(q["attendu"] == "abstention" for q in lignes)
    assert len({q["id"] for q in lignes}) == len(lignes), "ids dupliqués"
    assert all(q["categorie"] == "abstention" for q in lignes)


def test_abstention_repartition_par_raison():
    """Les trois raisons testent trois défaillances différentes : un split qui en néglige
    une mesurerait un système sur deux tiers de son périmètre de sûreté."""
    import collections
    raisons = collections.Counter(q["raison"] for q in _abstention())
    assert set(raisons) == RAISONS_ABSTENTION, f"raisons inattendues : {set(raisons)}"
    assert min(raisons.values()) >= 8, f"répartition déséquilibrée : {dict(raisons)}"


def test_abstention_nempiete_pas_sur_les_autres_splits():
    """Un id partagé avec dev ou test ferait basculer une question mesurée d'un split à
    l'autre selon l'ordre de chargement."""
    autres = set()
    for f in FILES:
        autres |= {json.loads(l)["id"] for l in
                   f.read_text(encoding="utf-8").splitlines() if l.strip()}
    partages = {q["id"] for q in _abstention()} & autres
    assert not partages, f"ids partagés avec dev/test : {sorted(partages)}"


def test_abstention_chaque_question_justifie_son_absence_du_corpus():
    """Le champ `notes` porte la preuve que la question n'a pas de réponse dans le corpus.

    Sans lui, une question d'abstention est une affirmation non vérifiable : rien ne
    distingue « le corpus ne répond pas » de « je n'ai pas cherché ».
    """
    for q in _abstention():
        assert len(q.get("notes", "")) >= 60, \
            f"{q['id']} : notes trop courtes pour justifier l'absence de réponse"


def _toutes() -> list[dict]:
    out = []
    for f in FILES + [ABSTENTION]:
        out += [json.loads(l) for l in
                f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def test_aucune_question_dupliquee_entre_les_splits():
    """Un énoncé présent dans deux splits ferait mesurer deux fois la même chose, et
    ferait fuiter une question du split gelé vers le split de développement."""
    textes = [q["question"].strip() for q in _toutes()]
    doublons = {t for t in textes if textes.count(t) > 1}
    assert not doublons, f"énoncés dupliqués : {sorted(doublons)[:3]}"


def test_les_divergences_2058a_gardent_leur_moitie_fiscale_marquee():
    """Ces questions ont deux réponses : la comptable, dans le corpus et goldée, et la
    fiscale, absente. La moitié manquante est MARQUÉE, jamais inventée — c'est ce qui
    permettra de lire le gain le jour où le corpus fiscal arrive."""
    marquees = [q for q in _toutes() if "gold_fiscal" in q]
    assert marquees, "aucune question de divergence 2058-A"
    for q in marquees:
        assert q["gold_fiscal"] == "a_completer", f"{q['id']} : marqueur inattendu"
        assert q["citations"], f"{q['id']} : la moitié comptable doit être goldée"
        assert "divergences 2058-A" in q.get("notes", ""), \
            f"{q['id']} : le thème doit être traçable dans les notes"


def test_le_split_de_validation_couvre_les_trois_categories():
    """Un split de validation qui n'en couvrirait qu'une mesurerait le système sur une
    fraction de son périmètre, tout en portant le nom de validation."""
    lignes = [json.loads(l) for l in
              VALIDATION.read_text(encoding="utf-8").splitlines() if l.strip()]
    cats = {q["categorie"] for q in lignes}
    assert cats == {"reference_directe", "regle", "vocabulaire_courant"}, cats
    themes = {t for q in lignes for t in ("fusions", "divergences 2058-A")
              if t in q.get("notes", "")}
    assert themes == {"fusions", "divergences 2058-A"}, \
        f"le split de validation doit couvrir les nouvelles sources : {themes}"
