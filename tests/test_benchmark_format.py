import json
import sqlite3
from pathlib import Path
import pytest

DEV = Path("benchmark/dev.jsonl")
TEST = Path("benchmark/test.jsonl")
FILES = [DEV, TEST]
DB = Path("data/corpus.db")

EXPECTED_COUNTS_PAR_FICHIER = {DEV: 61, TEST: 29}
EXPECTED_COUNTS_PAR_CATEGORIE = {
    "reference_directe": 10,
    "regle": 35,
    "vocabulaire_courant": 45,
}
# Stratification par fichier × catégorie (split gelé au jalon 2.5, cf. benchmark/README.md) :
# protège le gel du split lui-même, pas seulement les effectifs agrégés ci-dessus.
EXPECTED_COUNTS_PAR_FICHIER_ET_CATEGORIE = {
    DEV: {"reference_directe": 7, "regle": 23, "vocabulaire_courant": 31},
    TEST: {"reference_directe": 3, "regle": 12, "vocabulaire_courant": 14},
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
    assert total == 90
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
