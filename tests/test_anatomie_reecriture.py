"""Tests de l'anatomie du second correctif du jalon 3.

Les dérivations communes (marge parmi les présents, groupes de vecteurs identiques,
bascules) vivent dans le paquet et sont testées avec l'anatomie de fusion. Ce fichier
couvre ce qui est propre à ce correctif — et surtout la distinction sur laquelle il
repose : un gold ABSENT du pool n'est pas un gold mal classé, et les confondre effacerait
exactement le défaut que le levier vise.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "anatomie_reecriture", ROOT / "scripts/anatomie_reecriture.py")
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)


def _marge(rangs):
    return {"rangs": rangs, "n_gold_absent_de_la_fusion": sum(v is None for v in rangs.values())}


def _cfg(recall, pq, rangs, **extra):
    return {"recall@10": recall, "par_question": pq, "marge": _marge(rangs), **extra}


def _brut(rangs_ref, rangs_cfg):
    pq = {q: 1.0 for q in rangs_ref}
    return {"split": "dev", "n": len(pq), "contextes": {
        "reecriture": {
            "reference": _cfg(1.0, pq, rangs_ref),
            "poids_question=2": _cfg(1.0, pq, rangs_cfg,
                                     bootstrap_vs_reference={"p_amelioration": 0.99},
                                     adopte=True),
        },
        "livree": {
            "reference": _cfg(1.0, pq, rangs_ref),
            "poids_question=2": _cfg(1.0, pq, rangs_cfg,
                                     bootstrap_vs_reference={"p_amelioration": 0.4},
                                     adopte=False),
        },
    }}


def _lance(tmp_path, monkeypatch, brut):
    d = tmp_path / "m"
    d.mkdir()
    (d / "reecriture_dev.json").write_text(json.dumps(brut), encoding="utf-8")
    monkeypatch.setattr(ana, "IN_DIR", d)
    monkeypatch.setattr(ana, "OUT_DIR", d)
    monkeypatch.setattr("sys.argv", ["anatomie_reecriture.py", "--split", "dev"])
    ana.main()
    return json.loads((d / "anatomie_reecriture_dev.json").read_text(encoding="utf-8"))


def test_un_gold_hors_du_pool_est_rapporte_comme_tel_et_non_comme_un_rang():
    """La distinction qui porte tout le correctif : q080 casse parce que son gold SORT du
    pool, pas parce qu'il est mal classé. Publier un rang là où il n'y en a pas ferait
    disparaître le défaut mesuré."""
    m = ana.marge_parmi_les_golds_presents(_marge({"q080": None, "q025": 22}))
    assert m["n_golds_absents_du_pool"] == 1
    assert m["n_golds_presents_dans_la_fusion"] == 1


def test_les_trois_temoins_sont_ceux_etablis_au_jalon3():
    """Choisis dans `cloture_dev.json` (B vs C), pas après avoir vu le résultat."""
    cloture = json.loads(
        (ROOT / "docs/mesures/jalon3/cloture_dev.json").read_text(encoding="utf-8"))["configs"]
    B, C = cloture["B_rerank_jalon25"]["par_question"], \
        cloture["C_reecriture_rerank_jalon3"]["par_question"]
    assert set(ana.TEMOINS) == {q for q in B if C[q] < B[q]}


def test_les_rangs_des_temoins_sont_publies_y_compris_absents(tmp_path, monkeypatch):
    out = _lance(tmp_path, monkeypatch,
                 _brut({"q008": 1, "q025": 22, "q080": None},
                       {"q008": 1, "q025": 8, "q080": 26}))
    ref = out["contextes"]["reecriture"]["par_configuration"]["reference"]["rangs_des_temoins"]
    cfg = out["contextes"]["reecriture"]["par_configuration"]["poids_question=2"]["rangs_des_temoins"]
    assert ref == {"q008": 1, "q025": 22, "q080": None}
    assert cfg == {"q008": 1, "q025": 8, "q080": 26}


def test_une_question_temoin_routee_est_nommee_et_pas_confondue_avec_un_absent(tmp_path, monkeypatch):
    """Une question routée ne figure pas dans `rangs` : sans distinction explicite, elle
    se lirait comme un gold hors du pool, c'est-à-dire comme un échec."""
    out = _lance(tmp_path, monkeypatch,
                 _brut({"q008": 1, "q025": 3}, {"q008": 1, "q025": 3}))
    temoins = out["contextes"]["livree"]["par_configuration"]["reference"]["rangs_des_temoins"]
    assert temoins["q080"] == "routée"


def test_le_contexte_qui_decide_est_la_configuration_livree(tmp_path, monkeypatch):
    out = _lance(tmp_path, monkeypatch,
                 _brut({"q008": 1, "q025": 22, "q080": None},
                       {"q008": 1, "q025": 8, "q080": 26}))
    d = out["decision"]
    assert d["contexte_qui_decide"] == "livree"
    assert d["adoptees_dans_le_contexte_qui_decide"] == []
    assert d["franchissent_le_seuil_dans_le_contexte_mecanisme"] == ["poids_question=2"]
