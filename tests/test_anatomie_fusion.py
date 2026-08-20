"""Tests de l'anatomie du correctif de fusion.

Ce script ne mesure rien : il relit la campagne et en dérive des chiffres publiés. C'est
exactement le profil d'erreur le plus coûteux du dépôt — un agrégat faux calculé sur des
données justes, qu'aucun contrôle amont ne peut attraper parce que la campagne, elle, est
correcte. Le jalon 4 a livré un contrôle des chiffres qui lisait une clé inexistante,
avec une fixture portant la même faute : sept tests verts sur un schéma qui n'existait
pas. D'où le dernier test de ce fichier, qui compare la fixture à la vraie sortie.

La décomposition de la marge est le point sensible. `part_au_dela_de_25` mélange deux
causes distinctes — gold absent du pool (couverture) et gold présent mal classé
(classement) — et seule la seconde parle de la règle de fusion. Un test doit donc
distinguer les deux, sur un cas où les deux existent.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "anatomie_fusion", ROOT / "scripts/anatomie_fusion.py")
ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ana)


def _marge(rangs: dict) -> dict:
    """Construit une marge au format de `ablations_fusion.marge_avant_eviction`."""
    return {"rangs": rangs,
            "n_gold_absent_de_la_fusion": sum(v is None for v in rangs.values())}


# --- la décomposition qui sépare couverture et classement -----------------------------

def test_la_marge_parmi_les_presents_ignore_les_golds_absents_du_pool():
    """Le chiffre qui parle de la fusion ne doit compter que ce que la fusion peut classer."""
    m = ana.marge_parmi_les_golds_presents(_marge({"a": 1, "b": 30, "c": None, "d": None}))
    assert m["n_golds_presents_dans_la_fusion"] == 2
    assert m["n_golds_absents_du_pool"] == 2
    # 1 présent sur 2 au-delà de 25 — et non 3 sur 4, qui confondrait les deux causes.
    assert m["part_au_dela_de_25"] == 0.5
    assert m["n_au_dela_de_25"] == 1


def test_la_marge_parmi_les_presents_rapporte_ses_parts_aux_presents():
    m = ana.marge_parmi_les_golds_presents(_marge({f"q{i}": i for i in range(1, 41)}))
    assert m["n_golds_presents_dans_la_fusion"] == 40
    assert m["part_au_dela_de_10"] == round(30 / 40, 4)
    assert m["part_au_dela_de_25"] == round(15 / 40, 4)
    assert m["rang_max"] == 40 and m["rang_median"] == 20.5


def test_une_marge_sans_aucun_gold_present_ne_fabrique_pas_de_zero():
    """`None`, jamais 0,0 : un taux non défini qui s'affiche à zéro se lit comme un
    sans-faute. Même règle qu'au jalon 4 pour les taux de citation."""
    m = ana.marge_parmi_les_golds_presents(_marge({"a": None}))
    assert m["part_au_dela_de_25"] is None
    assert m["rang_median"] is None
    assert m["n_golds_presents_dans_la_fusion"] == 0


# --- l'égalité entre configurations : recall identique n'est pas classement identique --

def test_deux_configurations_au_meme_recall_mais_aux_rangs_differents_sont_distinguees():
    """Le contrôle qui tranche la coïncidence de la grille. Sans lui, « les deux leviers
    sont équivalents » serait une conclusion tirée d'un agrégat trop grossier."""
    configs = {
        "a": {"par_question": {"q1": 1.0}, "marge": _marge({"q1": 3})},
        "b": {"par_question": {"q1": 1.0}, "marge": _marge({"q1": 7})},
    }
    assert ana.vecteurs_identiques(configs, "par_question") == [["a", "b"]]
    assert ana.vecteurs_identiques(configs, "rangs") == []


def test_deux_configurations_reellement_identiques_sont_groupees_sur_les_deux_criteres():
    configs = {
        "a": {"par_question": {"q1": 1.0}, "marge": _marge({"q1": 3})},
        "b": {"par_question": {"q1": 1.0}, "marge": _marge({"q1": 3})},
        "c": {"par_question": {"q1": 0.0}, "marge": _marge({"q1": None})},
    }
    assert ana.vecteurs_identiques(configs, "par_question") == [["a", "b"]]
    assert ana.vecteurs_identiques(configs, "rangs") == [["a", "b"]]


# --- bascules -------------------------------------------------------------------------

def test_les_bascules_comptent_les_demi_reparations_dans_les_deux_sens():
    """Une question à deux citations peut passer de 0,5 à 1,0 : c'est un gain, pas un
    statu quo. Le jalon 3 avait dû corriger un compte qui les ignorait."""
    ref = {"q1": 0.0, "q2": 0.5, "q3": 1.0, "q4": 1.0}
    cfg = {"q1": 1.0, "q2": 1.0, "q3": 0.5, "q4": 1.0}
    b = ana.bascules(ref, cfg)
    assert b == {"gagnees": ["q1", "q2"], "perdues": ["q3"],
                 "n_gagnees": 2, "n_perdues": 1}


def test_aucune_bascule_quand_rien_ne_change():
    ref = {"q1": 1.0, "q2": 0.0}
    b = ana.bascules(ref, dict(ref))
    assert b["n_gagnees"] == 0 and b["n_perdues"] == 0


# --- le schéma : la fixture doit ressembler à la vraie sortie -------------------------

def test_la_fixture_de_marge_porte_les_memes_cles_que_la_vraie_mesure():
    """Le test qui manquait au jalon 4. Sept tests y sont passés contre un schéma
    inexistant parce que la fixture reproduisait la faute du code appelant."""
    _s = importlib.util.spec_from_file_location(
        "ablations_fusion", ROOT / "scripts/ablations_fusion.py")
    abl = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(abl)

    class FauxSearcher:
        def avant_rerank(self, question, mode="hybrid"):
            return [], ["pcg-1-1@2026-01-01"], {}

    vraie = abl.marge_avant_eviction(
        FauxSearcher(),
        [{"id": "q1", "question": "x", "citations": ["pcg-1-1"], "categorie": "regle"}],
        "hybrid")
    fixture = _marge({"q1": 1})
    assert set(fixture) <= set(vraie), set(fixture) - set(vraie)
    # Et la fonction consommatrice tourne sur la VRAIE structure, pas seulement la fixture.
    assert ana.marge_parmi_les_golds_presents(vraie)["n_golds_presents_dans_la_fusion"] == 1
