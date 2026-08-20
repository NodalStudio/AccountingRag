"""Tests du contrôle des chiffres du correctif de fusion.

Loi 5, appliquée à un contrôle qui est lui-même un contrôle : un vérificateur que
personne n'a vu refuser un artefact faux ne prouve rien, et ce dépôt a déjà livré un
contrôle de reproduction qui validait le bug qu'il devait attraper. Chaque test ci-dessous
corrompt l'artefact d'UNE façon et exige le refus.

La fixture est construite en appelant la VRAIE fonction de mesure pour la partie `marge`,
et non en recopiant à la main un schéma supposé — c'est la correction directe du défaut
du jalon 4, où sept tests sont passés contre une clé qui n'existait pas parce que la
fixture reproduisait la faute du code appelant.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _charger(nom, chemin):
    s = importlib.util.spec_from_file_location(nom, ROOT / chemin)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


ctrl = _charger("controle_chiffres_jalon3_fix", "scripts/controle_chiffres_jalon3_fix.py")
abl = _charger("ablations_fusion", "scripts/ablations_fusion.py")


class FauxSearcher:
    def __init__(self, plans):
        self._plans = plans

    def avant_rerank(self, question, mode="hybrid"):
        routes, classes = self._plans[question]
        return [{"record_id": r} for r in routes], list(classes), {}


def _marge_reelle(rangs_voulus: dict[str, int | None]) -> dict:
    """Fabrique une `marge` en passant par la vraie mesure, jamais à la main."""
    plans, questions = {}, []
    for qid, rang in rangs_voulus.items():
        gold = f"pcg-{qid}-1"
        classes = [f"pcg-bourrage-{j}@2026-01-01" for j in range(60)]
        if rang is not None:
            classes[rang - 1] = f"{gold}@2026-01-01"
        plans[qid] = ([], classes)
        questions.append({"id": qid, "question": qid, "citations": [gold],
                          "categorie": "regle"})
    return abl.marge_avant_eviction(FauxSearcher(plans), questions, "hybrid")


def _artefact() -> dict:
    """Campagne minuscule mais structurellement conforme : 4 questions, 2 configs."""
    ref_pq = {"q1": 1.0, "q2": 1.0, "q3": 0.0, "q4": 0.0}
    cfg_pq = {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 0.0}
    rangs_ref = {"q1": 1, "q2": 3, "q3": 30, "q4": None}
    rangs_cfg = {"q1": 1, "q2": 2, "q3": 8, "q4": None}
    return {
        "split": "dev",
        "n": 4,
        "contextes": {
            "hybrid": {
                "reference": {
                    "recall@10": round(sum(ref_pq.values()) / 4, 3),
                    "par_question": ref_pq,
                    "marge": _marge_reelle(rangs_ref),
                },
                "poids_consensus=0.1": {
                    "recall@10": round(sum(cfg_pq.values()) / 4, 3),
                    "par_question": cfg_pq,
                    "marge": _marge_reelle(rangs_cfg),
                    "bootstrap_vs_reference": {"delta": 0.25, "p_amelioration": 0.99},
                    "pire_categorie": {"categorie": "", "delta": 0.0},
                    "adopte": True,
                },
            }
        },
    }


def _ecrit(tmp_path, monkeypatch, artefact, rapport="") -> None:
    mesures = tmp_path / "mesures"
    mesures.mkdir()
    (mesures / "fusion_dev.json").write_text(json.dumps(artefact), encoding="utf-8")
    rap = tmp_path / "rapport.md"
    rap.write_text(rapport, encoding="utf-8")
    monkeypatch.setattr(ctrl, "MESURES", mesures)
    monkeypatch.setattr(ctrl, "RAPPORT", rap)


def test_un_artefact_coherent_passe(tmp_path, monkeypatch):
    _ecrit(tmp_path, monkeypatch, _artefact())
    ctrl.main()  # ne lève pas, ne sort pas


# --- les quatre corruptions ----------------------------------------------------------

def test_refuse_un_recall_qui_ne_correspond_pas_a_ses_questions(tmp_path, monkeypatch):
    a = _artefact()
    a["contextes"]["hybrid"]["reference"]["recall@10"] = 0.9
    _ecrit(tmp_path, monkeypatch, a)
    with pytest.raises(SystemExit) as e:
        ctrl.main()
    assert e.value.code == 1


def test_refuse_une_part_de_marge_qui_ne_correspond_pas_aux_rangs(tmp_path, monkeypatch):
    a = _artefact()
    a["contextes"]["hybrid"]["reference"]["marge"]["part_au_dela_de_25"] = 0.0
    _ecrit(tmp_path, monkeypatch, a)
    with pytest.raises(SystemExit) as e:
        ctrl.main()
    assert e.value.code == 1


def test_refuse_un_delta_de_bootstrap_qui_ne_correspond_pas_aux_vecteurs(tmp_path, monkeypatch):
    a = _artefact()
    a["contextes"]["hybrid"]["poids_consensus=0.1"]["bootstrap_vs_reference"]["delta"] = 0.5
    _ecrit(tmp_path, monkeypatch, a)
    with pytest.raises(SystemExit) as e:
        ctrl.main()
    assert e.value.code == 1


def test_refuse_une_adoption_que_le_critere_ne_soutient_pas(tmp_path, monkeypatch):
    """Le contrôle le plus important : c'est le drapeau qui décide de ce qu'on livre."""
    a = _artefact()
    a["contextes"]["hybrid"]["poids_consensus=0.1"]["bootstrap_vs_reference"]["p_amelioration"] = 0.5
    _ecrit(tmp_path, monkeypatch, a)
    with pytest.raises(SystemExit) as e:
        ctrl.main()
    assert e.value.code == 1


def test_refuse_une_adoption_malgre_une_categorie_qui_perd_trop(tmp_path, monkeypatch):
    a = _artefact()
    a["contextes"]["hybrid"]["poids_consensus=0.1"]["pire_categorie"]["delta"] = -0.2
    _ecrit(tmp_path, monkeypatch, a)
    with pytest.raises(SystemExit) as e:
        ctrl.main()
    assert e.value.code == 1


def test_refuse_un_compte_de_golds_absents_faux(tmp_path, monkeypatch):
    a = _artefact()
    a["contextes"]["hybrid"]["reference"]["marge"]["n_gold_absent_de_la_fusion"] = 0
    _ecrit(tmp_path, monkeypatch, a)
    with pytest.raises(SystemExit) as e:
        ctrl.main()
    assert e.value.code == 1


def test_refuse_un_repertoire_sans_artefact(tmp_path, monkeypatch):
    mesures = tmp_path / "vide"
    mesures.mkdir()
    monkeypatch.setattr(ctrl, "MESURES", mesures)
    with pytest.raises(SystemExit) as e:
        ctrl.main()
    assert e.value.code == 1


# --- confrontation au texte ----------------------------------------------------------

def test_un_chiffre_ecrit_a_la_francaise_est_reconnu_dans_le_rapport():
    trouves, manquants = ctrl.confronter_au_rapport(
        [0.747, 0.0323, 1.0], "recall 0,747 pour un delta de +0,0323 et une marge de 1,0")
    assert trouves == 3 and manquants == []


def test_un_chiffre_absent_du_rapport_est_remonte_sans_faire_echouer():
    """Le rapport ne cite pas tout le JSON ; l'échec est réservé aux incohérences."""
    trouves, manquants = ctrl.confronter_au_rapport([0.747, 0.123], "recall 0,747")
    assert trouves == 1 and manquants == [0.123]


def test_la_fixture_porte_le_meme_schema_de_marge_que_la_vraie_mesure():
    """Le test qui manquait au jalon 4 : la fixture est PRODUITE par la vraie fonction."""
    vraie = _marge_reelle({"q1": 2, "q2": None})
    assert vraie["rangs"] == {"q1": 2, "q2": None}
    assert vraie["n_gold_absent_de_la_fusion"] == 1
    assert "part_au_dela_de_25" in vraie and "n_exposees" in vraie


def test_un_negatif_ecrit_avec_le_moins_typographique_est_reconnu():
    """Le rapport écrit « −0,0122 » (U+2212), pas « -0,0122 ». Sans cette équivalence le
    contrôle signalait comme absent un chiffre bel et bien publié, et un contrôle qui
    crie au loup finit par ne plus être lu."""
    trouves, manquants = ctrl.confronter_au_rapport(
        [-0.0122], "pire catégorie −0,0122 sur vocabulaire_courant")
    assert trouves == 1 and manquants == []


def test_le_tiret_ascii_reste_reconnu():
    trouves, _ = ctrl.confronter_au_rapport([-0.0122], "delta -0,0122")
    assert trouves == 1
