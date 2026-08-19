"""Tests du contrôle des chiffres publiés au jalon 4.

Le « CONTRÔLE DES CHIFFRES OK » affiché par le script est une affirmation qui doit pouvoir
échouer, et de plusieurs manières distinctes : un agrégat qui ne suit plus ses données
brutes, un chiffre corrigé dans le JSON mais pas dans le rapport, un artefact manquant, un
seuil de calibration déplacé après coup. Le dépôt a déjà livré un contrôle de reproduction
qui validait le bug qu'il devait attraper ; celui-ci a un test par mode d'échec.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "controle_chiffres_jalon4",
    Path(__file__).resolve().parent.parent / "scripts/controle_chiffres_jalon4.py")
ctrl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctrl)


def _metriques(n_ok=8, n_inexistant=1, n_court=1, n_abstention=1, n_sans=1):
    """Bloc `metriques` cohérent : les agrégats découlent des données brutes."""
    total = n_ok + n_inexistant + n_court
    verdicts = {"ok": n_ok, "record_inexistant": n_inexistant,
                "extrait_trop_court": n_court}
    pq = {}
    i = 0
    for _ in range(n_abstention):
        i += 1; pq[f"q{i:03d}"] = "abstention"
    for _ in range(n_sans):
        i += 1; pq[f"q{i:03d}"] = "sans_citation"
    for _ in range(n_ok):
        i += 1; pq[f"q{i:03d}"] = "ok"
    n = len(pq)
    n_non_abst = n - n_abstention
    detail = {"q001": [{"record_id": "r", "verdict": "ok", "correspond_brut": True}
                       for _ in range(n_ok)]}
    return {
        "n": n, "n_citations": total, "verdicts": verdicts,
        "taux_citations_inexistantes": round(n_inexistant / total, 4),
        "taux_citations_non_portantes": round(n_court / total, 4),
        "taux_citations_version_omise": 0.0,
        "taux_reponses_sans_citation": round(n_sans / n_non_abst, 4),
        "taux_abstention": round(n_abstention / n, 4),
        "taux_correspondance_brute": round(n_ok / total, 4),
        "citations_par_reponse": {"min": 1, "max": 1, "median": 1, "moyenne": 1.0},
        "par_question": pq, "details": detail,
    }


def _calibration():
    detail = [{"question_id": f"q{i}", "cas_limite": "juste", "origine": "campagne",
               "note_humaine": v, "note_juge": v, "sur": 3, "par_critere": []}
              for i, v in enumerate([0, 1, 2, 3, 2, 1])]
    from accounting_rag.judge import accord
    a = accord({f"{d['question_id']}|{d['cas_limite']}": d["note_humaine"] for d in detail},
               {f"{d['question_id']}|{d['cas_limite']}": d["note_juge"] for d in detail})
    return {"seuil_kappa": 0.6, "cas": [{} for _ in detail], "detail": detail, "accord": a}


@pytest.fixture
def dossier(tmp_path, monkeypatch):
    """Quatre artefacts cohérents et un rapport qui cite tous leurs chiffres."""
    mesures = tmp_path / "mesures"
    mesures.mkdir()
    dev = _metriques()
    val = _metriques(n_ok=5, n_inexistant=0, n_court=0, n_abstention=0, n_sans=0)
    ab = _metriques(n_ok=1, n_inexistant=0, n_court=0, n_abstention=4, n_sans=0)
    ab["taux_abstention_correcte"] = ab["taux_abstention"]
    cal = _calibration()
    for nom, contenu in (("generation_dev", {"metriques": dev}),
                         ("generation_validation", {"metriques": val}),
                         ("generation_abstention", {"metriques": ab})):
        (mesures / f"{nom}.json").write_text(json.dumps(contenu, ensure_ascii=False),
                                             encoding="utf-8")
    (mesures / "calibration_juge.json").write_text(json.dumps(cal, ensure_ascii=False),
                                                   encoding="utf-8")

    rapport = tmp_path / "eval-jalon4.md"
    chiffres = [dev["taux_citations_inexistantes"], dev["taux_citations_non_portantes"],
                dev["taux_citations_version_omise"], dev["taux_correspondance_brute"],
                val["taux_citations_inexistantes"], val["taux_citations_non_portantes"],
                ab["taux_abstention_correcte"], cal["accord"]["kappa_pondere"]]
    effectifs = [dev["n"], val["n"], ab["n"], cal["accord"]["n"]]
    rapport.write_text(
        "# rapport\n" + " ".join(ctrl.fr(c) for c in chiffres)
        + "\n" + " ".join(str(n) for n in effectifs) + "\n", encoding="utf-8")

    monkeypatch.setattr(ctrl, "MESURES", mesures)
    monkeypatch.setattr(ctrl, "RAPPORT", rapport)
    return mesures, rapport


def test_le_controle_passe_quand_tout_concorde(dossier):
    assert ctrl.main() == 0


def test_un_agregat_qui_ne_suit_plus_ses_verdicts_echoue(dossier, capsys):
    """Le défaut le plus répété du dépôt : un taux figé quand ses données ont bougé."""
    mesures, _ = dossier
    f = mesures / "generation_dev.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    d["metriques"]["taux_citations_inexistantes"] = 0.42
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert ctrl.main() == 1
    assert "taux_citations_inexistantes" in capsys.readouterr().err


def test_un_chiffre_corrige_dans_le_json_mais_pas_dans_le_rapport_echoue(dossier, capsys):
    """Trois vagues de correctifs du jalon 3 sont nées de ce cas exact."""
    mesures, _ = dossier
    f = mesures / "generation_validation.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    m = d["metriques"]
    # On rend le JSON cohérent avec une nouvelle valeur, sans toucher au rapport.
    m["verdicts"] = {"ok": 4, "record_inexistant": 1}
    m["n_citations"] = 5
    m["taux_citations_inexistantes"] = 0.2
    m["taux_correspondance_brute"] = 0.8
    m["details"] = {"q001": [{"record_id": "r", "verdict": "ok", "correspond_brut": True}
                             for _ in range(4)]}
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert ctrl.main() == 1
    assert "rapport" in capsys.readouterr().err


def test_un_effectif_absent_du_rapport_echoue(dossier, capsys):
    _, rapport = dossier
    rapport.write_text("# rapport sans aucun chiffre\n", encoding="utf-8")
    assert ctrl.main() == 1
    assert "absent" in capsys.readouterr().err


def test_un_artefact_manquant_echoue(dossier, capsys):
    mesures, _ = dossier
    (mesures / "generation_validation.json").unlink()
    assert ctrl.main() == 1
    assert "manquant" in capsys.readouterr().err


def test_un_seuil_de_calibration_deplace_echoue(dossier, capsys):
    """Le seuil est fixé avant la mesure. Le déplacer après est le travers que tout le
    protocole du jalon existe pour empêcher."""
    mesures, _ = dossier
    f = mesures / "calibration_juge.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    d["seuil_kappa"] = 0.3
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert ctrl.main() == 1
    assert "seuil" in capsys.readouterr().err


def test_un_kappa_qui_ne_suit_plus_ses_notes_echoue(dossier, capsys):
    mesures, _ = dossier
    f = mesures / "calibration_juge.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    d["accord"]["kappa_pondere"] = 0.99
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert ctrl.main() == 1
    assert "kappa_pondere" in capsys.readouterr().err


def test_la_fixture_a_le_meme_schema_que_la_vraie_sortie():
    """Le contrôle a d'abord lu `detail` là où `citations.metriques` rend `details`, et la
    fixture portait la même faute : le test passait sur un schéma inexistant, et le contrôle
    plantait sur les artefacts réels. Une fixture qui invente son schéma ne teste rien.
    """
    import sqlite3
    from accounting_rag.citations import metriques
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE records (id TEXT PRIMARY KEY, texte TEXT)")
    con.execute("INSERT INTO records VALUES ('r1@2026-01-01', ?)",
                ("Le mode d'amortissement linéaire est appliqué à défaut de mode adapté.",))
    con.commit()
    chemin = Path(__file__).resolve().parent / "_schema_tmp.db"
    try:
        disque = sqlite3.connect(chemin)
        con.backup(disque)
        disque.close()
        vraie = metriques({"q1": {"abstention": False, "reponse": "a", "citations": [
            {"record_id": "r1@2026-01-01",
             "extrait": "Le mode d'amortissement linéaire est appliqué à défaut"}]}}, chemin)
    finally:
        chemin.unlink(missing_ok=True)
    fixture = _metriques()
    manquantes = set(fixture) - set(vraie)
    assert not manquantes, (
        f"la fixture porte des clés absentes de la vraie sortie : {sorted(manquantes)}")
