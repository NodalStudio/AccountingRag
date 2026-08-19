"""Tests du contrôle de fraîcheur de la campagne de génération.

Le contrôle refuse de dépenser un appel API si le retrieval neutre ne rend plus la
valeur publiée. Sans ces deux tests, le « contrôle OK » affiché serait une affirmation
non falsifiable.
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "eval_generation",
    Path(__file__).resolve().parent.parent / "scripts/eval_generation.py")
evg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evg)


def _perimetre(tmp_path, ids):
    """Écrit un faux JSON de clôture jalon 3 portant le périmètre publié."""
    import json
    f = tmp_path / "cloture_dev.json"
    f.write_text(json.dumps(
        {"configs": {"A_hybrid_neutre": {"par_question": {i: 1.0 for i in ids}}}}),
        encoding="utf-8")
    return f


def test_controle_passe_quand_le_recall_est_celui_publie(tmp_path, monkeypatch):
    monkeypatch.setattr(evg, "PERIMETRE_JALON3", _perimetre(tmp_path, ["q001", "q002"]))
    monkeypatch.setattr(evg, "load_benchmark",
                        lambda p: [{"id": "q001"}, {"id": "q002"}, {"id": "q1001"}])
    monkeypatch.setattr(evg, "Searcher", lambda *a, **k: "faux searcher")
    monkeypatch.setattr(evg, "evaluate",
                        lambda *a, **k: {"recall@10": evg.RECALL_NEUTRE_DEV})
    assert evg.controle_fraicheur(embedder=None) == evg.RECALL_NEUTRE_DEV


def test_controle_bloque_des_que_le_recall_derive(tmp_path, monkeypatch):
    """Un écart d'un millième suffit : la campagne ne doit rien dépenser."""
    monkeypatch.setattr(evg, "PERIMETRE_JALON3", _perimetre(tmp_path, ["q001"]))
    monkeypatch.setattr(evg, "load_benchmark", lambda p: [{"id": "q001"}])
    monkeypatch.setattr(evg, "Searcher", lambda *a, **k: "faux searcher")
    monkeypatch.setattr(evg, "evaluate",
                        lambda *a, **k: {"recall@10": evg.RECALL_NEUTRE_DEV - 0.001})
    with pytest.raises(SystemExit):
        evg.controle_fraicheur(embedder=None)


def test_le_controle_ne_mesure_que_le_perimetre_publie(tmp_path, monkeypatch):
    """L'extension du benchmark ne doit pas déplacer la référence.

    Défaut réel : le contrôle évaluait `benchmark/dev.jsonl` entier. Passé de 61 à 93
    questions, il a rendu 0,715 au lieu de 0,672 et bloqué la clôture. Déplacer la
    constante aurait détruit le contrôle ; il ne mesure donc que les questions du JSON qui
    porte le chiffre publié.
    """
    monkeypatch.setattr(evg, "PERIMETRE_JALON3", _perimetre(tmp_path, ["q001", "q002"]))
    monkeypatch.setattr(evg, "load_benchmark",
                        lambda p: [{"id": "q001"}, {"id": "q002"}, {"id": "q1001"}])
    monkeypatch.setattr(evg, "Searcher", lambda *a, **k: "faux searcher")
    vus = []

    def faux_evaluate(searcher, questions, **k):
        vus.append([q["id"] for q in questions])
        return {"recall@10": evg.RECALL_NEUTRE_DEV}

    monkeypatch.setattr(evg, "evaluate", faux_evaluate)
    evg.controle_fraicheur(embedder=None)
    assert vus[0] == ["q001", "q002"], f"le contrôle a mesuré {vus[0]}"
    # Le second appel est le chiffre informatif sur le split étendu.
    assert vus[1] == ["q001", "q002", "q1001"]


def test_une_question_du_perimetre_publie_disparue_bloque(tmp_path, monkeypatch):
    """Si une question publiée quitte dev, le chiffre de référence cesse d'être
    vérifiable : mieux vaut bloquer que mesurer sur un périmètre amputé."""
    monkeypatch.setattr(evg, "PERIMETRE_JALON3", _perimetre(tmp_path, ["q001", "q002"]))
    monkeypatch.setattr(evg, "load_benchmark", lambda p: [{"id": "q001"}])
    monkeypatch.setattr(evg, "Searcher", lambda *a, **k: "faux searcher")
    monkeypatch.setattr(evg, "evaluate",
                        lambda *a, **k: {"recall@10": evg.RECALL_NEUTRE_DEV})
    with pytest.raises(SystemExit):
        evg.controle_fraicheur(embedder=None)


def test_le_generateur_ne_voit_ni_gold_ni_notes():
    """Intégrité du benchmark : le script ne passe que question et passages.

    Contrôle textuel sur la source, parce que la campagne elle-même coûte de l'argent et
    ne peut pas tourner en test. Le seul appel à `repondre` doit passer `q["question"]`
    et `passages`, rien d'autre.
    """
    src = (Path(__file__).resolve().parent.parent
           / "scripts/eval_generation.py").read_text(encoding="utf-8")
    assert 'generateur.repondre(q["question"], passages)' in src
    for interdit in ('q["citations"]', "q['citations']", 'q["notes"]', "q['notes']"):
        assert interdit not in src


def test_le_cache_du_jalon4_reprend_a_lidentique_celui_du_jalon3(tmp_path, monkeypatch):
    """Les 61 questions communes doivent garder EXACTEMENT leur réécriture du jalon 3.

    Sinon le retrieval de ces questions changerait silencieusement et leurs résultats
    cesseraient d'être comparables d'un jalon à l'autre, sans qu'aucun chiffre ne le dise.
    Le contrôle porte sur l'égalité des valeurs, pas seulement sur la présence des clés.
    """
    import json
    ancrage = tmp_path / "jalon3.json"
    ancrage.write_text(json.dumps({
        "question A": "réécriture A",
        "question B": "réécriture B",
        "question hors split": "réécriture C",
    }, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "jalon4"
    monkeypatch.setattr(evg, "CACHE_REECRITURES", ancrage)
    monkeypatch.setattr(evg, "OUT_DIR", out)
    monkeypatch.setattr(evg, "load_benchmark",
                        lambda p: [{"question": "question A"}, {"question": "question B"},
                                   {"question": "question neuve"}])

    chemin = evg.cache_reecritures("dev")
    repris = json.loads(chemin.read_text(encoding="utf-8"))
    assert repris == {"question A": "réécriture A", "question B": "réécriture B"}, \
        "la reprise doit être identique, et limitée aux questions du split"
    assert "question neuve" not in repris


def test_lamorce_ne_recrit_pas_un_cache_existant(tmp_path, monkeypatch):
    """Une seconde exécution ne doit pas écraser les réécritures déjà produites."""
    import json
    ancrage = tmp_path / "jalon3.json"
    ancrage.write_text(json.dumps({"question A": "réécriture A"}, ensure_ascii=False),
                       encoding="utf-8")
    out = tmp_path / "jalon4"
    out.mkdir()
    (out / "reecritures_dev.json").write_text(
        json.dumps({"question A": "réécriture A", "question neuve": "réécriture neuve"},
                   ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(evg, "CACHE_REECRITURES", ancrage)
    monkeypatch.setattr(evg, "OUT_DIR", out)
    monkeypatch.setattr(evg, "load_benchmark", lambda p: [{"question": "question A"}])

    garde = json.loads((out / "reecritures_dev.json").read_text(encoding="utf-8"))
    evg.cache_reecritures("dev")
    assert json.loads((out / "reecritures_dev.json").read_text(encoding="utf-8")) == garde


def test_lancrage_du_jalon3_nest_jamais_ouvert_en_ecriture():
    """Contrôle textuel : aucun `Rewriter` du script ne pointe l'ancrage du jalon 3.

    La campagne coûte de l'argent et ne peut pas tourner en test ; c'est la loi 10 du
    dépôt (`docs/mesures/**` en lecture seule à l'exécution) qui est en jeu.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "scripts/eval_generation.py").read_text(encoding="utf-8")
    assert "Rewriter(cache_path=cache_reecritures(" in src
    assert "Rewriter(cache_path=CACHE_REECRITURES" not in src


def test_le_cout_distingue_un_rejeu_dune_campagne_gratuite():
    """Contrôle textuel : le JSON doit dire combien de réponses étaient déjà en cache.

    La campagne d'abstention a d'abord publié « 0 appel API » après un rejeu complet
    depuis le cache. Lu seul, ce chiffre se comprend comme une campagne gratuite ; il
    décrit en réalité un rejeu. Le coût est par exécution, jamais cumulé, et le JSON doit
    porter de quoi le savoir.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "scripts/eval_generation.py").read_text(encoding="utf-8")
    assert "reponses_deja_en_cache_avant" in src
    assert "rejeu_depuis_le_cache" in src
