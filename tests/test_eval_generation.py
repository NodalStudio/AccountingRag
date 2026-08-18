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


def test_controle_passe_quand_le_recall_est_celui_publie(monkeypatch):
    monkeypatch.setattr(evg, "load_benchmark", lambda p: ["peu importe"])
    monkeypatch.setattr(evg, "Searcher", lambda *a, **k: "faux searcher")
    monkeypatch.setattr(evg, "evaluate",
                        lambda *a, **k: {"recall@10": evg.RECALL_NEUTRE_DEV})
    assert evg.controle_fraicheur(embedder=None) == evg.RECALL_NEUTRE_DEV


def test_controle_bloque_des_que_le_recall_derive(monkeypatch):
    """Un écart d'un millième suffit : la campagne ne doit rien dépenser."""
    monkeypatch.setattr(evg, "load_benchmark", lambda p: ["peu importe"])
    monkeypatch.setattr(evg, "Searcher", lambda *a, **k: "faux searcher")
    monkeypatch.setattr(evg, "evaluate",
                        lambda *a, **k: {"recall@10": evg.RECALL_NEUTRE_DEV - 0.001})
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
