"""Tests du contrôle d'intégrité des réécritures (ablation G, T5, jalon 3).

Le test central est `test_detecte_une_fuite` : il injecte une réécriture qui cite le
numéro d'article attendu sans qu'il figure dans la question, et vérifie que l'audit la
signale. Sans ce test, le « CONTRÔLE D'INTÉGRITÉ OK » affiché par le script serait une
affirmation non falsifiable — exactement le travers documenté en T2 de ce jalon, où un
contrôle de reproduction avait validé le bug qu'il était censé attraper.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "audit_reecritures", Path(__file__).resolve().parent.parent / "scripts/audit_reecritures.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

CITATIONS = ["pcg-214-13@2026-01-01"]


def test_detecte_une_fuite():
    """Le numéro gold apparaît dans la réécriture SANS être dans la question → FUITE."""
    resultat = audit.classer(
        question="comment je repartis le cout d'une machine sur plusieurs annees ?",
        reecriture="amortissement lineaire article 214-13 plan d'amortissement",
        citations=CITATIONS,
    )
    assert resultat == [("214-13", audit.FUITE)]


def test_recopie_depuis_la_question_nest_pas_une_fuite():
    """Un numéro déjà présent dans la question est une recopie sans effet : le routeur
    lit de toute façon la question ORIGINALE, jamais la réécriture."""
    resultat = audit.classer(
        question="que dit l'article 214-13 du PCG sur le mode d'amortissement ?",
        reecriture="amortissement lineaire 214-13 rythme de consommation",
        citations=CITATIONS,
    )
    assert resultat == [("214-13", audit.RECOPIE)]


def test_numero_hors_gold_est_du_bruit_pas_une_fuite():
    resultat = audit.classer(
        question="comment amortir une machine ?",
        reecriture="amortissement article 999-42 immobilisation corporelle",
        citations=CITATIONS,
    )
    assert resultat == [("999-42", audit.INVENTE)]


def test_aucun_numero_ne_produit_aucune_ligne():
    assert audit.classer(
        question="comment amortir une machine ?",
        reecriture="amortissement immobilisation corporelle plan d'amortissement",
        citations=CITATIONS,
    ) == []


def test_numeros_golds_extrait_le_numero_sans_prefixe_ni_date():
    assert audit.numeros_golds(["pcg-1222-74@2026-01-01", "pcg-212-3"]) == {"1222-74", "212-3"}
