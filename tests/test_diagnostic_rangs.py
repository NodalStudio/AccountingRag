"""Tests unitaires des fonctions pures de `scripts/diagnostic_rangs.py` (rang/total du
gold dans un classement complet) — non-régression du bug de comptage trouvé en revue T2
(cf. docstring de `_rang_gold_lexical` dans le script) : le total ne doit JAMAIS être
égal au rang lui-même par construction, c'est le nombre RÉEL de candidats distincts.

`scripts/` n'étant pas un package installé, on l'ajoute explicitement à `sys.path`
(même patron que `ROOT = Path(__file__).resolve().parent.parent` dans les scripts
eux-mêmes) plutôt que d'introduire un `scripts/__init__.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from diagnostic_rangs import _rang_gold_dense, _rang_gold_lexical  # noqa: E402


def test_rang_gold_lexical_total_est_le_nombre_reel_de_candidats():
    """Le bug corrigé faisait que `total == rang` mécaniquement. Ici rang=3 (position
    de "c") et total=5 (taille de `ranked`) : les deux DOIVENT être différents pour
    que la non-régression soit visible."""
    ranked = ["a", "b", "c", "d", "e"]
    rang, total = _rang_gold_lexical(ranked, "c")
    assert rang == 3
    assert total == 5


def test_rang_gold_lexical_absent_total_est_toute_la_liste():
    ranked = ["a", "b", "c"]
    rang, total = _rang_gold_lexical(ranked, "zzz")
    assert rang is None
    assert total == 3


def test_rang_gold_lexical_gold_en_derniere_position():
    """Cas qui aurait été invisible avec le bug (rang == total par coïncidence) : on
    vérifie explicitement que `total` vient de `len(ranked)`, pas de `rang`, en
    variant la longueur de la liste indépendamment de la position du gold."""
    ranked = ["a", "b", "c", "d"]
    rang, total = _rang_gold_lexical(ranked, "d")
    assert rang == 4
    assert total == 4  # ici rang == total : le cas qui masquait le bug d'origine
    # ajout d'un candidat supplémentaire après le gold : le rang ne change pas, le
    # total, lui, doit augmenter — ce que le bug d'origine ne pouvait pas produire.
    rang2, total2 = _rang_gold_lexical(ranked + ["e"], "d")
    assert rang2 == 4
    assert total2 == 5


def test_rang_gold_dense_meme_convention_que_lexical():
    ranked = ["a", "b", "c", "d"]
    rang, total = _rang_gold_dense(ranked, "b")
    assert rang == 2
    assert total == 4
    assert _rang_gold_dense(ranked, "b") == _rang_gold_lexical(ranked, "b")
