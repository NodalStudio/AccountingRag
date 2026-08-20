"""Tests de `poids_question` : le poids de la question face à sa réécriture.

Le levier vise un défaut mesuré, pas supposé. La réécriture adoptée au jalon 3 répare
douze questions et en casse trois ; sur les deux qui cassent pour une raison lexicale, le
dégât est un glissement de rang bm25 — q080 du rang 7 au rang 84, donc hors du pool de
50, où plus aucun reranker ne peut la rattraper.

Deux exigences gouvernent ce fichier :

- **Loi 2, une variable à la fois.** Le levier ne doit toucher QUE la requête lexicale.
  S'il modifiait aussi le texte soumis au canal dense, il déplacerait l'embedding et
  mesurerait deux effets sous un seul nom. Les tests d'espionnage ci-dessous sont là pour
  ça, et ils échouent si un jour quelqu'un « simplifie » en réunifiant les deux requêtes.
- **Loi 5, un contrôle que personne n'a vu échouer ne prouve rien.** Un test qui
  vérifierait seulement « à 1, rien ne change » passerait à l'identique si le paramètre
  était purement décoratif. D'où le test sur corpus réel, qui exige que le rang du gold
  de q080 s'AMÉLIORE.
"""
import json
from pathlib import Path

import pytest

from accounting_rag.db import write_db
from accounting_rag.evalrag import match
from accounting_rag.index import build_index
from accounting_rag.search import Searcher
from conftest import _rec
from test_search import FakeEmbedder, FakeRewriter

ROOT = Path(__file__).resolve().parent.parent
DB_REELLE = ROOT / "data/corpus.db"
CACHE_REECRITURES = ROOT / "docs/mesures/jalon4/reecritures_dev.json"


def _base(tmp_path, nom="pq.db"):
    db = tmp_path / nom
    write_db([
        _rec("pcg-214-1@2026-01-01", "214-1", texte="Le credit-bail du locataire."),
        _rec("pcg-300-1@2026-01-01", "300-1", texte="Les stocks au cout d'achat."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    return db


def _requetes_vues(s) -> dict:
    """Espionne les deux canaux et renvoie le texte que chacun reçoit."""
    vues = {}
    vrai_bm25, vrai_dense = s._bm25, s._dense

    def bm25(q, limit=None):
        vues["lexicale"] = q
        return vrai_bm25(q, limit)

    def dense(q, limit=None):
        vues["dense"] = q
        return vrai_dense(q, limit)

    s._bm25, s._dense = bm25, dense
    return vues


# --- valeur neutre --------------------------------------------------------------------

def test_le_defaut_est_un(tmp_path):
    s = Searcher(_base(tmp_path), embedder=FakeEmbedder())
    assert s.poids_question == 1


def test_a_un_les_deux_canaux_recoivent_le_meme_texte(tmp_path):
    """La garantie « 1 = comportement inchangé » : avant ce levier, `search()` ne
    construisait qu'une seule requête pour les deux canaux."""
    s = Searcher(_base(tmp_path), embedder=FakeEmbedder(),
                 rewriter=FakeRewriter("amortissement derogatoire"), mode_reecriture="etend")
    vues = _requetes_vues(s)
    s.avant_rerank("credit-bail", mode="hybrid")
    assert vues["lexicale"] == vues["dense"] == "credit-bail amortissement derogatoire"


# --- loi 2 : le levier ne touche que le canal lexical ---------------------------------

def test_a_trois_seule_la_requete_lexicale_repete_la_question(tmp_path):
    s = Searcher(_base(tmp_path), embedder=FakeEmbedder(),
                 rewriter=FakeRewriter("amortissement derogatoire"),
                 mode_reecriture="etend", poids_question=3)
    vues = _requetes_vues(s)
    s.avant_rerank("credit-bail", mode="hybrid")
    assert vues["lexicale"] == "credit-bail credit-bail credit-bail amortissement derogatoire"
    assert vues["dense"] == "credit-bail amortissement derogatoire"


def test_le_canal_dense_est_identique_quel_que_soit_le_poids(tmp_path):
    """Le test qui échoue si quelqu'un réunifie les deux requêtes."""
    textes = set()
    for n in (1, 2, 5):
        s = Searcher(_base(tmp_path, f"pq{n}.db"), embedder=FakeEmbedder(),
                     rewriter=FakeRewriter("amortissement derogatoire"),
                     mode_reecriture="etend", poids_question=n)
        vues = _requetes_vues(s)
        s.avant_rerank("credit-bail", mode="hybrid")
        textes.add(vues["dense"])
    assert len(textes) == 1, textes


def test_en_mode_bm25_seul_la_requete_lexicale_est_utilisee(tmp_path):
    s = Searcher(_base(tmp_path), embedder=FakeEmbedder(),
                 rewriter=FakeRewriter("stocks"), mode_reecriture="etend", poids_question=2)
    vues = _requetes_vues(s)
    s.avant_rerank("credit-bail", mode="bm25")
    assert vues["lexicale"] == "credit-bail credit-bail stocks"
    assert "dense" not in vues


def test_sans_rewriter_le_levier_ne_change_pas_la_requete(tmp_path):
    """Sans réécriture il n'y a rien à pondérer : la question part telle quelle. C'est une
    propriété du code, pas une propriété de bm25 — donc elle se teste sur la chaîne."""
    s = Searcher(_base(tmp_path), embedder=FakeEmbedder(), poids_question=4)
    vues = _requetes_vues(s)
    s.avant_rerank("credit-bail", mode="hybrid")
    assert vues["lexicale"] == vues["dense"] == "credit-bail"


# --- validation des entrées -----------------------------------------------------------

def test_poids_question_zero_refuse(tmp_path):
    with pytest.raises(ValueError, match="poids_question"):
        Searcher(_base(tmp_path), embedder=FakeEmbedder(), poids_question=0)


def test_poids_question_avec_mode_remplace_refuse(tmp_path):
    """Un paramètre silencieusement inerte est un piège : en `remplace`, la question
    originale n'est pas transmise aux canaux, il n'y a rien à pondérer."""
    with pytest.raises(ValueError, match="remplace"):
        Searcher(_base(tmp_path), embedder=FakeEmbedder(),
                 mode_reecriture="remplace", poids_question=2)


def test_poids_question_a_un_reste_permis_en_mode_remplace(tmp_path):
    s = Searcher(_base(tmp_path), embedder=FakeEmbedder(), mode_reecriture="remplace")
    assert s.poids_question == 1


# --- loi 5 : sur corpus réel, le levier doit AMÉLIORER le cas qui l'a motivé ----------

@pytest.mark.skipif(not DB_REELLE.is_file() or not CACHE_REECRITURES.is_file(),
                    reason="corpus ou cache de réécriture absent")
def test_sur_corpus_reel_le_rang_lexical_du_gold_de_q080_sameliore(tmp_path):
    """q080 est la question qui a motivé ce levier : la réécriture fait sortir son gold
    du pool. Le test n'épingle aucun rang mesuré — il exige seulement que répéter la
    question rapproche le gold, ce qu'un paramètre décoratif ne pourrait pas faire.
    """
    question = next(
        json.loads(l) for l in open(ROOT / "benchmark/dev.jsonl", encoding="utf-8")
        if l.strip() and json.loads(l)["id"] == "q080")
    reecriture = json.loads(CACHE_REECRITURES.read_text(encoding="utf-8"))[question["question"]]
    s = Searcher(DB_REELLE, embedder=None, pool=400)

    def rang(n: int) -> int:
        texte = " ".join([question["question"]] * n + [reecriture])
        scores = s._bm25(texte, 400)
        ordre = sorted(scores, key=scores.get, reverse=True)
        return next(i + 1 for i, rid in enumerate(ordre)
                    if any(match(rid, c) for c in question["citations"]))

    assert rang(3) < rang(1), (rang(1), rang(3))
    # Et le dégât que le levier répare est bien réel : à 1, le gold est hors du pool
    # neutre de 50 candidats, donc hors de portée de tout reranker.
    assert rang(1) > 50
