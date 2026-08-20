"""Tests de la règle de fusion : `poids_consensus` et `rrf_k` (correctif du jalon 3).

Le défaut réparé ici est nommé et mesuré dans docs/eval-jalon3.md, § « Anatomie de
q023 » : la somme RRF récompense le consensus, pas l'excellence. Les chiffres de q023
utilisés ci-dessous ne sont pas inventés pour le test — ce sont les rangs persistés
dans `docs/mesures/jalon3/sondes.json` (champ `anatomie_q023`), et le test qui les lit
échoue si ce fichier change sans que ces tests soient revus.

Deux exigences de ce dépôt gouvernent ce fichier :

- **Loi 8, défauts neutres.** `Searcher()` sans argument doit reproduire la baseline
  publiée. Ici cela demande plus qu'un « à l'arrondi près » : la fusion neutre doit
  redonner la somme RRF historique BIT À BIT, sinon un ex aequo peut basculer et le
  contrôle de non-régression à 0,672 se met à mentir de temps en temps.
- **Loi 5, un contrôle que personne n'a vu échouer ne prouve rien.** Un test qui
  vérifierait seulement « le neutre reproduit l'ancien » passerait à l'identique si le
  levier ne faisait strictement RIEN. D'où les tests de direction : à faible
  `poids_consensus`, le classement de q023 doit s'INVERSER.
"""
import json
from pathlib import Path

import pytest

from accounting_rag.db import write_db
from accounting_rag.index import build_index
from accounting_rag.search import Searcher, _RRF_K
from conftest import _rec
from test_search import FakeEmbedder

ANATOMIE = json.loads(
    (Path(__file__).resolve().parent.parent / "docs/mesures/jalon3/sondes.json")
    .read_text(encoding="utf-8"))["anatomie_q023"]


def _canal(ids: list[str]) -> dict[str, float]:
    """Canal factice : des scores strictement décroissants, donc des rangs sans ex aequo."""
    return {rid: 1000.0 - i for i, rid in enumerate(ids)}


def _somme_rrf_historique(rankings, k=_RRF_K) -> dict[str, float]:
    """La formule EXACTE du jalon 3, recopiée avant modification, comme témoin."""
    fused: dict[str, float] = {}
    for scores in rankings:
        ordered = sorted(scores, key=scores.get, reverse=True)
        for rank, rid in enumerate(ordered):
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (k + rank + 1)
    return fused


# --- Loi 8 : le neutre reproduit l'existant, bit à bit -------------------------------

def test_le_neutre_reproduit_la_somme_rrf_bit_a_bit():
    """Exhaustif sur le domaine que le code peut atteindre : deux canaux, rangs 0..400.

    C'est ce test qui a fait retirer le court-circuit `if poids_consensus == 1.0:` prévu
    au design : l'identité `max + 1.0 * (somme - max) == somme` tient sans exception, donc
    la branche spéciale n'aurait été qu'une ligne qu'aucune mutation ne pouvait mettre en
    défaut. Le domaine est borné volontairement — l'identité repose sur le fait qu'il n'y
    a que DEUX canaux, où `somme - max` redonne exactement le second terme.
    """
    for k in (_RRF_K, 20, 5, 1, 0):
        for a in range(0, 401):
            for b in range(0, 401):
                if a == b:
                    continue
                x, y = 1.0 / (k + a + 1), 1.0 / (k + b + 1)
                meilleure = x if x > y else y
                assert meilleure + 1.0 * ((x + y) - meilleure) == x + y, (k, a, b)


def test_le_neutre_reproduit_la_somme_rrf_sur_deux_canaux_realistes():
    bm = _canal([f"r{i}" for i in range(50)])
    dense = _canal([f"r{i}" for i in range(25, 75)])
    attendu = _somme_rrf_historique([bm, dense])
    obtenu = Searcher._rrf([bm, dense])
    assert obtenu == attendu
    # Bit à bit, pas seulement « proche » : `==` sur des flottants est ici l'assertion
    # voulue, et `pytest.approx` masquerait exactement ce qu'on veut interdire.
    assert list(obtenu) == list(attendu)  # et le même ordre d'insertion (départage stable)


def test_le_defaut_de_searcher_est_la_valeur_neutre(tmp_path):
    db = tmp_path / "neutre.db"
    write_db([_rec("pcg-1-1@2026-01-01", "1-1", texte="x")], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder())
    assert s.poids_consensus == 1.0
    assert s.rrf_k == _RRF_K == 60


# --- Loi 5 : le levier doit pouvoir changer quelque chose ----------------------------

def test_un_poids_consensus_faible_change_reellement_le_classement():
    """Sans ce test, tous les autres passeraient encore si `poids_consensus` était ignoré."""
    bm = _canal(["excellent_solo", "moyen_double"])
    dense = _canal(["autre", "moyen_double"])
    neutre = Searcher._rrf([bm, dense], _RRF_K, 1.0)
    excellence = Searcher._rrf([bm, dense], _RRF_K, 0.0)
    assert neutre["moyen_double"] > neutre["excellent_solo"]
    assert excellence["excellent_solo"] > excellence["moyen_double"]


def test_le_score_dun_candidat_mono_canal_ne_depend_pas_du_poids():
    """Le levier ne doit toucher QUE ce qui dépasse la meilleure contribution."""
    bm = _canal(["solo", "autre"])
    dense = _canal(["autre"])
    scores = [Searcher._rrf([bm, dense], _RRF_K, p)["solo"] for p in (1.0, 0.5, 0.1, 0.0)]
    assert len(set(scores)) == 1


# --- q023 : le cas réel, sur les rangs persistés du jalon 3 --------------------------

def _rangs_q023():
    """Reconstruit les deux canaux de q023 à partir des rangs publiés (1-indexés)."""
    gold = ANATOMIE["gold"]
    rival = ANATOMIE["trois_premiers_apres_fusion"][0]
    assert gold["rang_dense"] is None, "q023 n'isole le défaut que si le gold est mono-canal"
    taille = 80
    bm = [f"bourrage_bm{i}" for i in range(taille)]
    bm[gold["rang_bm25"] - 1] = gold["record"]
    bm[rival["rang_bm25"] - 1] = rival["record"]
    dense = [f"bourrage_de{i}" for i in range(taille)]
    dense[rival["rang_dense"] - 1] = rival["record"]
    return _canal(bm), _canal(dense), gold["record"], rival["record"]


def test_q023_le_gold_perd_au_poids_neutre():
    """Le DÉFAUT lui-même, reproduit : sans ce test, la réparation ne prouverait rien."""
    bm, dense, gold, rival = _rangs_q023()
    scores = Searcher._rrf([bm, dense], _RRF_K, 1.0)
    assert scores[rival] > scores[gold]


def test_q023_le_gold_gagne_quand_le_consensus_pese_peu():
    bm, dense, gold, rival = _rangs_q023()
    scores = Searcher._rrf([bm, dense], _RRF_K, 0.025)
    assert scores[gold] > scores[rival]


def test_q023_la_bascule_est_bien_ou_larithmetique_la_place():
    """La grille de mesure est fixée d'avance ; ce test dit seulement où tombe le seuil.

    Il documente que le point de bascule (~0,05) est une conséquence des rangs mesurés,
    pas un réglage choisi pour faire réussir q023.
    """
    bm, dense, gold, rival = _rangs_q023()
    def gold_gagne(p):
        s = Searcher._rrf([bm, dense], _RRF_K, p)
        return s[gold] > s[rival]
    assert gold_gagne(0.04)
    assert not gold_gagne(0.06)


def test_rrf_k_ne_repare_q023_quaux_valeurs_extremes():
    """Le levier ATTENDU, et sa disqualification par le calcul (loi 6 : on mesure).

    Régler l'escompte de rang ne suffit pas : il faut descendre à `rrf_k <= 1` pour que
    le meilleur candidat lexical du corpus repasse devant un candidat 5ᵉ et 6ᵉ. À k=2 il
    reperd. C'est ce qui fait dire au rapport que le défaut est structurel à la somme,
    et non un k mal choisi.
    """
    bm, dense, gold, rival = _rangs_q023()
    def gold_gagne(k):
        s = Searcher._rrf([bm, dense], k, 1.0)
        return s[gold] > s[rival]
    assert not gold_gagne(60)
    assert not gold_gagne(20)
    assert not gold_gagne(5)
    assert not gold_gagne(2)
    assert gold_gagne(1)
    assert gold_gagne(0)


# --- Validation des entrées ----------------------------------------------------------

def test_poids_consensus_negatif_refuse(tmp_path):
    db = tmp_path / "neg.db"
    write_db([_rec("pcg-1-1@2026-01-01", "1-1", texte="x")], db)
    build_index(db, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="poids_consensus"):
        Searcher(db, embedder=FakeEmbedder(), poids_consensus=-0.1)


def test_rrf_k_negatif_refuse(tmp_path):
    db = tmp_path / "negk.db"
    write_db([_rec("pcg-1-1@2026-01-01", "1-1", texte="x")], db)
    build_index(db, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="rrf_k"):
        Searcher(db, embedder=FakeEmbedder(), rrf_k=-1)


def test_les_deux_leviers_sont_transmis_a_la_fusion(tmp_path):
    """Espionne `_rrf` : un paramètre accepté mais jamais transmis serait invisible."""
    db = tmp_path / "transmis.db"
    write_db([
        _rec("pcg-214-1@2026-01-01", "214-1", texte="Le credit-bail est comptabilise."),
        _rec("pcg-300-1@2026-01-01", "300-1", texte="Les stocks sont valorises."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder(), poids_consensus=0.25, rrf_k=7)
    vus = {}
    vrai = Searcher._rrf

    def espion(rankings, k=_RRF_K, poids_consensus=1.0):
        vus["k"] = k
        vus["poids"] = poids_consensus
        return vrai(rankings, k, poids_consensus)

    s._rrf = espion
    s.search("credit-bail", mode="hybrid")
    assert vus == {"k": 7, "poids": 0.25}


# --- avant_rerank : le chemin exposé à la mesure est celui qui est exécuté -----------

def test_avant_rerank_est_coherent_avec_search(tmp_path):
    """La marge avant éviction se mesure sur `avant_rerank` ; si elle divergeait de
    `search`, elle décrirait un système que personne n'exécute.

    Ce test est la raison pour laquelle `avant_rerank` a été EXTRAIT de `search` au lieu
    d'être reconstitué dans le script de mesure : une reconstitution parallèle passe ce
    test le jour où elle est écrite, puis dérive en silence.
    """
    db = tmp_path / "coherence.db"
    write_db([
        _rec("pcg-214-1@2026-01-01", "214-1", texte="Le credit-bail est comptabilise chez le locataire."),
        _rec("pcg-300-1@2026-01-01", "300-1", texte="Les stocks sont valorises au cout d'achat."),
        _rec("pcg-400-1@2026-01-01", "400-1", texte="Le credit et le bail sont deux notions distinctes."),
    ], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder())
    routes, classes, scores = s.avant_rerank("credit-bail", mode="hybrid")
    attendu = [h["record_id"] for h in s.search("credit-bail", k=10, mode="hybrid")]
    assert [r["record_id"] for r in routes] + classes[:10 - len(routes)] == attendu
    assert set(classes).isdisjoint({r["record_id"] for r in routes})


def test_avant_rerank_refuse_un_mode_inconnu(tmp_path):
    db = tmp_path / "mode2.db"
    write_db([_rec("pcg-1-1@2026-01-01", "1-1", texte="x")], db)
    build_index(db, embedder=FakeEmbedder())
    s = Searcher(db, embedder=FakeEmbedder())
    with pytest.raises(ValueError, match="mode inconnu"):
        s.avant_rerank("x", mode="bogus")
