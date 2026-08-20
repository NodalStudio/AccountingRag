"""Tests du script de mesure du correctif de fusion.

Deux choses sont contrôlées ici, et une seule des deux est du calcul.

La première est la **marge avant éviction**, le livrable central de ce correctif : le
rang du gold dans la fusion avant reranking. Un agrégat faux y serait invisible — il
n'existe aucun chiffre publié auquel le comparer, puisque c'est la première fois que le
projet le mesure.

La seconde est le **contrôle de fraîcheur**, et il tombe sous la loi 5 : un contrôle que
personne n'a vu échouer ne prouve rien. Ce dépôt a déjà livré un contrôle de reproduction
qui validait le bug qu'il devait attraper. Les tests ci-dessous le font donc échouer.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ablations_fusion", ROOT / "scripts/ablations_fusion.py")
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)


class FauxSearcher:
    """Renvoie un classement fixé par question, sans base ni modèle."""

    def __init__(self, plans: dict[str, tuple[list[str], list[str]]]):
        self._plans = plans

    def avant_rerank(self, question, mode="hybrid"):
        routes, classes = self._plans[question]
        return [{"record_id": r} for r in routes], list(classes), {}


def _q(qid, question, citations, categorie="regle"):
    return {"id": qid, "question": question, "citations": citations,
            "categorie": categorie}


# --- marge avant éviction ------------------------------------------------------------

def test_marge_compte_les_rangs_en_1_indexe():
    questions = [_q("q1", "a", ["pcg-1-1"])]
    s = FauxSearcher({"a": ([], ["pcg-9-9@2026-01-01", "pcg-1-1@2026-01-01"])})
    m = abl.marge_avant_eviction(s, questions, "hybrid")
    assert m["rangs"] == {"q1": 2}
    assert m["rang_median_des_golds_trouves"] == 2


def test_une_question_routee_est_comptee_a_part_jamais_comme_un_succes():
    """Un gold routé n'est pas exposé au défaut de fusion : le compter comme un rang 1
    ferait baisser artificiellement la marge médiane et masquerait le risque."""
    questions = [_q("q1", "a", ["pcg-1-1"]), _q("q2", "b", ["pcg-2-2"])]
    s = FauxSearcher({
        "a": (["pcg-1-1@2026-01-01"], ["pcg-8-8@2026-01-01"]),
        "b": ([], ["pcg-2-2@2026-01-01"]),
    })
    m = abl.marge_avant_eviction(s, questions, "hybrid")
    assert m["n_routees"] == 1 and m["questions_routees"] == ["q1"]
    assert m["n_exposees"] == 1
    assert "q1" not in m["rangs"]


def test_un_gold_absent_de_la_fusion_est_compte_et_ne_fausse_pas_la_mediane():
    questions = [_q("q1", "a", ["pcg-1-1"]), _q("q2", "b", ["pcg-2-2"])]
    s = FauxSearcher({
        "a": ([], ["pcg-1-1@2026-01-01"]),
        "b": ([], ["pcg-7-7@2026-01-01"]),
    })
    m = abl.marge_avant_eviction(s, questions, "hybrid")
    assert m["rangs"] == {"q1": 1, "q2": None}
    assert m["n_gold_absent_de_la_fusion"] == 1
    assert m["rang_median_des_golds_trouves"] == 1
    # La médiane ne voit pas le gold absent : c'est pour cela qu'elle ne se publie
    # qu'accompagnée du compte des absents, et que les deux parts ci-dessous, elles,
    # le comptent.
    assert m["part_au_dela_de_10"] == round(1 / 2, 4)


def test_les_deux_seuils_publies_comptent_bien_ce_quils_annoncent():
    """10 = le top-k rendu, 25 = la fenêtre du reranker livré. Les parts sont rapportées
    au nombre de questions EXPOSÉES, pas au nombre de golds trouvés : un gold absent de
    la fusion est au-delà de tous les seuils, l'oublier flatterait la marge."""
    plans, questions = {}, []
    for i, rang in enumerate([1, 10, 11, 25, 26, None], start=1):
        qid, texte, gold = f"q{i}", f"t{i}", f"pcg-{i}-{i}"
        classes = [f"pcg-bourrage-{j}@2026-01-01" for j in range(40)]
        if rang is not None:
            classes[rang - 1] = f"{gold}@2026-01-01"
        plans[texte] = ([], classes)
        questions.append(_q(qid, texte, [gold]))
    m = abl.marge_avant_eviction(FauxSearcher(plans), questions, "hybrid")
    assert m["n_exposees"] == 6
    assert m["part_au_dela_de_10"] == round(4 / 6, 4)   # rangs 11, 25, 26 + l'absent
    assert m["part_au_dela_de_25"] == round(2 / 6, 4)   # rang 26 + l'absent
    assert m["n_gold_absent_de_la_fusion"] == 1


def test_la_marge_utilise_le_meme_appariement_que_le_recall():
    """`match()` accepte un id abrégé et les chunks de commentaire ; la marge doit
    l'utiliser, sinon elle compterait « gold absent » là où le recall compte 1,0."""
    questions = [_q("q1", "a", ["pcg-214-22"])]
    s = FauxSearcher({"a": ([], ["pcg-214-22-c1@2026-01-01"])})
    assert abl.marge_avant_eviction(s, questions, "hybrid")["rangs"] == {"q1": 1}


# --- contrôle de fraîcheur : loi 5, on le fait échouer --------------------------------

def _fraicheur_avec(monkeypatch, valeurs: dict):
    monkeypatch.setattr(abl, "_searcher", lambda *a, **k: object())
    appels = iter(["hybrid", "livree"])

    def faux_evaluate(searcher, questions, mode, k=10):
        return {"recall@10": valeurs[next(appels)]}

    monkeypatch.setattr(abl, "evaluate", faux_evaluate)
    return abl.controle_fraicheur(None, None, None)


def test_le_controle_passe_quand_les_chiffres_publies_sont_redonnes(monkeypatch):
    publie = json.loads(abl.PERIMETRE_JALON3.read_text(encoding="utf-8"))["configs"]
    out = _fraicheur_avec(monkeypatch, {
        "hybrid": publie["A_hybrid_neutre"]["recall@10"],
        "livree": publie["C_reecriture_rerank_jalon3"]["recall@10"],
    })
    assert out["hybrid"]["conforme"] and out["livree"]["conforme"]


def test_le_controle_BLOQUE_si_le_retrieval_a_bouge(monkeypatch):
    """Le test qui rend le « contrôle OK » publiable. Sans lui, la ligne de fraîcheur
    serait une affirmation invérifiable."""
    publie = json.loads(abl.PERIMETRE_JALON3.read_text(encoding="utf-8"))["configs"]
    with pytest.raises(SystemExit) as e:
        _fraicheur_avec(monkeypatch, {
            "hybrid": round(publie["A_hybrid_neutre"]["recall@10"] + 0.001, 3),
            "livree": publie["C_reecriture_rerank_jalon3"]["recall@10"],
        })
    assert e.value.code == 1


def test_le_controle_BLOQUE_si_une_question_du_perimetre_a_disparu(monkeypatch, tmp_path):
    publie = json.loads(abl.PERIMETRE_JALON3.read_text(encoding="utf-8"))
    publie["configs"]["A_hybrid_neutre"]["par_question"]["q_fantome"] = 1.0
    faux = tmp_path / "perimetre.json"
    faux.write_text(json.dumps(publie), encoding="utf-8")
    monkeypatch.setattr(abl, "PERIMETRE_JALON3", faux)
    with pytest.raises(SystemExit) as e:
        abl.controle_fraicheur(None, None, None)
    assert e.value.code == 1


# --- protocole : ce qui doit être vrai AVANT de mesurer -------------------------------

def test_chaque_grille_contient_sa_valeur_neutre():
    """Sans le neutre dans la grille, la référence ne serait pas dans la même famille
    que les configurations comparées."""
    for grille, (levier, valeurs) in abl.GRILLES.items():
        assert abl.NEUTRE[levier] in valeurs, grille


def test_le_neutre_est_bien_le_defaut_de_searcher():
    """Loi 8 : le levier rejeté reste exposé à une valeur neutre, et cette valeur est
    le défaut. Si quelqu'un changeait le défaut de `Searcher` sans toucher ce script,
    la « référence » cesserait d'être la baseline publiée en silence."""
    from accounting_rag.search import Searcher
    import inspect
    defauts = {k: v.default for k, v in
               inspect.signature(Searcher.__init__).parameters.items()}
    for levier, valeur in abl.NEUTRE.items():
        assert defauts[levier] == valeur, levier


def test_le_critere_dadoption_est_celui_du_depot():
    assert abl.SEUIL_ADOPTION == 0.95
    assert abl.GARDE_CATEGORIE == -0.05


def test_pire_perte_categorie_trouve_la_categorie_qui_regresse():
    questions = [_q("q1", "a", ["x"], "regle"), _q("q2", "b", ["x"], "regle"),
                 _q("q3", "c", ["x"], "vocabulaire_courant")]
    ref = {"q1": 1.0, "q2": 1.0, "q3": 0.0}
    cfg = {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    cat, delta = abl._pire_perte_categorie(ref, cfg, questions)
    assert cat == "regle" and delta == -0.5


def test_pire_perte_categorie_rend_zero_quand_rien_ne_regresse():
    questions = [_q("q1", "a", ["x"], "regle")]
    cat, delta = abl._pire_perte_categorie({"q1": 0.0}, {"q1": 1.0}, questions)
    assert (cat, delta) == ("", 0.0)


def test_les_reecritures_sont_lues_dans_un_ancrage_versionne():
    """Loi 10 : `docs/mesures/**` est en lecture seule à l'exécution. Ce script ne doit
    jamais pouvoir faire grossir l'ancrage du jalon 4 ni appeler l'API payante."""
    source = (ROOT / "scripts/ablations_fusion.py").read_text(encoding="utf-8")
    assert "ecrire_cache=False" in source
    assert source.count("Rewriter(") == source.count("ecrire_cache=False")
    assert abl.CACHE_REECRITURES.is_file()
