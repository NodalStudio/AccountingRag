import json
import pytest
from accounting_rag.judge import Judge, accord

BAREME = ["cite le mode linéaire", "dit qu'il s'applique à défaut de mieux adapté"]
REPONSE = {"abstention": False,
           "reponse": "Le mode linéaire s'applique à défaut de mode mieux adapté.",
           "citations": [{"record_id": "pcg-214-13@2026-01-01",
                          "extrait": "Le mode d'amortissement linéaire est appliqué"}]}


class FauxBloc:
    def __init__(self, texte):
        self.type, self.text = "text", texte


class FauxMessage:
    def __init__(self, payload, stop_reason="end_turn"):
        self.content = [FauxBloc(json.dumps(payload, ensure_ascii=False))]
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class FauxClient:
    def __init__(self, payload=None):
        self.payload = payload or {
            "note": 2, "sur": 2,
            "par_critere": [{"critere": c, "acquis": True, "justification": "ok"}
                            for c in BAREME]}
        self.appels = []
        self.messages = self
        self.stop_reason = "end_turn"

    def create(self, **kwargs):
        self.appels.append(kwargs)
        return FauxMessage(self.payload, self.stop_reason)


def test_accord_parfait():
    a = {"c1": 3, "c2": 1, "c3": 2}
    r = accord(a, a)
    assert r["exact"] == 1.0 and r["ecart_moyen"] == 0.0 and r["kappa_pondere"] == 1.0


def test_accord_nul_quand_le_juge_note_a_lenvers():
    """Deux notations exactement inversées : kappa pondéré = -1,0 (vérifié à la main)."""
    r = accord({"c1": 0, "c2": 3}, {"c1": 3, "c2": 0})
    assert r["kappa_pondere"] == -1.0 and r["exact"] == 0.0


def test_ecart_moyen_est_bien_une_moyenne():
    r = accord({"c1": 3, "c2": 3}, {"c1": 2, "c2": 3})
    assert r["ecart_moyen"] == 0.5 and r["exact"] == 0.5


def test_accord_refuse_des_ensembles_de_cles_differents():
    """Un kappa calculé sur des questions qui ne se correspondent pas est un faux chiffre."""
    with pytest.raises(ValueError, match="mêmes questions"):
        accord({"c1": 1}, {"c2": 1})


def test_accord_de_deux_notations_constantes_identiques_vaut_un():
    """Cas dégénéré : l'accord attendu est nul, donc kappa est 0/0. Doit rendre 1,0,
    pas planter et pas 0,0 — les deux notations sont identiques."""
    r = accord({"c1": 2, "c2": 2}, {"c1": 2, "c2": 2})
    assert r["kappa_pondere"] == 1.0


def test_le_juge_ne_voit_ni_gold_ni_passages(tmp_path):
    """Frontière de la loi 9 : barème et réponse oui, golds et passages non."""
    client = FauxClient()
    j = Judge(cache_path=tmp_path / "cache.json", client=client)
    j.noter("comment amortir ?", REPONSE, BAREME)
    envoye = json.dumps(client.appels[0], default=str, ensure_ascii=False)
    assert "cite le mode linéaire" in envoye          # le barème passe
    assert "Le mode linéaire s'applique" in envoye    # la réponse passe
    for interdit in ("citations_attendues", "gold", "passages", "chemin", "score"):
        assert interdit not in envoye


def test_le_cache_evite_un_second_appel(tmp_path):
    client = FauxClient()
    j = Judge(cache_path=tmp_path / "cache.json", client=client)
    a = j.noter("comment amortir ?", REPONSE, BAREME)
    b = j.noter("comment amortir ?", REPONSE, BAREME)
    assert a == b and len(client.appels) == 1


def test_lecture_seule_leve_et_nappelle_pas_lapi(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text("{}", encoding="utf-8")
    client = FauxClient()
    j = Judge(cache_path=cache, client=client, ecrire_cache=False)
    with pytest.raises(RuntimeError, match="lecture seule"):
        j.noter("inconnue", REPONSE, BAREME)
    assert client.appels == []


def test_note_incoherente_avec_le_bareme_leve(tmp_path):
    """Une note supérieure au nombre de critères est un chiffre faux : ne pas la cacher."""
    client = FauxClient({"note": 5, "sur": 2, "par_critere": []})
    j = Judge(cache_path=tmp_path / "cache.json", client=client)
    with pytest.raises(RuntimeError, match="incohérente"):
        j.noter("comment amortir ?", REPONSE, BAREME)
    assert not (tmp_path / "cache.json").exists()


def test_troncature_leve_avant_analyse(tmp_path):
    client = FauxClient()
    client.stop_reason = "max_tokens"
    j = Judge(cache_path=tmp_path / "cache.json", client=client)
    with pytest.raises(RuntimeError, match="tronquée"):
        j.noter("comment amortir ?", REPONSE, BAREME)


def test_le_garde_de_seuil_sort_en_erreur_sous_le_seuil(tmp_path, monkeypatch):
    """Le seuil est lu dans le JSON, pas codé en dur : il ne peut pas être déplacé
    silencieusement, et un kappa sous le seuil doit faire sortir le script en erreur."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "calibrer_juge", Path(__file__).resolve().parent.parent / "scripts/calibrer_juge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({
        "seuil_kappa": 0.6,
        "cas": [{"question_id": "q1", "question_texte": "q ?", "cas_limite": "juste",
                 "bareme": BAREME, "note_humaine": 0},
                {"question_id": "q2", "question_texte": "q2 ?", "cas_limite": "juste",
                 "bareme": BAREME, "note_humaine": 2}],
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "CALIBRATION", calib)
    monkeypatch.setattr(mod, "CACHE_JUGE", tmp_path / "cache.json")
    monkeypatch.setattr(mod, "charger_reponses", lambda: {"q1": REPONSE, "q2": REPONSE})
    monkeypatch.setattr(mod, "Judge", lambda **kw: _JugeFixe())

    with pytest.raises(SystemExit):
        mod.main()


class _JugeFixe:
    """Note toujours 2 : en désaccord avec la note humaine 0 du premier cas."""
    modele, appels, tokens_entree, tokens_sortie = "faux", 0, 0, 0

    def noter(self, question, reponse, bareme):
        return {"note": 2, "sur": len(bareme),
                "par_critere": [{"critere": c, "acquis": True, "justification": "ok"}
                                for c in bareme]}


def test_un_sur_incoherent_leve_meme_quand_la_note_est_dans_les_bornes(tmp_path):
    """Isole la clause `out["sur"] != len(bareme)` du contrôle de cohérence.

    Relevé par l'implémenteur : le test précédent utilisait `note=5, sur=2` sur un barème
    de 2 critères, ce qui viole DÉJÀ la seconde clause (`note` hors bornes). La première
    clause n'avait donc aucun test capable de la faire échouer — un contrôle que personne
    n'a vu échouer ne prouve rien (loi 5). Ici `note=1` est dans les bornes et seul `sur`
    est faux.
    """
    client = FauxClient({"note": 1, "sur": 3, "par_critere": []})
    j = Judge(cache_path=tmp_path / "cache.json", client=client)
    with pytest.raises(RuntimeError, match="incohérente"):
        j.noter("comment amortir ?", REPONSE, BAREME)
    assert not (tmp_path / "cache.json").exists()


def test_la_calibration_ne_confond_pas_deux_cas_du_meme_enonce(tmp_path, monkeypatch):
    """Six énoncés du jeu de calibration apparaissent DEUX fois : une fois avec leur
    réponse réelle, une fois avec une abstention fabriquée. Si la clé d'indexation était
    le seul `question_id`, la seconde écraserait la première et le kappa porterait
    silencieusement sur 24 cas au lieu de 30 — un chiffre faux qui ne se signalerait pas.
    """
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "calibrer_juge",
        Path(__file__).resolve().parent.parent / "scripts/calibrer_juge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({
        "seuil_kappa": -2.0,  # volontairement inatteignable par le bas : on teste le n
        "cas": [
            {"question_id": "q1", "question_texte": "q ?", "cas_limite": "juste",
             "origine": "campagne", "bareme": BAREME, "note_humaine": 2},
            {"question_id": "q1", "question_texte": "q ?", "cas_limite": "abstention_excessive",
             "origine": "perturbation", "bareme": BAREME, "note_humaine": 0,
             "reponse": {"abstention": True, "reponse": "rien", "citations": []}},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "CALIBRATION", calib)
    monkeypatch.setattr(mod, "CACHE_JUGE", tmp_path / "cache.json")
    monkeypatch.setattr(mod, "charger_reponses", lambda: {"q1": REPONSE})
    monkeypatch.setattr(mod, "Judge", lambda **kw: _JugeFixe())

    mod.main()
    ecrit = json.loads(calib.read_text(encoding="utf-8"))
    assert ecrit["accord"]["n"] == 2, \
        f"les deux cas du même énoncé ont fusionné : n={ecrit['accord']['n']}"
    assert ecrit["notes_juge"] == {"q1|juste": 2, "q1|abstention_excessive": 2}


def test_une_reponse_en_ligne_dispense_du_cache_de_campagne(tmp_path, monkeypatch):
    """Un cas fabriqué porte sa réponse : il ne doit pas être déclaré manquant."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "calibrer_juge",
        Path(__file__).resolve().parent.parent / "scripts/calibrer_juge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    calib = tmp_path / "calibration.json"
    calib.write_text(json.dumps({
        "seuil_kappa": -2.0,
        "cas": [{"question_id": "absente_du_cache", "question_texte": "q ?",
                 "cas_limite": "fausse_bien_citee", "origine": "perturbation",
                 "bareme": BAREME, "note_humaine": 0,
                 "reponse": {"abstention": False, "reponse": "faux", "citations": []}},
                {"question_id": "q2", "question_texte": "q2 ?", "cas_limite": "juste",
                 "origine": "campagne", "bareme": BAREME, "note_humaine": 2}],
    }, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(mod, "CALIBRATION", calib)
    monkeypatch.setattr(mod, "CACHE_JUGE", tmp_path / "cache.json")
    monkeypatch.setattr(mod, "charger_reponses", lambda: {"q2": REPONSE})
    monkeypatch.setattr(mod, "Judge", lambda **kw: _JugeFixe())

    mod.main()  # ne doit pas lever SystemExit
    assert json.loads(calib.read_text(encoding="utf-8"))["accord"]["n"] == 2
