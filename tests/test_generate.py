import json
import pytest
from accounting_rag.generate import Generator

PASSAGES = [
    {"record_id": "pcg-214-13@2026-01-01", "article": "214-13", "chemin": "Livre II",
     "texte": "Le mode d'amortissement linéaire est appliqué à défaut de mode mieux adapté.",
     "score": 1.0, "source": "fusion"},
]


class FauxBloc:
    def __init__(self, texte):
        self.type, self.text = "text", texte


class FauxMessage:
    def __init__(self, payload, stop_reason="end_turn", texte_brut=None):
        texte = texte_brut if texte_brut is not None else json.dumps(payload,
                                                                     ensure_ascii=False)
        self.content = [FauxBloc(texte)]
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class FauxClient:
    """Client Anthropic factice : enregistre les appels, ne sort jamais sur le réseau."""

    def __init__(self, payload=None):
        self.payload = payload or {
            "abstention": False,
            "reponse": "Le mode linéaire s'applique à défaut de mode mieux adapté.",
            "citations": [{"record_id": "pcg-214-13@2026-01-01",
                           "extrait": "Le mode d'amortissement linéaire est appliqué"}],
        }
        self.appels = []
        self.messages = self
        self.stop_reason = "end_turn"
        self.texte_brut = None

    def create(self, **kwargs):
        self.appels.append(kwargs)
        return FauxMessage(self.payload, self.stop_reason, self.texte_brut)


def test_repondre_rend_une_reponse_citee(tmp_path):
    client = FauxClient()
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    out = g.repondre("comment amortir ?", PASSAGES)
    assert out["abstention"] is False
    assert out["citations"][0]["record_id"] == "pcg-214-13@2026-01-01"
    assert len(client.appels) == 1


def test_le_cache_evite_un_second_appel(tmp_path):
    client = FauxClient()
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    a = g.repondre("comment amortir ?", PASSAGES)
    b = g.repondre("comment amortir ?", PASSAGES)
    assert a == b
    assert len(client.appels) == 1


def test_abstention_est_transmise_telle_quelle(tmp_path):
    client = FauxClient({"abstention": True, "reponse": "Le corpus ne le dit pas.",
                         "citations": []})
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    out = g.repondre("quel est le taux d'IS ?", PASSAGES)
    assert out["abstention"] is True and out["citations"] == []


def test_le_generateur_ne_recoit_que_question_et_passages(tmp_path):
    """Intégrité du benchmark : ni gold, ni citations attendues dans le prompt."""
    client = FauxClient()
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    g.repondre("comment amortir ?", PASSAGES)
    envoye = json.dumps(client.appels[0], default=str)
    assert "comment amortir ?" in envoye
    for interdit in ("citations_attendues", "gold", "expected", "benchmark"):
        assert interdit not in envoye


def test_lecture_seule_leve_et_nappelle_pas_lapi(tmp_path):
    """`ecrire_cache=False` : question absente -> lève, sans appel ni écriture."""
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({}), encoding="utf-8")
    avant = cache.read_bytes()
    client = FauxClient()
    g = Generator(cache_path=cache, client=client, ecrire_cache=False)
    with pytest.raises(RuntimeError, match="lecture seule"):
        g.repondre("inconnue", PASSAGES)
    assert client.appels == [] and cache.read_bytes() == avant


def test_reponse_vide_leve_au_lieu_detre_mise_en_cache(tmp_path):
    client = FauxClient({"abstention": False, "reponse": "   ", "citations": []})
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    with pytest.raises(RuntimeError, match="réponse vide"):
        g.repondre("comment amortir ?", PASSAGES)
    assert not (tmp_path / "cache.json").exists()


def test_troncature_leve_avant_analyse_et_nest_pas_mise_en_cache(tmp_path):
    """`stop_reason=max_tokens` : le JSON est amputé, il ne doit jamais entrer au cache.

    Cas réel mesuré à max_tokens=2000 sur q001 (cf. sonde_verbatim). Le payload est ici
    volontairement VALIDE : le contrôle doit tenir sur stop_reason, pas sur l'échec de
    json.loads, sinon une troncature tombant sur une accolade fermante passerait.
    """
    client = FauxClient()
    client.stop_reason = "max_tokens"
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    with pytest.raises(RuntimeError, match="tronquée"):
        g.repondre("comment amortir ?", PASSAGES)
    assert not (tmp_path / "cache.json").exists()


def test_json_illisible_leve_une_erreur_diagnosticable(tmp_path):
    """Un JSON cassé doit dire le modèle, la question et le stop_reason."""
    client = FauxClient()
    client.texte_brut = '{"abstention": false, "reponse": "coup'
    g = Generator(cache_path=tmp_path / "cache.json", client=client)
    with pytest.raises(RuntimeError, match="JSON illisible"):
        g.repondre("comment amortir ?", PASSAGES)
    assert not (tmp_path / "cache.json").exists()
