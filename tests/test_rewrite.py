import json
from accounting_rag.config import charge_env
from accounting_rag.rewrite import Rewriter


class FauxBloc:
    def __init__(self, texte):
        self.type = "text"
        self.text = texte


class FauxMessage:
    def __init__(self, texte):
        self.content = [FauxBloc(texte)]


class FauxClient:
    """Client Anthropic factice : enregistre les appels, ne sort jamais sur le réseau."""

    def __init__(self, reponse="amortissement immobilisation corporelle"):
        self.reponse = reponse
        self.appels = []
        self.messages = self

    def create(self, **kwargs):
        self.appels.append(kwargs)
        return FauxMessage(self.reponse)


def test_charge_env_sans_ecraser(tmp_path, monkeypatch):
    fichier = tmp_path / ".env"
    fichier.write_text("ANTHROPIC_API_KEY=depuis-le-fichier\nAUTRE=x\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("AUTRE", "deja-defini")
    charge_env(fichier)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "depuis-le-fichier"
    assert os.environ["AUTRE"] == "deja-defini"  # jamais écrasé


def test_charge_env_silencieux_si_absent(tmp_path):
    charge_env(tmp_path / "inexistant")  # ne doit pas lever


def test_reecrire_appelle_le_modele_et_met_en_cache(tmp_path):
    client = FauxClient()
    r = Rewriter(cache_path=tmp_path / "cache.json", client=client)
    a = r.reecrire("comment je répartis le coût d'une machine ?")
    b = r.reecrire("comment je répartis le coût d'une machine ?")
    assert a == b == "amortissement immobilisation corporelle"
    assert len(client.appels) == 1  # deuxième appel servi par le cache
    cache = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert cache["comment je répartis le coût d'une machine ?"] == a


def test_cache_relu_depuis_le_disque(tmp_path):
    (tmp_path / "cache.json").write_text(
        json.dumps({"q": "reecriture-en-cache"}), encoding="utf-8"
    )
    client = FauxClient()
    r = Rewriter(cache_path=tmp_path / "cache.json", client=client)
    assert r.reecrire("q") == "reecriture-en-cache"
    assert client.appels == []  # aucun appel API


def test_reecriture_vide_leve_et_ne_pollue_pas_le_cache(tmp_path):
    """Une réponse sans texte doit échouer bruyamment, pas devenir une requête vide.

    Le cache étant committé, une réécriture vide mise en cache serait rejouée
    silencieusement par toutes les campagnes suivantes.
    """
    import pytest
    client = FauxClient(reponse="   ")
    cache = tmp_path / "cache.json"
    r = Rewriter(cache_path=cache, client=client)
    with pytest.raises(RuntimeError, match="réécriture vide"):
        r.reecrire("ma question")
    assert not cache.exists()
    assert r._cache == {}


def test_le_rewriter_ne_recoit_que_la_question(tmp_path):
    """Intégrité du benchmark : ni gold, ni corpus, ni résultats dans le prompt."""
    client = FauxClient()
    r = Rewriter(cache_path=tmp_path / "cache.json", client=client)
    r.reecrire("ma question")
    envoye = json.dumps(client.appels[0], default=str)
    assert "ma question" in envoye
    for interdit in ("pcg-", "citations", "gold", "record_id"):
        assert interdit not in envoye
