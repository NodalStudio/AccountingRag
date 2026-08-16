from accounting_rag.evalrag import match, evaluate, paired_bootstrap


def test_match_prefixe():
    assert match("pcg-214-1@2026-01-01", "pcg-214-1")
    assert match("pcg-214-1-c2@2026-01-01", "pcg-214-1")
    assert match("pcg-214-1@2026-01-01#2", "pcg-214-1")
    assert not match("pcg-214-10@2026-01-01", "pcg-214-1")


class FakeSearcher:
    def search(self, query, k=10, mode="hybrid"):
        # Retourne 10 résultats pour garantir recall@10 indépendant de k
        # Premier résultat toujours pcg-214-1@e, les autres ne matchent rien
        results = [{"record_id": "pcg-214-1@e", "score": 1.0, "source": "bm25",
                    "article": "214-1", "chemin": "", "texte": ""}]
        for i in range(1, 10):
            results.append({"record_id": f"pcg-000-{i}@e", "score": 1.0 - i*0.01, "source": "bm25",
                           "article": "000", "chemin": "", "texte": ""})
        return results


def test_evaluate_recall_parfait_et_nul():
    qs = [
        {"id": "q1", "question": "x", "categorie": "regle", "citations": ["pcg-214-1"]},
        {"id": "q2", "question": "y", "categorie": "regle", "citations": ["pcg-999-9"]},
    ]
    m = evaluate(FakeSearcher(), qs, mode="bm25", k=10)
    assert m["recall@10"] == 0.5
    assert m["mrr"] == 0.5
    assert m["n"] == 2


def test_evaluate_recall10_independent_of_k():
    """recall@10 doit être indépendant de k puisque on récupère toujours k >= 10."""
    qs = [
        {"id": "q1", "question": "x", "categorie": "regle", "citations": ["pcg-214-1"]},
        {"id": "q2", "question": "y", "categorie": "regle", "citations": ["pcg-999-9"]},
    ]
    m_k5 = evaluate(FakeSearcher(), qs, mode="bm25", k=5)
    m_k10 = evaluate(FakeSearcher(), qs, mode="bm25", k=10)
    assert m_k5["recall@10"] == m_k10["recall@10"]


def test_evaluate_expose_par_question():
    qs = [
        {"id": "q1", "question": "x", "categorie": "regle", "citations": ["pcg-214-1"]},
        {"id": "q2", "question": "y", "categorie": "regle", "citations": ["pcg-999-9"]},
    ]
    m = evaluate(FakeSearcher(), qs, mode="bm25", k=10)
    assert m["par_question"] == {"q1": 1.0, "q2": 0.0}


def test_paired_bootstrap_deterministe():
    a = {f"q{i}": 0.0 for i in range(20)}
    b = {f"q{i}": 1.0 for i in range(20)}
    r = paired_bootstrap(a, b)
    assert r["delta"] == 1.0
    assert r["p_amelioration"] == 1.0
    assert r["ic95"] == (1.0, 1.0)
    r2 = paired_bootstrap(a, a)
    assert r2["delta"] == 0.0
    assert r2["p_amelioration"] < 0.95
