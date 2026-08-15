from accounting_rag.evalrag import match, evaluate


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
