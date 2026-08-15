from accounting_rag.evalrag import match, evaluate


def test_match_prefixe():
    assert match("pcg-214-1@2026-01-01", "pcg-214-1")
    assert match("pcg-214-1-c2@2026-01-01", "pcg-214-1")
    assert match("pcg-214-1@2026-01-01#2", "pcg-214-1")
    assert not match("pcg-214-10@2026-01-01", "pcg-214-1")


class FakeSearcher:
    def search(self, query, k=10, mode="hybrid"):
        return [{"record_id": "pcg-214-1@e", "score": 1.0, "source": "bm25",
                 "article": "214-1", "chemin": "", "texte": ""}]


def test_evaluate_recall_parfait_et_nul():
    qs = [
        {"id": "q1", "question": "x", "categorie": "regle", "citations": ["pcg-214-1"]},
        {"id": "q2", "question": "y", "categorie": "regle", "citations": ["pcg-999-9"]},
    ]
    m = evaluate(FakeSearcher(), qs, mode="bm25", k=10)
    assert m["recall@10"] == 0.5
    assert m["mrr"] == 0.5
    assert m["n"] == 2
