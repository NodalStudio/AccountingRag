from accounting_rag.model import Line, Kind
from accounting_rag.parse import parse, _Builder


def _line(text, page=1):
    return Line(text=text, size=10.0, bold=True, font="Tahoma", x=99.0, y=300.0, page=page)


def test_pages_40_41(recueil_path):
    records, anomalies = parse(recueil_path)
    by_id = {r.id: r for r in records}

    art = by_id["pcg-212-5@2026-01-01"]
    assert art.type == "reglementaire"
    assert art.texte.startswith("Le titulaire d'un contrat de crédit-bail")
    assert art.page_debut == 40
    assert "Sous-section" not in art.texte           # les titres ne fuient pas dans le texte

    # le commentaire qui suit 212-5 lui est rattaché, avec sa provenance
    c = by_id.get("pcg-212-5-c1@2026-01-01")
    assert c is not None and c.type == "commentaire_ANC"
    assert c.opposable is False
    assert "avis-cu-2006-C" in [r.cible for r in c.renvois]

    # la série d'articles de la sous-section 2 est présente
    for num in ("212-6", "212-7", "212-8", "212-9", "212-10", "212-11"):
        assert f"pcg-{num}@2026-01-01" in by_id

    # le chemin porte la sous-section pour 212-6
    assert "Sous-section 2" in by_id["pcg-212-6@2026-01-01"].chemin


def test_articles_reglementaires_opposables_non(recueil_path):
    records, _ = parse(recueil_path)
    assert all(r.opposable is False for r in records)  # rien d'opposable dans l'ANC (≠ BOFiP)


def test_collision_id_suffixee_et_signalee():
    # Cas synthétique minimal : deux « articles » distincts qui produisent le
    # même numéro (ex. cur_article resté figé faute d'en-tête « Art. NNN-N »
    # reconnu, comme p.660-662 « Article 1er » dans le PDF réel).
    b = _Builder("2026-01-01")
    b.feed(_line("Art. 100-1", page=1), Kind.ARTICLE_HEADER)
    b.feed(_line("Premier texte réglementaire.", page=1), Kind.REGLEMENTAIRE)
    b.feed(_line("Art. 100-1", page=2), Kind.ARTICLE_HEADER)
    b.feed(_line("Second texte, article distinct mais même numéro.", page=2), Kind.REGLEMENTAIRE)
    b.flush()

    ids = [r.id for r in b.records]
    assert ids == ["pcg-100-1@2026-01-01", "pcg-100-1@2026-01-01#2"]
    assert len(set(ids)) == len(ids)
    collisions = [a for a in b.anomalies if a.raison.startswith("collision d'identifiant")]
    assert len(collisions) == 1
    assert collisions[0].ligne == "pcg-100-1@2026-01-01"


def test_ids_uniques_sur_corpus_reel(recueil_path):
    records, anomalies = parse(recueil_path)
    ids = [r.id for r in records]
    assert len(set(ids)) == len(ids)  # aucun id dupliqué après désambiguïsation

    collisions = [a for a in anomalies if a.raison.startswith("collision d'identifiant")]
    # Auto-cohérence du mécanisme : chaque collision produit exactement un id
    # suffixé "#n" et réciproquement (pas de suffixage silencieux, pas
    # d'anomalie orpheline).
    suffixed = [i for i in ids if "#" in i.rsplit("@", 1)[-1]]
    assert len(collisions) == len(suffixed)
    assert len(collisions) > 0

    # Vérifié empiriquement sur le corpus réel (662 pages) : 292 collisions au
    # total, dont seulement 2 correspondent au cas anticipé (pcg-500-2@2026-01-01,
    # « Article 1er » d'annexe hors format Art. NNN-N). Les 290 restantes ont une
    # cause distincte, plus large : de longues zones sans aucun ARTICLE_HEADER
    # (plan de comptes / annexes d'exemples chiffrés) où cur_article reste figé
    # pendant que le contenu alterne entre les strates REGLEMENTAIRE et
    # COMMENTAIRE, générant de nombreux fragments réglementaires qui partagent
    # le même id de base. Compteur figé ici comme garde de régression — voir
    # task-7-report.md pour l'analyse complète.
    assert len(collisions) == 292
    assert sum(1 for a in collisions if a.ligne == "pcg-500-2@2026-01-01") == 2
