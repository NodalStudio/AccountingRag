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

    # Compteur mis à jour après Rulings 17/18 (SECTION_HEADER ferme l'article
    # courant ; le contenu hors article devient un record pcg-na-<n> au lieu
    # d'une anomalie-poubelle). Avant ces rulings : 292 collisions, dont 290
    # provoquées par la fuite d'annexes/tableaux entiers sur le dernier
    # article valide (34 % des records réglementaires mal attribués).
    #
    # Après Rulings 17/18, vérifié empiriquement : 80 collisions. Le cas
    # anticipé « Article 1er » (pcg-500-2@2026-01-01, Titres 7/8) tombe à
    # ZÉRO — pas à ~2 comme prévu à la consigne : Ruling 17 ferme déjà
    # l'article sur le SECTION_HEADER « Titre 7 »/« Titre 8 » qui précède
    # cette prose, donc ce contenu devient nativement un record pcg-na-<n>
    # (voir test_zone_plan_de_comptes_produit_des_records_na) plutôt qu'une
    # collision. Les 80 collisions résiduelles sont un phénomène RÉSIDUEL
    # et DISTINCT, confirmé par investigation manuelle : un même article
    # encore ouvert peut légitimement produire plusieurs fragments
    # réglementaires non contigus (alinéas interrompus par une strate de
    # commentaire intermédiaire, sans SECTION_HEADER intercalé) — l'id
    # réglementaire n'a pas de compteur (contrairement au commentaire qui a
    # com_count). Sur les 80 : ~41 sont de purs artefacts de folio (un
    # numéro de page isolé classé par erreur dans la strate REGLEMENTAIRE,
    # ex. pcg-628-18@2026-01-01#2 dont le texte est "302" — défaut de
    # classify.py, hors périmètre T7) et ~39 portent du texte réglementaire
    # réel (ex. pcg-223-4@2026-01-01#2, pcg-512-1@2026-01-01#2). Compteur
    # figé en garde de régression — voir task-7-report.md pour l'analyse
    # complète.
    assert len(collisions) == 80
    assert sum(1 for a in collisions if a.ligne == "pcg-500-2@2026-01-01") == 0

    # Ruling 18 : plus aucune anomalie-poubelle « texte avant tout article »
    # ni « commentaire orphelin » — tout ce contenu est désormais un record.
    assert not any(a.raison.startswith("texte avant tout article") for a in anomalies)
    assert not any(a.raison.startswith("commentaire orphelin") for a in anomalies)

    # Les vrais articles réglementaires retombent près de la prévision du
    # contrôleur (~554 = 845 − 291 anciens fragments mal attribués).
    reg_avec_article = {r.article for r in records if r.type == "reglementaire" and r.article is not None}
    assert len(reg_avec_article) == 554

    # Les deux clusters massifs de fuite d'annexe (plan de comptes des
    # coopératives / fonctionnement des comptes) ne portent plus la
    # centaine de fragments observée avant Rulings 17/18.
    assert sum(1 for r in records if r.article == "628-18") < 5
    assert sum(1 for r in records if r.article == "1231-89") < 5


def test_section_header_ferme_larticle_courant():
    # Ruling 17 : un SECTION_HEADER ferme le périmètre de l'article courant.
    # Le texte qui suit une section intercalée, sans nouvel « Art. », ne doit
    # jamais hériter de l'article précédent.
    b = _Builder("2026-01-01")
    b.feed(_line("Chapitre 1 – Premier chapitre", page=1), Kind.SECTION_HEADER)
    b.feed(_line("Art. 100-1", page=1), Kind.ARTICLE_HEADER)
    b.feed(_line("Premier texte réglementaire, dans l'article 100-1.", page=1), Kind.REGLEMENTAIRE)
    b.feed(_line("Chapitre 2 – Second chapitre", page=2), Kind.SECTION_HEADER)
    b.feed(_line("Second texte, ne doit pas porter l'article 100-1.", page=2), Kind.REGLEMENTAIRE)
    b.flush()

    by_text = {r.texte: r for r in b.records}
    premier = by_text["Premier texte réglementaire, dans l'article 100-1."]
    second = by_text["Second texte, ne doit pas porter l'article 100-1."]

    assert premier.article == "100-1"
    assert premier.id == "pcg-100-1@2026-01-01"

    assert second.article is None                      # Ruling 17 : pas d'héritage
    assert second.id == "pcg-na-1@2026-01-01"           # Ruling 18 : record ancré à la section
    assert "Chapitre 2" in second.chemin
    assert not any(a.raison.startswith("collision") for a in b.anomalies)


def test_zone_plan_de_comptes_produit_des_records_na(recueil_path):
    # Ruling 18 : le contenu hors article (plans de comptes en tables,
    # annexes d'exemples) devient un record pcg-na-<n> ancré à la section
    # courante, plutôt qu'une anomalie-poubelle ou un article mal attribué.
    records, _ = parse(recueil_path)
    na_recs = [r for r in records if r.id.startswith("pcg-na-")]
    plan_recs = [r for r in na_recs if "plan de comptes" in r.chemin.lower()]

    assert plan_recs, "aucun record na-* rattaché à une zone « plan de comptes »"
    assert all(r.article is None for r in plan_recs)
    assert all(r.chemin for r in plan_recs)   # chemin non vide malgré l'absence d'article
