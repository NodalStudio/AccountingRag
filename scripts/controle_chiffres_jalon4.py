"""Recalcule chaque chiffre publié au jalon 4 depuis les JSON, et échoue sur écart.

C'est la leçon la plus chère du jalon 3 : trois vagues de correctifs y ont été consacrées,
et les trois revues ont trouvé zéro chiffre de résultat faux pour une douzaine
d'affirmations fausses *à propos* de ces chiffres. Un rapport dont la prose dérive de ses
JSON est aussi trompeur qu'un JSON faux.

Le contrôle fait deux choses différentes, et les deux comptent :

  1. **Cohérence interne des JSON** : chaque agrégat est recalculé depuis les données
     brutes du même fichier (`verdicts`, `par_question`, `details`). Un agrégat figé qui ne
     suit plus ses données brutes est le défaut le plus répété du dépôt.
  2. **Cohérence de la prose** : chaque chiffre de la liste `publies` de `main()` doit
     apparaître littéralement dans `docs/eval-jalon4.md`, et sa valeur doit être celle
     recalculée. Un chiffre corrigé dans le JSON mais pas dans le rapport échoue ici.

Le contrôle est gratuit et hors ligne. Il doit échouer si on casse un chiffre : c'est la
loi 5 du dépôt, et `tests/test_controle_chiffres.py` en fait la démonstration.

Usage : uv run python scripts/controle_chiffres_jalon4.py
Sortie : code 0 si tout concorde, 1 sinon (avec la liste des écarts).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESURES = ROOT / "docs/mesures/jalon4"
RAPPORT = ROOT / "docs/eval-jalon4.md"


def _court(chemin: Path) -> str:
    """Chemin relatif au dépôt quand c'est possible, sinon tel quel (cas des tests)."""
    try:
        return str(chemin.relative_to(ROOT))
    except ValueError:
        return str(chemin)


def fr(x: float) -> str:
    """Format français d'un nombre : la virgule décimale, comme dans le rapport."""
    return f"{x}".replace(".", ",")


def recalcule_metriques(m: dict) -> list[str]:
    """Recalcule les agrégats d'un bloc `metriques` depuis ses données brutes."""
    ecarts = []
    verdicts = m["verdicts"]
    total = sum(verdicts.values())
    if total != m["n_citations"]:
        ecarts.append(f"n_citations={m['n_citations']} mais la somme des verdicts "
                      f"vaut {total}")

    def attendu(numerateur: int, denominateur: int):
        return round(numerateur / denominateur, 4) if denominateur else None

    controles = {
        "taux_citations_inexistantes": attendu(verdicts.get("record_inexistant", 0), total),
        "taux_citations_non_portantes": attendu(
            verdicts.get("extrait_absent", 0) + verdicts.get("extrait_trop_court", 0),
            total),
        "taux_citations_version_omise": attendu(
            verdicts.get("version_omise", 0) + verdicts.get("version_ambigue", 0), total),
    }
    for cle, valeur in controles.items():
        if m[cle] != valeur:
            ecarts.append(f"{cle}={m[cle]} mais les verdicts donnent {valeur}")

    # Les taux calculés sur les RÉPONSES, recalculés depuis `par_question`.
    pq = m["par_question"]
    n_abst = sum(1 for v in pq.values() if v == "abstention")
    n_sans = sum(1 for v in pq.values() if v == "sans_citation")
    n_non_abst = len(pq) - n_abst
    if m["taux_abstention"] != attendu(n_abst, len(pq)):
        ecarts.append(f"taux_abstention={m['taux_abstention']} mais par_question donne "
                      f"{attendu(n_abst, len(pq))}")
    if m["taux_reponses_sans_citation"] != attendu(n_sans, n_non_abst):
        ecarts.append(f"taux_reponses_sans_citation={m['taux_reponses_sans_citation']} "
                      f"mais par_question donne {attendu(n_sans, n_non_abst)}")
    if len(pq) != m["n"]:
        ecarts.append(f"n={m['n']} mais par_question porte {len(pq)} questions")

    # Le taux de correspondance brute se recalcule depuis `details`.
    # La clé est au PLURIEL : `citations.metriques` rend `details`. Une première
    # version lisait `detail`, et sa fixture de test employait la même faute — le
    # test passait donc sur un schéma qui n'existe pas. Un test dont la fixture
    # invente le schéma ne teste rien.
    brut = sum(1 for cs in m["details"].values() for c in cs
               if c["correspond_brut"])
    if m["taux_correspondance_brute"] != attendu(brut, total):
        ecarts.append(f"taux_correspondance_brute={m['taux_correspondance_brute']} mais "
                      f"le détail donne {attendu(brut, total)}")
    return ecarts


def recalcule_calibration(c: dict) -> list[str]:
    """Recalcule l'accord juge/humain depuis le détail par cas."""
    sys.path.insert(0, str(ROOT / "src"))
    from accounting_rag.judge import accord
    ecarts = []
    detail = c["detail"]
    cle = lambda d: f"{d['question_id']}|{d['cas_limite']}"  # noqa: E731
    refait = accord({cle(d): d["note_humaine"] for d in detail},
                    {cle(d): d["note_juge"] for d in detail})
    for k in ("n", "exact", "ecart_moyen", "kappa_pondere"):
        if c["accord"][k] != refait[k]:
            ecarts.append(f"accord.{k}={c['accord'][k]} mais le détail donne {refait[k]}")
    if len(detail) != len(c["cas"]):
        ecarts.append(f"{len(c['cas'])} cas notés à la main mais {len(detail)} notés "
                      f"par le juge")
    # Le seuil ne doit pas avoir bougé : il est fixé avant la mesure.
    if c["seuil_kappa"] != 0.6:
        ecarts.append(f"seuil_kappa={c['seuil_kappa']} : le seuil du jalon est 0,6 et "
                      f"n'est pas révisable")
    return ecarts


def main() -> int:
    ecarts: list[str] = []
    charge = {}
    for nom in ("generation_dev", "generation_validation", "generation_abstention",
                "calibration_juge"):
        f = MESURES / f"{nom}.json"
        if not f.is_file():
            ecarts.append(f"{_court(f)} manquant : chiffre non recalculable")
            continue
        charge[nom] = json.loads(f.read_text(encoding="utf-8"))

    for nom, d in charge.items():
        if nom == "calibration_juge":
            ecarts += [f"[{nom}] {e}" for e in recalcule_calibration(d)]
        else:
            ecarts += [f"[{nom}] {e}" for e in recalcule_metriques(d["metriques"])]

    # --- Cohérence de la prose --------------------------------------------------------
    if not RAPPORT.is_file():
        ecarts.append(f"{_court(RAPPORT)} manquant")
    elif len(charge) == 4:
        texte = RAPPORT.read_text(encoding="utf-8")
        dev = charge["generation_dev"]["metriques"]
        val = charge["generation_validation"]["metriques"]
        ab = charge["generation_abstention"]["metriques"]
        cal = charge["calibration_juge"]
        # (libellé, valeur recalculée). Toute valeur doit figurer littéralement dans le
        # rapport, au format français.
        publies = [
            ("dev citations inexistantes", dev["taux_citations_inexistantes"]),
            ("dev citations non portantes", dev["taux_citations_non_portantes"]),
            ("dev version omise", dev["taux_citations_version_omise"]),
            ("dev correspondance brute", dev["taux_correspondance_brute"]),
            ("validation citations inexistantes", val["taux_citations_inexistantes"]),
            ("validation citations non portantes", val["taux_citations_non_portantes"]),
            ("abstention correcte", ab["taux_abstention_correcte"]),
            ("kappa pondéré du juge", cal["accord"]["kappa_pondere"]),
        ]
        for libelle, valeur in publies:
            if valeur is None:
                continue
            if fr(valeur) not in texte and f"{valeur}" not in texte:
                ecarts.append(f"[rapport] {libelle} : {fr(valeur)} absent de "
                              f"{RAPPORT.name} — chiffre publié non recalculable ou périmé")
        # Effectifs cités : ils dérivent des JSON, donc ils doivent en découler.
        for libelle, n in (("n dev", dev["n"]), ("n validation", val["n"]),
                           ("n abstention", ab["n"]),
                           ("n calibration", cal["accord"]["n"])):
            if not re.search(rf"\b{n}\b", texte):
                ecarts.append(f"[rapport] {libelle} = {n} absent de {RAPPORT.name}")

    if ecarts:
        print(f"CONTRÔLE DES CHIFFRES EN ÉCHEC — {len(ecarts)} écart(s) :", file=sys.stderr)
        for e in ecarts:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"CONTRÔLE DES CHIFFRES OK : {len(charge)} artefact(s) recalculé(s), agrégats "
          f"conformes à leurs données brutes, et chaque chiffre publié dans "
          f"{RAPPORT.name} retrouvé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
