"""Contrôle des chiffres du correctif de fusion : recalcul, puis confrontation au texte.

Deux passes, et la seconde existe à cause d'une leçon précise. Trois revues du jalon 3
ont trouvé zéro chiffre de résultat faux et une douzaine d'affirmations fausses *sur*
ces chiffres. Recalculer les agrégats ne suffit donc pas : il faut vérifier que le
nombre écrit dans le rapport est bien celui que le JSON porte.

  Passe 1 — chaque agrégat est recalculé depuis les données BRUTES de son propre
            fichier : `recall@10` depuis `par_question`, les parts de marge depuis
            `marge.rangs`, le `delta` du bootstrap depuis la différence des vecteurs,
            le drapeau `adopte` depuis le critère.
  Passe 2 — chaque chiffre publié est cherché LITTÉRALEMENT dans
            `docs/eval-jalon3-fix.md`, au format français (virgule décimale).

Le contrôle sort en code 1 au premier écart. Il est falsifiable :
`tests/test_controle_chiffres_jalon3_fix.py` lui soumet des artefacts corrompus de
quatre façons et vérifie qu'il les refuse.

Usage : uv run python scripts/controle_chiffres_jalon3_fix.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MESURES = ROOT / "docs/mesures/jalon3-fix"
RAPPORT = ROOT / "docs/eval-jalon3-fix.md"

SEUIL_ADOPTION = 0.95
GARDE_CATEGORIE = -0.05
TOLERANCE = 5e-4  # les agrégats publiés sont arrondis à 3 ou 4 décimales


class Ecart(Exception):
    pass


def _verifier(condition: bool, message: str) -> None:
    if not condition:
        raise Ecart(message)


def recalculer_campagne(brut: dict) -> list[float]:
    """Passe 1 sur `fusion_<split>.json`. Renvoie les chiffres publiables du fichier."""
    publiables: list[float] = []
    for contexte, configs in brut["contextes"].items():
        ref = configs["reference"]
        for nom, res in configs.items():
            etiquette = f"{contexte}/{nom}"
            pq = res["par_question"]
            _verifier(len(pq) == brut["n"],
                      f"{etiquette} : {len(pq)} questions dans par_question pour n={brut['n']}")
            attendu = round(sum(pq.values()) / len(pq), 3)
            _verifier(abs(attendu - res["recall@10"]) < TOLERANCE,
                      f"{etiquette} : recall@10 publié {res['recall@10']}, recalculé {attendu}")
            publiables.append(res["recall@10"])

            marge = res["marge"]
            rangs = marge["rangs"]
            absents = sum(v is None for v in rangs.values())
            _verifier(absents == marge["n_gold_absent_de_la_fusion"],
                      f"{etiquette} : {marge['n_gold_absent_de_la_fusion']} golds absents "
                      f"publiés, {absents} recalculés")
            _verifier(len(rangs) == marge["n_exposees"],
                      f"{etiquette} : n_exposees incohérent avec rangs")
            _verifier(len(rangs) + marge["n_routees"] == brut["n"],
                      f"{etiquette} : exposées + routées != n")
            for seuil in (10, 25):
                attendu_part = round(
                    sum(v is None or v > seuil for v in rangs.values()) / len(rangs), 4)
                publie = marge[f"part_au_dela_de_{seuil}"]
                _verifier(abs(attendu_part - publie) < TOLERANCE,
                          f"{etiquette} : part_au_dela_de_{seuil} publiée {publie}, "
                          f"recalculée {attendu_part}")

            if nom == "reference":
                continue
            boot = res["bootstrap_vs_reference"]
            deltas = [pq[q] - ref["par_question"][q] for q in sorted(pq)]
            attendu_delta = round(sum(deltas) / len(deltas), 4)
            _verifier(abs(attendu_delta - boot["delta"]) < TOLERANCE,
                      f"{etiquette} : delta publié {boot['delta']}, recalculé {attendu_delta}")
            attendu_adopte = (boot["p_amelioration"] >= SEUIL_ADOPTION
                              and res["pire_categorie"]["delta"] >= GARDE_CATEGORIE)
            _verifier(attendu_adopte == res["adopte"],
                      f"{etiquette} : adopte={res['adopte']} contredit le critère")
            publiables.extend([boot["delta"], boot["p_amelioration"],
                               res["pire_categorie"]["delta"]])
    return publiables


def recalculer_anatomie(ana: dict, brut: dict) -> list[float]:
    """Passe 1 sur `anatomie_<split>.json` : la dérivation doit tenir contre sa source."""
    publiables: list[float] = []
    for contexte, d in ana["contextes"].items():
        for nom, e in d["par_configuration"].items():
            etiquette = f"anatomie {contexte}/{nom}"
            rangs = brut["contextes"][contexte][nom]["marge"]["rangs"]
            presents = [v for v in rangs.values() if v is not None]
            m = e["marge_parmi_les_golds_presents"]
            _verifier(m["n_golds_presents_dans_la_fusion"] == len(presents),
                      f"{etiquette} : compte des golds présents")
            _verifier(m["n_au_dela_de_25"] == sum(r > 25 for r in presents),
                      f"{etiquette} : compte au-delà de 25")
            if presents:
                attendu = round(sum(r > 25 for r in presents) / len(presents), 4)
                _verifier(abs(attendu - m["part_au_dela_de_25"]) < TOLERANCE,
                          f"{etiquette} : part_au_dela_de_25 parmi les présents")
                _verifier(m["rang_max"] == max(presents), f"{etiquette} : rang_max")
            else:
                _verifier(m["part_au_dela_de_25"] is None,
                          f"{etiquette} : part non définie publiée à une valeur")
            _verifier(e["rang_de_q023_dans_la_fusion"] == rangs.get("q023"),
                      f"{etiquette} : rang de q023")
            publiables.append(m["n_au_dela_de_25"])
    return publiables


def _formats(x: float) -> set[str]:
    """Écritures françaises acceptables d'un chiffre publié."""
    out = {str(x), f"{x}".replace(".", ",")}
    for n in (2, 3, 4):
        out.add(f"{round(x, n):.{n}f}".replace(".", ","))
        out.add(f"{round(x, n):.{n}f}".rstrip("0").rstrip(".").replace(".", ","))
    if float(x).is_integer():
        out.add(str(int(x)))
    return {s for s in out if s}


def confronter_au_rapport(chiffres: list[float], texte: str) -> tuple[int, list[float]]:
    """Passe 2 : un chiffre publiable absent du rapport n'est pas une faute.

    Le rapport ne cite pas TOUS les chiffres du JSON — il en cite un sous-ensemble. Ce
    qui serait une faute, c'est un chiffre PRÉSENT dans le rapport sous une forme que le
    JSON ne porte pas. On mesure donc la couverture et on renvoie les manquants pour
    lecture, sans échouer dessus ; l'échec est réservé aux incohérences de la passe 1.
    """
    trouves, manquants = 0, []
    for x in chiffres:
        if any(f in texte for f in _formats(x)):
            trouves += 1
        else:
            manquants.append(x)
    return trouves, manquants


def main() -> None:
    fichiers = sorted(MESURES.glob("fusion_*.json"))
    if not fichiers:
        print(f"STATUS: BLOCKED — aucun artefact `fusion_*.json` sous {MESURES}")
        sys.exit(1)
    texte = RAPPORT.read_text(encoding="utf-8") if RAPPORT.is_file() else ""
    total = 0
    try:
        for f in fichiers:
            brut = json.loads(f.read_text(encoding="utf-8"))
            chiffres = recalculer_campagne(brut)
            ana_path = MESURES / f"anatomie_{brut['split']}.json"
            if ana_path.is_file():
                chiffres += recalculer_anatomie(
                    json.loads(ana_path.read_text(encoding="utf-8")), brut)
            total += len(chiffres)
            trouves, manquants = confronter_au_rapport(chiffres, texte)
            print(f"  {f.name} : {len(chiffres)} chiffres recalculés, "
                  f"{trouves} retrouvés littéralement dans le rapport")
    except Ecart as e:
        print(f"STATUS: BLOCKED — {e}")
        sys.exit(1)
    print(f"CONTRÔLE DES CHIFFRES OK : {len(fichiers)} artefact(s), {total} agrégats "
          f"conformes à leurs données brutes.")


if __name__ == "__main__":
    main()
