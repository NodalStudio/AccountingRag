"""Anatomie du correctif de fusion : ce que la grille brute ne dit pas encore.

Ne mesure RIEN — relit `docs/mesures/jalon3-fix/fusion_<split>.json` et en dérive les
chiffres que le tableau agrégé laisse ambigus. Aucun modèle, aucun GPU, aucune API :
tout est recalculé depuis les vecteurs `par_question` et `marge.rangs` déjà persistés,
ce qui rend chaque chiffre publié reproductible sans re-dépenser une campagne (loi 1).

Trois questions, chacune posée parce que la grille brute invite à une conclusion fausse :

1. **Deux leviers différents rendent des chiffres identiques.** `rrf_k=5` et
   `poids_consensus=0,1` donnent le même recall, le même delta et le même
   `p_amelioration` à la quatrième décimale. Un `p` identique impose des deltas
   par question identiques — mais cela n'impose PAS des classements identiques.
   Conclure « les deux leviers sont équivalents » sans regarder les rangs serait
   exactement le genre de mécanisme plausible et faux que le jalon 3 a livré trois fois.

2. **La marge au-delà de 25 ne bouge presque pas.** La lecture tentante — « la règle de
   fusion ne change pas la marge » — confondrait deux causes. Un gold ABSENT du pool
   n'est pas évincé par la fusion : aucune règle de fusion ne peut classer un candidat
   qui n'est pas là. Le seul chiffre qui parle de la fusion est la marge parmi les golds
   PRÉSENTS dans la fusion.

3. **Quelles questions basculent**, et q023 en particulier — le cas qui a motivé tout
   ce correctif.

Usage : uv run python scripts/anatomie_fusion.py [--split dev]
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = OUT_DIR = ROOT / "docs/mesures/jalon3-fix"
SONDES_JALON3 = ROOT / "docs/mesures/jalon3/sondes.json"


def vecteurs_identiques(configs: dict, cle: str) -> list[list[str]]:
    """Groupes de configurations dont le vecteur `cle` est identique, valeur par valeur."""
    groupes: dict[str, list[str]] = {}
    for nom, res in configs.items():
        source = res[cle] if cle in res else res["marge"]["rangs"]
        empreinte = json.dumps(source, sort_keys=True)
        groupes.setdefault(empreinte, []).append(nom)
    return [sorted(noms) for noms in groupes.values() if len(noms) > 1]


def marge_parmi_les_golds_presents(marge: dict) -> dict:
    """La seule décomposition qui parle de la RÈGLE DE FUSION.

    `part_au_dela_de_25` mélange deux causes qui n'ont rien à voir : un gold absent du
    pool (défaut de COUVERTURE, que la fusion ne peut pas corriger) et un gold présent
    mais mal classé (défaut de CLASSEMENT, le sujet de ce correctif). Le jalon 3 avait
    déjà nommé ce découplage sur la couverture du pool ; l'oublier ici referait la même
    erreur d'un cran plus loin.
    """
    presents = [r for r in marge["rangs"].values() if r is not None]
    n = len(presents)
    return {
        "n_golds_presents_dans_la_fusion": n,
        "n_golds_absents_du_pool": marge["n_gold_absent_de_la_fusion"],
        "rang_median": statistics.median(presents) if presents else None,
        "rang_moyen": round(statistics.fmean(presents), 2) if presents else None,
        "rang_max": max(presents) if presents else None,
        "part_au_dela_de_10": round(sum(r > 10 for r in presents) / n, 4) if n else None,
        "part_au_dela_de_25": round(sum(r > 25 for r in presents) / n, 4) if n else None,
        "n_au_dela_de_25": sum(r > 25 for r in presents),
    }


def bascules(ref: dict, cfg: dict) -> dict:
    """Questions dont le recall@10 change entre la référence et une configuration."""
    gagnees = sorted(q for q in ref if cfg[q] > ref[q])
    perdues = sorted(q for q in ref if cfg[q] < ref[q])
    return {"gagnees": gagnees, "perdues": perdues,
            "n_gagnees": len(gagnees), "n_perdues": len(perdues)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    brut = json.loads((IN_DIR / f"fusion_{args.split}.json").read_text(encoding="utf-8"))
    q023_gold = json.loads(SONDES_JALON3.read_text(encoding="utf-8"))["anatomie_q023"]

    sortie = {"split": args.split, "source": f"fusion_{args.split}.json", "contextes": {}}
    for contexte, configs in brut["contextes"].items():
        ref = configs["reference"]
        par_contexte = {
            "configurations_au_vecteur_de_recall_identique":
                vecteurs_identiques(configs, "par_question"),
            "configurations_au_classement_identique":
                vecteurs_identiques(configs, "rangs"),
            "par_configuration": {},
        }
        for nom, res in configs.items():
            entree = {
                "recall@10": res["recall@10"],
                "marge_parmi_les_golds_presents":
                    marge_parmi_les_golds_presents(res["marge"]),
                "rang_de_q023_dans_la_fusion": res["marge"]["rangs"].get("q023"),
            }
            if nom != "reference":
                entree["bascules"] = bascules(ref["par_question"], res["par_question"])
                entree["p_amelioration"] = res["bootstrap_vs_reference"]["p_amelioration"]
                entree["adopte"] = res["adopte"]
            par_contexte["par_configuration"][nom] = entree
        sortie["contextes"][contexte] = par_contexte

    sortie["q023_publie_au_jalon3"] = {
        "gold": q023_gold["gold"]["record"],
        "rang_apres_fusion_publie": q023_gold["gold"]["rang_apres_fusion"],
        "note": ("le rang publié au jalon 3 vaut pour le mode `hybrid` SANS réécriture, "
                 "sur le pool de 81 candidats de la sonde — il n'est comparable qu'au "
                 "contexte `hybrid` ci-dessus, et seulement à titre indicatif : dev "
                 "compte 93 questions ici contre 61 au jalon 3, mais q023 est la même "
                 "question et son gold le même record."),
    }

    chemin = OUT_DIR / f"anatomie_{args.split}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=2, sort_keys=True),
                      encoding="utf-8")

    for contexte, d in sortie["contextes"].items():
        print(f"\n=== {contexte} ===")
        print(f"  recall identique : {d['configurations_au_vecteur_de_recall_identique']}")
        print(f"  classement identique : {d['configurations_au_classement_identique']}")
        for nom, e in d["par_configuration"].items():
            m = e["marge_parmi_les_golds_presents"]
            b = e.get("bascules", {})
            print(f"  {nom:22s} recall={e['recall@10']:<6} "
                  f"golds_presents={m['n_golds_presents_dans_la_fusion']:<3} "
                  f"med={m['rang_median']:<5} max={m['rang_max']:<4} "
                  f">25={m['n_au_dela_de_25']:<3} q023={e['rang_de_q023_dans_la_fusion']} "
                  f"{'+' + str(b['n_gagnees']) + '/-' + str(b['n_perdues']) if b else ''}")
    print(f"\n[anatomie_fusion] écrit {chemin.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
