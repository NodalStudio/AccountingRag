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
from pathlib import Path

from accounting_rag.ablation import bascules, marge_parmi_les_golds_presents, vecteurs_identiques

ROOT = Path(__file__).resolve().parent.parent


def _affiche(chemin: Path) -> str:
    """Chemin relatif au dépôt quand c'est possible — un test qui redirige la sortie
    vers un `tmp_path` ne doit pas faire échouer un simple message d'information."""
    try:
        return str(chemin.relative_to(ROOT))
    except ValueError:
        return str(chemin)
IN_DIR = OUT_DIR = ROOT / "docs/mesures/jalon3-fix"
SONDES_JALON3 = ROOT / "docs/mesures/jalon3/sondes.json"


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

    # Quel contexte DÉCIDE. `fusion_<split>.json` liste `configurations_adoptees` en
    # mêlant les deux contextes, parce que le script de campagne calcule le drapeau par
    # configuration sans savoir lequel engage le projet. Laisser cette liste parler seule
    # ferait lire une adoption là où il n'y a qu'un mécanisme : le contexte `hybrid` est
    # une fusion nue que PERSONNE n'exécute — ni la démo, ni les campagnes, ni le jalon 4.
    # Le protocole, fixé avant mesure, désigne `livree` comme le contexte d'adoption.
    decide = "livree"
    sortie["decision"] = {
        "contexte_qui_decide": decide,
        "pourquoi": ("le critère d'adoption porte sur la configuration livrée au jalon 3, "
                     "seule configuration réellement exécutée ; le contexte `hybrid` "
                     "mesure le MÉCANISME et n'adopte rien."),
        "adoptees_dans_le_contexte_qui_decide": sorted(
            nom for nom, res in brut["contextes"].get(decide, {}).items()
            if res.get("adopte")),
        "franchissent_le_seuil_dans_le_contexte_mecanisme": sorted(
            nom for nom, res in brut["contextes"].get("hybrid", {}).items()
            if res.get("adopte")),
    }

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
    d = sortie["decision"]
    print(f"\nadoptées dans le contexte qui décide ({d['contexte_qui_decide']}) : "
          f"{d['adoptees_dans_le_contexte_qui_decide'] or 'AUCUNE'}")
    print(f"franchissent le seuil dans le contexte mécanisme (hybrid) : "
          f"{d['franchissent_le_seuil_dans_le_contexte_mecanisme'] or 'aucune'}")
    print(f"\n[anatomie_fusion] écrit {_affiche(chemin)}")


if __name__ == "__main__":
    main()
