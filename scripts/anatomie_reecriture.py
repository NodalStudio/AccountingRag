"""Anatomie du second correctif du jalon 3 : ce que la grille agrégée ne dit pas.

Ne mesure rien — relit `docs/mesures/jalon3-fix/reecriture_<split>.json` et en dérive les
chiffres qui décident de la lecture. Aucun modèle, aucun GPU, aucune API.

Le chiffre qui compte ici n'est pas le même que pour le correctif de fusion, et c'est le
point. La fusion réordonne l'intérieur du pool : sa métrique était la profondeur du gold
dans le classement. Ce levier-ci change la COMPOSITION du pool — q080 casse parce que son
gold en SORT (rang bm25 7 -> 84, pool de 50). La métrique décisive est donc le nombre de
golds **absents du pool**, qu'aucun reranker ne peut rattraper.

Trois questions témoins, celles que la réécriture casse, avec leurs trois mécanismes
distincts :

  - q080 : gold hors du pool ;
  - q025 : gold dans la fenêtre du reranker (rang 22) mais non retenu ;
  - q008 : rang lexical inchangé — sa dégradation relève de l'éviction par la fusion,
           mesurée dans l'autre correctif. Elle est suivie ici pour que le recoupement
           entre les deux dettes reste visible plutôt que raconté.

Usage : uv run python scripts/anatomie_reecriture.py [--split dev]
"""
import argparse
import json

from accounting_rag.ablation import (ROOT, bascules, marge_parmi_les_golds_presents,
                                     vecteurs_identiques, _affiche)

IN_DIR = OUT_DIR = ROOT / "docs/mesures/jalon3-fix"

# Les trois questions que la réécriture casse dans la configuration livrée, établies au
# jalon 3 (`docs/mesures/jalon3/cloture_dev.json`, B vs C) et non choisies après coup.
TEMOINS = ("q008", "q025", "q080")
CONTEXTE_QUI_DECIDE = "livree"
CONTEXTE_MECANISME = "reecriture"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    brut = json.loads((IN_DIR / f"reecriture_{args.split}.json").read_text(encoding="utf-8"))
    sortie = {"split": args.split, "source": f"reecriture_{args.split}.json",
              "questions_temoins": list(TEMOINS), "contextes": {}}

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
            rangs = res["marge"]["rangs"]
            entree = {
                "recall@10": res["recall@10"],
                "marge_parmi_les_golds_presents": marge_parmi_les_golds_presents(res["marge"]),
                # `None` = gold absent du pool, donc hors de portée de tout reranker.
                # C'est la distinction que ce levier vise, et elle disparaîtrait si on
                # ne publiait qu'un rang médian.
                "rangs_des_temoins": {q: rangs.get(q, "routée") for q in TEMOINS},
            }
            if nom != "reference":
                entree["bascules"] = bascules(ref["par_question"], res["par_question"])
                entree["p_amelioration"] = res["bootstrap_vs_reference"]["p_amelioration"]
                entree["adopte"] = res["adopte"]
            par_contexte["par_configuration"][nom] = entree
        sortie["contextes"][contexte] = par_contexte

    sortie["decision"] = {
        "contexte_qui_decide": CONTEXTE_QUI_DECIDE,
        "pourquoi": ("le critère d'adoption porte sur la configuration livrée au jalon 3, "
                     "seule configuration réellement exécutée ; le contexte "
                     f"`{CONTEXTE_MECANISME}` mesure le MÉCANISME et n'adopte rien."),
        "adoptees_dans_le_contexte_qui_decide": sorted(
            nom for nom, res in brut["contextes"].get(CONTEXTE_QUI_DECIDE, {}).items()
            if res.get("adopte")),
        "franchissent_le_seuil_dans_le_contexte_mecanisme": sorted(
            nom for nom, res in brut["contextes"].get(CONTEXTE_MECANISME, {}).items()
            if res.get("adopte")),
    }

    chemin = OUT_DIR / f"anatomie_reecriture_{args.split}.json"
    chemin.write_text(json.dumps(sortie, ensure_ascii=False, indent=2, sort_keys=True),
                      encoding="utf-8")

    for contexte, d in sortie["contextes"].items():
        print(f"\n=== {contexte} ===")
        print(f"  recall identique : {d['configurations_au_vecteur_de_recall_identique']}")
        print(f"  classement identique : {d['configurations_au_classement_identique']}")
        for nom, e in d["par_configuration"].items():
            m = e["marge_parmi_les_golds_presents"]
            b = e.get("bascules", {})
            temoins = " ".join(f"{q}={e['rangs_des_temoins'][q]}" for q in TEMOINS)
            print(f"  {nom:20s} recall={e['recall@10']:<6} "
                  f"hors_pool={m['n_golds_absents_du_pool']:<3} "
                  f">25={m['n_au_dela_de_25']:<3} max={m['rang_max']:<5} {temoins:34s} "
                  f"{'+' + str(b['n_gagnees']) + '/-' + str(b['n_perdues']) if b else ''}")
    d = sortie["decision"]
    print(f"\nadoptées dans le contexte qui décide ({d['contexte_qui_decide']}) : "
          f"{d['adoptees_dans_le_contexte_qui_decide'] or 'AUCUNE'}")
    print(f"franchissent le seuil dans le contexte mécanisme ({CONTEXTE_MECANISME}) : "
          f"{d['franchissent_le_seuil_dans_le_contexte_mecanisme'] or 'aucune'}")
    print(f"\n[anatomie_reecriture] écrit {_affiche(chemin)}")


if __name__ == "__main__":
    main()
