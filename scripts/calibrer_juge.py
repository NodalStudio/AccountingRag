"""Calibration du LLM-juge : mesure son accord avec 30 notes humaines.

Le seuil est fixé AVANT la mesure et lu dans le JSON de calibration lui-même
(`seuil_kappa`), pour qu'il ne puisse pas être déplacé après avoir vu le résultat. Sous
le seuil, le script sort en code 1 et le juge ne publie aucun chiffre : l'échec est un
résultat négatif publiable, pas un bug à contourner.

Les notes humaines sont la RÉFÉRENCE : le script ne les récrit jamais.

Cache d'exécution : `data/juge-calibration-cache.json` (gitignoré).

Usage : uv run python scripts/calibrer_juge.py
"""
import json
import sys
from pathlib import Path

from accounting_rag.judge import Judge, accord

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = ROOT / "docs/mesures/jalon4/calibration_juge.json"
SPLITS = ("dev", "abstention")  # le jeu de calibration puise dans les deux
CACHE_JUGE = ROOT / "data/juge-calibration-cache.json"


def charger_reponses() -> dict[str, dict]:
    """Le cache du générateur est indexé par une clé JSON [question, [record_id…]].

    On le réindexe par question_id en relisant les benchmarks, seule source qui relie un
    id de question à son texte. Les deux splits sont chargés : le jeu de calibration tire
    ses cas « abstention correcte » du split d'abstention et tous les autres de dev.
    """
    from accounting_rag.evalrag import load_benchmark
    par_question: dict[str, dict] = {}
    for split in SPLITS:
        cache = ROOT / f"docs/mesures/jalon4/reponses_{split}.json"
        if not cache.is_file():
            continue
        brut = json.loads(cache.read_text(encoding="utf-8"))
        par_texte = {json.loads(cle)[0]: valeur for cle, valeur in brut.items()}
        for q in load_benchmark(ROOT / f"benchmark/{split}.jsonl"):
            if q["question"] in par_texte:
                par_question[q["id"]] = par_texte[q["question"]]
    return par_question


def main() -> None:
    calib = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    seuil = calib["seuil_kappa"]
    reponses = charger_reponses()

    # Deux des cinq cas limites n'ont aucune instance réelle dans la campagne (« fausse
    # mais bien citée » : 0 citation hallucinée sur 422 ; « abstention excessive » :
    # l'unique abstention de dev est fondée). Ces cas portent donc une réponse `reponse`
    # en ligne, fabriquée à partir d'une réponse réelle, et le champ `origine` la
    # distingue de `campagne`. Un jeu de calibration éprouve la discrimination du juge ;
    # il n'échantillonne pas le comportement du système.
    manquantes = [c["question_id"] for c in calib["cas"]
                  if "reponse" not in c and c["question_id"] not in reponses]
    if manquantes:
        print(f"STATUS: BLOCKED — réponses absentes du cache pour {manquantes}. "
              f"Lancer d'abord scripts/eval_generation.py sur les splits "
              f"{', '.join(SPLITS)}.", file=sys.stderr)
        raise SystemExit(1)

    juge = Judge(cache_path=CACHE_JUGE)
    humaines, notes_juge, detail = {}, {}, []
    for cas in calib["cas"]:
        qid = cas["question_id"]
        reponse = cas.get("reponse") or reponses[qid]
        out = juge.noter(cas["question_texte"], reponse, cas["bareme"])
        # Clé = question_id + cas limite : six énoncés apparaissent deux fois, une fois
        # avec leur réponse réelle et une fois avec une abstention fabriquée. Indexer sur
        # le seul question_id ferait silencieusement tomber le kappa de 30 cas à 24.
        cle = f"{qid}|{cas['cas_limite']}"
        humaines[cle] = cas["note_humaine"]
        notes_juge[cle] = out["note"]
        detail.append({"question_id": qid, "cas_limite": cas["cas_limite"],
                       "origine": cas.get("origine", "campagne"),
                       "note_humaine": cas["note_humaine"], "note_juge": out["note"],
                       "sur": out["sur"], "par_critere": out["par_critere"]})

    a = accord(humaines, notes_juge)

    # Désaccord par cas limite : c'est ce qui rend un échec exploitable plutôt que subi.
    par_cas: dict[str, list[int]] = {}
    for d in detail:
        par_cas.setdefault(d["cas_limite"], []).append(
            abs(d["note_humaine"] - d["note_juge"]))
    a["ecart_moyen_par_cas_limite"] = {
        k: round(sum(v) / len(v), 4) for k, v in sorted(par_cas.items())}

    # Accord sur le sous-ensemble `campagne` seul : un kappa porté par des cas fabriqués
    # ne dit pas la même chose qu'un kappa mesuré sur ce que le système produit vraiment.
    camp = [d for d in detail if d.get("origine", "campagne") == "campagne"]
    if camp:
        a["accord_campagne_seule"] = accord(
            {f"{d['question_id']}|{d['cas_limite']}": d["note_humaine"] for d in camp},
            {f"{d['question_id']}|{d['cas_limite']}": d["note_juge"] for d in camp})

    calib["notes_juge"] = notes_juge          # les notes humaines restent intactes
    calib["accord"] = a
    calib["detail"] = detail
    calib["modele_juge"] = juge.modele
    calib["cout"] = {"appels_api": juge.appels, "tokens_entree": juge.tokens_entree,
                     "tokens_sortie": juge.tokens_sortie}
    CALIBRATION.write_text(json.dumps(calib, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"\n[calibration] n={a['n']} accord exact={a['exact']} "
          f"écart moyen={a['ecart_moyen']} kappa pondéré={a['kappa_pondere']}")
    for cas, ecart in a["ecart_moyen_par_cas_limite"].items():
        print(f"  {cas:26s} écart moyen {ecart}")
    if a["kappa_pondere"] < seuil:
        print(f"\n[calibration] kappa {a['kappa_pondere']} < seuil {seuil} : LE JUGE NE "
              f"PUBLIE AUCUN CHIFFRE. Documenter l'échec comme résultat négatif dans "
              f"docs/eval-jalon4.md, avec les cas limites où le désaccord se concentre. "
              f"Ne pas réviser le seuil.", file=sys.stderr)
        raise SystemExit(1)
    print(f"[calibration] kappa {a['kappa_pondere']} >= seuil {seuil} : le juge sert.")


if __name__ == "__main__":
    main()
