"""Machinerie commune des ablations à deux contextes (correctifs du jalon 3).

Extraite de `scripts/ablations_fusion.py` quand un second correctif a eu besoin des
mêmes pièces. Elle vit dans le paquet, et pas recopiée d'un script à l'autre, pour une
raison que ce dépôt a déjà payée : une reconstitution parallèle passe ses tests le jour
où elle est écrite, puis dérive en silence. C'est l'argument qui avait fait extraire
`Searcher.avant_rerank` de `search()` au lieu de le réécrire dans un script de mesure.

Trois pièces y sont sensibles :

- `controle_fraicheur` refuse de mesurer si le retrieval n'est plus celui qui a été
  publié. Il lit son périmètre dans le JSON qui porte le chiffre, jamais dans une
  constante — un compte figé mesurerait la taille du benchmark et non le périmètre
  audité, et bloquerait dès que `dev` grandit (défaut constaté au jalon 4).
- `marge_avant_eviction` produit le livrable central de ces correctifs : le rang du gold
  DANS la fusion, avant reranking. Le recall dit que le reranker rattrape ; il ne dit pas
  de combien.
- `run_grilles` impose deux règles du protocole : la référence n'est mesurée qu'UNE fois
  et jamais deux, et le drapeau `adopte` découle du critère au lieu d'être écrit à la
  main.

Les grilles et la valeur neutre restent dans le script appelant : ce sont elles qui
définissent l'expérience, et elles doivent être lisibles dans le fichier qui la porte.
"""
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from .embed import Embedder
from .evalrag import evaluate, load_benchmark, match, paired_bootstrap
from .rerank import Reranker
from .rewrite import Rewriter
from .search import Searcher

# `parent.parent.parent` : src/accounting_rag/ablation.py -> racine du dépôt.
ROOT = Path(__file__).resolve().parent.parent.parent

def _affiche(chemin: Path) -> str:
    """Chemin relatif au dépôt quand c'est possible — un test qui redirige la sortie
    vers un `tmp_path` ne doit pas faire échouer un simple message d'information."""
    try:
        return str(chemin.relative_to(ROOT))
    except ValueError:
        return str(chemin)


DB = ROOT / "data/corpus.db"
# Ancrages en LECTURE SEULE (loi 10) : cette machinerie ne réécrit jamais
# `docs/mesures/**` — seuls les scripts appelants écrivent, dans leur propre répertoire.
PERIMETRE_JALON3 = ROOT / "docs/mesures/jalon3/cloture_dev.json"
CACHE_REECRITURES = ROOT / "docs/mesures/jalon4/reecritures_dev.json"

SEUIL_ADOPTION = 0.95
GARDE_CATEGORIE = -0.05

def searcher_du_contexte(contexte: str, embedder, reranker, rewriter, **leviers) -> Searcher:
    if contexte == "hybrid":
        return Searcher(DB, embedder=embedder, **leviers)
    if contexte == "reecriture":
        # Réécriture `etend` SANS reranking : le contexte de l'ablation G du jalon 3.
        # C'est le contexte mécanisme des leviers qui agissent sur la requête envoyée aux
        # canaux — dans `hybrid` nu il n'y a pas de réécriture, donc rien à pondérer, et
        # une grille y produirait une colonne de zéros sans rien mesurer.
        return Searcher(DB, embedder=embedder, rewriter=rewriter,
                        mode_reecriture="etend", **leviers)
    if contexte == "livree":
        return Searcher(DB, embedder=embedder, reranker=reranker, rewriter=rewriter,
                        mode_reecriture="etend", n_rerank=25, pool=50, **leviers)
    raise ValueError(f"contexte inconnu : {contexte!r}")


MODE = {"hybrid": "hybrid", "reecriture": "hybrid", "livree": "hybrid+rerank"}

# Chiffre publié au jalon 3 pour chaque contexte, et le fichier qui le porte. Le
# périmètre est TOUJOURS lu dans ce fichier, jamais figé en constante.
ABLATION_G = ROOT / "docs/mesures/jalon3/G_dev.json"


def _publie_cloture(cle: str) -> tuple[float, set[str]]:
    d = json.loads(PERIMETRE_JALON3.read_text(encoding="utf-8"))["configs"][cle]
    return d["recall@10"], set(d["par_question"])


def _publie_ablation_g(label: str) -> tuple[float, set[str]]:
    """`G_dev.json` range ses configurations dans une LISTE, pas un dict — d'où un
    accesseur distinct plutôt qu'une clé indexée au hasard."""
    configs = json.loads(ABLATION_G.read_text(encoding="utf-8"))["configs"]
    d = next(c for c in configs if c["label"] == label)
    return d["recall@10"], set(d["par_question"])


PUBLIE = {
    "hybrid": lambda: _publie_cloture("A_hybrid_neutre"),
    "reecriture": lambda: _publie_ablation_g("réécriture, mode etend"),
    "livree": lambda: _publie_cloture("C_reecriture_rerank_jalon3"),
}


def controle_fraicheur(embedder, reranker, rewriter, contextes=None) -> dict:
    """Chaque configuration publiée au jalon 3 doit redonner son chiffre exact.

    Le périmètre n'est PAS une constante : il est lu dans le JSON qui porte le chiffre
    publié. Un contrôle qui figerait « 61 questions » mesurerait la taille du benchmark
    au lieu du périmètre audité, et se mettrait à bloquer dès que dev grandit — ce qui
    est arrivé au jalon 4, où dev est passé de 61 à 93 questions.

    `contextes=None` contrôle les trois contextes connus. Un appelant qui n'en mesure que
    certains peut restreindre la liste : contrôler un contexte qu'on ne mesurera pas coûte
    du temps de reranker sans rien garantir de plus.
    """
    toutes = load_benchmark(ROOT / "benchmark/dev.jsonl")
    noms = list(PUBLIE) if contextes is None else list(contextes)
    out = {}
    for contexte in noms:
        attendu, ids = PUBLIE[contexte]()
        questions = [q for q in toutes if q["id"] in ids]
        if len(questions) != len(ids):
            manquants = sorted(ids - {q["id"] for q in questions})
            print(f"STATUS: BLOCKED — {len(manquants)} question(s) du périmètre publié "
                  f"absente(s) de benchmark/dev.jsonl : {manquants[:5]}", flush=True)
            sys.exit(1)
        s = searcher_du_contexte(contexte, embedder, reranker, rewriter)
        obtenu = evaluate(s, questions, mode=MODE[contexte])["recall@10"]
        etat = "OK" if obtenu == attendu else "ÉCART"
        print(f"[fraîcheur] {contexte:7s} sur {len(questions)} questions : "
              f"{obtenu} (publié {attendu}) — {etat}", flush=True)
        out[contexte] = {"n": len(questions), "attendu": attendu, "obtenu": obtenu,
                         "conforme": obtenu == attendu}
        if obtenu != attendu:
            print("STATUS: BLOCKED — le retrieval a bougé depuis les chiffres publiés ; "
                  "aucune ablation ne serait interprétable.", flush=True)
            sys.exit(1)
    return out


def marge_avant_eviction(s: Searcher, questions: list[dict], mode: str) -> dict:
    """Rang du gold DANS LA FUSION, avant reranking — le livrable central.

    Le recall@10 de la configuration livrée dit que le reranker rattrape les golds que
    la fusion évince. Il ne dit pas de combien. Cette mesure-ci le dit : tant que le
    gold évincé reste dans les `n_rerank` premiers de la fusion, le rattrapage tient ;
    au-delà, il ne tient plus, et rien ne garantit qu'il tiendra sur un corpus plus
    grand. Une question dont le gold est ROUTÉ (référence d'article explicite) n'est
    pas exposée au défaut et est comptée à part, jamais comme un succès de la fusion.
    """
    rangs: dict[str, int | None] = {}
    routees: list[str] = []
    for q in questions:
        routes, classes, _ = s.avant_rerank(q["question"], mode=mode)
        if any(match(r["record_id"], c) for r in routes for c in q["citations"]):
            routees.append(q["id"])
            continue
        rang = next((i + 1 for i, rid in enumerate(classes)
                     if any(match(rid, c) for c in q["citations"])), None)
        rangs[q["id"]] = rang
    trouves = [r for r in rangs.values() if r is not None]
    n = len(rangs)
    return {
        "n_exposees": n,
        "n_routees": len(routees),
        "questions_routees": sorted(routees),
        "n_gold_absent_de_la_fusion": n - len(trouves),
        # Médiane des rangs TROUVÉS seulement — donc une statistique partielle, qui
        # flatte d'autant plus qu'il y a de golds absents. Elle ne se lit qu'à côté de
        # `n_gold_absent_de_la_fusion`, jamais seule.
        "rang_median_des_golds_trouves": statistics.median(trouves) if trouves else None,
        # Les deux seuils qui décident quelque chose : 10 est le top-k rendu, 25 est la
        # fenêtre du reranker de la configuration livrée. Un gold ABSENT de la fusion
        # compte au-delà de tous les seuils — il est hors de portée du reranker au même
        # titre qu'un gold au rang 300, et l'omettre ferait sous-estimer précisément le
        # risque que cette mesure existe pour rendre visible.
        "part_au_dela_de_10": round(sum(r is None or r > 10 for r in rangs.values()) / n, 4) if n else None,
        "part_au_dela_de_25": round(sum(r is None or r > 25 for r in rangs.values()) / n, 4) if n else None,
        "rangs": rangs,
    }


def pire_perte_categorie(ref: dict, cfg: dict, questions: list[dict]) -> tuple[str, float]:
    cat_of = {q["id"]: q["categorie"] for q in questions}
    a: dict[str, list[float]] = defaultdict(list)
    b: dict[str, list[float]] = defaultdict(list)
    for qid, v in ref.items():
        a[cat_of[qid]].append(v)
    for qid, v in cfg.items():
        b[cat_of[qid]].append(v)
    pire_cat, pire = "", 0.0
    for cat in a:
        delta = sum(b[cat]) / len(b[cat]) - sum(a[cat]) / len(a[cat])
        if delta < pire:
            pire, pire_cat = delta, cat
    return pire_cat, round(pire, 4)


def mesurer(contexte: str, leviers: dict, questions: list[dict],
            embedder, reranker, rewriter) -> dict:
    s = searcher_du_contexte(contexte, embedder, reranker, rewriter, **leviers)
    t0 = time.time()
    res = evaluate(s, questions, mode=MODE[contexte])
    duree = time.time() - t0
    res["latence_s_par_question"] = duree / len(questions)
    res["marge"] = marge_avant_eviction(s, questions, mode=MODE[contexte])
    res["leviers"] = dict(leviers)
    return res


def run_grilles(split: str, contextes: list[str], grilles: list[str],
                grilles_def: dict, neutre: dict) -> dict:
    questions = load_benchmark(ROOT / f"benchmark/{split}.jsonl")
    embedder = Embedder()
    reranker = Reranker()
    # LECTURE SEULE : lève sur une question absente plutôt que d'appeler l'API payante
    # et de réécrire un ancrage versionné (loi 10).
    rewriter = Rewriter(cache_path=CACHE_REECRITURES, ecrire_cache=False)

    sortie = {
        "split": split,
        "n": len(questions),
        "grilles": {k: {"levier": v[0], "valeurs": v[1]} for k, v in grilles_def.items()},
        "neutre": neutre,
        "critere_adoption": {"p_amelioration_min": SEUIL_ADOPTION,
                             "garde_categorie": GARDE_CATEGORIE,
                             "metrique": "recall@10", "n_boot": 10000, "seed": 42},
        # Contrôle de fraîcheur sur EXACTEMENT les contextes mesurés : en contrôler
        # un qu'on ne mesurera pas coûte du temps de reranker sans rien garantir,
        # et en omettre un qu'on mesure laisserait une campagne partir sur un
        # retrieval non vérifié.
        "fraicheur": controle_fraicheur(embedder, reranker, rewriter,
                                        contextes=contextes),
        "contextes": {},
    }

    for contexte in contextes:
        print(f"\n=== contexte {contexte} ({MODE[contexte]}) ===", flush=True)
        ref = mesurer(contexte, dict(neutre), questions, embedder, reranker, rewriter)
        print(f"  référence (neutre) : recall@10={ref['recall@10']} "
              f"marge médiane={ref['marge']['rang_median_des_golds_trouves']}", flush=True)
        configs = {"reference": ref}
        for grille in grilles:
            levier, valeurs = grilles_def[grille]
            for valeur in valeurs:
                if valeur == neutre[levier]:
                    continue  # la référence, déjà mesurée — jamais deux fois
                leviers = dict(neutre, **{levier: valeur})
                res = mesurer(contexte, leviers, questions, embedder, reranker, rewriter)
                boot = paired_bootstrap(ref["par_question"], res["par_question"])
                cat, perte = pire_perte_categorie(ref["par_question"],
                                                   res["par_question"], questions)
                res["bootstrap_vs_reference"] = boot
                res["pire_categorie"] = {"categorie": cat, "delta": perte}
                res["adopte"] = (boot["p_amelioration"] >= SEUIL_ADOPTION
                                 and perte >= GARDE_CATEGORIE)
                configs[f"{levier}={valeur}"] = res
                print(f"  {levier}={valeur:<6} recall@10={res['recall@10']:<6} "
                      f"delta={boot['delta']:+.4f} p={boot['p_amelioration']:.4f} "
                      f"pire_cat={perte:+.4f} marge_med={res['marge']['rang_median_des_golds_trouves']} "
                      f"au-delà de 25={res['marge']['part_au_dela_de_25']}", flush=True)
        sortie["contextes"][contexte] = configs

    adoptes = [f"{c}/{n}" for c, cfgs in sortie["contextes"].items()
               for n, r in cfgs.items() if r.get("adopte")]
    sortie["configurations_adoptees"] = adoptes
    return sortie
