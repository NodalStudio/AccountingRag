"""Campagne de génération : mesure ce que le système RÉPOND, pas ce qu'il retrouve.

Trois jalons ont amélioré le retrieval sans jamais mesurer la réponse. Ce script produit
la première mesure de génération du projet : les taux de citation de la brique
`citations.py` (aucun LLM, aucun jugement) sur la configuration de retrieval livrée au
jalon 3, sans y toucher.

Le générateur ne reçoit que la question et les passages — jamais les citations attendues
du benchmark (loi 9 du dépôt). Le contrôle de fraîcheur en tête de script refuse de
dépenser un centime d'API si le retrieval n'est plus celui qui a été publié.

Caches :
  - réécritures : `docs/mesures/jalon3/reecritures.json` en LECTURE SEULE ;
  - réponses : `docs/mesures/jalon4/reponses_{split}.json`, écrit par ce script et par
    lui seul, puis versionné comme ancrage de reproductibilité.

Coût : un appel API par question (61 sur dev), gratuit à toute ré-exécution.

Usage :
  uv run python scripts/eval_generation.py --split dev --controle-seul   # gratuit
  uv run python scripts/eval_generation.py --split dev
"""
import argparse
import json
import sys
from pathlib import Path

from accounting_rag.citations import metriques
from accounting_rag.embed import Embedder
from accounting_rag.evalrag import evaluate, load_benchmark
from accounting_rag.generate import Generator
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
OUT_DIR = ROOT / "docs/mesures/jalon4"
CACHE_REECRITURES = ROOT / "docs/mesures/jalon3/reecritures.json"

# Baseline publiée du retrieval neutre (jalon 3, docs/eval-jalon3.md) : recall@10 = 0,672
# SUR LES 61 QUESTIONS DU SPLIT DEV DU JALON 3. Le contrôle de fraîcheur compare à cette
# valeur AVANT tout appel payant.
#
# Le périmètre du contrôle est lu dans le JSON qui porte le chiffre publié, et non déduit
# du contenu courant de `benchmark/dev.jsonl` : l'extension du benchmark au jalon 4 a porté
# dev à 93 questions, et le contrôle a alors correctement refusé de mesurer (0,715 au lieu
# de 0,672). Réagir en déplaçant la constante aurait détruit le contrôle ; le lier au
# périmètre publié le garde vrai à travers toutes les extensions futures.
RECALL_NEUTRE_DEV = 0.672
PERIMETRE_JALON3 = ROOT / "docs/mesures/jalon3/cloture_dev.json"

CONFIG = {
    "retrieval": "hybrid+rerank",
    "reranker": "BAAI/bge-reranker-v2-m3",
    "n_rerank": 25,
    "pool": 50,
    "mode_reecriture": "etend",
    "k": 10,
}


def cache_reecritures(split: str) -> Path:
    """Cache de réécriture du split, versionné sous `docs/mesures/jalon4/`.

    L'ancrage du jalon 3 (`docs/mesures/jalon3/reecritures.json`) reste en LECTURE SEULE :
    c'est l'ancrage de reproductibilité des chiffres publiés au jalon 3, et le faire
    grossir hors de la campagne qui l'a produit le viderait de son sens. Le jalon 4 mesure
    un benchmark différent — 150 questions au lieu de 90 — donc il a son propre ancrage.

    Celui de `dev` est AMORCÉ depuis l'ancrage du jalon 3 pour les 61 questions communes :
    la réécriture y est reprise à l'identique, donc le retrieval de ces questions est
    inchangé et leurs résultats restent comparables d'un jalon à l'autre. Les 32 questions
    ajoutées au jalon 4 n'y figurent pas et sont réécrites par cette campagne.
    """
    chemin = OUT_DIR / f"reecritures_{split}.json"
    if not chemin.is_file() and CACHE_REECRITURES.is_file():
        amorce = json.loads(CACHE_REECRITURES.read_text(encoding="utf-8"))
        textes = {q["question"] for q in
                  load_benchmark(ROOT / f"benchmark/{split}.jsonl")}
        repris = {q: r for q, r in amorce.items() if q in textes}
        if repris:
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(json.dumps(repris, ensure_ascii=False, indent=2,
                                         sort_keys=True), encoding="utf-8")
            print(f"[eval_generation] {len(repris)} réécriture(s) reprise(s) à "
                  f"l'identique depuis l'ancrage du jalon 3", flush=True)
    return chemin


def controle_fraicheur(embedder) -> float:
    """`Searcher()` neutre sur dev doit rendre exactement la valeur publiée.

    Gratuit (aucun appel API). S'il échoue, quelque chose a bougé dans l'index ou dans
    le retrieval, et la campagne mesurerait un système différent de celui décrit.
    """
    toutes = load_benchmark(ROOT / "benchmark/dev.jsonl")
    ids_publies = set(json.loads(PERIMETRE_JALON3.read_text(encoding="utf-8"))
                      ["configs"]["A_hybrid_neutre"]["par_question"])
    questions = [q for q in toutes if q["id"] in ids_publies]
    if len(questions) != len(ids_publies):
        manquantes = sorted(ids_publies - {q["id"] for q in questions})
        print(f"STATUS: BLOCKED — {len(manquantes)} question(s) du périmètre publié au "
              f"jalon 3 ont disparu de benchmark/dev.jsonl : {manquantes[:5]}. Le chiffre "
              f"de référence n'est plus vérifiable.", file=sys.stderr)
        raise SystemExit(1)
    r = evaluate(Searcher(DB, embedder=embedder), questions, mode="hybrid", k=10)
    obtenu = r["recall@10"]
    if obtenu != RECALL_NEUTRE_DEV:
        print(f"STATUS: BLOCKED — contrôle de fraîcheur en échec : recall@10 neutre sur "
              f"dev = {obtenu}, attendu {RECALL_NEUTRE_DEV}. Aucun appel API n'a été "
              f"fait. L'index ou le retrieval a changé : ne pas mesurer avant de savoir "
              f"quoi.", file=sys.stderr)
        raise SystemExit(1)
    print(f"[eval_generation] contrôle OK : recall@10 = {obtenu} sur les "
          f"{len(questions)} questions du périmètre publié au jalon 3", flush=True)
    # Chiffre informatif, jamais un seuil : le recall neutre sur le dev ÉTENDU.
    etendu = evaluate(Searcher(DB, embedder=embedder), toutes, mode="hybrid", k=10)
    print(f"[eval_generation] pour information, recall@10 neutre sur les "
          f"{len(toutes)} questions de dev étendu : {etendu['recall@10']}", flush=True)
    return obtenu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev",
                    choices=("dev", "test", "validation", "abstention"))
    ap.add_argument("--controle-seul", action="store_true",
                    help="n'exécute que le contrôle de fraîcheur, gratuit")
    args = ap.parse_args()

    embedder = Embedder()
    recall_neutre = controle_fraicheur(embedder)
    if args.controle_seul:
        return

    questions = load_benchmark(ROOT / f"benchmark/{args.split}.jsonl")
    cache_reponses = OUT_DIR / f"reponses_{args.split}.json"
    rewriter = Rewriter(cache_path=cache_reecritures(args.split))
    searcher = Searcher(DB, embedder=embedder, reranker=Reranker(),
                        rewriter=rewriter,
                        mode_reecriture=CONFIG["mode_reecriture"])
    generateur = Generator(cache_path=cache_reponses)
    # Nombre d'entrées DÉJÀ en cache avant la campagne : sans lui, une ré-exécution
    # complète depuis le cache publierait « 0 appel API », ce qui se lit comme une
    # campagne gratuite au lieu d'un rejeu. Le coût est par exécution, pas cumulé.
    n_deja_en_cache = len(generateur._cache)

    reponses: dict[str, dict] = {}
    for i, q in enumerate(questions, 1):
        passages = searcher.search(q["question"], mode=CONFIG["retrieval"],
                                   k=CONFIG["k"])
        # Le générateur ne voit QUE la question et les passages : ni les citations
        # gold du benchmark, ni ses notes, ni sa catégorie.
        reponses[q["id"]] = generateur.repondre(q["question"], passages)
        print(f"[eval_generation] {i}/{len(questions)} {q['id']} "
              f"abstention={reponses[q['id']]['abstention']} "
              f"citations={len(reponses[q['id']]['citations'])}", flush=True)

    m = metriques(reponses, DB)
    if args.split == "abstention":
        # Sur ce split, toute question DOIT recevoir une abstention : `taux_abstention`
        # est donc le taux d'abstention CORRECTE. Les non-abstentions sont nommées, pour
        # que le rapport ne puisse pas les résumer en un taux.
        #
        # Elles ne sont PAS appelées « réponses inventées », et c'est une correction
        # apportée après lecture de la première campagne. L'unique non-abstention de dev
        # (qa014) n'inventait rien : elle ouvrait sur « les passages fournis ne détaillent
        # pas les écritures de retraitement, qui relèvent du règlement ANC n° 2020-01 »,
        # citait six extraits tous verbatim et existants, et concluait sur ce qui manquait.
        # Seul le drapeau `abstention` était faux — ce qui reste un défaut réel, puisqu'un
        # appelant qui s'y fie présenterait une réponse à une question sans réponse — mais
        # l'appeler « inventée » serait une accusation que la mesure ne soutient pas.
        non_abst = sorted(q for q, r in reponses.items() if not r["abstention"])
        propres = sorted(q for q in non_abst
                         if m["par_question"].get(q) in ("ok", "version_omise"))
        m["taux_abstention_correcte"] = m["taux_abstention"]
        m["non_abstentions"] = non_abst
        m["n_non_abstentions"] = len(non_abst)
        # Une non-abstention dont toutes les citations sont propres n'a rien fabriqué :
        # elle a répondu hors périmètre en restant sourcée. Le distinguer d'une invention
        # est la différence entre un défaut de drapeau et une hallucination.
        m["non_abstentions_sans_faute_de_citation"] = propres
        m["n_fabrications"] = len(non_abst) - len(propres)
        par_raison = {q["id"]: q["raison"] for q in questions}
        m["non_abstentions_par_raison"] = {
            r: sorted(q for q in non_abst if par_raison[q] == r)
            for r in sorted({q["raison"] for q in questions})}
    resultat = {
        "split": args.split,
        "n": len(questions),
        "config": CONFIG,
        "controle_fraicheur": {"recall@10_neutre_dev": recall_neutre,
                               "attendu": RECALL_NEUTRE_DEV},
        "modele_generation": generateur.modele,
        "metriques": m,
        "cout": {"appels_api": generateur.appels,
                 "tokens_entree": generateur.tokens_entree,
                 "tokens_sortie": generateur.tokens_sortie,
                 # Le coût ci-dessus est celui de CETTE exécution. Une valeur nulle avec un
                 # cache déjà plein est un rejeu gratuit, pas une campagne gratuite.
                 "reponses_deja_en_cache_avant": n_deja_en_cache,
                 "rejeu_depuis_le_cache": generateur.appels == 0 and n_deja_en_cache > 0},
        "reponses_cache": str(cache_reponses.relative_to(ROOT)),
        "reecritures_cache": str(cache_reecritures(args.split).relative_to(ROOT)),
        "cout_reecriture": {"appels_api": rewriter.appels},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sortie = OUT_DIR / f"generation_{args.split}.json"
    sortie.write_text(json.dumps(resultat, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    print(f"\n[eval_generation] === {args.split}, n={len(questions)} ===")
    for cle in ("n_citations", "taux_citations_inexistantes",
                "taux_citations_non_portantes", "taux_reponses_sans_citation",
                "taux_citations_version_omise", "taux_abstention",
                "taux_correspondance_brute"):
        print(f"  {cle:32s} {m[cle]}")
    print(f"  citations_par_reponse            {m['citations_par_reponse']}")
    print(f"[eval_generation] écrit {sortie.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
