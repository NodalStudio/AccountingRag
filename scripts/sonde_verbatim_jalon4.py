"""Sonde : un extrait VERBATIM du corpus est-il une contrainte que le modèle peut tenir ?

Pourquoi cette sonde existe AVANT la vérification des citations (tâche 2 du jalon 4) :
le schéma du générateur exige un `extrait` « recopié caractère pour caractère » du texte
du passage. La tâche 2 doit décider comment comparer cet extrait au texte du corpus, et
ce choix n'est pas neutre :

  - si l'on compare brut et que le corpus contient des artefacts de mise en page héritée
    du PDF (retours à la ligne au milieu d'une phrase, apostrophes courbes, césures),
    alors le « taux de citations non portantes » mesurerait la typographie du corpus et
    non l'honnêteté du système — un défaut de conception déguisé en résultat ;
  - si l'on normalise trop (casse, ponctuation, accents), on rend le contrôle incapable
    d'échouer, ce que la loi 5 du dépôt interdit.

La sonde mesure donc le taux de correspondance à quatre niveaux de normalisation
croissante, sur des réponses réelles produites par le générateur avec la configuration
de retrieval adoptée au jalon 3. Le niveau retenu par la tâche 2 est celui que cette
mesure justifie, et pas un choix a priori.

Le générateur ne voit que la question et les passages : la sonde ne lui transmet jamais
les citations attendues du benchmark (loi 9).

Cache d'exécution : `data/sonde-verbatim-cache.json` (gitignoré, loi 10).
Le cache de réécriture versionné est ouvert en LECTURE SEULE.

Sortie : `docs/mesures/jalon4/sonde_verbatim.json`.

Coût : un appel API par question (12 questions), gratuit au second passage.

Usage : uv run python scripts/sonde_verbatim_jalon4.py
"""
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from accounting_rag.embed import Embedder
from accounting_rag.evalrag import load_benchmark
from accounting_rag.generate import Generator
from accounting_rag.rerank import Reranker
from accounting_rag.rewrite import Rewriter
from accounting_rag.search import Searcher

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/corpus.db"
CACHE_REECRITURES = ROOT / "docs/mesures/jalon3/reecritures.json"
CACHE_GENERATION = ROOT / "data/sonde-verbatim-cache.json"
OUT = ROOT / "docs/mesures/jalon4/sonde_verbatim.json"

N_QUESTIONS = 12  # une par catégorie autant que possible, choix déterministe

# Les quatre niveaux, du plus strict au plus permissif. L'ordre compte : le niveau
# rapporté pour une citation est le PREMIER qui la fait correspondre.
_APOSTROPHES = {"’": "'", "‘": "'", "ʼ": "'"}
_GUILLEMETS = {"“": '"', "”": '"', "«": '"', "»": '"'}
_TIRETS = {"‐": "-", "‑": "-", "–": "-", "—": "-"}


def n_brut(s: str) -> str:
    return s


def n_espaces(s: str) -> str:
    """Réduit tout blanc (retour à la ligne, insécable, tabulation) à une espace simple."""
    return re.sub(r"\s+", " ", s).strip()


def n_typographie(s: str) -> str:
    """Espaces + apostrophes courbes, guillemets et tirets ramenés à l'ASCII."""
    s = n_espaces(s)
    for table in (_APOSTROPHES, _GUILLEMETS, _TIRETS):
        for k, v in table.items():
            s = s.replace(k, v)
    return s


def n_casse_accents(s: str) -> str:
    """Typographie + casse + accents. Niveau volontairement trop permissif : il sert de
    borne haute, pour savoir si un extrait qui échoue échoue vraiment sur le CONTENU."""
    s = n_typographie(s).casefold()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


NIVEAUX = [("brut", n_brut), ("espaces", n_espaces),
           ("typographie", n_typographie), ("casse_accents", n_casse_accents)]


def choisir_questions(questions: list[dict]) -> list[dict]:
    """Une question par catégorie en tourniquet, ordre stable, jusqu'à N_QUESTIONS."""
    par_cat: dict[str, list[dict]] = {}
    for q in questions:
        par_cat.setdefault(q["categorie"], []).append(q)
    choix, i = [], 0
    while len(choix) < N_QUESTIONS:
        ajoute = False
        for cat in sorted(par_cat):
            if i < len(par_cat[cat]) and len(choix) < N_QUESTIONS:
                choix.append(par_cat[cat][i])
                ajoute = True
        if not ajoute:
            break
        i += 1
    return choix


def classer(extrait: str, texte: str) -> str:
    """Le premier niveau de normalisation qui fait figurer l'extrait dans le texte."""
    for nom, f in NIVEAUX:
        if f(extrait) and f(extrait) in f(texte):
            return nom
    return "absent"


def main() -> None:
    con = sqlite3.connect(DB)
    textes = dict(con.execute("SELECT id, texte FROM records").fetchall())

    questions = choisir_questions(load_benchmark(ROOT / "benchmark/dev.jsonl"))
    print(f"[sonde] {len(questions)} questions : "
          f"{', '.join(q['id'] for q in questions)}", flush=True)

    embedder = Embedder()
    reranker = Reranker()
    rewriter = Rewriter(cache_path=CACHE_REECRITURES, ecrire_cache=False)
    searcher = Searcher(DB, embedder=embedder, reranker=reranker,
                        rewriter=rewriter, mode_reecriture="etend")
    generateur = Generator(cache_path=CACHE_GENERATION)

    detail, par_niveau = [], {nom: 0 for nom, _ in NIVEAUX}
    par_niveau["absent"] = 0
    n_abstentions = 0

    for q in questions:
        passages = searcher.search(q["question"], mode="hybrid+rerank", k=10)
        # Le générateur ne reçoit que la question et les passages (loi 9).
        out = generateur.repondre(q["question"], passages)
        if out["abstention"]:
            n_abstentions += 1
        for c in out["citations"]:
            texte = textes.get(c["record_id"])
            niveau = "inexistant" if texte is None else classer(c["extrait"], texte)
            par_niveau[niveau] = par_niveau.get(niveau, 0) + 1
            detail.append({
                "question_id": q["id"],
                "record_id": c["record_id"],
                "record_existe": texte is not None,
                "niveau": niveau,
                "longueur_extrait": len(c["extrait"]),
                "extrait_traverse_un_retour_ligne": (
                    texte is not None and niveau in ("espaces", "typographie")
                    and "\n" in texte and bool(re.search(r"\s", c["extrait"]))),
                "extrait": c["extrait"],
            })
        print(f"[sonde] {q['id']}: abstention={out['abstention']} "
              f"citations={len(out['citations'])}", flush=True)

    n_cit = len(detail)
    resultat = {
        "n_questions": len(questions),
        "question_ids": [q["id"] for q in questions],
        "n_abstentions": n_abstentions,
        "n_citations": n_cit,
        "modele": generateur.modele,
        "appels_api": generateur.appels,
        "tokens": {"entree": generateur.tokens_entree, "sortie": generateur.tokens_sortie},
        "par_niveau": par_niveau,
        "taux_par_niveau": ({k: round(v / n_cit, 4) for k, v in par_niveau.items()}
                            if n_cit else {}),
        "cumule_brut": round(par_niveau.get("brut", 0) / n_cit, 4) if n_cit else None,
        "cumule_espaces": (round((par_niveau.get("brut", 0)
                                  + par_niveau.get("espaces", 0)) / n_cit, 4)
                           if n_cit else None),
        "cumule_typographie": (round((par_niveau.get("brut", 0)
                                      + par_niveau.get("espaces", 0)
                                      + par_niveau.get("typographie", 0)) / n_cit, 4)
                               if n_cit else None),
        "detail": detail,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(resultat, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[sonde] {n_cit} citations sur {len(questions)} questions")
    for nom in [n for n, _ in NIVEAUX] + ["absent", "inexistant"]:
        v = par_niveau.get(nom, 0)
        if v:
            print(f"  {nom:15s} {v:4d}  {100 * v / n_cit:5.1f}%")
    print(f"[sonde] écrit {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
