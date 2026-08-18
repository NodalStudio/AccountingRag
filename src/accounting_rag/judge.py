"""LLM-juge, et la fonction qui mesure son accord avec des notes humaines.

Le juge est la brique risquée du jalon, et le risque a un nom : un juge non calibré
produit des chiffres qui ont l'air rigoureux sans l'être. Le jalon 3 a puni trois fois ce
travers. D'où la règle posée avant le code : le juge est lui-même mesuré avant de servir,
contre 30 réponses notées à la main, et sous kappa pondéré 0,60 il ne publie rien.

Ce que le juge voit : la question, la réponse générée, et un BARÈME écrit à la main. Ce
qu'il ne voit jamais : les citations attendues du benchmark, les passages remontés par le
retrieval, le corpus. Le barème porte la référence de justesse ; la justesse des
citations est déjà mesurée sans LLM par `citations.py`.
"""
import json
import os
from pathlib import Path

_DEFAUT = "claude-opus-5"

# Même leçon que sur le générateur : le thinking adaptatif est actif par défaut sur les
# modèles 5 et partage le budget avec le texte. Une notation est bien plus courte qu'une
# réponse, mais la marge reste large parce qu'une sortie tronquée coûte un appel perdu.
_MAX_TOKENS = 4000

_SCHEMA = {
    "type": "object",
    "properties": {
        "note": {"type": "integer",
                 "description": "Nombre de critères du barème effectivement acquis."},
        "sur": {"type": "integer",
                "description": "Nombre total de critères du barème."},
        "par_critere": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "critere": {"type": "string",
                                "description": "Le critère du barème, recopié."},
                    "acquis": {"type": "boolean"},
                    "justification": {"type": "string",
                                      "description": "Une phrase, appuyée sur la réponse."},
                },
                "required": ["critere", "acquis", "justification"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["note", "sur", "par_critere"],
    "additionalProperties": False,
}

_SYSTEME = (
    "Tu corriges la réponse d'un candidat à une question de comptabilité française, "
    "comme un jury d'examen : critère par critère, selon le barème fourni et lui seul. "
    "Un critère est acquis si la réponse le contient explicitement ; une formulation "
    "différente mais équivalente compte comme acquise, une affirmation absente ne "
    "s'invente pas. Une abstention explicite du candidat n'acquiert aucun critère, sauf "
    "si le barème prévoit précisément l'abstention. N'ajoute aucun critère au barème et "
    "n'en retire aucun."
)


def _kappa_pondere(paires: list[tuple[int, int]]) -> float:
    """Kappa de Cohen à poids quadratiques sur des notes entières.

    Poids `(i-j)² / (max-min)²` sur l'échelle observée. Si l'accord attendu par hasard
    est nul — les deux notations sont constantes — le kappa est 0/0 : il vaut 1,0 quand
    les notations coïncident, sinon -1,0. Renvoyer 0,0 dans ce cas serait un faux
    chiffre : deux notations identiques ne sont pas un accord de hasard.
    """
    valeurs = sorted({v for paire in paires for v in paire})
    etendue = (valeurs[-1] - valeurs[0]) ** 2
    if etendue == 0:  # toutes les notes égales de part et d'autre
        return 1.0

    n = len(paires)
    def poids(i: int, j: int) -> float:
        return (i - j) ** 2 / etendue

    observe = sum(poids(h, j) for h, j in paires) / n
    marge_h = {v: sum(1 for h, _ in paires if h == v) / n for v in valeurs}
    marge_j = {v: sum(1 for _, j in paires if j == v) / n for v in valeurs}
    attendu = sum(poids(i, j) * marge_h[i] * marge_j[j] for i in valeurs for j in valeurs)
    if attendu == 0:
        return 1.0 if observe == 0 else -1.0
    return round(1 - observe / attendu, 4)


def accord(humaines: dict[str, int], juge: dict[str, int]) -> dict:
    """Accord entre notes humaines et notes du juge, sur les MÊMES questions."""
    if set(humaines) != set(juge):
        raise ValueError(
            "l'accord se calcule sur les mêmes questions de part et d'autre : "
            f"{sorted(set(humaines) ^ set(juge))} n'est pas noté des deux côtés")
    cles = sorted(humaines)
    paires = [(humaines[c], juge[c]) for c in cles]
    n = len(paires)
    return {
        "n": n,
        "exact": round(sum(1 for h, j in paires if h == j) / n, 4),
        "ecart_moyen": round(sum(abs(h - j) for h, j in paires) / n, 4),
        "kappa_pondere": _kappa_pondere(paires),
    }


class Judge:
    def __init__(self, cache_path: str | Path, modele: str | None = None, client=None,
                 ecrire_cache: bool = True):
        self.cache_path = Path(cache_path)
        if modele is None:
            from .config import charge_env
            charge_env()  # AVANT de lire la variable, cf. Rewriter et Generator
            modele = os.environ.get("ACCRAG_JUDGE_MODEL", _DEFAUT)
        self.modele = modele
        self.ecrire_cache = ecrire_cache
        self._client = client
        self._cache: dict[str, dict] = {}
        if self.cache_path.is_file():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.appels = 0
        self.tokens_entree = 0
        self.tokens_sortie = 0

    @property
    def client(self):
        if self._client is None:
            import anthropic  # import paresseux : s'importe sans clé ni réseau
            from .config import charge_env
            charge_env()
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _cle(question: str, reponse: dict, bareme: list[str]) -> str:
        """La note dépend de la question, du texte de la réponse ET du barème.

        Les `record_id` cités entrent dans la clé parce qu'une même réponse citant
        d'autres articles est une autre réponse à noter.
        """
        return json.dumps(
            [question, reponse.get("abstention"), reponse.get("reponse"),
             [c["record_id"] for c in reponse.get("citations") or []], bareme],
            ensure_ascii=False, sort_keys=True)

    def noter(self, question: str, reponse: dict, bareme: list[str]) -> dict:
        cle = self._cle(question, reponse, bareme)
        if cle in self._cache:
            return self._cache[cle]
        if not self.ecrire_cache:
            raise RuntimeError(
                f"Judge en lecture seule (ecrire_cache=False) et entrée absente du cache "
                f"{self.cache_path} : {question!r}. Aucun appel API ni écriture.")
        if self.appels >= 400:
            raise RuntimeError("garde-fou : plus de 400 appels API dans une exécution")

        criteres = "\n".join(f"- {c}" for c in bareme)
        # Ni les golds du benchmark, ni les passages du retrieval : seulement ce que le
        # candidat a écrit et le barème.
        citees = ", ".join(c["record_id"] for c in reponse.get("citations") or []) or "aucune"
        candidat = (f"Abstention : {'oui' if reponse.get('abstention') else 'non'}\n"
                    f"Réponse : {reponse.get('reponse', '')}\n"
                    f"Articles cités : {citees}")
        message = (f"Question : {question}\n\nBarème ({len(bareme)} critères) :\n{criteres}"
                   f"\n\nRéponse du candidat :\n{candidat}")

        msg = self.client.messages.create(
            model=self.modele, max_tokens=_MAX_TOKENS, system=_SYSTEME,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": message}],
        )
        self.appels += 1
        usage = getattr(msg, "usage", None)
        if usage is not None:
            self.tokens_entree += getattr(usage, "input_tokens", 0) or 0
            self.tokens_sortie += getattr(usage, "output_tokens", 0) or 0

        stop = getattr(msg, "stop_reason", None)
        if stop == "max_tokens":
            raise RuntimeError(
                f"notation tronquée par max_tokens={_MAX_TOKENS} pour : {question!r} "
                f"({usage}). Une note tronquée ne doit jamais entrer au cache.")
        texte = "".join(b.text for b in msg.content
                        if getattr(b, "type", None) == "text").strip()
        if not texte:
            raise RuntimeError(
                f"notation vide renvoyée par {self.modele} pour : {question!r} "
                f"(blocs : {[getattr(b, 'type', None) for b in msg.content]}, "
                f"stop_reason : {stop!r})")
        try:
            out = json.loads(texte)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"JSON illisible renvoyé par {self.modele} pour : {question!r} "
                f"(stop_reason : {stop!r}, {len(texte)} caractères, erreur : {e})") from e

        if out["sur"] != len(bareme) or not 0 <= out["note"] <= len(bareme):
            # Une note hors barème est un chiffre faux. Elle ne doit pas entrer au cache,
            # qui est l'ancrage de reproductibilité de la calibration.
            raise RuntimeError(
                f"notation incohérente avec le barème pour {question!r} : "
                f"note={out['note']} sur={out['sur']}, barème de {len(bareme)} critères")

        self._cache[cle] = out
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        return out
