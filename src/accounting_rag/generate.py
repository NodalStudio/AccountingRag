"""Génération d'une réponse comptable citée, à partir des passages retrouvés.

Le générateur ne produit que DEUX formes de sortie : une réponse dont chaque
affirmation porte une citation, ou une abstention explicite. La contrainte est
appliquée par sortie structurée (`output_config.format`), pas par prompt : un
schéma JSON garantit que la réponse est analysable programmatiquement, ce qui est
la condition pour que la vérification des citations (citations.py) soit possible.

Le modèle ne reçoit QUE la question et les passages — jamais les citations
attendues du benchmark, jamais le corpus entier.
"""
import json
import os
from pathlib import Path

_DEFAUT = "claude-opus-5"

# Mesuré, pas deviné : à 2000 la première question du benchmark (q001, 10 passages,
# 6200 tokens d'entrée) sort avec stop_reason=max_tokens et un JSON tronqué — le
# modèle produit un bloc thinking (le thinking adaptatif est actif par défaut sur les
# modèles 5, et il partage le budget max_tokens avec le texte) puis 6276 caractères de
# JSON, soit 2586 tokens de sortie. Cf. docs/mesures/jalon4/sonde_verbatim.json.
_MAX_TOKENS = 8000

_SCHEMA = {
    "type": "object",
    "properties": {
        "abstention": {
            "type": "boolean",
            "description": "true si les passages fournis ne permettent pas de répondre",
        },
        "reponse": {
            "type": "string",
            "description": "La réponse, ou la raison de l'abstention.",
        },
        "citations": {
            "type": "array",
            "description": "Une entrée par affirmation. Vide si abstention.",
            "items": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "L'identifiant exact du passage cité, recopié tel quel.",
                    },
                    "extrait": {
                        "type": "string",
                        "description": (
                            "Un extrait VERBATIM du texte du passage, recopié "
                            "caractère pour caractère, qui soutient l'affirmation."
                        ),
                    },
                },
                "required": ["record_id", "extrait"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["abstention", "reponse", "citations"],
    "additionalProperties": False,
}

_SYSTEME = (
    "Tu réponds à une question de comptabilité française en t'appuyant EXCLUSIVEMENT "
    "sur les passages du Plan comptable général qui te sont fournis. "
    "Chaque affirmation de ta réponse doit être soutenue par une citation : le "
    "record_id du passage, recopié exactement, et un extrait VERBATIM de son texte. "
    "N'invente jamais un numéro d'article ni un extrait. "
    "Si les passages fournis ne permettent pas de répondre — parce que la réponse "
    "n'y figure pas, ou parce que la question relève de la fiscalité et non de la "
    "comptabilité — mets abstention à true et explique en une phrase ce qui manque. "
    "Une abstention correcte vaut mieux qu'une réponse inventée."
)


class Generator:
    def __init__(self, cache_path: str | Path, modele: str | None = None, client=None,
                 ecrire_cache: bool = True):
        self.cache_path = Path(cache_path)
        if modele is None:
            # charge_env() AVANT de lire la variable : sinon une valeur placée dans
            # .env n'aurait aucun effet, le module étant importé avant tout
            # chargement du fichier (défaut corrigé sur Rewriter au jalon 3).
            from .config import charge_env
            charge_env()
            modele = os.environ.get("ACCRAG_GEN_MODEL", _DEFAUT)
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
            import anthropic  # import paresseux : le module s'importe sans clé ni réseau
            from .config import charge_env
            charge_env()
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _cle(question: str, passages: list[dict]) -> str:
        """Clé de cache : la question ET les passages, puisque la réponse dépend des deux."""
        return json.dumps([question, [p["record_id"] for p in passages]],
                          ensure_ascii=False, sort_keys=True)

    def repondre(self, question: str, passages: list[dict]) -> dict:
        cle = self._cle(question, passages)
        if cle in self._cache:
            return self._cache[cle]
        if not self.ecrire_cache:
            raise RuntimeError(
                f"Generator en lecture seule (ecrire_cache=False) et entrée absente du "
                f"cache {self.cache_path} : {question!r}. Aucun appel API ni écriture.")
        if self.appels >= 400:
            raise RuntimeError("garde-fou : plus de 400 appels API dans une exécution")

        contexte = "\n\n".join(
            f"[{p['record_id']}] ({p['chemin']}, article {p['article']})\n{p['texte']}"
            for p in passages)
        reponse = self.client.messages.create(
            model=self.modele,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEME,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user",
                       "content": f"Question : {question}\n\nPassages :\n{contexte}"}],
        )
        self.appels += 1
        usage = getattr(reponse, "usage", None)
        if usage is not None:
            self.tokens_entree += getattr(usage, "input_tokens", 0) or 0
            self.tokens_sortie += getattr(usage, "output_tokens", 0) or 0

        stop = getattr(reponse, "stop_reason", None)
        if stop == "max_tokens":
            # Une sortie tronquée doit lever AVANT l'analyse : le JSON tronqué est
            # tantôt illisible (JSONDecodeError opaque), tantôt analysable et donc
            # silencieusement amputé de ses dernières citations. Ni l'un ni l'autre ne
            # doit entrer dans le cache, qui est l'ancrage de reproductibilité.
            raise RuntimeError(
                f"réponse tronquée par max_tokens={_MAX_TOKENS} pour : {question!r} "
                f"({getattr(reponse, 'usage', None)}). Augmenter _MAX_TOKENS ; ne pas "
                f"réduire le nombre de passages, ce serait changer deux variables.")

        texte = "".join(b.text for b in reponse.content
                        if getattr(b, "type", None) == "text").strip()
        if not texte:
            raise RuntimeError(
                f"réponse vide renvoyée par {self.modele} pour : {question!r} "
                f"(blocs : {[getattr(b, 'type', None) for b in reponse.content]}, "
                f"stop_reason : {stop!r})")
        try:
            out = json.loads(texte)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"JSON illisible renvoyé par {self.modele} pour : {question!r} "
                f"(stop_reason : {stop!r}, {len(texte)} caractères, erreur : {e})") from e
        if not out.get("reponse", "").strip():
            # Ne JAMAIS mettre en cache une réponse vide : le cache étant committé,
            # elle serait rejouée silencieusement par toutes les campagnes suivantes.
            raise RuntimeError(
                f"réponse vide dans le JSON renvoyé par {self.modele} pour : {question!r}")

        self._cache[cle] = out
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8")
        return out
