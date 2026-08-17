"""Réécriture d'une question en langage courant vers le vocabulaire du PCG (via l'API Claude).

Le diagnostic du jalon 3 a montré que les questions « grand public » ne partagent aucun
token de contenu avec l'article qui y répond : le canal lexical est alors muet. La
réécriture vise ce cas précis. Le modèle ne voit QUE la question — jamais le corpus,
jamais les citations attendues.
"""
import json
import os
from pathlib import Path

_DEFAUT = "claude-sonnet-5"

_SYSTEME = (
    "Tu traduis une question de comptabilité posée en langage courant vers le vocabulaire "
    "technique du Plan comptable général français. Réponds UNIQUEMENT par une liste de "
    "termes et expressions normalisés, séparés par des espaces, sans phrase, sans "
    "explication, sans ponctuation superflue. N'invente aucun numéro d'article. "
    "Exemple d'entrée : « comment je répartis le coût d'une machine sur plusieurs années ». "
    "Exemple de sortie : amortissement immobilisation corporelle plan d'amortissement "
    "durée d'utilisation base amortissable."
)


class Rewriter:
    def __init__(self, cache_path: str | Path, modele: str | None = None, client=None,
                 ecrire_cache: bool = True):
        self.cache_path = Path(cache_path)
        # ecrire_cache=False : mode LECTURE SEULE, pour tout contexte de mesure qui pointe
        # un cache versionné. `reecrire()` lève alors sur une question absente au lieu
        # d'appeler l'API et de réécrire le fichier. Sans ce garde-fou, un script de
        # mesure pointant docs/mesures/jalon3/reecritures.json réécrirait l'ancrage de
        # reproductibilité des chiffres publiés dès qu'une question serait ajoutée au
        # benchmark — inoffensif par ÉTAT (tout est en cache) mais pas par GARANTIE, ce
        # que trois revues successives ont relevé sous des formes différentes.
        self.ecrire_cache = ecrire_cache
        if modele is None:
            # `charge_env()` AVANT de lire ACCRAG_REWRITE_MODEL : sinon une variable
            # placée dans .env (le seul endroit où ce projet met sa configuration
            # sensible) n'aurait aucun effet, le module étant importé avant tout
            # chargement du fichier. Lecture de fichier uniquement, aucun réseau.
            from .config import charge_env
            charge_env()
            modele = os.environ.get("ACCRAG_REWRITE_MODEL", _DEFAUT)
        self.modele = modele
        self._client = client
        self._cache: dict[str, str] = {}
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

    def reecrire(self, question: str) -> str:
        if question in self._cache:
            return self._cache[question]
        if not self.ecrire_cache:
            raise RuntimeError(
                f"Rewriter en lecture seule (ecrire_cache=False) et question absente du "
                f"cache {self.cache_path} : {question!r}. Aucun appel API ni écriture ne "
                f"sera fait. Utilisez un cache d'exécution accessible en écriture.")
        if self.appels >= 200:
            raise RuntimeError("garde-fou : plus de 200 appels API dans une exécution")
        reponse = self.client.messages.create(
            model=self.modele,
            # `thinking` DÉSACTIVÉ explicitement. Sur les modèles de la série 5, le
            # thinking adaptatif est actif PAR DÉFAUT quand le paramètre est omis
            # (changement silencieux par rapport à la série 4.6, qui ne pensait pas sans
            # configuration explicite), et `max_tokens` plafonne thinking + texte de
            # réponse ENSEMBLE. Constaté à la première campagne de l'ablation G : le
            # budget de 200 tokens était intégralement consommé par le raisonnement et
            # la réponse ne contenait qu'un bloc `thinking`, aucun bloc `text` — d'où
            # l'échec bruyant ci-dessous. Traduire du vocabulaire courant vers le
            # vocabulaire du PCG ne demande aucun raisonnement multi-étapes : le
            # désactiver supprime la cause à la racine plutôt que d'élargir le budget.
            thinking={"type": "disabled"},
            max_tokens=400,
            system=_SYSTEME,
            messages=[{"role": "user", "content": question}],
        )
        self.appels += 1
        usage = getattr(reponse, "usage", None)
        if usage is not None:
            self.tokens_entree += getattr(usage, "input_tokens", 0) or 0
            self.tokens_sortie += getattr(usage, "output_tokens", 0) or 0
        texte = " ".join(
            bloc.text.strip() for bloc in reponse.content if getattr(bloc, "type", None) == "text"
        ).strip()
        # Une réécriture vide ne doit JAMAIS être mise en cache : en mode "remplace" elle
        # deviendrait la requête envoyée aux canaux (MATCH vide côté FTS5), et le cache
        # étant committé, la campagne suivante rejouerait silencieusement la même requête
        # vide sans qu'aucun chiffre ne signale l'anomalie. Échec bruyant plutôt que
        # dégradation invisible : une mesure faussée sans trace est le pire des deux.
        if not texte:
            raise RuntimeError(
                f"réécriture vide renvoyée par {self.modele} pour la question : {question!r} "
                f"(blocs reçus : {[getattr(b, 'type', None) for b in reponse.content]}, "
                f"stop_reason : {getattr(reponse, 'stop_reason', None)!r}). Si les blocs "
                f"reçus contiennent 'thinking' et que stop_reason vaut 'max_tokens', le "
                f"budget a été consommé par le raisonnement avant tout texte."
            )
        self._cache[question] = texte
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        return texte
