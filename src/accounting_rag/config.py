"""Chargement d'un fichier .env sans dépendance externe (les secrets ne sont jamais versionnés)."""
import os
from pathlib import Path


def charge_env(chemin: str | Path = ".env") -> None:
    """Charge les paires CLE=valeur dans os.environ, sans écraser l'existant.

    Silencieux si le fichier est absent : l'environnement peut déjà porter les variables.
    """
    p = Path(chemin)
    if not p.is_file():
        return
    for ligne in p.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne or ligne.startswith("#") or "=" not in ligne:
            continue
        cle, _, valeur = ligne.partition("=")
        cle, valeur = cle.strip(), valeur.strip().strip('"').strip("'")
        os.environ.setdefault(cle, valeur)
