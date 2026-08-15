"""Télécharge le Recueil des normes comptables françaises depuis anc.gouv.fr."""
import hashlib
import urllib.request
from pathlib import Path

RECUEIL_URL = (
    "https://www.anc.gouv.fr/files/anc/files/"
    "1_Normes_fran%C3%A7aises/recueil/2026/Recueil-PCG-Janvier-2026.pdf"
)
DEST = Path("data/raw/recueil-pcg-2026.pdf")
TIMEOUT_S = 60
# Empreinte du PDF validé (recueil-pcg-2026.pdf tel que téléchargé et vérifié
# manuellement pour ce jalon). Toute divergence — fichier corrompu en cours de
# téléchargement, ou l'ANC ayant republié un contenu différent sous la même
# URL — doit être détectée plutôt que silencieusement parsée.
EXPECTED_SHA256 = "ae21e019a6295beb7825f562643f67977cbcc0ec90ccab57f20cefcf0e105ec1"


class ChecksumMismatch(RuntimeError):
    """Le PDF téléchargé (ou déjà présent) ne correspond pas au sha256 attendu."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(dest: Path) -> None:
    digest = _sha256(dest)
    if digest != EXPECTED_SHA256:
        raise ChecksumMismatch(
            f"sha256 invalide pour {dest} :\n"
            f"  attendu : {EXPECTED_SHA256}\n"
            f"  obtenu  : {digest}\n"
            "Le fichier est peut-être corrompu (téléchargement interrompu) ou l'ANC a "
            "republié un contenu différent sous la même URL. Supprimez le fichier et "
            "relancez ce script pour retélécharger, ou — après vérification manuelle du "
            "nouveau contenu — mettez à jour EXPECTED_SHA256 dans scripts/download_data.py."
        )


def download(dest: Path = DEST) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        # Fichier déjà présent : on ne retélécharge pas, mais on vérifie
        # quand même son intégrité avant de laisser le pipeline continuer.
        _verify(dest)
        return dest
    try:
        with urllib.request.urlopen(RECUEIL_URL, timeout=TIMEOUT_S) as resp:
            data = resp.read()
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    dest.write_bytes(data)
    try:
        _verify(dest)
    except ChecksumMismatch:
        # Ne pas laisser un téléchargement corrompu/inattendu traîner comme
        # un fichier "présent" qui tromperait la prochaine exécution.
        dest.unlink(missing_ok=True)
        raise
    return dest


if __name__ == "__main__":
    p = download()
    print(f"OK: {p} ({p.stat().st_size / 1e6:.1f} Mo)")
