"""Télécharge le Recueil des normes comptables françaises depuis anc.gouv.fr."""
import urllib.request
from pathlib import Path

RECUEIL_URL = (
    "https://www.anc.gouv.fr/files/anc/files/"
    "1_Normes_fran%C3%A7aises/recueil/2026/Recueil-PCG-Janvier-2026.pdf"
)
DEST = Path("data/raw/recueil-pcg-2026.pdf")


def download(dest: Path = DEST) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(RECUEIL_URL, dest)
    return dest


if __name__ == "__main__":
    p = download()
    print(f"OK: {p} ({p.stat().st_size / 1e6:.1f} Mo)")
