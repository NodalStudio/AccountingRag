from pathlib import Path
import pytest

RECUEIL = Path("data/raw/recueil-pcg-2026.pdf")


@pytest.fixture(scope="session")
def recueil_path() -> Path:
    if not RECUEIL.exists():
        pytest.skip("PDF absent — lancer scripts/download_data.py")
    return RECUEIL
