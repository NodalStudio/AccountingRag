from pathlib import Path
import pytest
from accounting_rag.model import Record, Renvoi

RECUEIL = Path("data/raw/recueil-pcg-2026.pdf")


@pytest.fixture(scope="session")
def recueil_path() -> Path:
    if not RECUEIL.exists():
        pytest.skip("PDF absent — lancer scripts/download_data.py")
    return RECUEIL


def _rec(rid, article, chemin="Livre II > Titre I", texte="x", renvois=()):
    """Fabrique de Record pour les tests d'intégrité et DB."""
    return Record(
        id=rid,
        article=article,
        chemin=chemin,
        texte=texte,
        type="reglementaire",
        nature="comptable",
        opposable=False,
        valide_du="2026-01-01",
        valide_au=None,
        source_citation=None,
        page_debut=1,
        page_fin=1,
        renvois=list(renvois),
    )
