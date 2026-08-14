from dataclasses import dataclass, field, asdict
from enum import Enum, auto


@dataclass(frozen=True)
class Line:
    text: str
    size: float
    bold: bool
    font: str
    x: float
    y: float
    page: int  # 1-indexé


class Kind(Enum):
    REGLEMENTAIRE = auto()
    COMMENTAIRE = auto()
    ARTICLE_HEADER = auto()
    COMMENTAIRE_TITRE = auto()
    SECTION_HEADER = auto()
    PUCE = auto()
    BRUIT = auto()
    INCONNU = auto()


@dataclass(frozen=True)
class Renvoi:
    cible: str
    famille: str  # interne | externe_legal | historique


@dataclass
class Record:
    id: str
    article: str | None
    chemin: str
    texte: str
    type: str          # reglementaire | commentaire_ANC
    nature: str        # comptable
    opposable: bool
    valide_du: str
    valide_au: str | None
    source_citation: str | None
    page_debut: int
    page_fin: int
    renvois: list[Renvoi] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["renvois"] = [asdict(r) for r in self.renvois]
        return d
