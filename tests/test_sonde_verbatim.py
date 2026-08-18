"""Tests de la classification de la sonde verbatim du jalon 4.

La sonde conclut que 77 extraits sur 77 figurent dans le corpus SANS normalisation. Ce
chiffre n'a de valeur que si `classer` sait rendre autre chose que « brut » : un
classement qui renvoie toujours le niveau le plus strict produirait exactement le même
100 % sur des extraits inventés. C'est la loi 5 du dépôt — un contrôle que personne n'a
vu échouer ne prouve rien — et le dépôt a déjà livré un contrôle de reproduction qui
validait le bug qu'il devait attraper.

Chaque test fixe donc un niveau différent, y compris « absent ».
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "sonde_verbatim_jalon4",
    Path(__file__).resolve().parent.parent / "scripts/sonde_verbatim_jalon4.py")
sonde = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sonde)

# Texte de corpus réaliste : apostrophe courbe et retour à la ligne au milieu d'une
# énumération, les deux artefacts effectivement présents dans data/corpus.db
# (79,6 % des records portent une apostrophe courbe, 77,5 % un retour à la ligne).
TEXTE = ("Technique : l’évolution technique entraîne une obsolescence de l’actif ;\n"
         "- Juridique : l’utilisation est limitée dans le temps.")


def test_extrait_recopie_a_lidentique_est_brut():
    assert sonde.classer("l’évolution technique entraîne une obsolescence", TEXTE) == "brut"


def test_extrait_traversant_un_retour_ligne_est_brut_sil_le_recopie():
    assert sonde.classer("de l’actif ;\n- Juridique :", TEXTE) == "brut"


def test_retour_ligne_remplace_par_une_espace_tombe_au_niveau_espaces():
    assert sonde.classer("de l’actif ; - Juridique :", TEXTE) == "espaces"


def test_apostrophe_droite_au_lieu_de_courbe_tombe_au_niveau_typographie():
    assert sonde.classer("l'évolution technique", TEXTE) == "typographie"


def test_casse_et_accents_perdus_tombent_au_dernier_niveau():
    assert sonde.classer("L'EVOLUTION TECHNIQUE", TEXTE) == "casse_accents"


def test_extrait_invente_est_absent():
    """Le cas qui rend le 100 % publiable : un extrait plausible mais absent du texte."""
    assert sonde.classer("l’actif est amorti sur dix ans", TEXTE) == "absent"


def test_extrait_vide_ne_compte_pas_comme_present():
    """Sans la garde `if f(extrait)`, la chaîne vide figurerait dans tout texte."""
    assert sonde.classer("", TEXTE) == "absent"
    assert sonde.classer("   \n ", TEXTE) == "absent"
