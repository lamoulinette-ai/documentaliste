"""Trois issues au lieu de deux, et ce que la troisième empêche de compter à l'envers.

Une source qui CONSTATE une absence répond ; une source qui se TAIT ne répond pas. La
consigne confondait les deux — elle demandait de refuser « quand les extraits traitent du
sujet sans trancher » — et dix-neuf refus sur cinquante-trois s'en sont suivis alors que
l'extrait portait la réponse.

Le constat d'absence ne peut être fondu ni dans la réponse ni dans le refus : sur une
question que la HAS déclare sans réponse établie, rapporter qu'elle ne tranche pas est le
bon comportement, et le compter comme une réponse le ferait passer pour une régression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from documentaliste.probes.reponse import CONSIGNE, SCHEMA, Affirmation, Reponse
from documentaliste.probes.reponse import main as _main


def main_reponse(arguments: list[str]) -> None:
    """Lance la sonde avec des arguments donnés, sans passer par la ligne de commande."""
    ancien, sys.argv = sys.argv, ["sonde-reponse", *arguments]
    try:
        _main()
    finally:
        sys.argv = ancien


def _reponse(nature: str = "", affirmations: int = 1, refus: str = "") -> Reponse:
    return Reponse(
        question="q",
        refus=refus,
        affirmations=tuple(Affirmation(f"a{i}", 1) for i in range(affirmations)),
        nature=nature,
    )


class TestIssue:
    """L'issue se lit sur ce qui a été produit, jamais sur ce que le modèle déclare."""

    def test_une_reponse_ordinaire(self) -> None:
        assert _reponse("reponse").issue == "reponse"

    def test_un_constat_d_absence_n_est_pas_une_reponse_ordinaire(self) -> None:
        assert _reponse("constat_absence").issue == "constat_absence"

    def test_un_constat_d_absence_n_est_pas_un_refus(self) -> None:
        """Il porte une affirmation citée : « la source écrit qu'il n'y a pas de donnée »."""
        constat = _reponse("constat_absence")
        assert not constat.a_refuse and constat.a_constate_une_absence

    def test_sans_affirmation_c_est_un_refus_quoi_qu_il_declare(self) -> None:
        """Déclarer un constat sans rien citer ne prouve rien : c'est un refus habillé."""
        vide = _reponse("constat_absence", affirmations=0, refus="rien ne répond")
        assert vide.issue == "refus" and not vide.a_constate_une_absence

    def test_avec_des_affirmations_ce_n_est_pas_un_refus_quoi_qu_il_declare(self) -> None:
        """Le champ de refus devient un commentaire dès qu'une affirmation est produite."""
        assert _reponse("refus", affirmations=2, refus="il manque le reste").issue == "reponse"

    def test_une_nature_absente_reste_lisible(self) -> None:
        """Les 195 réponses conservées avant cet axe ne portent pas le champ."""
        assert _reponse("").issue == "reponse"
        assert _reponse("", affirmations=0).issue == "refus"


class TestConsigne:
    """Ce que la consigne doit dire, et ce qu'elle ne doit plus dire."""

    def test_la_clause_qui_produisait_les_refus_excessifs_a_disparu(self) -> None:
        """« Refuse quand les extraits traitent du sujet sans trancher » visait les non
        tranchées et attrapait les sources qui constatent elles-mêmes une absence."""
        assert "sans trancher la question posée" not in CONSIGNE

    def test_l_absence_constatee_est_nommee_comme_une_reponse(self) -> None:
        assert "ABSENCE CONSTATÉE PAR LA SOURCE EST UNE RÉPONSE" in CONSIGNE

    def test_la_distinction_constate_taire_est_explicite(self) -> None:
        """Sans elle, la clause précédente ferait répondre à tout."""
        aplatie = " ".join(CONSIGNE.split())
        assert "la source qui CONSTATE une absence répond" in aplatie
        assert "la source qui se TAIT ne répond pas" in aplatie

    def test_la_reponse_partielle_se_donne(self) -> None:
        """Consigne de RÉPONSE, distincte de l'axe de pertinence du juge : le système doit
        affirmer ce qu'il peut et dire ce qui manque, plutôt que refuser en bloc."""
        assert "RÉPONSE PARTIELLE SE DONNE" in CONSIGNE


class TestSchema:
    def test_la_nature_est_obligatoire(self) -> None:
        """Facultative, le modèle l'omettrait et les trois issues retomberaient à deux."""
        assert "nature" in SCHEMA["required"]

    def test_les_trois_issues_sont_closes(self) -> None:
        assert SCHEMA["properties"]["nature"]["enum"] == [
            "reponse",
            "constat_absence",
            "refus",
        ]


class TestCampagneConservee:
    """Une campagne payée ne se recouvre pas, et pas seulement pour l'argent.

    `fixtures/reponses.json` est aussi la référence de la relecture : `aligner` apparie ses
    paires sur (question, affirmation). Régénérer le fichier rendrait introuvables les 34
    paires des sections A et B, et `sonde-juge` s'arrêterait sur `PaireIntrouvable`.
    """

    def test_la_sonde_refuse_de_recouvrir(self, tmp_path: Path) -> None:
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        (fixtures / "reponses.json").write_text("[]", encoding="utf-8")
        with pytest.raises(SystemExit, match="existe déjà"):
            main_reponse(["--racine", str(tmp_path)])

    def test_une_marque_permet_d_ecrire_a_cote(self, tmp_path: Path) -> None:
        """Sans elle, la seule issue serait de déplacer un fichier à la main avant chaque
        campagne — et de l'oublier une fois.

        La sonde s'arrête ici faute de questions, et c'est le point : elle est allée
        au-delà du garde-fou au lieu de buter dessus.
        """
        fixtures = tmp_path / "fixtures"
        fixtures.mkdir()
        (fixtures / "reponses.json").write_text("[]", encoding="utf-8")
        (fixtures / "questions_reformulees.json").write_text("[]", encoding="utf-8")
        with pytest.raises(SystemExit, match="Aucune question"):
            main_reponse(["--racine", str(tmp_path), "--marque", "trois_issues"])
