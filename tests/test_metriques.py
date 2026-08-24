"""Ce qui juge une recherche : rang du bon document, rang réciproque, plafond."""

from __future__ import annotations

from pathlib import Path

from documentaliste.probes.metrics import (
    ndcg,
    plafond_de_rappel,
    rang_du_bon_document,
    rang_reciproque_moyen,
    rappels,
)
from documentaliste.probes.retrieval import Passage, Question

RACINE = Path(__file__).resolve().parent.parent


class TestRappel:
    def test_rang_du_bon_document(self) -> None:
        resultats = [
            Passage("autre", 1, "x"),
            Passage("attendu", 4, "y"),
            Passage("attendu", 5, "z"),
        ]
        assert rang_du_bon_document(resultats, "attendu") == 2

    def test_document_absent(self) -> None:
        """Renvoyer None plutôt que zéro : « pas trouvé » n'est pas « trouvé au rang 0 »."""
        assert rang_du_bon_document([Passage("autre", 1, "x")], "attendu") is None

    def test_aucun_resultat(self) -> None:
        assert rang_du_bon_document([], "attendu") is None


class TestRangReciproque:
    """La mesure qui départage les configurations : elle doit privilégier la tête."""

    QUESTIONS = [
        Question("q1", frozenset({"attendu"})),
        Question("q2", frozenset({"attendu"})),
    ]

    def test_reponse_en_tete_vaut_un(self) -> None:
        trouves = [[Passage("attendu", 1, "x")], [Passage("attendu", 1, "y")]]
        assert rang_reciproque_moyen(self.QUESTIONS, trouves) == 1.0

    def test_document_absent_ne_vaut_rien(self) -> None:
        """Non trouvé compte zéro, et non « trouvé très loin »."""
        trouves = [[Passage("autre", 1, "x")], [Passage("autre", 1, "y")]]
        assert rang_reciproque_moyen(self.QUESTIONS, trouves) == 0.0

    def test_le_rang_pese(self) -> None:
        """Deuxième rang vaut la moitié du premier : 1,0 et 0,5 donnent 0,75."""
        trouves = [
            [Passage("attendu", 1, "x")],
            [Passage("autre", 1, "y"), Passage("attendu", 2, "z")],
        ]
        assert rang_reciproque_moyen(self.QUESTIONS, trouves) == 0.75

    def test_departage_ce_que_la_somme_des_rappels_confondait(self) -> None:
        """Le cas qui justifie le changement de critère."""
        loin = [Passage("autre", i, "x") for i in range(1, 10)]
        tete_ou_loin = [[Passage("attendu", 1, "x")], [*loin, Passage("attendu", 10, "y")]]
        toujours_second = [
            [Passage("autre", 1, "x"), Passage("attendu", 2, "y")],
            [Passage("autre", 1, "x"), Passage("attendu", 2, "y")],
        ]
        somme = lambda t: sum(rappels(self.QUESTIONS, t, 10).values())  # noqa: E731
        assert somme(toujours_second) > somme(tete_ou_loin)
        assert rang_reciproque_moyen(self.QUESTIONS, tete_ou_loin) > rang_reciproque_moyen(
            self.QUESTIONS, toujours_second
        )


class TestPlafondDeRappel:
    """Ce qu'une fusion parfaite pourrait au mieux atteindre."""

    QUESTIONS = [
        Question("q1", frozenset({"a"})),
        Question("q2", frozenset({"b"})),
    ]

    def test_bras_complementaires(self) -> None:
        """Chaque bras n'en sert qu'une, mais leur union sert tout."""
        par_bras = {
            "vectoriel": [[Passage("a", 1, "x")], [Passage("zzz", 1, "y")]],
            "lexical": [[Passage("zzz", 1, "x")], [Passage("b", 1, "y")]],
        }
        assert plafond_de_rappel(self.QUESTIONS, par_bras) == 1.0

    def test_question_hors_de_portee(self) -> None:
        """Une question qu'aucun bras ne ramène plafonne la fusion, quoi qu'on reclasse."""
        par_bras = {
            "vectoriel": [[Passage("a", 1, "x")], [Passage("zzz", 1, "y")]],
            "lexical": [[Passage("a", 1, "x")], [Passage("zzz", 1, "y")]],
        }
        assert plafond_de_rappel(self.QUESTIONS, par_bras) == 0.5


class TestNdcg:
    """Ce que le rang réciproque et le rappel ne voient ni l'un ni l'autre."""

    def _question(self) -> Question:
        return Question("q", ["doc"], None)

    def _passages(self, documents: list[str]) -> list[Passage]:
        return [Passage(d, 1, "t") for d in documents]

    def test_tous_les_utiles_en_tete_donne_un(self) -> None:
        lot = self._passages(["doc", "doc", "autre"])
        assert ndcg([self._question()], [lot], 10, "document") == 1.0

    def test_le_meme_nombre_d_utiles_plus_bas_vaut_moins(self) -> None:
        """Le rappel@10 serait identique dans les deux cas, et le rang réciproque aussi
        dès que le premier utile ne bouge pas."""
        haut = self._passages(["doc", "doc", "autre", "autre"])
        bas = self._passages(["doc", "autre", "autre", "doc"])
        assert ndcg([self._question()], [haut], 10, "document") > ndcg(
            [self._question()], [bas], 10, "document"
        )

    def test_aucun_utile_donne_zero(self) -> None:
        lot = self._passages(["autre", "autre"])
        assert ndcg([self._question()], [lot], 10, "document") == 0.0

    def test_la_profondeur_est_respectee(self) -> None:
        """Un passage utile au rang onze ne compte pas dans un nDCG@10."""
        lot = self._passages(["autre"] * 10 + ["doc"])
        assert ndcg([self._question()], [lot], 10, "document") == 0.0

    def test_sans_question_il_ne_leve_pas(self) -> None:
        assert ndcg([], [], 10) == 0.0
