"""Réponse à citation obligatoire : ce qui se prouve, et ce qu'on refuse d'affirmer.

Un assistant clinique qui invente une posologie est plus dangereux qu'un assistant qui se
tait. Ces tests portent sur les vérifications que l'on peut mener sans juger le sens, et
sur le refus de présenter comme mesuré ce qui ne l'est pas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from documentaliste.probes.controle import (
    ORIGINES,
    Defaut,
    ancrer,
    controlables,
    croiser,
    quantites,
    rapporter,
    verifier,
)
from documentaliste.probes.reponse import (
    CONSIGNE,
    PASSAGES,
    SCHEMA,
    Affirmation,
    Cas,
    Extrait,
    Reponse,
    composer,
    controler,
    deserialiser,
    lire,
    prelever,
    serialiser,
)
from outils import code_seul, corps

RACINE = Path(__file__).resolve().parent.parent
REPONSE = RACINE / "src" / "documentaliste" / "probes" / "reponse.py"
CONTROLE = RACINE / "src" / "documentaliste" / "probes" / "controle.py"


@dataclass(frozen=True)
class FauxPassage:
    """Un passage réduit à ce que la vérification en regarde."""

    document: str
    texte: str


PASSAGE = FauxPassage("reco_diabete", "La metformine est débutée à 500 mg par jour.")


def _reponse(*affirmations: tuple[str, int], refus: str = "") -> Reponse:
    return Reponse("q", refus, tuple(Affirmation(t, e) for t, e in affirmations))


class TestSchema:
    """Le refus doit être une sortie prévue, pas un cas d'erreur."""

    def test_le_refus_est_dans_le_schema(self) -> None:
        """Aucun seuil sur la recherche ne sépare les questions sans réponse : le refus
        doit venir de la rédaction, donc le schéma doit le permettre."""
        assert "refus" in SCHEMA["properties"]
        assert "refus" in SCHEMA["required"]

    def test_chaque_affirmation_porte_son_extrait(self) -> None:
        """Une affirmation sans citation ne se vérifie par aucun moyen."""
        item = SCHEMA["properties"]["affirmations"]["items"]
        assert item["required"] == ["texte", "extrait"]

    def test_le_schema_est_ferme(self) -> None:
        """Un champ libre laisserait le modèle rendre ce qu'il veut à côté du contrat."""
        assert SCHEMA["additionalProperties"] is False

    def test_la_consigne_dit_quand_refuser(self) -> None:
        """« Refuse si tu ne sais pas » ne suffit pas : il faut nommer les cas.

        « sans trancher » figurait ici et a été retiré après relecture des 53 refus : la
        clause visait les questions non tranchées et attrapait aussi les sources qui
        constatent elles-mêmes une absence, lesquelles répondent. `test_issues` tient
        désormais ce bord.
        """
        for cas in ("ne font qu'aborder le sujet", "population", "sort du champ"):
            assert cas in CONSIGNE, cas

    def test_la_consigne_dit_que_refuser_est_bien(self) -> None:
        """Sans cela, le modèle répond toujours : c'est ce qu'on attend d'un assistant."""
        assert "pas un échec" in CONSIGNE


class TestRefus:
    """Un refus est l'absence d'affirmation, jamais la présence d'un texte."""

    def test_sans_affirmation_c_est_un_refus(self) -> None:
        assert _reponse(refus="les extrait ne tranchent pas").a_refuse

    def test_un_refus_motivé_mais_suivi_d_affirmations_n_en_est_pas_un(self) -> None:
        """Le modèle peut commenter *et* répondre. Se fier au champ ferait compter comme
        refus des réponses bel et bien données."""
        assert not _reponse(("la metformine est débutée", 1), refus="incertain").a_refuse

    def test_une_affirmation_vide_ne_compte_pas(self) -> None:
        assert lire("q", {"refus": "", "affirmations": [{"texte": "  ", "extrait": 1}]}).a_refuse

    def test_les_champs_absents_ne_font_pas_echouer(self) -> None:
        """Le schéma est strict, mais une sortie tronquée ne doit pas faire tomber la sonde
        au milieu d'un lot payé."""
        assert lire("q", {}).a_refuse


class TestVerificationMecanique:
    """Ce qui se prouve sans modèle se vérifie en premier."""

    def test_un_extrait_hors_perimetre_est_signale(self) -> None:
        """Le modèle cite un numéro qu'on ne lui a pas donné : détectable sans rien lire."""
        defauts = verifier(_reponse(("peu importe", 7)), [PASSAGE])
        assert len(defauts) == 1 and "hors des 1 fournis" in defauts[0].motif

    def test_un_numero_nul_est_signale(self) -> None:
        """Les extraits sont numérotés à partir de 1 : zéro ne désigne rien."""
        assert verifier(_reponse(("peu importe", 0)), [PASSAGE])

    def test_une_quantite_absente_de_l_extrait_est_signalee(self) -> None:
        """La faute la plus grave d'un assistant clinique, et l'une des rares qui se
        démontre : la posologie citée ne figure pas dans le passage cité."""
        defauts = verifier(_reponse(("On débute à 850 mg par jour.", 1)), [PASSAGE])
        assert len(defauts) == 1 and "850" in defauts[0].motif

    def test_une_quantite_presente_ne_declenche_rien(self) -> None:
        assert verifier(_reponse(("On débute à 500 mg.", 1)), [PASSAGE]) == []

    def test_une_affirmation_sans_quantite_ne_declenche_rien(self) -> None:
        assert verifier(_reponse(("La metformine est le traitement de fond.", 1)), [PASSAGE]) == []

    def test_les_ecritures_decimales_se_rejoignent(self) -> None:
        """« 7,5 » et « 7.5 » sont le même seuil ; les distinguer inventerait des fautes."""
        assert quantites("HbA1c 7,5 %") == quantites("HbA1c 7.5 %")

    def test_les_zeros_de_queue_ne_creent_pas_de_faute(self) -> None:
        assert quantites("500 mg") == quantites("500.0 mg")

    def test_un_chiffre_de_sigle_n_est_pas_une_quantite(self) -> None:
        """« HbA1c » rendait le nombre « 1 ». Le modèle aurait été accusé d'un nombre
        inventé chaque fois qu'il nomme un examen que le passage désigne autrement."""
        assert quantites("Objectif HbA1c") == set()
        assert quantites("HbA1c 7,5 %") == {"7.5"}

    def test_un_ordinal_n_est_pas_une_quantite(self) -> None:
        assert quantites("en 1re intention") == set()

    def test_les_zeros_d_un_entier_sont_conserves(self) -> None:
        """Les retirer partout ramènerait « 500 mg » à « 5 mg » : deux posologies sur
        trois cesseraient de se distinguer, dans le sens qui absout le modèle."""
        assert quantites("500 mg") == {"500"}
        assert quantites("500 mg") != quantites("50 mg")

    def test_un_refus_n_a_rien_a_verifier(self) -> None:
        assert verifier(_reponse(refus="rien ici"), [PASSAGE]) == []


class TestInvite:
    """Les extraits doivent être numérotés comme la consigne l'annonce."""

    def test_les_extraits_sont_numerotes_a_partir_de_un(self) -> None:
        invite = composer("Quelle posologie ?", [PASSAGE, PASSAGE])
        assert "[1]" in invite and "[2]" in invite and "[0]" not in invite

    def test_la_question_precede_les_extraits(self) -> None:
        invite = composer("Quelle posologie ?", [PASSAGE])
        assert invite.index("Question") < invite.index("Extraits")

    def test_la_profondeur_est_celle_du_reclassement(self) -> None:
        assert PASSAGES == 10


class TestCroisement:
    """Un refus n'est un défaut que si la réponse était dans les extraits."""

    def _cas(self, servie: bool, refuse: bool, origine: str = "positive") -> Cas:
        reponse = _reponse() if refuse else _reponse(("affirmation", 1))
        return Cas("q", servie, origine, reponse, (), ())

    def test_le_refus_n_est_pas_moyenne_avec_la_disponibilite(self) -> None:
        """Le plafond de rappel est de 82 % : sur une question sur cinq, les extraits ne
        contiennent pas la réponse et le refus y est correct. Les mélanger ferait porter
        au modèle le manque de la recherche."""
        rendu = "\n".join(
            croiser([self._cas(servie=True, refuse=False), self._cas(servie=False, refuse=True)])
        )
        assert "document attendu présent" in rendu and "document attendu absent" in rendu

    def test_le_bavardage_est_nomme(self) -> None:
        """Répondre sans avoir de quoi est la faute qu'aucun seuil sur la recherche ne
        pouvait prévenir : elle doit être désignée, pas noyée dans un taux."""
        assert "bavardage" in "\n".join(croiser([self._cas(servie=False, refuse=False)]))

    def test_chaque_origine_negative_a_sa_ligne(self) -> None:
        """Une question hors périmètre est facile à écarter, une question non tranchée ne
        l'est pas : les moyenner cacherait le cas difficile."""
        rendu = "\n".join(
            croiser(
                [
                    self._cas(False, True, "non_tranchee"),
                    self._cas(False, True, "hors_perimetre"),
                ]
            )
        )
        assert "non_tranchee" in rendu and "hors_perimetre" in rendu

    def test_une_situation_sans_cas_ne_rend_pas_un_taux(self) -> None:
        """Zéro sur zéro affiché en pourcentage se lirait comme un échec total."""
        assert "—" in "\n".join(croiser([self._cas(servie=True, refuse=False)]))


class TestPrelevement:
    """Un lot d'épreuve doit voir le cas difficile, sinon il ne mesure rien."""

    NEGATIVES = (
        [{"question": f"h{i}", "origine": "hors_perimetre"} for i in range(8)]
        + [{"question": f"n{i}", "origine": "non_tranchee"} for i in range(20)]
        + [{"question": f"p{i}", "origine": "absente_de_l_archive"} for i in range(4)]
    )

    def test_les_trois_origines_sont_representees(self) -> None:
        """Le premier lot a pris les dix premières du fichier, ordonné par origine : huit
        hors périmètre, deux postérieures, zéro non tranchée. Le 100 % de refus obtenu ne
        portait que sur les catégories faciles."""
        tirees = prelever(self.NEGATIVES, 6)
        assert {n["origine"] for n in tirees} == ORIGINES

    def test_le_compte_demande_est_respecte(self) -> None:
        assert len(prelever(self.NEGATIVES, 7)) == 7

    def test_une_origine_epuisee_ne_bloque_pas_le_tirage(self) -> None:
        """« postérieure à l'archive » ne compte que quatre questions : au-delà, le tirage
        doit continuer sur les autres plutôt que de rendre un lot tronqué."""
        assert len(prelever(self.NEGATIVES, 20)) == 20

    def test_tout_demander_rend_tout(self) -> None:
        assert prelever(self.NEGATIVES, 0) == self.NEGATIVES
        assert prelever(self.NEGATIVES, 999) == self.NEGATIVES

    def test_une_origine_absente_est_signalee(self) -> None:
        """Un taux de refus qui n'a pas vu les non tranchées ne dit rien du refus."""
        cas = [Cas("q", False, "hors_perimetre", _reponse(), (), ())]
        assert "non_tranchee" in "\n".join(croiser(cas))
        assert "non couverte(s)" in "\n".join(croiser(cas))

    def test_un_lot_complet_ne_signale_rien(self) -> None:
        cas = [Cas("q", False, o, _reponse(), (), ()) for o in ORIGINES]
        assert "non couverte" not in "\n".join(croiser(cas))


class TestPiece:
    """Un rapport qui accuse doit produire la pièce."""

    def test_le_defaut_porte_l_extrait_cite(self) -> None:
        """« PillCam COLON 2 » : chiffre inventé ou nom de produit ? Sans l'extrait, le
        rapport accuse sans permettre de trancher."""
        defauts = verifier(_reponse(("On débute à 850 mg.", 1)), [PASSAGE])
        assert PASSAGE.texte[:20] in defauts[0].piece

    def test_l_extrait_est_centre_sur_le_sujet_commun(self) -> None:
        """On ne peut pas centrer sur le nombre reproché : il est absent du passage, c'est
        l'objet même du reproche. Le mot le plus long en commun sert d'ancre."""
        long = "bla " * 200 + "la metformine est débutée à 500 mg" + " bla" * 200
        piece = ancrer(long, "On débute la metformine à 850 mg.")
        assert "metformine" in piece and len(piece) < len(long) / 2

    def test_sans_mot_commun_le_debut_sert_d_ancre(self) -> None:
        """Mieux vaut le début du passage que rien du tout."""
        assert ancrer("un texte court sans rapport", "totalement autre chose").startswith("un")

    def test_les_mots_courts_ne_servent_pas_d_ancre(self) -> None:
        """« dans » ou « pour » sont partout : ils n'indiqueraient rien."""
        piece = ancrer("début du passage. " + "x " * 200 + "dans le tableau", "dans")
        assert piece.startswith("début")

    def test_les_quantites_de_l_extrait_sont_donnees(self) -> None:
        """C'est leur confrontation qui distingue une posologie fausse — l'extrait dit 500,
        l'affirmation 850 — d'un nom de produit, où l'extrait ne porte aucun nombre."""
        defaut = verifier(_reponse(("On débute à 850 mg.", 1)), [PASSAGE])[0]
        assert "l'extrait porte : 500" in defaut.motif

    def test_un_extrait_sans_quantite_le_dit(self) -> None:
        """Face à une posologie avancée, « l'extrait porte : aucune » dit tout de suite
        que le passage ne parlait pas de doses."""
        sans = FauxPassage("d", "La metformine est le traitement de fond.")
        defaut = verifier(_reponse(("On débute à 850 mg.", 1)), [sans])[0]
        assert "l'extrait porte : aucune" in defaut.motif

    def test_un_numero_de_modele_n_est_plus_reproche(self) -> None:
        """Le premier lot d'épreuve a produit sept fautes, toutes fausses, toutes le même
        « 2 » de « PillCam™ COLON 2 ». Zéro faute réelle, sept inventées."""
        sans = FauxPassage("d", "Les contre-indications de la capsule colique sont …")
        assert verifier(_reponse(("PillCamTM COLON 2 est contre-indiquée.", 1)), [sans]) == []

    def test_un_extrait_inexistant_n_a_pas_de_piece(self) -> None:
        assert verifier(_reponse(("x", 9)), [PASSAGE])[0].piece == ""

    def test_le_rapport_reproduit_la_piece(self, tmp_path: Path) -> None:
        cas = [
            Cas(
                "q",
                True,
                "positive",
                _reponse(("On débute à 850 mg.", 1)),
                (),
                tuple(verifier(_reponse(("On débute à 850 mg.", 1)), [PASSAGE])),
            )
        ]
        chemin = tmp_path / "r.md"
        rapporter(cas, 0, PASSAGES, "m", chemin)
        assert "extrait cité :" in chemin.read_text(encoding="utf-8")


class TestDenominateur:
    """Un contrôle qui n'a rien eu à contrôler doit le dire."""

    def _cas(self, *textes: str) -> list[Cas]:
        reponse = _reponse(*[(t, 1) for t in textes])
        return [Cas("q", True, "positive", reponse, (), ())]

    def test_seules_les_affirmations_chiffrees_sont_controlables(self) -> None:
        """« Zéro défaut sur 42 affirmations » se lit comme une mesure. Rapporté aux
        trois qui portaient une quantité, il dit ce qu'il vaut."""
        assert controlables(self._cas("On débute à 500 mg.", "C'est le traitement.")) == 1

    def test_aucune_quantite_rend_un_denominateur_nul(self) -> None:
        assert controlables(self._cas("Le dépistage est recommandé.")) == 0

    def test_le_rapport_expose_le_denominateur(self, tmp_path: Path) -> None:
        chemin = tmp_path / "r.md"
        rapporter(self._cas("On débute à 500 mg.", "C'est le traitement."), 0, 10, "m", chemin)
        rendu = chemin.read_text(encoding="utf-8")
        assert "**1** avancent une" in rendu and "contrôlables" in rendu


class TestSymetrieDuRapport:
    """Ne détailler que ce qui disculpe ferait pencher le rapport du côté flatteur."""

    def _bavarde(self, origine: str, servie: bool) -> Cas:
        reponse = _reponse(("La metformine est recommandée.", 2))
        return Cas("Question ?", servie, origine, reponse, (), ())

    def test_une_reponse_sans_document_est_detaillee(self, tmp_path: Path) -> None:
        chemin = tmp_path / "r.md"
        rapporter([self._bavarde("positive", servie=False)], 0, 10, "m", chemin)
        rendu = chemin.read_text(encoding="utf-8")
        assert "sans avoir de quoi" in rendu and "La metformine est recommandée." in rendu

    def test_une_negative_non_refusee_est_detaillee(self, tmp_path: Path) -> None:
        """C'est là qu'on voit ce que le modèle invente quand il invente."""
        chemin = tmp_path / "r.md"
        rapporter([self._bavarde("non_tranchee", servie=False)], 0, 10, "m", chemin)
        assert "La metformine est recommandée." in chemin.read_text(encoding="utf-8")

    def test_l_extrait_cite_accompagne_l_affirmation(self, tmp_path: Path) -> None:
        """Sans le numéro, on ne peut pas remonter à ce que le modèle a lu."""
        chemin = tmp_path / "r.md"
        rapporter([self._bavarde("positive", servie=False)], 0, 10, "m", chemin)
        assert "[2]" in chemin.read_text(encoding="utf-8")

    def test_sans_faute_la_section_disparait(self, tmp_path: Path) -> None:
        chemin = tmp_path / "r.md"
        rapporter([Cas("q", True, "positive", _reponse(("a", 1)), (), ())], 0, 10, "m", chemin)
        assert "sans avoir de quoi" not in chemin.read_text(encoding="utf-8")


class TestPersistance:
    """Corriger une règle de vérification ne doit rien coûter en appels modèle."""

    CAS = [
        Cas(
            "Quelle posologie ?",
            True,
            "positive",
            Reponse("Quelle posologie ?", "", (Affirmation("On débute à 850 mg.", 1),)),
            (Extrait("reco", 3, PASSAGE.texte),),
        )
    ]

    def test_un_aller_retour_conserve_tout(self) -> None:
        """La première correction de la règle des nombres a coûté 55 000 jetons pour
        régénérer des réponses identiques. Ce qui n'est pas conservé se repaie."""
        relu = deserialiser(serialiser(self.CAS))
        assert relu[0].question == self.CAS[0].question
        assert relu[0].reponse == self.CAS[0].reponse
        assert relu[0].extraits == self.CAS[0].extraits
        assert relu[0].servie is True and relu[0].origine == "positive"

    def test_les_extraits_soumis_sont_conserves(self) -> None:
        """Sans eux, aucun contrôle ne peut être rejoué : c'est la pièce du dossier."""
        assert serialiser(self.CAS)[0]["extraits"][0]["texte"] == PASSAGE.texte

    def test_le_controle_se_rejoue_sans_appel_modele(self) -> None:
        rejoues = controler(deserialiser(serialiser(self.CAS)))
        assert len(rejoues[0].defauts) == 1
        assert "850" in rejoues[0].defauts[0].motif

    def test_les_defauts_ne_sont_pas_conserves(self) -> None:
        """Ils sont le produit d'une règle qui change ; les relire figerait le verdict
        d'hier et rendrait le rejeu sans effet."""
        assert "defauts" not in serialiser(self.CAS)[0]

    def test_le_controle_n_appelle_pas_le_modele(self) -> None:
        assert "completer_json" not in corps(REPONSE, "controler")


class TestRapport:
    """Le rapport doit dire ce qu'il ignore aussi clairement que ce qu'il montre."""

    CAS = [
        Cas("q1", True, "positive", _reponse(("a", 1)), (), ()),
        Cas(
            "q2",
            True,
            "positive",
            _reponse(("b 850 mg", 1)),
            (),
            (Defaut("q2", "b 850 mg", "quantité(s) absente(s) de l'extrait cité : 850"),),
        ),
    ]

    def _rendu(self, tmp_path: Path) -> str:
        chemin = tmp_path / "reponse.md"
        rapporter(self.CAS, 1234, PASSAGES, "m", chemin)
        return chemin.read_text(encoding="utf-8")

    def test_le_soutien_semantique_est_declare_non_mesure(self, tmp_path: Path) -> None:
        """Sans cette réserve, un lecteur conclurait que les citations sont fondées."""
        assert "ne mesure pas" in self._rendu(tmp_path)

    def test_les_defauts_sont_nommes_un_par_un(self, tmp_path: Path) -> None:
        """Un décompte agrégé ne permet pas de vérifier le verdict sur pièce."""
        assert "850" in self._rendu(tmp_path)

    def test_le_cout_en_jetons_est_consigne(self, tmp_path: Path) -> None:
        assert "1234 jetons" in self._rendu(tmp_path)

    def test_aucun_juge_modele_n_est_appele(self) -> None:
        """Le lot n'a pas de juge validé : en introduire un ici rendrait le rapport
        dépendant d'une mesure elle-même non mesurée. Un seul site d'appel, celui qui
        rédige — le module de contrôle, lui, ne consulte personne."""
        assert code_seul(REPONSE).count("completer_json(") == 1
        assert "completer_json" not in code_seul(CONTROLE)
        assert "mistral" not in code_seul(CONTROLE)
