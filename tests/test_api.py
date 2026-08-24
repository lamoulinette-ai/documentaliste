"""Le garde-fou de dépense, et la dégradation qu'il doit produire.

Deux propriétés valent tous les autres tests de ce fichier :

- **une question déjà posée ne rappelle pas le modèle** — c'est ce qui rend un budget de
  quelques cents suffisant pour une démonstration ;
- **le plafond atteint ne prive jamais des passages retrouvés** — la recherche a lieu avant
  toute considération de budget, et une démonstration qui dégrade vaut mieux qu'une
  démonstration qui meurt.

La seconde est celle qui se casserait en silence : un refus renvoyé sans passages ressemble
à un refus normal, et rien dans les journaux ne le distinguerait.
"""

from __future__ import annotations

from pathlib import Path

from documentaliste.api.budget import Budget, cle, normaliser


class _Passage:
    def __init__(self, document: str, page: int, texte: str) -> None:
        self.document, self.page, self.texte = document, page, texte


class TestNormalisation:
    """Deux formulations de la même question ne doivent pas coûter deux appels."""

    def test_la_casse_ne_distingue_pas(self) -> None:
        assert normaliser("Quel Dépistage ?") == normaliser("quel dépistage ?")

    def test_les_accents_ne_distinguent_pas(self) -> None:
        assert normaliser("quel dépistage") == normaliser("quel depistage")

    def test_les_espaces_multiples_ne_distinguent_pas(self) -> None:
        assert normaliser("quel   dépistage  ?") == normaliser("quel dépistage ?")

    def test_deux_questions_differentes_restent_differentes(self) -> None:
        assert normaliser("dépistage du cancer") != normaliser("dépistage du diabète")

    def test_la_cle_n_expose_pas_la_question(self) -> None:
        """Le texte servirait de nom de fichier lisible, et écrirait les questions des
        visiteurs en clair sur le disque du serveur."""
        empreinte = cle("Quel est le dépistage recommandé ?")
        assert "dépistage" not in empreinte and len(empreinte) == 32


class TestBudget:
    def test_le_compteur_part_de_zero(self, tmp_path: Path) -> None:
        assert Budget(tmp_path).appels == 0

    def test_le_compteur_persiste(self, tmp_path: Path) -> None:
        """Il survit au redémarrage du conteneur : sinon le plafond se remettrait à zéro
        à chaque incident, et ne plafonnerait plus rien."""
        Budget(tmp_path).compter()
        assert Budget(tmp_path).appels == 1

    def test_le_plafond_se_declenche(self, tmp_path: Path) -> None:
        budget = Budget(tmp_path, plafond=2)
        budget.compter()
        assert not budget.epuise
        budget.compter()
        assert budget.epuise

    def test_un_compteur_illisible_ne_leve_pas(self, tmp_path: Path) -> None:
        """Un fichier abîmé doit dégrader vers zéro, pas casser le service."""
        (tmp_path / "appels.txt").write_text("bruit", encoding="utf-8")
        assert Budget(tmp_path).appels == 0

    def test_le_cache_rend_ce_qu_il_a_recu(self, tmp_path: Path) -> None:
        budget = Budget(tmp_path)
        budget.ecrire("Quel dépistage ?", {"question": "Quel dépistage ?", "issue": "reponse"})
        assert budget.lire("quel  depistage ?")["issue"] == "reponse"

    def test_une_question_inconnue_rend_none(self, tmp_path: Path) -> None:
        assert Budget(tmp_path).lire("jamais posée") is None


class TestDegradation:
    """Ce que le service fait quand il ne peut plus rédiger."""

    def test_le_plafond_rend_les_passages(  # noqa: ANN001
        self, tmp_path, monkeypatch
    ) -> None:
        """La propriété centrale : un refus de budget montre ce que la recherche a trouvé.

        Sans elle, l'épuisement du crédit produirait un écran vide indiscernable d'une
        panne — et d'un refus légitime du système.
        """
        from documentaliste.api import service

        passages = [_Passage("has_1234", 12, "Le dépistage est recommandé à partir de 50 ans.")]
        monkeypatch.setattr(service, "rechercher", lambda question, k=10: (passages, {}))

        budget = Budget(tmp_path, plafond=0)
        rendue = service.repondre("Quel dépistage ?", budget)

        assert rendue.redaction_indisponible
        assert rendue.issue == "refus"
        assert len(rendue.passages) == 1
        assert rendue.passages[0].document == "has_1234"
        assert rendue.passages[0].numero == 1

    def test_le_cache_evite_l_appel(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        """Si le cache ne court-circuitait pas, la recherche serait relancée — et le modèle
        avec elle."""
        from documentaliste.api import service

        appels = []
        monkeypatch.setattr(service, "rechercher", lambda q, k=10: (appels.append(q) or [], {}))

        budget = Budget(tmp_path)
        budget.ecrire("Quel dépistage ?", {"question": "Quel dépistage ?", "issue": "reponse"})
        rendue = service.repondre("Quel dépistage ?", budget)

        assert rendue.issue == "reponse"
        assert appels == []

    def test_une_panne_degrade_pareil(  # noqa: ANN001
        self, tmp_path, monkeypatch
    ) -> None:
        """Une panne du fournisseur ne doit pas être plus grave qu'un budget épuisé."""
        from documentaliste.api import service
        from documentaliste.probes.mistral import MistralIndisponible

        def tomber(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise MistralIndisponible("panne")

        monkeypatch.setattr(
            service, "rechercher", lambda q, k=10: ([_Passage("has_1", 1, "t")], {})
        )
        monkeypatch.setattr(service, "completer_json", tomber)

        rendue = service.repondre("Quel dépistage ?", Budget(tmp_path))
        assert rendue.redaction_indisponible and len(rendue.passages) == 1


class TestCitation:
    """Un nom de fichier n'est pas une citation.

    `2007-05-03_rpc_sftg_insomnie_-_argumentaire_mel` est vérifiable et illisible : devant
    un professionnel de santé, il dessert la traçabilité qu'il est censé établir. Le titre
    publié et le lien vers la fiche HAS sont ce qui distingue une citation d'une allégation.
    """

    def test_le_titre_et_le_lien_sont_rendus(self) -> None:
        from documentaliste.api.service import _rendus

        meta = {
            "has_1": {
                "titre": "Prise en charge de l'insomnie — Argumentaire",
                "url_fiche": "https://www.has-sante.fr/jcms/p_1234/fr/insomnie",
                "url_pdf": "https://www.has-sante.fr/upload/x.pdf",
                "type_publication": "Recommandation",
                "mise_en_ligne": "03 mai 2007",
            }
        }
        (rendu,) = _rendus([_Passage("has_1", 15, "texte")], meta)
        assert rendu.titre.startswith("Prise en charge")
        assert rendu.url.endswith("/insomnie")
        assert rendu.document == "has_1"

    def test_la_fiche_prime_sur_le_pdf(self) -> None:
        """La fiche porte le contexte de publication et n'impose pas de téléchargement."""
        from documentaliste.api.service import _rendus

        meta = {"has_1": {"url_fiche": "", "url_pdf": "https://has.fr/x.pdf"}}
        (rendu,) = _rendus([_Passage("has_1", 1, "t")], meta)
        assert rendu.url == "https://has.fr/x.pdf"

    def test_un_document_inconnu_ne_leve_pas(self) -> None:
        """Le manifeste couvre 99,3 % du corpus : les 0,7 % restants doivent s'afficher."""
        from documentaliste.api.service import _rendus

        (rendu,) = _rendus([_Passage("orphelin", 1, "t")], {})
        assert rendu.titre == "" and rendu.url == "" and rendu.document == "orphelin"
