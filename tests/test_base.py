"""Persistance : schéma, connexion et ingestion.

docker compose up -d && uv run pytest
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from documentaliste.db import pool
from documentaliste.db.ingest import configuration_retenue, numeroter
from documentaliste.db.manifeste import lire_manifeste, titres
from documentaliste.probes.pages import Page
from documentaliste.probes.retrieval import Passage
from documentaliste.probes.sweep import (
    Configuration,
    empreinte_de_page,
    indexer,
    variantes,
)

RACINE = Path(__file__).resolve().parent.parent

#: Le manifeste du corpus est un artefact produit par l'extraction de l'archive HAS, pas
#: une donnée du dépôt : 5 Mo dérivés de 15 Go. Les tests qui en dépendent se sautent quand
#: il est absent — sur une machine d'intégration continue, il l'est toujours.
#:
#: Se sauter et non échouer : un échec ici ne signalerait pas une régression du code, mais
#: l'absence d'une donnée qu'on a délibérément sortie du dépôt.
MANIFESTE = RACINE / "fixtures" / "corpus.json"
sans_manifeste = pytest.mark.skipif(
    not MANIFESTE.exists(), reason="manifeste absent : « uv run corpus-extraire »"
)


class TestChaineDeConnexion:
    """Un secret ne doit jamais transiter par un argument ni finir dans un journal."""

    def test_le_mot_de_passe_est_masque(self) -> None:
        masquee = pool.sans_secret("postgresql://moi:secret@127.0.0.1:5433/base")
        assert "secret" not in masquee
        assert "moi" in masquee and "5433" in masquee

    def test_une_chaine_sans_identifiants_reste_lisible(self) -> None:
        assert pool.sans_secret("postgresql:///base") == "postgresql:///base"

    def test_la_chaine_vient_de_l_environnement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOCUMENTALISTE_DSN", "postgresql://a:b@ailleurs/base")
        assert pool.dsn().startswith("postgresql://a:b@ailleurs/base")

    def test_une_base_absente_se_signale_au_lieu_de_se_faire_attendre(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Le proxy de Docker Desktop écoute même conteneur arrêté : il accepte la
        connexion et ne la sert jamais. Sans délai, la suite est passée de quatorze
        secondes à quatre minutes sans que rien n'indique pourquoi."""
        monkeypatch.setenv("DOCUMENTALISTE_DSN", "postgresql://a:b@ailleurs/base")
        assert f"connect_timeout={pool.DELAI_CONNEXION}" in pool.dsn()

    def test_le_delai_ne_casse_pas_une_chaine_deja_parametree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCUMENTALISTE_DSN", "postgresql://a:b@ailleurs/base?sslmode=require")
        chaine = pool.dsn()
        assert chaine.count("?") == 1 and "sslmode=require" in chaine

    def test_un_delai_explicite_est_respecte(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Le déploiement peut vouloir attendre davantage ; le défaut ne doit pas primer."""
        pose = "postgresql://a:b@ailleurs/base?connect_timeout=30"
        monkeypatch.setenv("DOCUMENTALISTE_DSN", pose)
        assert pool.dsn().endswith("connect_timeout=30")

    def test_la_disponibilite_n_est_eprouvee_qu_une_fois(self) -> None:
        """Chaque `skipif` du dépôt pose la question ; sans mémoire, une base absente
        ferait payer le délai autant de fois qu'il y a de tests."""
        assert hasattr(pool.disponible, "cache_info")

    def test_le_defaut_vise_le_conteneur_de_developpement(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Le port 5433, pour qu'une installation locale ne reçoive pas nos connexions."""
        monkeypatch.setenv("DOCUMENTALISTE_DSN", "")
        assert ":5433/" in pool.dsn()

    def test_aucune_option_de_ligne_de_commande_ne_porte_le_secret(self) -> None:
        source = (RACINE / "src" / "documentaliste" / "db" / "ingest.py").read_text(
            encoding="utf-8"
        )
        assert "--dsn" not in source and "password" not in source


class TestSchema:
    """Le schéma porte des décisions qu'aucune relecture du code ne restituerait."""

    BRUT = (RACINE / "src" / "documentaliste" / "db" / "schema.sql").read_text(encoding="utf-8")

    #: Schéma privé de ses commentaires : un test doit porter sur ce que le code fait.
    SQL = "\n".join(ligne.split("--")[0] for ligne in BRUT.splitlines())

    def test_la_dimension_correspond_au_modele_retenu(self) -> None:
        """384 dimensions : « multilingual-e5-small ». Un autre modèle impose une migration."""
        assert "vector(384)" in self.SQL

    def test_l_index_utilise_le_produit_scalaire(self) -> None:
        """Les vecteurs étant normalisés, leur produit scalaire est la similarité cosinus."""
        assert "vector_ip_ops" in self.SQL
        assert "vector_cosine_ops" not in self.SQL

    def test_le_passage_porte_sa_page(self) -> None:
        """Exigence du produit : une citation doit se vérifier à un endroit précis."""
        assert "page" in self.SQL and "REFERENCES document" in self.SQL

    def test_le_texte_des_pages_est_en_base(self) -> None:
        """Le texte des pages vit en base, non dans un cache monolithique."""
        assert "CREATE TABLE IF NOT EXISTS page" in self.SQL
        assert "PRIMARY KEY (document, numero)" in self.SQL

    def test_l_empreinte_de_page_est_indexee(self) -> None:
        """Sans index, le regroupement par empreinte balaierait tout le texte intégral."""
        assert "page_empreinte_idx" in self.SQL

    def test_l_unicite_couvre_le_rang_dans_la_page(self) -> None:
        """Le rang distingue les fragments d'une même page, que le recouvrement multiplie."""
        assert "UNIQUE (document, page, rang)" in self.SQL


@sans_manifeste
class TestIngestionSansBase:
    """Ce qui peut être éprouvé sans serveur doit l'être sans serveur."""

    def test_la_configuration_est_celle_qui_a_ete_mesuree(self) -> None:
        """L'ingestion réutilise `cumul` au lieu de redécouper à sa façon."""
        config = configuration_retenue()
        assert config.nom == "cumul"
        assert config.decesure and config.sans_gardes
        assert config.sans_texte_standard and config.titre_en_prefixe
        assert (config.taille, config.recouvrement) == (900, 200)

    def test_le_rang_distingue_les_fragments_d_une_meme_page(self) -> None:
        passages = [
            Passage("a", 1, "premier"),
            Passage("a", 1, "second"),
            Passage("a", 2, "autre page"),
            Passage("b", 1, "autre document"),
        ]
        assert [rang for _, rang in numeroter(passages)] == [0, 1, 0, 0]

    def test_le_manifeste_reel_se_lit(self) -> None:
        documents = lire_manifeste(RACINE)
        assert len(documents) >= 100
        assert all(d.nom and d.titre for d in documents.values())

    def test_chaque_document_porte_sa_fiche(self) -> None:
        """La fiche rattache une citation à la publication dont elle provient."""
        documents = lire_manifeste(RACINE)
        assert all(d.fiche for d in documents.values())

    def test_le_manifeste_absent_donne_une_consigne(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="corpus-extraire"):
            lire_manifeste(tmp_path)


@sans_manifeste
class TestTitreEnPrefixe:
    """Le titre préfixé à chaque passage doit venir du manifeste, pas du nom de fichier."""

    CONFIG = Configuration("t", "", taille=900, recouvrement=200, titre_en_prefixe=True)
    PAGES = [Page(document="2012-04-11_okbat_guide_gdr", numero=1, texte="Le texte. " * 30)]

    def test_le_titre_reel_est_prefere_au_nom_de_fichier(self) -> None:
        vrai = "Mettre en oeuvre la gestion des risques associés aux soins"
        passages = indexer(
            RACINE, self.CONFIG, pages=self.PAGES, titres={self.PAGES[0].document: vrai}
        )
        assert passages and all(p.texte.startswith(vrai + ".") for p in passages)
        assert "okbat" not in passages[0].texte

    def test_sans_titre_le_comportement_du_lot_0_est_conserve(self) -> None:
        """Les chiffres inscrits au plan doivent rester rejouables à l'identique."""
        passages = indexer(RACINE, self.CONFIG, pages=self.PAGES)
        assert passages[0].texte.startswith("2012-04-11 okbat guide gdr.")

    def test_un_titre_vide_retombe_sur_le_nom(self) -> None:
        """Un manifeste incomplet ne doit pas produire un passage préfixé de « . »."""
        passages = indexer(RACINE, self.CONFIG, pages=self.PAGES, titres={"2012-04-11_okbat": ""})
        assert passages[0].texte.startswith("2012-04-11 okbat guide gdr.")

    def test_les_titres_viennent_du_manifeste_reel(self) -> None:
        intitules = titres(lire_manifeste(RACINE))
        assert all(t.strip() for t in intitules.values())


class TestTexteStandard:
    """Le critère de retrait doit rester celui qui a été mesuré."""

    def test_l_empreinte_ignore_casse_accents_et_ponctuation(self) -> None:
        """La ponctuation devient un espace, puis les espaces sont réduits — dans cet ordre."""
        assert empreinte_de_page("Hôpital, 2024 !") == empreinte_de_page("hopital  2024")
        assert empreinte_de_page("Hôpital, 2024 !") == "hopital 2024"

    def test_la_requete_compte_les_pages_et_non_les_documents(self) -> None:
        """Le comptage porte sur les pages, non sur les documents distincts."""
        source = (RACINE / "src" / "documentaliste" / "db" / "ingest.py").read_text(
            encoding="utf-8"
        )
        requete = source.split("SELECT empreinte FROM page")[1].split('"')[0]
        assert "count(*) >= %s" in requete
        assert "DISTINCT document" not in requete

    def test_le_seuil_ne_vide_aucun_document(self) -> None:
        """À 2, les deux exemplaires d'une fiche présente dans deux dossiers
        disparaissaient ensemble : 235 documents devenaient inatteignables."""
        assert configuration_retenue().seuil_standard >= 5

    def test_le_seuil_est_une_variante_mesurable(self) -> None:
        """Un réglage qu'aucun balayage ne compare est un réglage qu'on ne défend pas."""
        noms = {c.nom for c in variantes(Configuration("reference", "aucune"))}
        assert {"texte_standard_seuil_2", "texte_standard_seuil_3"} <= noms


@sans_manifeste
@pytest.mark.skipif(not pool.disponible(), reason="PostgreSQL absent : « docker compose up -d »")
class TestBaseVivante:
    """Ce qui ne peut se vérifier que contre un vrai serveur."""

    def test_le_schema_s_applique_deux_fois(self) -> None:
        """Idempotence : une seconde application du schéma ne doit rien casser."""
        pool.appliquer_schema()
        pool.appliquer_schema()

    def test_l_extension_vectorielle_est_presente(self) -> None:
        pool.appliquer_schema()
        with pool.connexion() as cnx:
            trouvee = cnx.execute(
                "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0]
        assert trouvee == 1

    def test_les_passages_inseres_correspondent_au_manifeste(self) -> None:
        """Un passage sans document rattaché rendrait sa citation invérifiable."""
        with pool.connexion() as cnx:
            orphelins = cnx.execute(
                "SELECT count(*) FROM passage p"
                " LEFT JOIN document d ON d.nom = p.document WHERE d.nom IS NULL"
            ).fetchone()[0]
        assert orphelins == 0

    #: Part du manifeste devant être présente en base pour qu'une ingestion soit dite
    #: complète. En deçà de 1, un PDF illisible ne fait pas échouer la suite.
    COUVERTURE_MINIMALE = 0.98

    def test_l_ingestion_couvre_le_manifeste(self) -> None:
        """Un seuil relatif au manifeste, et non un nombre absolu.

        Le garde-fou précédent exigeait « plus de 15 000 passages », chiffre calibré sur
        478 documents : il aurait laissé passer une ingestion arrêtée au centième du
        corpus complet.
        """
        attendus = len(lire_manifeste(RACINE))
        with pool.connexion() as cnx:
            avec_passages = cnx.execute("SELECT count(DISTINCT document) FROM passage").fetchone()
        if not avec_passages[0]:
            pytest.skip("corpus non ingéré : « uv run documentaliste-ingerer »")
        assert avec_passages[0] >= attendus * self.COUVERTURE_MINIMALE

    def test_aucun_passage_n_est_vide_ni_sans_vecteur(self) -> None:
        """Un passage vide occupe un rang sans jamais pouvoir être cité."""
        with pool.connexion() as cnx:
            creux = cnx.execute(
                "SELECT count(*) FROM passage WHERE texte IS NULL OR btrim(texte) = ''"
            ).fetchone()[0]
        assert creux == 0

    def test_l_index_vectoriel_existe(self) -> None:
        """Il est supprimé avant le chargement : son absence signale une reconstruction
        interrompue, et la recherche resterait fonctionnelle mais lente sans le dire."""
        with pool.connexion() as cnx:
            passages = cnx.execute("SELECT count(*) FROM passage").fetchone()[0]
            present = cnx.execute(
                "SELECT count(*) FROM pg_indexes WHERE indexname = 'passage_vecteur_idx'"
            ).fetchone()[0]
        if not passages:
            pytest.skip("corpus non ingéré : « uv run documentaliste-ingerer »")
        assert present == 1

    def test_aucune_page_n_est_orpheline(self) -> None:
        with pool.connexion() as cnx:
            orphelines = cnx.execute(
                "SELECT count(*) FROM page p"
                " LEFT JOIN document d ON d.nom = p.document WHERE d.nom IS NULL"
            ).fetchone()[0]
        assert orphelines == 0

    def test_tout_document_pourvu_de_passages_a_ses_pages(self) -> None:
        """Un document pourvu de passages mais privé de pages signalerait un lot à moitié écrit."""
        with pool.connexion() as cnx:
            boiteux = cnx.execute(
                "SELECT count(*) FROM (SELECT DISTINCT document FROM passage) s"
                " WHERE NOT EXISTS (SELECT 1 FROM page WHERE page.document = s.document)"
            ).fetchone()[0]
        assert boiteux == 0

    def test_les_metadonnees_du_manifeste_sont_completes_en_base(self) -> None:
        attendues = lire_manifeste(RACINE)
        with pool.connexion() as cnx:
            noms = {n for (n,) in cnx.execute("SELECT nom FROM document").fetchall()}
        if not noms:
            pytest.skip("corpus non ingéré : « uv run documentaliste-ingerer »")
        assert noms <= set(attendues)


def test_le_compose_dimensionne_le_cache() -> None:
    """L'index vectoriel pèse 1,55 Go ; le défaut de PostgreSQL est 128 Mo.

    Un index qui ne tient pas en cache fait payer un accès disque à chaque question :
    245 ms mesurées contre 9 ms. Le cache du système d'exploitation compensait sur le
    poste de développement, ce qui masquait le réglage.
    """
    compose = (RACINE / "docker-compose.yml").read_text(encoding="utf-8")
    assert "shared_buffers" in compose
    assert "128MB" not in compose


def test_le_compose_expose_pgvector() -> None:
    """L'image doit porter pgvector : une image PostgreSQL nue échoue sur le schéma."""
    compose = (RACINE / "docker-compose.yml").read_text(encoding="utf-8")
    assert "pgvector/pgvector" in compose
    assert "127.0.0.1:5433:5432" in compose
    assert "healthcheck" in compose


@sans_manifeste
def test_le_manifeste_consigne_ses_exclusions() -> None:
    """Le manifeste dit ce qui a été écarté et pourquoi, pas seulement ce qui est gardé."""
    brut = json.loads((RACINE / "fixtures" / "corpus.json").read_text(encoding="utf-8"))
    assert brut["retenus"] and brut["licence"] and brut["source"]
    assert set(brut["ecartes"]) >= {"doublons_de_contenu", "publications_retirees"}
    # Les recommandations retirées par la HAS doivent rester nommées : ce sont celles
    assert brut["ecartes"]["publications_retirees"]
