"""Connexion à PostgreSQL : une seule porte d'entrée, et un diagnostic utile.

DOCUMENTALISTE_DSN=postgresql://documentaliste:documentaliste@127.0.0.1:5433/documentaliste
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from documentaliste.probes.mistral import charger_env

#: Chaîne de connexion du conteneur de développement décrit par `docker-compose.yml`.
DSN_DEFAUT = "postgresql://documentaliste:documentaliste@127.0.0.1:5433/documentaliste"

#: Emplacement du schéma, livré avec le paquet.
SCHEMA = Path(__file__).with_name("schema.sql")


class BaseIndisponible(RuntimeError):
    """La base n'a pas répondu, ou le pilote n'est pas installé."""


#: Secondes d'attente avant de déclarer la base injoignable.
#:
#: Sans ce délai, un conteneur arrêté ne produit pas une erreur mais une attente : le proxy
#: de Docker Desktop continue d'écouter sur le port publié et accepte la connexion, que
#: personne ne servira jamais. La suite de tests est alors passée de quatorze secondes à
#: quatre minutes, sans que rien n'indique pourquoi.
#:
#: Trois secondes suffisent largement pour une base locale, et transforment une attente
#: inexplicable en un message qui dit quoi faire.
DELAI_CONNEXION = 3


def dsn() -> str:
    """Chaîne de connexion, lue dans l'environnement puis dans un `.env`.

    Un délai d'attente est ajouté s'il n'y en a pas : une base absente doit se signaler,
    pas se faire attendre.
    """
    charger_env()
    chaine = os.environ.get("DOCUMENTALISTE_DSN", "").strip() or DSN_DEFAUT
    if "connect_timeout" in chaine:
        return chaine
    return f"{chaine}{'&' if '?' in chaine else '?'}connect_timeout={DELAI_CONNEXION}"


def _pilote():  # noqa: ANN202 - type fourni par psycopg
    """Module `psycopg`, importé tardivement avec un message qui dit quoi faire."""
    try:
        import psycopg
    except ImportError as erreur:  # pragma: no cover - dépend de l'installation
        raise BaseIndisponible(
            "psycopg n'est pas installé. Lancer « uv sync --extra bdd »."
        ) from erreur
    return psycopg


#: Largeur du parcours de l'index vectoriel, posée à chaque connexion.
#:
#: Mesuré sur 796 000 vecteurs, rappel des vrais plus proches voisins à 10 : le défaut de
#: pgvector, 40, n'en retrouve que 63,5 %. 100 en retrouve 86,0 %, 200 en retrouve 96,5 %,
#: 400 en retrouve 98,0 %. La courbe s'aplatit après 200 : doubler le travail pour un point
#: et demi n'en vaut pas la latence.
EF_SEARCH = 200


@contextmanager
def connexion(autocommit: bool = False) -> Iterator[object]:
    """Connexion à la base, refermée quoi qu'il arrive."""
    psycopg = _pilote()
    chaine = dsn()
    try:
        with psycopg.connect(chaine, autocommit=autocommit) as cnx:
            cnx.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(EF_SEARCH),))
            yield cnx
    except psycopg.Error as erreur:
        raise BaseIndisponible(
            f"Connexion impossible à {sans_secret(chaine)} : {erreur}\n"
            "Démarrer la base avec « docker compose up -d »."
        ) from erreur


def sans_secret(chaine: str) -> str:
    """Chaîne de connexion privée de son mot de passe, pour l'affichage et les journaux."""
    if "@" not in chaine or "://" not in chaine:
        return chaine
    schema, reste = chaine.split("://", 1)
    identifiants, hote = reste.rsplit("@", 1)
    utilisateur = identifiants.split(":", 1)[0]
    return f"{schema}://{utilisateur}:***@{hote}"


def appliquer_schema() -> None:
    """Crée l'extension, les tables et les index s'ils n'existent pas déjà."""
    with connexion(autocommit=True) as cnx:
        cnx.execute(SCHEMA.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def disponible() -> bool:
    """Vrai si la base répond. Sert aux tests d'intégration, qui se sautent sinon.

    Le résultat est retenu : la question est posée par chaque `skipif` du dépôt, et une
    base absente ferait payer le délai d'attente autant de fois qu'il y a de tests. Une
    base qui apparaît ou disparaît en cours d'exécution est un cas qu'on ne cherche pas à
    servir — relancer suffit.
    """
    try:
        with connexion() as cnx:
            cnx.execute("SELECT 1")
        return True
    except BaseIndisponible:
        return False
