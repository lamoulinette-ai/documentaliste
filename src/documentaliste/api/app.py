"""Le service HTTP : santé, périmètre, et une question à la fois.

Lancement : `uvicorn documentaliste.api.app:app --host 0.0.0.0 --port 8005`.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from documentaliste.api.budget import Budget
from documentaliste.api.schemas import Demande, Perimetre, Reponse
from documentaliste.api.service import moteur, repondre
from documentaliste.encodage import MODELE, charger_modele
from documentaliste.probes.mistral import charger_env

# Avant toute lecture de l'environnement. En conteneur les variables viennent de
# `env_file` et sont déjà là ; hors conteneur, sans cet appel, tout ce module lirait des
# valeurs par défaut en croyant lire le `.env` — et rien ne le signalerait.
charger_env()

# Uvicorn n'installe de gestionnaire que sur ses propres loggers, pas sur la racine. Sans
# cette ligne, les avertissements de ce module passeraient par le gestionnaire de dernier
# recours et les messages d'information ne s'afficheraient pas du tout — or ce sont eux qui
# confirment au démarrage que le modèle est chargé et quel budget reste.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s: %(name)s - %(message)s",
)

journal = logging.getLogger("documentaliste")

#: Origines autorisées quand `CORS_ORIGINS` n'est pas renseigné : celle du serveur de
#: développement de Quasar, et rien d'autre. Un défaut permissif (« * ») ouvrirait l'API à
#: n'importe quelle page ; un défaut vide bloque le développement sans dire pourquoi.
ORIGINES_DEFAUT = ("http://localhost:9000", "http://127.0.0.1:9000")

#: Ce que le corpus contient. Rendu par l'API pour que le front n'ait rien à coder en dur,
#: et mis à jour ici le jour où l'archive change.
PERIMETRE = Perimetre(
    archive="18 juin 2026",
    documents=6504,
    passages=796019,
    thematiques=[
        "Bon usage du médicament",
        "Cancérologie",
        "Maladies chroniques",
        "Organisation des parcours",
        "Périnatalité et pédiatrie",
        "Personnes âgées",
        "Psychiatrie et santé mentale",
        "Santé publique et prévention",
    ],
)


def _origines() -> list[str]:
    """Origines autorisées, celles du développement à défaut.

    Le journal dit laquelle des deux sources a servi : une origine manquante ne se voit
    autrement que dans la console du navigateur, jamais côté serveur, et la panne ressemble
    alors à une défaillance de l'API.
    """
    brut = os.environ.get("CORS_ORIGINS", "").strip()
    declarees = [o.strip() for o in brut.split(",") if o.strip()]
    if declarees:
        return declarees
    journal.warning("CORS_ORIGINS absent — repli sur %s", ", ".join(ORIGINES_DEFAUT))
    return list(ORIGINES_DEFAUT)


limiteur = Limiter(
    key_func=get_remote_address,
    enabled=os.environ.get("RATELIMIT_ENABLED", "true").lower() == "true",
)


@asynccontextmanager
async def cycle(_app: FastAPI):  # noqa: ANN201
    """Charge le modèle au démarrage, et refuse de démarrer s'il manque quelque chose.

    Deux raisons, et la seconde est celle qui compte un jour de démonstration :

    - la première question ne paie plus les secondes de chargement des poids ;
    - **une dépendance absente se voit au lancement, pas devant l'utilisateur.** Sans ce
      préchargement, un extra oublié à l'installation produit une trace de trente lignes à
      la première question posée, et rien avant.

    Échouer ici est le bon comportement : le healthcheck du conteneur le voit, et le
    déploiement s'arrête. Basculer en silence sur un autre moteur ferait tourner le service
    dans une configuration qui n'a jamais été mesurée.
    """
    charger_modele(MODELE, moteur())
    journal.info("modèle « %s » chargé (moteur %s)", MODELE, moteur())
    if not os.environ.get("MISTRAL_API_KEY", "").strip():
        # Averti, pas fatal : la recherche fonctionne sans clé, et un service qui rend ses
        # passages vaut mieux qu'un service qui refuse de démarrer. Mais sans cette ligne,
        # l'absence de clé se confond avec un budget épuisé, question après question.
        journal.warning("MISTRAL_API_KEY absente — la recherche répondra, rédaction coupée")
    journal.info(
        "plafond de rédaction : %d appels, déjà consommés : %d", budget.plafond, budget.appels
    )
    yield


app = FastAPI(title="Documentaliste HAS", docs_url=None, redoc_url=None, lifespan=cycle)
app.state.limiter = limiteur
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def panne(_request: Request, erreur: Exception) -> JSONResponse:
    """Toute défaillance imprévue rend un message lisible, jamais une trace.

    La trace part au journal, où elle sert au diagnostic. Ce qui arrive à l'écran est une
    phrase : devant un professionnel de santé, une pile d'appels Python ne dit rien d'utile
    et donne le sentiment que rien n'a été prévu.
    """
    journal.exception("défaillance imprévue", exc_info=erreur)
    return JSONResponse(
        status_code=503,
        content={"detail": "Le service rencontre une difficulté. Réessayez dans un instant."},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=_origines(),
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

#: Construit une fois : le dossier de cache et le compteur sont partagés par les requêtes.
budget = Budget.depuis_l_environnement()

#: Fenêtre du limiteur, lue dans l'environnement pour ne pas figer un choix d'exploitation.
CADENCE = (
    f"{os.environ.get('RATELIMIT_IP_MAX', '20')}"
    f"/{os.environ.get('RATELIMIT_IP_WINDOW', '3600')}second"
)


@app.get("/health")
def sante() -> dict:
    """Sonde du conteneur. Ne touche ni la base ni le modèle : elle dit que le processus
    répond, pas que tout va bien — c'est exactement ce qu'un healthcheck doit dire."""
    return {"status": "ok", "appels": budget.appels, "plafond": budget.plafond}


@app.get("/perimetre", response_model=Perimetre)
def perimetre() -> Perimetre:
    """Ce que le corpus couvre, pour que l'interface puisse l'afficher."""
    return PERIMETRE


@app.post("/question", response_model=Reponse)
@limiteur.limit(CADENCE)
def question(demande: Demande, request: Request) -> Reponse:  # noqa: ARG001
    """Une question, des passages, et une réponse citée quand le corpus en porte une.

    `request` n'est pas utilisé ici mais doit figurer dans la signature : slowapi y lit
    l'adresse d'appel. Le retirer désactiverait le limiteur sans rien casser d'apparent.
    """
    return repondre(demande.question, budget)
