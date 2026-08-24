"""Client Mistral minimal, réduit à ce dont les sondes ont besoin.

MISTRAL_API_KEY=...   (dans .env à la racine du dépôt, voir .env.example)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://api.mistral.ai/v1/chat/completions"

#: Modèle retenu : le plus petit qui reformule proprement en français.
MODELE = "mistral-small-latest"

#: Température nulle et graine fixe : deux exécutions doivent donner le même étalon.
TEMPERATURE = 0.0
GRAINE = 7

_DELAIS = httpx.Timeout(connect=15.0, read=120.0, write=15.0, pool=15.0)

#: Codes qui valent « réessaie » : débit dépassé, ou panne passagère du service.
#:
#: Les autres 4xx — clé invalide, requête malformée — n'ont aucune chance d'aboutir à la
#: seconde tentative, et les réessayer ne ferait que retarder le diagnostic de cinq
#: attentes.
_REESSAYABLES = frozenset({429, 500, 502, 503, 504})

#: Tentatives par appel, la première comprise.
TENTATIVES = 5

#: Attente initiale, doublée à chaque échec. Cinq tentatives couvrent ainsi une trentaine
#: de secondes, ce qu'une limite de débit par minute suffit à franchir.
ATTENTE_INITIALE = 1.0


def attente(reponse: httpx.Response, tentative: int) -> float:
    """Secondes avant de réessayer.

    `Retry-After` est préféré quand l'API le fournit : elle sait quand sa fenêtre se
    rouvre, nous ne faisons que le supposer.
    """
    entete = reponse.headers.get("Retry-After", "").strip()
    if entete.isdigit():
        return float(entete)
    return ATTENTE_INITIALE * 2**tentative


class MistralIndisponible(RuntimeError):
    """L'appel n'a pas abouti, ou la réponse ne respecte pas le contrat."""


#: Profondeur de remontée à la recherche d'un `.env`.
_REMONTEE = 2


def fichiers_env(depart: Path | None = None) -> list[Path]:
    """Emplacements de `.env` examinés, du plus proche au plus lointain."""
    base = (depart or Path.cwd()).resolve()
    return [dossier / ".env" for dossier in [base, *base.parents[:_REMONTEE]]]


def charger_env(depart: Path | None = None) -> list[Path]:
    """Verse les variables des `.env` trouvés dans l'environnement, sans rien écraser."""
    trouves = []
    for chemin in fichiers_env(depart):
        if not chemin.is_file():
            continue
        trouves.append(chemin)
        # Un `.env` rédigé sous Windows peut être en cp1252 ; on ne bloque pas dessus.
        for ligne in chemin.read_text(encoding="utf-8", errors="replace").splitlines():
            ligne = ligne.strip()
            if not ligne or ligne.startswith("#") or "=" not in ligne:
                continue
            nom, _, valeur = ligne.partition("=")
            nom = nom.strip().removeprefix("export ").strip()
            if nom:
                os.environ.setdefault(nom, valeur.strip().strip('"').strip("'"))
    return trouves


def cle_api() -> str:
    """Clé lue dans l'environnement, `.env` compris, avec un diagnostic utile."""
    charger_env()
    cle = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not cle:
        examines = "\n  ".join(str(c) for c in fichiers_env())
        raise MistralIndisponible(
            "MISTRAL_API_KEY introuvable, ni dans l'environnement ni dans un .env.\n"
            f"Fichiers examinés :\n  {examines}\n"
            "Voir .env.example ; ne jamais passer la clé en argument."
        )
    return cle


def _poster(charge: dict[str, Any], entetes: dict[str, str]) -> httpx.Response:
    """Poste la requête, en réessayant ce qui mérite de l'être.

    Une campagne de deux cents appels franchit forcément une limite de débit. Abandonner au
    premier 429 jetait le tour entier pour une attente d'une seconde.
    """
    dernier = ""
    with httpx.Client(timeout=_DELAIS) as client:
        for tentative in range(TENTATIVES):
            try:
                reponse = client.post(API_URL, headers=entetes, json=charge)
            except httpx.HTTPError as erreur:
                raise MistralIndisponible(f"appel impossible : {erreur}") from erreur
            if reponse.status_code == 200:
                return reponse
            dernier = f"HTTP {reponse.status_code} : {reponse.text[:300]}"
            if reponse.status_code not in _REESSAYABLES or tentative == TENTATIVES - 1:
                break
            time.sleep(attente(reponse, tentative))
    raise MistralIndisponible(dernier)


def completer_json(
    consigne: str, demande: str, schema: dict[str, Any], modele: str = MODELE
) -> tuple[dict[str, Any], int]:
    """Renvoie l'objet JSON produit par le modèle et le nombre de jetons facturés."""
    charge = {
        "model": modele,
        "temperature": TEMPERATURE,
        "random_seed": GRAINE,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "sortie", "schema": schema, "strict": True},
        },
        "messages": [
            {"role": "system", "content": consigne},
            {"role": "user", "content": demande},
        ],
    }
    entetes = {"Authorization": f"Bearer {cle_api()}", "Content-Type": "application/json"}
    reponse = _poster(charge, entetes)
    corps = reponse.json()
    jetons = int(corps.get("usage", {}).get("total_tokens", 0))
    try:
        return json.loads(corps["choices"][0]["message"]["content"]), jetons
    except (KeyError, IndexError, json.JSONDecodeError) as erreur:
        raise MistralIndisponible(f"réponse inexploitable : {erreur}") from erreur
