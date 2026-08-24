"""Ce qui mérite d'être réessayé, et ce qui ne le mérite pas.

Une campagne de deux cents appels franchit forcément une limite de débit. Abandonner au
premier 429 jetait le tour entier pour une attente d'une seconde ; réessayer une clé
invalide retarderait le diagnostic de cinq attentes pour rien.
"""

from __future__ import annotations

import httpx
import pytest

from documentaliste.probes import mistral
from documentaliste.probes.mistral import (
    ATTENTE_INITIALE,
    TENTATIVES,
    MistralIndisponible,
    _poster,
    attente,
)


class FausseReponse:
    """Le strict nécessaire de `httpx.Response` pour la boucle de reprise."""

    def __init__(self, code: int, texte: str = "", entetes: dict[str, str] | None = None) -> None:
        self.status_code, self.text, self.headers = code, texte, entetes or {}


@pytest.fixture
def poster(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Rend une fonction qui joue une suite de réponses et compte les tentatives."""

    def preparer(*reponses: FausseReponse) -> list[int]:
        suite, appels = list(reponses), []

        class Client:
            def __init__(self, **_: object) -> None: ...

            def __enter__(self) -> Client:
                return self

            def __exit__(self, *_: object) -> bool:
                return False

            def post(self, *_: object, **__: object) -> FausseReponse:
                appels.append(1)
                return suite.pop(0)

        monkeypatch.setattr(httpx, "Client", Client)
        monkeypatch.setattr(mistral, "attente", lambda *_: 0.0)
        return appels

    return preparer


class TestAttente:
    def test_elle_double_a_chaque_echec(self) -> None:
        valeurs = [attente(FausseReponse(429), n) for n in range(4)]
        assert valeurs == [ATTENTE_INITIALE * 2**n for n in range(4)]

    def test_retry_after_est_prefere(self) -> None:
        """L'API sait quand sa fenêtre se rouvre ; nous ne faisons que le supposer."""
        assert attente(FausseReponse(429, entetes={"Retry-After": "12"}), 0) == 12.0

    def test_un_retry_after_non_chiffre_ne_casse_pas(self) -> None:
        """La forme date HTTP est licite : on retombe sur le doublement plutôt qu'on échoue."""
        entetes = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
        assert attente(FausseReponse(429, entetes=entetes), 2) == ATTENTE_INITIALE * 4


class TestPoster:
    def test_un_429_est_reessaye(self, poster) -> None:  # noqa: ANN001
        appels = poster(FausseReponse(429), FausseReponse(429), FausseReponse(200))
        assert _poster({}, {}).status_code == 200
        assert len(appels) == 3

    def test_une_cle_invalide_n_est_pas_reessayee(self, poster) -> None:  # noqa: ANN001
        """Un 401 n'aboutira pas davantage à la cinquième tentative."""
        appels = poster(FausseReponse(401, "clé invalide"))
        with pytest.raises(MistralIndisponible, match="401"):
            _poster({}, {})
        assert len(appels) == 1

    def test_une_panne_du_service_est_reessayee(self, poster) -> None:  # noqa: ANN001
        appels = poster(FausseReponse(503), FausseReponse(200))
        assert _poster({}, {}).status_code == 200
        assert len(appels) == 2

    def test_les_tentatives_sont_bornees(self, poster) -> None:  # noqa: ANN001
        appels = poster(*[FausseReponse(429, "sature")] * TENTATIVES)
        with pytest.raises(MistralIndisponible, match="429"):
            _poster({}, {})
        assert len(appels) == TENTATIVES

    def test_le_dernier_message_est_celui_rendu(self, poster) -> None:  # noqa: ANN001
        """Rendre le premier échec cacherait la raison de l'abandon."""
        poster(FausseReponse(429, "trop vite"), FausseReponse(400, "requête malformée"))
        with pytest.raises(MistralIndisponible, match="400"):
            _poster({}, {})
