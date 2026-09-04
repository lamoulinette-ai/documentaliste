"""Ce qui empêche une démonstration de brûler son budget avant le rendez-vous.

Le crédit se compte en cents. Trois protections, dans l'ordre où elles agissent :

1. **Le cache** — une question déjà posée ne rappelle pas le modèle. Deux personnes qui
   essaient le même exemple ne coûtent qu'une fois.
2. **Le compteur** — les appels effectivement passés sont comptés, et le compte survit à un
   redémarrage du conteneur tant que le volume tient.
3. **Le plafond** — au-delà, la rédaction se coupe.

Le point important est ce que la coupure ne fait pas : **la recherche continue de
fonctionner.** L'utilisateur voit toujours les passages retrouvés, il perd la synthèse. Une
démonstration qui dégrade vaut mieux qu'une démonstration qui meurt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

#: Plafond par défaut, en appels au modèle. Choisi contre un budget réel : à l'ordre de
#: grandeur de 0,0009 $ la question sur `mistral-small`, 200 appels tiennent sous 20 cents.
PLAFOND_DEFAUT = 200

_ESPACES = re.compile(r"\s+")


def normaliser(question: str) -> str:
    """Forme sur laquelle deux questions sont tenues pour la même.

    Casse, accents et espaces multiples sont écartés : « Quel dépistage ? » et « quel
    depistage ? » ne doivent pas coûter deux appels.
    """
    sans_accent = unicodedata.normalize("NFKD", question.casefold())
    sans_accent = "".join(c for c in sans_accent if not unicodedata.combining(c))
    return _ESPACES.sub(" ", sans_accent).strip()


def cle(question: str) -> str:
    """Nom de fichier du cache : une empreinte, jamais le texte de la question.

    Le texte servirait de nom de fichier lisible, et écrirait les questions des visiteurs
    en clair sur le disque du serveur. Une empreinte suffit à retrouver l'entrée.
    """
    return hashlib.sha256(normaliser(question).encode()).hexdigest()[:32]


class Budget:
    """Le cache, le compteur et le plafond, autour d'un dossier."""

    def __init__(self, dossier: Path, plafond: int = PLAFOND_DEFAUT) -> None:
        self.dossier = dossier
        self.plafond = plafond
        self.dossier.mkdir(parents=True, exist_ok=True)
        self._compteur = self.dossier / "appels.txt"

    @classmethod
    def depuis_l_environnement(cls) -> Budget:
        """Construit le garde-fou d'après `.env`, avec les défauts du module."""
        dossier = Path(os.environ.get("DOCUMENTALISTE_CACHE", "/tmp/cache"))
        plafond = int(os.environ.get("DOCUMENTALISTE_PLAFOND_APPELS", PLAFOND_DEFAUT))
        return cls(dossier, plafond)

    @property
    def inscriptible(self) -> bool:
        """Le dossier accepte-t-il réellement une écriture ?

        `compter` et `ecrire` avalent les erreurs d'écriture, pour qu'un disque plein ne
        casse pas une réponse. La même clémence masquerait à jamais un dossier appartenant
        à un autre utilisateur : le cache paraîtrait fonctionner et ne garderait rien.

        Cette propriété existe pour que le démarrage puisse le dire une fois.
        """
        essai = self.dossier / ".inscriptible"
        try:
            essai.write_text("", encoding="utf-8")
            essai.unlink()
        except OSError:
            return False
        return True

    @property
    def appels(self) -> int:
        """Appels au modèle déjà passés."""
        try:
            return int(self._compteur.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            return 0

    @property
    def epuise(self) -> bool:
        """Le plafond est-il atteint ?"""
        return self.appels >= self.plafond

    def compter(self) -> None:
        """Enregistre un appel. Un échec d'écriture ne doit pas casser la réponse.

        Le compteur est un garde-fou, pas une comptabilité : mieux vaut servir la question
        avec un compte approximatif que refuser parce qu'un disque est plein.
        """
        try:
            self._compteur.write_text(str(self.appels + 1), encoding="utf-8")
        except OSError:  # pragma: no cover - dépend du système de fichiers
            pass

    def lire(self, question: str) -> dict | None:
        """La réponse déjà rendue pour cette question, ou None."""
        chemin = self.dossier / f"{cle(question)}.json"
        try:
            return json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def ecrire(self, question: str, charge: dict) -> None:
        """Conserve une réponse. Comme le compteur, l'échec est sans conséquence."""
        try:
            (self.dossier / f"{cle(question)}.json").write_text(
                json.dumps(charge, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:  # pragma: no cover - dépend du système de fichiers
            pass
