"""Bras lexical de la recherche : BM25, écrit à la main, sans dépendance."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

#: Paramètres canoniques de BM25 : saturation d'un terme, correction par longueur.
_K1 = 1.5
_B = 0.75

#: Mots vides du français, restreints à ceux qui ne portent aucun sens clinique.
_VIDES = frozenset(
    """
    a au aux avec ce ces dans de des du elle en et eux il je la le les leur lui ma mais
    me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son
    sur ta te tes toi ton tu un une vos votre vous c d j l m n s t y est sont etre quel
    quelle quels quelles quoi comment lorsque
    """.split()
)

_MOT = re.compile(r"[a-z0-9]+")


def normaliser(texte: str) -> list[str]:
    """Minuscules, accents retirés, mots vides écartés."""
    plie = unicodedata.normalize("NFD", texte.lower())
    plie = "".join(c for c in plie if unicodedata.category(c) != "Mn")
    return [m for m in _MOT.findall(plie) if m not in _VIDES and len(m) > 1]


def rarete(passages_contenant: int, total: int) -> float:
    """Poids d'un terme selon sa rareté dans la collection."""
    return math.log(1 + (total - passages_contenant + 0.5) / (passages_contenant + 0.5))


def contribution(frequence: int, longueur_relative: float, poids: float) -> float:
    """Apport d'un terme au score d'un passage.

    Seul endroit du projet où la formule est écrite. La recherche en base réutilise cette
    fonction avec des statistiques venues de PostgreSQL : une seconde implémentation, même
    fidèle le jour où on l'écrit, dériverait au premier correctif porté d'un seul côté.
    """
    return poids * (frequence * (_K1 + 1) / (frequence + _K1 * (1 - _B + _B * longueur_relative)))


class Bm25:
    """Index lexical d'une collection de passages."""

    def __init__(self, documents: list[str]) -> None:
        self.jetons = [normaliser(d) for d in documents]
        self.longueurs = [len(j) for j in self.jetons]
        self.longueur_moyenne = sum(self.longueurs) / max(len(self.jetons), 1)
        # Index inversé : pour chaque terme, les passages où il figure et son compte.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, jetons in enumerate(self.jetons):
            for terme, compte in Counter(jetons).items():
                self.postings.setdefault(terme, []).append((index, compte))
        n = len(self.jetons)
        self.rarete = {terme: rarete(len(p), n) for terme, p in self.postings.items()}

    def scores(self, requete: str) -> list[float]:
        """Score BM25 de chaque passage pour cette requête."""
        moyenne = max(self.longueur_moyenne, 1e-9)
        resultats = [0.0] * len(self.jetons)
        for terme in normaliser(requete):
            postings = self.postings.get(terme)
            if not postings:
                continue
            poids = self.rarete[terme]
            for index, f in postings:
                resultats[index] += contribution(f, self.longueurs[index] / moyenne, poids)
        return resultats


#: Constante de la fusion par rangs réciproques. Elle décide de ce que le rang pèse face au
#: poids : à profondeur d, le rang 1 vaut `(k+d)/(k+1)` fois le rang d. Grande, elle écrase
#: cet écart et le poids cesse de doser pour choisir un bras. La valeur est à confronter au
#: balayage, jamais à reprendre d'un article dont les listes n'avaient pas notre profondeur.
_K_RRF = 60


def fusion_rrf(
    *classements: list[int], k: int = _K_RRF, poids: tuple[float, ...] | None = None
) -> list[int]:
    """Fusionne plusieurs classements d'indices en un seul.

    Les ex æquo sont départagés par le rang du document dans le classement où il est le
    mieux placé. Sans ce départage, l'ordre des arguments trancherait — et à poids égaux,
    où tout document de rang *r* d'un bras égale exactement le rang *r* de l'autre, la
    fusion produirait un entrelacement décidé par rien.

    Le dernier recours reste l'identifiant, arbitraire mais déterministe. Qui veut un
    entrelacement le demande à `entrelacer`, qui l'assume.
    """
    facteurs = poids or (1.0,) * len(classements)
    points: dict[int, float] = {}
    meilleurs: dict[int, int] = {}
    for classement, facteur in zip(classements, facteurs, strict=True):
        for rang, index in enumerate(classement, 1):
            points[index] = points.get(index, 0.0) + facteur / (k + rang)
            meilleurs[index] = min(meilleurs.get(index, rang), rang)
    return [
        index
        for index, _ in sorted(points.items(), key=lambda kv: (-kv[1], meilleurs[kv[0]], kv[0]))
    ]


def entrelacer(*classements: list[int]) -> list[int]:
    """Alterne les classements un à un, en ignorant les doublons déjà pris.

    Stratégie distincte de la fusion pondérée : elle ne suppose aucune commensurabilité
    entre les scores des bras, seulement que chacun sait ce qu'il place en tête. C'est ce
    que la fusion produisait par accident à poids exactement égaux, et qui s'est révélé
    meilleur que toute pondération — mieux vaut l'écrire que le laisser émerger d'une
    égalité de flottants.
    """
    ordre: list[int] = []
    vus: set[int] = set()
    for rang in range(max((len(c) for c in classements), default=0)):
        for classement in classements:
            if rang < len(classement) and classement[rang] not in vus:
                vus.add(classement[rang])
                ordre.append(classement[rang])
    return ordre
