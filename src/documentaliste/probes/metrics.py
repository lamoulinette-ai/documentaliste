"""Mesures de qualité d'une recherche documentaire."""

from __future__ import annotations

from math import log2
from typing import TYPE_CHECKING

from documentaliste.probes.lexical import normaliser

if TYPE_CHECKING:  # pragma: no cover - annotations seules
    # Importés pour l'annotation seule : à l'exécution, le cycle avec `retrieval` existerait.
    from documentaliste.probes.retrieval import Passage, Question


#: Les trois lectures d'une même collecte, de la plus stricte à la plus large.
#:
#: Elles s'emboîtent : un passage de la page source appartient au document source, qui
#: appartient au dossier. Le rappel est donc croissant de l'une à l'autre, et l'écart entre
#: « page » et « dossier » borne ce qu'une mesure au document laisse indéterminé.
GRANULARITES = ("page", "document", "dossier")


def _atteint(passage: Passage, question: Question, granularite: str) -> bool:
    """Ce passage compte-t-il comme une réponse, à cette granularité ?"""
    if granularite == "dossier" or question.source is None:
        return passage.document in question.documents
    document, page = question.source
    if granularite == "document":
        return passage.document == document
    return passage.document == document and passage.page == page


def rang_du_bon_document(
    resultats: list[Passage],
    attendus: frozenset[str],
    question: Question | None = None,
    granularite: str = "dossier",
) -> int | None:
    """Rang du premier passage qui répond, ou None.

    `attendus` reste le paramètre des appelants qui n'ont pas de `Question` sous la main ;
    dès qu'une question est fournie, c'est sa granularité qui décide.
    """
    for rang, passage in enumerate(resultats, 1):
        if question is None:
            if passage.document in attendus:
                return rang
        elif _atteint(passage, question, granularite):
            return rang
    return None


def rangs(
    questions: list[Question], trouves: list[list[Passage]], granularite: str = "dossier"
) -> list[int | None]:
    """Rang du premier passage répondant, question par question."""
    return [
        rang_du_bon_document(r, q.documents, q, granularite)
        for q, r in zip(questions, trouves, strict=True)
    ]


def rappels(
    questions: list[Question],
    trouves: list[list[Passage]],
    k: int,
    granularite: str = "dossier",
) -> dict[int, float]:
    rangs_ = rangs(questions, trouves, granularite)
    return {
        seuil: sum(1 for r in rangs_ if r is not None and r <= seuil) / len(questions)
        for seuil in sorted({1, 3, 5, k})
        if seuil <= k
    }


def ndcg(
    questions: list[Question],
    trouves: list[list[Passage]],
    k: int = 10,
    granularite: str = "dossier",
) -> float:
    """nDCG@k, gain binaire, moyenné sur les questions.

    Le rang réciproque ne compte que le premier passage utile ; le rappel@k ne compte pas
    où ils sont. Le nDCG fait les deux : chaque passage pertinent rapporte d'autant moins
    qu'il est loin, et la normalisation par l'idéal rend les questions comparables entre
    elles quel que soit leur nombre de passages utiles.

    Il est ici la monnaie d'échange du domaine, pas une mesure de plus : nos rang réciproque
    et rappel@k ne se comparent à rien de publié. **Cela ne rend pas le chiffre comparable
    pour autant** — l'étalon reste le nôtre. Le nDCG rend notre résultat lisible par
    d'autres ; il ne le met pas sur la même échelle qu'un résultat obtenu sur un autre jeu.
    """
    if not questions:
        return 0.0
    total = 0.0
    for question, resultats in zip(questions, trouves, strict=True):
        utiles = [
            rang
            for rang, passage in enumerate(resultats[:k], 1)
            if _atteint(passage, question, granularite)
        ]
        if not utiles:
            continue
        obtenu = sum(1 / log2(rang + 1) for rang in utiles)
        ideal = sum(1 / log2(rang + 1) for rang in range(1, len(utiles) + 1))
        total += obtenu / ideal
    return total / len(questions)


def aire_sous_la_courbe(scores: list[float], positifs: list[bool]) -> float:
    """AUC-ROC, calculée par le rang moyen des positifs — équivalente et sans tri instable.

    Elle répond à : « ce score, à lui seul, sépare-t-il les questions auxquelles il faut
    répondre de celles où il faut se taire ? » 0,5 signifie qu'il ne sépare rien, et c'est
    la réponse qu'il faut pouvoir écrire sans détour si elle vient.

    Les ex æquo reçoivent leur rang moyen : sans cela, l'ordre d'entrée des questions
    déciderait de la valeur.
    """
    combien = len(scores)
    vrais = sum(positifs)
    if not vrais or vrais == combien:
        return 0.5
    ordre = sorted(range(combien), key=lambda i: scores[i])
    rangs_moyens = [0.0] * combien
    debut = 0
    while debut < combien:
        fin = debut
        while fin + 1 < combien and scores[ordre[fin + 1]] == scores[ordre[debut]]:
            fin += 1
        moyen = (debut + fin) / 2 + 1
        for position in range(debut, fin + 1):
            rangs_moyens[ordre[position]] = moyen
        debut = fin + 1
    somme = sum(r for r, positif in zip(rangs_moyens, positifs, strict=True) if positif)
    return (somme - vrais * (vrais + 1) / 2) / (vrais * (combien - vrais))


def part_litterale(questions: list[Question], passages: list[Passage]) -> float:
    """Part des questions figurant **mot pour mot** dans le corpus indexé."""
    if not questions:
        return 0.0
    corpus = " ".join(" ".join(normaliser(p.texte)) for p in passages)
    aiguilles = [" ".join(normaliser(q.question)) for q in questions]
    return sum(1 for a in aiguilles if a and a in corpus) / len(questions)


def _pourcent(valeur: float | None) -> str:
    """« — » plutôt qu'un chiffre quand la mesure n'a pas de quoi se prononcer."""
    return "—" if valeur is None else f"{valeur:.0%}"


def rang_reciproque_moyen(
    questions: list[Question], trouves: list[list[Passage]], granularite: str = "dossier"
) -> float:
    """Moyenne de 1/rang du premier document attendu, 0 quand il ne vient pas."""
    if not questions:
        return 0.0
    return sum(1 / r for r in rangs(questions, trouves, granularite) if r is not None) / len(
        questions
    )


def plafond_de_rappel(questions: list[Question], par_bras: dict[str, list[list[Passage]]]) -> float:
    """Part des questions qu'**au moins un** bras sait servir."""
    if not questions:
        return 0.0
    servies = sum(
        1
        for index, q in enumerate(questions)
        if any(
            rang_du_bon_document(trouves[index], q.documents) is not None
            for trouves in par_bras.values()
        )
    )
    return servies / len(questions)


def part_lexicale(questions: list[Question], trouves: list[list[Passage]]) -> float | None:
    """Part des questions dont un terme attendu figure dans les 3 premiers passages."""
    avec_cles = [q for q in questions if q.mots_cles]
    if not avec_cles:
        return None
    exacts = 0
    for q, resultats in zip(questions, trouves, strict=True):
        if not q.mots_cles:
            continue
        tete = " ".join(p.texte.lower() for p in resultats[:3])
        if any(mot.lower() in tete for mot in q.mots_cles):
            exacts += 1
    return exacts / len(avec_cles)
