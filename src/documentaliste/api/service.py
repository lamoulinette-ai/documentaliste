"""Enchaîne la recherche et la rédaction, sans logique métier neuve.

Tout ce qui décide existe déjà et a été mesuré : `encoder` pour la requête, `chercher` pour
la fusion des deux bras, `composer`/`lire` pour la rédaction citée, `verifier` pour les
contrôles mécaniques. Ce module câble, il n'arbitre pas — c'est ce qui garantit que le
service rend ce que les rapports décrivent.
"""

from __future__ import annotations

import logging
import os

from documentaliste.api.budget import Budget
from documentaliste.api.schemas import (
    AffirmationRendue,
    DefautRendu,
    PassageRendu,
    Reponse,
)
from documentaliste.db.pool import connexion
from documentaliste.db.recherche import chercher, documents
from documentaliste.encodage import encoder
from documentaliste.probes.controle import verifier
from documentaliste.probes.mistral import MODELE, MistralIndisponible, completer_json
from documentaliste.probes.reponse import CONSIGNE, PASSAGES, SCHEMA, composer, lire

#: Motif rendu quand le plafond de dépense est atteint. Explicite plutôt que muet : un
#: utilisateur doit pouvoir distinguer « le système ne sait pas » de « le service est bridé ».
BUDGET_EPUISE = (
    "La rédaction automatique est momentanément indisponible. Les passages retrouvés "
    "dans le corpus sont affichés ci-dessous."
)

journal = logging.getLogger("documentaliste")


def _degrader(question: str, rendus: list[PassageRendu], motif: str) -> Reponse:
    """Rend les passages sans la rédaction, en NOMMANT la cause dans le journal.

    L'utilisateur voit toujours la même phrase — la cause ne le regarde pas et ne l'aide
    pas. Le journal, lui, doit distinguer un plafond atteint d'une clé absente ou d'une
    panne du fournisseur : sans cela, une erreur de configuration ressemble trait pour
    trait à un fonctionnement nominal en fin de budget.
    """
    journal.warning("rédaction non produite (%s) — question : %s", motif, question)
    return Reponse(
        question=question,
        issue="refus",
        refus=BUDGET_EPUISE,
        passages=rendus,
        redaction_indisponible=True,
    )


def _rendus(
    passages: list, metadonnees: dict[str, dict[str, str]] | None = None
) -> list[PassageRendu]:
    """Passages prêts pour l'affichage, enrichis du titre publié quand on le connaît."""
    connus = metadonnees or {}
    rendus = []
    for numero, p in enumerate(passages, 1):
        meta = connus.get(p.document, {})
        rendus.append(
            PassageRendu(
                numero=numero,
                document=p.document,
                page=p.page,
                texte=p.texte,
                titre=meta.get("titre", ""),
                # La fiche plutôt que le PDF : elle porte le contexte de publication, et
                # ouvre le document sans imposer un téléchargement.
                url=meta.get("url_fiche") or meta.get("url_pdf", ""),
                type_publication=meta.get("type_publication", ""),
                mise_en_ligne=meta.get("mise_en_ligne", ""),
            )
        )
    return rendus


def moteur() -> str:
    """Moteur d'inférence local, celui avec lequel l'image a été construite."""
    return os.environ.get("DOCUMENTALISTE_MOTEUR", "onnx")


def rechercher(question: str, k: int = PASSAGES) -> tuple[list, dict[str, dict[str, str]]]:
    """Les passages que la fusion des deux bras rend, et de quoi les citer proprement.

    Les deux lectures se font sur la même connexion : les métadonnées ne concernent qu'une
    dizaine de documents, et rouvrir une connexion pour elles coûterait plus que la requête.
    """
    (vecteur,) = encoder([question], moteur())
    with connexion() as cnx:
        passages = chercher(cnx, question, vecteur, k)
        return passages, documents(cnx, [p.document for p in passages])


def repondre(question: str, budget: Budget) -> Reponse:
    """La réponse complète : recherche, puis rédaction si le budget le permet.

    L'ordre compte. La recherche a lieu **avant** toute considération de budget, pour que
    l'épuisement du crédit ne prive jamais l'utilisateur des passages retrouvés.
    """
    if (connu := budget.lire(question)) is not None:
        # Journalisé comme les autres chemins : une question servie par le cache ne coûte
        # rien et ne produit aucune trace, ce qui donne l'impression que rien ne s'est
        # passé. Le compteur d'appels n'ayant pas bougé, c'est même le comportement voulu.
        journal.info("servie par le cache — question : %s", question)
        return Reponse(**connu)

    passages, metadonnees = rechercher(question)
    rendus = _rendus(passages, metadonnees)

    if budget.epuise:
        return _degrader(question, rendus, f"plafond atteint : {budget.appels}/{budget.plafond}")

    try:
        brut, _ = completer_json(
            CONSIGNE,
            composer(question, passages),
            SCHEMA,
            os.environ.get("DOCUMENTALISTE_MODELE", MODELE),
        )
    except MistralIndisponible as erreur:
        # Même dégradation vue de l'utilisateur, cause nommée dans le journal : une clé
        # absente et un fournisseur en panne lèvent la même exception.
        return _degrader(question, rendus, f"appel au modèle impossible — {erreur}")
    budget.compter()

    reponse = lire(question, brut)
    rendue = Reponse(
        question=question,
        issue=reponse.issue,
        refus=reponse.refus,
        affirmations=[
            AffirmationRendue(texte=a.texte, extrait=a.extrait) for a in reponse.affirmations
        ],
        passages=rendus,
        defauts=[
            DefautRendu(affirmation=d.affirmation, motif=d.motif, piece=d.piece)
            for d in verifier(reponse, passages)
        ],
    )
    budget.ecrire(question, rendue.model_dump())
    # Le chemin nominal se journalise aussi : sans cette ligne, « le modèle a répondu par un
    # refus » et « le modèle n'a pas été appelé » se ressemblent vus de l'écran.
    journal.info(
        "issue=%s affirmations=%d defauts=%d appels=%d/%d",
        rendue.issue,
        len(rendue.affirmations),
        len(rendue.defauts),
        budget.appels,
        budget.plafond,
    )
    return rendue
