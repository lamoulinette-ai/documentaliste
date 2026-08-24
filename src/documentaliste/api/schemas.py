"""Ce que l'API reçoit et ce qu'elle rend.

Une décision d'interface est inscrite ici plutôt que dans le front : **les passages
retrouvés sont rendus dans tous les cas, y compris sur refus**. Dix-neuf refus sur
cinquante-trois ont été mesurés comme excessifs ; un refus nu ressemble à une panne, un
refus qui montre ce qu'il a trouvé et dit pourquoi il ne tranche pas ressemble à de la
prudence — ce qu'il est censé être. Laisser ce choix au front reviendrait à pouvoir
l'oublier.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Demande(BaseModel):
    """La question posée."""

    question: str = Field(min_length=3, max_length=500)


class PassageRendu(BaseModel):
    """Un passage du corpus, avec de quoi le retrouver dans la publication d'origine."""

    numero: int
    #: Nom de fichier. Conservé : c'est l'identifiant vérifiable, et il ne change jamais.
    document: str
    page: int
    texte: str
    #: Titre publié par la HAS. Vide si le document n'est pas au manifeste — auquel cas
    #: l'interface retombe sur le nom de fichier plutôt que d'afficher un blanc.
    titre: str = ""
    #: Fiche HAS, pour aller lire la source. C'est ce qui distingue une citation d'une
    #: allégation : le lecteur peut vérifier sans nous croire sur parole.
    url: str = ""
    type_publication: str = ""
    mise_en_ligne: str = ""


class AffirmationRendue(BaseModel):
    """Une phrase et le passage qui doit l'établir."""

    texte: str
    extrait: int


class DefautRendu(BaseModel):
    """Ce qu'une affirmation a de vérifiablement faux, contrôlé sans modèle."""

    affirmation: str
    motif: str
    piece: str = ""


class Perimetre(BaseModel):
    """Ce que le corpus contient, pour que le front n'ait rien à coder en dur.

    Un professionnel de santé pose une question hors périmètre dans les cinq premières
    minutes. Qu'il l'apprenne de l'écran plutôt que d'un refus sec.
    """

    archive: str
    documents: int
    passages: int
    thematiques: list[str]


class Reponse(BaseModel):
    """Ce que l'API rend pour une question."""

    question: str
    #: « reponse », « constat_absence » ou « refus ». Lue sur ce qui a été produit, jamais
    #: sur ce que le modèle déclare.
    issue: str
    #: Motif, quand le système se tait. Vide sinon.
    refus: str = ""
    affirmations: list[AffirmationRendue] = []
    #: Toujours renseigné, refus compris.
    passages: list[PassageRendu] = []
    #: Contrôles mécaniques : citation hors périmètre, quantité absente de l'extrait cité.
    defauts: list[DefautRendu] = []
    #: Vrai quand la rédaction a été coupée faute de budget. La recherche, elle, a eu lieu :
    #: les passages sont là, seule la synthèse manque.
    redaction_indisponible: bool = False
