"""Ce qu'on peut reprocher à une réponse sans juger le sens, et comment le rendre.

Trois vérifications, de la plus dure à la plus molle. L'extrait cité existe et fait partie
de ceux qui ont été soumis — cela se prouve. Les **quantités** avancées figurent dans
l'extrait cité — cela se prouve aussi. Que l'extrait *soutienne* l'affirmation — cela ne se
prouve pas ici, et le rapport le dit.

**Seuls les nombres porteurs d'unité sont contrôlés.** Le premier lot d'épreuve a signalé
sept fautes, toutes fausses, toutes le même « 2 » de « PillCam™ COLON 2 » — un numéro de
version. Zéro faute réelle, sept inventées. Un danger clinique est une quantité : `850 mg`,
`7,5 %`, `65 ans`. Une année, un sigle, un numéro de modèle n'en sont pas.

Ce que la règle laisse passer : « grade 2 », « type 2 », « stade III ». Ce sont des
catégories, pas des quantités ; les vérifier demanderait de distinguer « diabète de type 2 »
de « PillCam COLON 2 », ce qu'aucun motif ne fait.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

#: Origines du jeu négatif. Un lot qui n'en couvre pas une doit le dire : les trois testent
#: trois échecs distincts, et le taux de refus n'a pas de sens moyenné sur elles.
ORIGINES = frozenset({"non_tranchee", "hors_perimetre", "absente_de_l_archive"})

#: Unités qui font d'un nombre une quantité. Masses, volumes, concentrations, durées,
#: pressions — ce dont une erreur change la conduite à tenir.
_UNITES = (
    r"%|mg|µg|mcg|ng|kg|g|mL|ml|cL|dL|L|UI|U|mmHg|mmol|mol|mEq|kcal|Gy|Bq|"
    r"ans?|années?|mois|semaines?|jours?|heures?|min|secondes?"
)

#: Une quantité : un nombre, un espace facultatif, une unité. Le nombre ne doit pas être
#: collé à une lettre — « HbA1c » n'est pas une quantité — et l'unité doit s'achever, sans
#: quoi « 2 gouttes » se lirait comme « 2 g ».
_QUANTITE = re.compile(rf"(?<![A-Za-zÀ-ÿ0-9])(\d+(?:[.,]\d+)?)\s?(?:{_UNITES})(?![A-Za-zÀ-ÿ])")


def quantites(texte: str) -> set[str]:
    """Nombres porteurs d'unité, sous forme repliée.

    Les zéros de queue ne se retirent qu'après une virgule décimale. Les retirer partout
    ramènerait « 500 mg » à « 5 mg », et deux posologies sur trois cesseraient de se
    distinguer — dans le sens qui absout le modèle.
    """
    valeurs = set()
    for brut in _QUANTITE.findall(texte):
        valeur = brut.replace(",", ".")
        if "." in valeur:
            valeur = valeur.rstrip("0").rstrip(".")
        valeurs.add(valeur or "0")
    return valeurs


#: Caractères de contexte rendus de part et d'autre du terme d'ancrage.
CONTEXTE = 120

#: Longueur minimale d'un mot pour servir d'ancre. En dessous, on tomberait sur « dans » ou
#: « pour », présents partout et qui n'indiquent rien.
ANCRE_MINIMALE = 5

_MOT = re.compile(rf"[A-Za-zÀ-ÿ]{{{ANCRE_MINIMALE},}}")


def _fenetre(texte: str, centre: int) -> str:
    debut = max(0, centre - CONTEXTE)
    fin = min(len(texte), centre + CONTEXTE)
    return ("…" if debut else "") + texte[debut:fin].strip() + ("…" if fin < len(texte) else "")


def ancrer(texte: str, affirmation: str) -> str:
    """Fenêtre du passage autour du terme le plus long qu'il partage avec l'affirmation.

    On ne peut pas centrer sur la quantité reprochée : elle est absente du passage, c'est
    tout l'objet du reproche. Le mot le plus long en commun est un ancrage grossier mais
    qui tombe presque toujours sur le sujet — molécule, examen, pathologie.
    """
    minuscules = texte.lower()
    for mot in sorted(set(_MOT.findall(affirmation)), key=len, reverse=True):
        position = minuscules.find(mot.lower())
        if position >= 0:
            return _fenetre(texte, position)
    return _fenetre(texte, 0)


@dataclass(frozen=True)
class Defaut:
    """Ce qu'une affirmation a de vérifiablement faux, avec de quoi en juger."""

    question: str
    affirmation: str
    motif: str
    #: Extrait du passage cité, autour de ce qui est reproché. Vide quand le passage cité
    #: n'existe pas. Sans lui, le rapport accuse sans produire la pièce.
    piece: str = ""


def verifier(reponse, passages: list) -> list[Defaut]:  # noqa: ANN001
    """Défauts prouvables sans modèle : citation hors périmètre, quantité absente."""
    defauts = []
    for affirmation in reponse.affirmations:
        if not 1 <= affirmation.extrait <= len(passages):
            defauts.append(
                Defaut(
                    reponse.question,
                    affirmation.texte,
                    f"extrait [{affirmation.extrait}] hors des {len(passages)} fournis",
                )
            )
            continue
        source = passages[affirmation.extrait - 1]
        presentes = quantites(source.texte)
        inventees = sorted(quantites(affirmation.texte) - presentes)
        if inventees:
            # Les quantités de l'extrait sont données à côté de celles qu'on reproche :
            # c'est leur confrontation qui rend le verdict jugeable.
            porte = ", ".join(sorted(presentes)) or "aucune"
            defauts.append(
                Defaut(
                    reponse.question,
                    affirmation.texte,
                    f"quantité(s) absente(s) de l'extrait cité : {', '.join(inventees)} "
                    f"(l'extrait porte : {porte})",
                    ancrer(source.texte, affirmation.texte),
                )
            )
    return defauts


def _part(compte: int, total: int) -> str:
    """« 7/8 (88 %) », ou « — » quand il n'y a rien à rapporter."""
    if not total:
        return "—"
    return f"{compte}/{total} ({compte / total:.0%})"


def controlables(cas: list) -> int:
    """Affirmations portant au moins une quantité, seules à passer sous le contrôle.

    C'est le dénominateur du « zéro défaut ». Sans lui, un contrôle qui n'a rien eu à
    contrôler se lit comme un contrôle réussi.
    """
    return sum(1 for c in cas for a in c.reponse.affirmations if quantites(a.texte))


def _affirmations(cas) -> list[str]:  # noqa: ANN001
    """Ce que le modèle a affirmé, pour qu'on voie ce qu'il invente quand il invente."""
    lignes = [f"- « {cas.question} »"]
    for affirmation in cas.reponse.affirmations:
        lignes.append(f"  — [{affirmation.extrait}] {affirmation.texte}")
    return lignes


def croiser(cas: list) -> list[str]:
    """Le refus confronté à ce que la recherche avait réellement fourni.

    Le plafond de rappel est de 82 % : sur une question sur cinq, les passages soumis ne
    contiennent pas la réponse. Un refus y est **correct**, et le compter comme échec
    ferait passer pour un défaut du modèle ce qui est une limite de la recherche.
    """
    positives = [c for c in cas if c.origine == "positive"]
    servies = [c for c in positives if c.servie]
    aveugles = [c for c in positives if not c.servie]
    lignes = [
        "",
        "## Le refus, croisé avec ce que la recherche avait fourni",
        "",
        "Un refus n'est un défaut que si la réponse était dans les extraits. Les moyenner",
        "ferait porter au modèle le manque de la recherche.",
        "",
        "| situation | attendu | mesuré | |",
        "| --- | --- | ---: | --- |",
        f"| document attendu présent | répondre | "
        f"{_part(sum(1 for c in servies if not c.reponse.a_refuse), len(servies))} | |",
        f"| document attendu absent | refuser | "
        f"{_part(sum(1 for c in aveugles if c.reponse.a_refuse), len(aveugles))} |"
        " ⚠ non concluante |",
    ]
    par_origine: dict[str, list] = {}
    for c in cas:
        if c.origine != "positive":
            par_origine.setdefault(c.origine, []).append(c)
    for origine, lot in sorted(par_origine.items()):
        refus = sum(1 for c in lot if c.reponse.a_refuse)
        lignes.append(f"| question {origine} | refuser | {_part(refus, len(lot))} | |")
    lignes += _rendre_les_trois_issues(cas)
    lignes += _rendre_les_non_tranchees(par_origine.get("non_tranchee", []))
    lignes += [
        "",
        "La ligne « document attendu absent » mesure le bavardage : le modèle a répondu",
        "alors qu'il n'avait pas de quoi. C'est la faute la plus grave de ce tableau, et",
        "celle qu'aucun seuil sur la recherche ne pouvait prévenir.",
        "",
        "**Le refus excessif a été mesuré à la main** : sur les 53 questions où le document",
        "attendu figurait et où le modèle a refusé, 19 refus étaient excessifs — la réponse",
        "était dans les passages. Onze des trente-quatre refus justifiés tenaient à une",
        "absence réelle du corpus ; les autres, à la page servie ou à l'étalon.",
        "",
        "**La ligne « document attendu absent » ne conclut rien, et c'est mesuré.** Elle",
        "suppose que SEUL le document attendu peut répondre. La relecture des 192 issues a",
        "montré le contraire : la plupart des réponses produites sans lui viennent d'autres",
        "référentiels HAS et sont correctes — « la démarche pour initier un traitement contre",
        "l'ostéoporose » en est le cas type. Ce qu'elle compte comme du bavardage est",
        "majoritairement une réponse juste tirée d'une source que la vérité de terrain",
        "ignorait. Le rappel au **dossier HAS**, mesuré par `sonde-mesure`, est la lecture qui",
        "ne souffre pas de ce défaut.",
    ]
    bavardes = [c for c in aveugles if not c.reponse.a_refuse]
    bavardes += [c for c in cas if c.origine != "positive" and not c.reponse.a_refuse]
    if bavardes:
        lignes += [
            "",
            "### Ce que le modèle a affirmé sans avoir de quoi",
            "",
            "Les fautes sont nommées comme les refus. Ne détailler que ce qui disculpe le",
            "modèle ferait pencher le rapport du côté flatteur.",
            "",
        ]
        for c in bavardes:
            lignes += _affirmations(c)

    refusees = [c for c in servies if c.reponse.a_refuse]
    if refusees:
        lignes += [
            "",
            "### Les refus alors que le document attendu était là",
            "",
            "À lire une par une : elles diront si le refus était excessif, ou si le bon",
            "document figurait par une page qui ne répondait pas.",
            "",
        ]
        for c in refusees:
            lignes.append(f"- « {c.question} »")
            if c.reponse.refus:
                lignes.append(f"  — motif du modèle : « {c.reponse.refus} »")
    manquantes = ORIGINES - set(par_origine)
    if manquantes:
        lignes += [
            "",
            f"**Origine(s) non couverte(s) par ce lot : {', '.join(sorted(manquantes))}.** "
            "Les questions que la HAS déclare non tranchées sont le seul cas difficile — "
            "leur sujet est traité par le corpus, seule la réponse manque. Un taux de refus "
            "qui ne les a pas vues ne dit rien du refus.",
        ]
    return lignes


#: Affirmations rendues par cas dans la liste à trier. Assez pour juger, pas de quoi noyer.
AFFIRMATIONS_MONTREES = 3


def _rendre_les_trois_issues(cas: list) -> list[str]:
    """Réponse, constat d'absence, refus — par origine.

    Le constat d'absence ne peut pas être fondu dans la réponse. Sur une question que la HAS
    déclare sans réponse établie, rapporter qu'elle ne tranche pas est le bon comportement ;
    le compter comme une réponse le ferait passer pour une régression, et le compter comme
    un refus effacerait ce qui distingue une source qui constate d'une source qui se tait.
    """
    lignes = [
        "",
        "### Trois issues, et non deux",
        "",
        "Une source qui CONSTATE une absence répond ; une source qui se TAIT ne répond pas.",
        "Les confondre a produit des refus là où l'extrait portait la réponse.",
        "",
        "| origine | réponse | constat d'absence | refus |",
        "| --- | ---: | ---: | ---: |",
    ]
    par_origine: dict[str, list] = {}
    for c in cas:
        par_origine.setdefault(c.origine, []).append(c)
    for origine, lot in sorted(par_origine.items()):
        compte = Counter(c.reponse.issue for c in lot)
        lignes.append(
            f"| {origine} | {_part(compte['reponse'], len(lot))} | "
            f"{_part(compte['constat_absence'], len(lot))} | "
            f"{_part(compte['refus'], len(lot))} |"
        )
    return lignes


def _rendre_les_non_tranchees(lot: list) -> list[str]:
    """Les non tranchées auxquelles le modèle a répondu, à trier à la lecture.

    Un taux de refus seul y confond deux conduites opposées : conclure là où la HAS ne
    conclut pas, et répondre en nommant ce que les données ne permettent pas d'établir. La
    seconde est celle qu'on recherche.

    Les séparer par motif a été tenté et abandonné : le motif qui repère les non tranchées
    dans les documents de la HAS — « données insuffisantes pour », « ne permet pas de
    conclure » — ne reconnaît pas les tournures du rédacteur, « il n'y a pas de données
    évaluant », « ne permet pas d'identifier ». Il rendait zéro signalement sur onze là où
    la lecture en trouvait. L'élargir sur les deux tournures aperçues aurait été l'ajuster
    sur ce qu'on venait de voir.
    """
    repondues = [c for c in lot if not c.reponse.a_refuse]
    if not repondues:
        return []
    lignes = [
        "",
        "### Les non tranchées auxquelles le modèle a répondu",
        "",
        f"{len(repondues)} sur {len(lot)}. La HAS y déclare que les données ne permettent pas",
        "de conclure, et un refus n'est pas la seule conduite juste : **répondre en nommant ce",
        "qui n'est pas établi en est une autre, préférable**. Le taux de refus les confond.",
        "",
        "Pour chacune : cette réponse reconnaît-elle que la HAS ne conclut pas, ou conclut-elle",
        "à sa place ? Seul le second cas est un défaut.",
        "",
    ]
    for cas in repondues:
        lignes.append(f"#### {cas.question}")
        lignes.append("")
        lignes += [f"- {a.texte}" for a in cas.reponse.affirmations[:AFFIRMATIONS_MONTREES]]
        reste = len(cas.reponse.affirmations) - AFFIRMATIONS_MONTREES
        if reste > 0:
            lignes.append(f"- *…et {reste} autre(s) affirmation(s).*")
        lignes.append("")
    return lignes


def rapporter(cas: list, jetons: int, passages: int, modele: str, chemin: Path) -> None:
    """Consigne ce qui est prouvé, ce qui est compté, et ce qui ne l'est pas."""
    defauts = [d for c in cas for d in c.defauts]
    affirmations = sum(len(c.reponse.affirmations) for c in cas)
    avec_quantite = controlables(cas)
    cout = f"{jetons} jetons facturés" if jetons else "aucun appel modèle, contrôle rejoué"
    lignes = [
        "# Réponse à citation obligatoire",
        "",
        f"{len(cas)} question(s), {passages} extraits soumis par question, "
        f"modèle `{modele}`, {cout}.",
        "",
        "## Ce qui se prouve sans modèle",
        "",
        f"{affirmations} affirmation(s) produite(s), dont **{avec_quantite}** avancent une",
        f"quantité. Ce sont elles, et elles seules, que le contrôle peut juger : "
        f"**{len(defauts)}** défaut(s).",
        "",
        "Le second nombre est le seul qui compte. Un « zéro défaut » rapporté aux",
        f"{affirmations} affirmations se lirait comme une mesure ; rapporté aux",
        f"{avec_quantite} contrôlables, il dit ce qu'il vaut — et si ce dénominateur est",
        "petit, le contrôle n'a presque rien contrôlé.",
        "",
        "Seuls les nombres porteurs d'unité sont retenus — `850 mg`, `7,5 %`, `65 ans`.",
        "Une année, un sigle ou un numéro de modèle ne sont pas des quantités : les",
        "compter avait produit sept fausses accusations pour zéro faute réelle, toutes le",
        "même « 2 » de « PillCam™ COLON 2 ».",
        "",
        "Restent hors contrôle « grade 2 », « type 2 », « stade III » : ce sont des",
        "catégories, et aucun motif ne les distingue d'un numéro de modèle.",
    ]
    if defauts:
        lignes += [
            "",
            "### Les défauts, un par un",
            "",
            "L'extrait cité est reproduit sous chaque reproche, ancré sur le terme qu'il",
            "partage avec l'affirmation : sans lui, l'accusation n'est pas jugeable.",
            "",
        ]
        for d in defauts:
            lignes += [f"- « {d.affirmation} »", f"  — {d.motif}"]
            if d.piece:
                lignes.append(f"  — extrait cité : « {d.piece} »")
            lignes.append(f"  (question : {d.question})")
    lignes += croiser(cas)
    lignes += [
        "",
        "## Ce que ce rapport ne mesure pas",
        "",
        "Que l'extrait cité **soutienne** l'affirmation. Une phrase peut n'inventer aucune",
        "quantité, citer un extrait bien réel, et lui faire dire autre chose. Établir cela",
        "demande un juge, et un juge non confronté à des jugements humains serait lui-même",
        "une mesure non mesurée.",
    ]
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"\nRapport écrit -> {chemin}")
