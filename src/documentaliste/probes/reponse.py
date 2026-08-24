"""Répondre en citant, ou refuser : ce que le lot 1c doit rendre vérifiable.

**Le refus est une sortie de plein droit, pas un cas d'erreur.** La mesure de séparabilité
a établi qu'aucun seuil sur la recherche ne sépare les questions répondables des autres :
les questions que la HAS déclare non tranchées obtiennent une similarité supérieure au
premier décile des positives. Rien en amont ne peut donc trier. Le refus doit être produit
par la rédaction, à partir des passages lus, et le schéma doit le permettre explicitement.

**Une affirmation sans citation valide est rejetée avant lecture.** Chaque affirmation porte
l'identifiant du passage qui la fonde. Un identifiant absent des passages soumis se détecte
sans modèle et sans jugement.

**Tout est conservé, pour que corriger un contrôle ne coûte rien.** Les réponses brutes et
les extraits soumis sont écrits dans `fixtures/reponses.json`. Ce fichier ne s'écrase pas :
il porte une campagne payée, et sert de référence à la relecture. Changer une règle de
vérification se rejoue alors avec `--controler`, sans un seul appel modèle. La première
correction de la règle des nombres avait coûté 55 000 jetons pour régénérer des réponses
identiques.

    uv run sonde-reponse --essai        # n'envoie rien, montre ce qui serait envoyé
    uv run sonde-reponse --combien 9    # lot d'épreuve, trois par origine négative
    uv run sonde-reponse                # les deux étalons entiers
    uv run sonde-reponse --controler    # rejoue les contrôles sur ce qui est conservé
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from documentaliste.db.pool import connexion
from documentaliste.db.recherche import CANDIDATS, chercher
from documentaliste.probes.controle import Defaut, rapporter, verifier
from documentaliste.probes.mistral import MODELE, MistralIndisponible, completer_json

#: Passages soumis au modèle. C'est la profondeur que le reclassement rend, et au-delà de
#: laquelle le rappel de la recherche ne progresse plus.
PASSAGES = 10

CONSIGNE = (
    "Tu réponds à des questions cliniques en t'appuyant EXCLUSIVEMENT sur des extraits de "
    "référentiels de la Haute Autorité de Santé qui te sont fournis.\n"
    "\n"
    "Deux réponses sont possibles, et une seule à la fois.\n"
    "\n"
    "1. Si les extraits répondent à la question, produis une liste d'affirmations. Chaque "
    "affirmation est une phrase autonome, et porte le numéro de l'extrait qui l'établit. "
    "N'affirme rien qui ne soit écrit dans l'extrait cité : ni déduction, ni complément "
    "venu de tes connaissances, ni généralisation.\n"
    "\n"
    "2. Si les extraits ne répondent pas, renvoie un refus expliquant en une phrase ce qui "
    "manque. Refuse quand les extraits ne font qu'aborder le sujet, quand ils portent sur "
    "une population ou une situation différente, ou quand la question sort du champ de ces "
    "référentiels.\n"
    "\n"
    "UNE ABSENCE CONSTATÉE PAR LA SOURCE EST UNE RÉPONSE, PAS UN MOTIF DE REFUS. Si un "
    "extrait écrit lui-même qu'il n'existe pas de donnée, pas de consensus, pas de "
    "recommandation, ou que les données ne permettent pas de conclure ou de privilégier une "
    "option, alors la question a une réponse : c'est celle-là. Affirme-la en citant "
    "l'extrait qui la porte, et pose « nature » à « constat_absence ».\n"
    "Ne confonds pas les deux : la source qui CONSTATE une absence répond ; la source qui se "
    "TAIT ne répond pas.\n"
    "\n"
    "UNE RÉPONSE PARTIELLE SE DONNE. Si les extraits établissent une partie de ce qui est "
    "demandé — un bénéfice quand la question demande bénéfices et risques, un examen quand "
    "elle demande la liste —, affirme cette partie et dis dans le champ « refus » ce qui "
    "reste sans réponse. Ne refuse pas en bloc parce qu'il manque le reste.\n"
    "\n"
    "Refuser quand les extraits ne suffisent pas est un bon comportement, pas un échec. "
    "Une affirmation non fondée est une faute grave.\n"
    "Reprends les nombres — posologies, seuils, grades, années — exactement tels qu'ils "
    "figurent dans l'extrait cité.\n"
    "Réponds uniquement selon le schéma demandé."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "nature": {
            "type": "string",
            "enum": ["reponse", "constat_absence", "refus"],
            "description": (
                "« constat_absence » quand un extrait constate lui-même qu'il n'existe "
                "pas de donnée ou de recommandation : c'est une réponse, pas un refus."
            ),
        },
        "refus": {
            "type": "string",
            "description": "Ce qui reste sans réponse, ou ce qui manque si tu refuses.",
        },
        "affirmations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "texte": {"type": "string"},
                    "extrait": {
                        "type": "integer",
                        "description": "Numéro de l'extrait qui établit l'affirmation.",
                    },
                },
                "required": ["texte", "extrait"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["nature", "refus", "affirmations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Affirmation:
    """Une phrase et l'extrait qui doit l'établir."""

    texte: str
    extrait: int


@dataclass(frozen=True)
class Reponse:
    """Ce que le modèle a rendu pour une question, avant toute vérification."""

    question: str
    refus: str
    affirmations: tuple[Affirmation, ...]
    #: « reponse », « constat_absence » ou « refus », tels que le modèle les déclare.
    nature: str = ""

    @property
    def a_refuse(self) -> bool:
        """Un refus est l'absence d'affirmation, jamais la seule présence d'un texte.

        Le modèle peut motiver un refus *et* produire des affirmations ; c'est alors une
        réponse, et le champ de refus n'est qu'un commentaire. Se fier au champ ferait
        compter comme refus des réponses bel et bien données.
        """
        return not self.affirmations

    @property
    def a_constate_une_absence(self) -> bool:
        """La source dit elle-même qu'il n'y a pas de donnée, et le modèle l'a rapporté.

        Troisième issue, indispensable à la mesure : sans elle, « la HAS n'a pas tranché »
        se compterait comme une réponse ordinaire. Sur les questions déclarées sans réponse
        établie, ce serait compter une régression là où le comportement est le bon.
        """
        return self.nature == "constat_absence" and not self.a_refuse

    @property
    def issue(self) -> str:
        """L'issue effective, mesurée sur ce qui a été produit et non sur ce qui est déclaré.

        Un modèle qui annonce « refus » en produisant des affirmations a répondu ; un modèle
        qui annonce « constat_absence » sans rien affirmer a refusé. La déclaration ne sert
        qu'à distinguer les deux formes de réponse.
        """
        if self.a_refuse:
            return "refus"
        return "constat_absence" if self.nature == "constat_absence" else "reponse"


@dataclass(frozen=True)
class Extrait:
    """Un passage tel qu'il a été soumis, conservé pour rejouer les contrôles."""

    document: str
    page: int
    texte: str


@dataclass
class Cas:
    """Une question, ce que la recherche a rendu, et ce que le modèle en a fait."""

    question: str
    #: Vrai si un document attendu figure parmi les passages soumis. Sans lui, un refus est
    #: correct : le modèle n'avait pas de quoi répondre.
    servie: bool
    origine: str
    reponse: Reponse
    extraits: tuple[Extrait, ...]
    defauts: tuple[Defaut, ...] = ()


def composer(question: str, passages: list) -> str:
    """Met la question et les extraits numérotés dans la forme attendue par la consigne."""
    lignes = [f"Question : {question}", "", "Extraits :"]
    for numero, passage in enumerate(passages, 1):
        lignes.append(f"[{numero}] {passage.texte}")
    return "\n".join(lignes)


def lire(question: str, brut: dict) -> Reponse:
    """Convertit la sortie du modèle, en tolérant les champs absents du schéma strict."""
    affirmations = tuple(
        Affirmation(a["texte"], a["extrait"])
        for a in brut.get("affirmations") or []
        if a.get("texte", "").strip()
    )
    return Reponse(
        question,
        (brut.get("refus") or "").strip(),
        affirmations,
        (brut.get("nature") or "").strip(),
    )


def _servie(passages: list, attendus: frozenset[str]) -> bool:
    return any(p.document in attendus for p in passages)


#: Signes par jeton, ordre de grandeur pour du français. Sert à situer un coût avant de
#: l'engager, jamais à le facturer : le décompte réel est rendu par l'API après coup.
SIGNES_PAR_JETON = 4

#: Première invite composée, gardée pour être montrée en exemple à l'essai.
_APERCU: list[str] = []


def devis(tailles: list[int], apercu: str = "") -> list[str]:
    """Ce que la campagne coûterait, avant de l'engager.

    L'invite entière était affichée pour chaque question : cent quatre-vingt-quinze
    invites déversées, et pas le seul chiffre qui décide. Un essai doit rendre une taille,
    pas un volume.
    """
    if not tailles:
        return ["Aucune question."]
    exemple = ["", "Une invite, en exemple :", "-" * 78, apercu, "-" * 78, ""] if apercu else []
    ordonnees = sorted(tailles)
    total = sum(tailles)
    return exemple + [
        f"{len(tailles)} appel(s) · {total // 1000} k signes au total",
        f"médiane {ordonnees[len(ordonnees) // 2]} signes · maximum {ordonnees[-1]}",
        f"soit environ {total // SIGNES_PAR_JETON // 1000} k jetons d'entrée,"
        f" à {SIGNES_PAR_JETON} signes par jeton",
        "Le décompte réel viendra de l'API : celui-ci situe l'ordre de grandeur.",
    ]


def traiter(
    cnx: object, question: str, attendus: frozenset[str], origine: str, vecteur, essai: bool
) -> tuple[Cas | None, int]:  # noqa: ANN001
    """Interroge la base, puis le modèle. La vérification vient après, séparément.

    En essai, rien n'est envoyé et la taille de l'invite est rendue à la place du coût :
    c'est elle qui permet de chiffrer la campagne avant de la payer.
    """
    passages = chercher(cnx, question, vecteur, PASSAGES, candidats=CANDIDATS)
    invite = composer(question, passages)
    if essai:
        # Une seule invite est montrée, la première : les cent quatre-vingt-quinze déversées
        # noyaient le seul chiffre qui décide.
        if not _APERCU:
            _APERCU.append(invite)
        return None, len(invite)
    brut, jetons = completer_json(CONSIGNE, invite, SCHEMA, MODELE)
    extraits = tuple(Extrait(p.document, p.page, p.texte) for p in passages)
    return (
        Cas(question, _servie(passages, attendus), origine, lire(question, brut), extraits),
        jetons,
    )


def prelever(negatives: list[dict], combien: int) -> list[dict]:
    """Tire `combien` négatives en passant par toutes les origines, à tour de rôle.

    Prendre les premières du fichier revient à prendre une origine : il est ordonné par
    catégorie. Le premier lot d'épreuve n'a ainsi vu aucune question non tranchée — la
    seule catégorie difficile, dont la similarité dépasse le premier décile des positives.
    Un jeu qui ne peut pas mettre le système en défaut ne le mesure pas.
    """
    if combien <= 0 or combien >= len(negatives):
        return negatives
    par_origine: dict[str, list[dict]] = {}
    for negative in negatives:
        par_origine.setdefault(negative["origine"], []).append(negative)
    tirees: list[dict] = []
    rang = 0
    while len(tirees) < combien:
        ajoutees = False
        for origine in sorted(par_origine):
            lot = par_origine[origine]
            if rang < len(lot) and len(tirees) < combien:
                tirees.append(lot[rang])
                ajoutees = True
        if not ajoutees:
            break
        rang += 1
    return tirees


def entrelacer(travaux: list[tuple]) -> list[tuple]:
    """Alterne les origines, pour qu'un tour interrompu reste représentatif.

    Les positives d'abord et les négatives ensuite : une interruption à la vingt-troisième
    question sur cent quatre-vingt-quinze n'avait donné aucune négative, donc aucune mesure
    de refus — la moitié du sujet. L'ordre de passage n'est pas un détail quand la campagne
    peut s'arrêter en cours.
    """
    par_origine: dict[str, list[tuple]] = {}
    for travail in travaux:
        par_origine.setdefault(travail[2], []).append(travail)
    entrelaces: list[tuple] = []
    for rang in range(max((len(lot) for lot in par_origine.values()), default=0)):
        for origine in sorted(par_origine):
            if rang < len(par_origine[origine]):
                entrelaces.append(par_origine[origine][rang])
    return entrelaces


def restantes(travaux: list[tuple], acquis: list[Cas]) -> list[tuple]:
    """Les travaux dont la réponse n'est pas déjà payée.

    Rejouer une question déjà obtenue la facture une seconde fois pour un résultat que le
    fichier contient déjà.
    """
    faites = {cas.question for cas in acquis}
    return [travail for travail in travaux if travail[0] not in faites]


def serialiser(cas: list[Cas]) -> list[dict]:
    """Forme JSON des cas, extraits compris : c'est ce qui rend les contrôles rejouables."""
    return [
        {
            "question": c.question,
            "servie": c.servie,
            "origine": c.origine,
            "nature": c.reponse.nature,
            "refus": c.reponse.refus,
            "affirmations": [
                {"texte": a.texte, "extrait": a.extrait} for a in c.reponse.affirmations
            ],
            "extraits": [
                {"document": e.document, "page": e.page, "texte": e.texte} for e in c.extraits
            ],
        }
        for c in cas
    ]


def deserialiser(brut: list[dict]) -> list[Cas]:
    """Relit les cas conservés, sans leurs défauts : ils seront recalculés."""
    return [
        Cas(
            c["question"],
            c["servie"],
            c["origine"],
            Reponse(
                c["question"],
                c["refus"],
                tuple(Affirmation(a["texte"], a["extrait"]) for a in c["affirmations"]),
                c.get("nature", ""),
            ),
            tuple(Extrait(e["document"], e["page"], e["texte"]) for e in c["extraits"]),
        )
        for c in brut
    ]


def controler(cas: list[Cas]) -> list[Cas]:
    """Rejoue les vérifications sur des cas déjà obtenus. Aucun appel modèle."""
    for c in cas:
        c.defauts = tuple(verifier(c.reponse, list(c.extraits)))
    return cas


def main() -> None:
    parser = argparse.ArgumentParser(prog="sonde-reponse")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--negatives", type=Path, default=None)
    parser.add_argument("--combien", type=int, default=0, help="questions par étalon, 0 = tout")
    parser.add_argument("--moteur", choices=("torch", "onnx"), default="onnx")
    parser.add_argument("--essai", action="store_true", help="n'envoie rien, montre l'invite")
    parser.add_argument(
        "--controler", action="store_true", help="rejoue les contrôles sur ce qui est conservé"
    )
    parser.add_argument(
        "--marque",
        default="",
        help="suffixe des fichiers écrits, pour ne pas recouvrir une campagne précédente",
    )
    args = parser.parse_args()

    marque = f"_{args.marque}" if args.marque else ""
    conserve = args.racine / "fixtures" / f"reponses{marque}.json"
    rapport = args.racine / "reports" / f"reponse{marque}.md"

    if conserve.is_file() and not (args.controler or args.essai):
        raise SystemExit(
            f"{conserve} existe déjà et porte une campagne payée.\n"
            "Ce fichier est aussi la référence de la relecture : « sonde-juge » apparie ses "
            "paires sur (question, affirmation), et les régénérer les rendrait introuvables.\n"
            "Déplacer l'ancien, ou relancer avec « --marque » pour écrire à côté."
        )

    if args.controler:
        if not conserve.is_file():
            raise SystemExit(f"Rien à contrôler ({conserve}). Lancer « uv run sonde-reponse ».")
        cas = controler(deserialiser(json.loads(conserve.read_text(encoding="utf-8"))))
        defauts = sum(len(c.defauts) for c in cas)
        print(f"{len(cas)} question(s) · {defauts} défaut(s) · aucun appel modèle")
        rapporter(cas, 0, PASSAGES, MODELE, rapport)
        return

    from documentaliste.encodage import encoder
    from documentaliste.probes.retrieval import charger_questions

    chemin = args.questions or args.racine / "fixtures" / "questions_reformulees.json"
    positives = charger_questions(chemin)
    negatives_chemin = args.negatives or args.racine / "fixtures" / "questions_negatives.json"
    negatives = (
        json.loads(negatives_chemin.read_text(encoding="utf-8"))
        if negatives_chemin.exists()
        else []
    )
    if args.combien:
        positives = positives[: args.combien]
        negatives = prelever(negatives, args.combien)
    if not positives and not negatives:
        raise SystemExit("Aucune question. Lancer d'abord « uv run sonde-reformuler ».")

    travaux = [(q.question, q.documents, "positive") for q in positives]
    travaux += [(n["question"], frozenset(), n["origine"]) for n in negatives]
    travaux = entrelacer(travaux)

    acquis = (
        deserialiser(json.loads(conserve.read_text(encoding="utf-8")))
        if conserve.is_file() and not args.essai
        else []
    )
    if acquis:
        travaux = restantes(travaux, acquis)
        print(f"{len(acquis)} réponse(s) déjà obtenues, {len(travaux)} restante(s)")
    if not travaux:
        print("Rien à demander : tout est déjà conservé.")
        rapporter(controler(acquis), 0, PASSAGES, MODELE, rapport)
        return
    vecteurs = encoder([t[0] for t in travaux], args.moteur)

    cas: list[Cas] = list(acquis)
    jetons = 0
    tailles: list[int] = []
    try:
        with connexion() as cnx:
            for (question, attendus, origine), vecteur in tqdm(
                list(zip(travaux, vecteurs, strict=True)), desc="réponses", unit="question"
            ):
                resultat, cout = traiter(cnx, question, attendus, origine, vecteur, args.essai)
                if args.essai:
                    tailles.append(cout)
                    continue
                jetons += cout
                if resultat:
                    cas.append(resultat)
    except MistralIndisponible as erreur:
        # Ce qui est acquis vaut mieux que rien : les cas obtenus sont conservés et rendus.
        print(f"\nInterrompu : {erreur}")
    if args.essai:
        print("\n".join(devis(tailles, _APERCU[0] if _APERCU else "")))
        return
    if not cas:
        return

    conserve.parent.mkdir(parents=True, exist_ok=True)
    conserve.write_text(json.dumps(serialiser(cas), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRéponses conservées -> {conserve}")

    controler(cas)
    refus = sum(1 for c in cas if c.reponse.a_refuse)
    defauts = sum(len(c.defauts) for c in cas)
    print(f"{len(cas)} question(s) · {refus} refus · {defauts} défaut(s) · {jetons} jetons")
    rapporter(cas, jetons, PASSAGES, MODELE, rapport)


if __name__ == "__main__":
    main()
