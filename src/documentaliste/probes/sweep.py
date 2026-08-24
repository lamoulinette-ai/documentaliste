"""Balayage : quel réglage maximise le rappel, un facteur à la fois.

uv run sonde-balayage
uv run sonde-balayage --seulement reference,titre_en_prefixe
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from documentaliste.probes.lexical import Bm25, fusion_rrf
from documentaliste.probes.metrics import rang_reciproque_moyen, rappels
from documentaliste.probes.pages import Page
from documentaliste.probes.pages import extraire as extraire_pages
from documentaliste.probes.retrieval import (
    MODELE,
    Passage,
    Question,
    charger_modele,
    charger_questions,
    decesurer,
    decouper,
    est_page_de_garde,
    vectoriser,
)

#: Modèle plus large, même famille, utilisable sur processeur.
MODELE_BASE = "intfloat/multilingual-e5-base"

#: Longueur du début de page servant d'empreinte pour repérer le texte standard.
_EMPREINTE = 400

_ACCENTS = re.compile(r"[^a-z0-9 ]+")


def empreinte_de_page(texte: str) -> str:
    """Forme repliée du début d'une page, servant à repérer le texte standard répété."""
    plie = unicodedata.normalize("NFD", texte.lower())
    plie = "".join(c for c in plie if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", _ACCENTS.sub(" ", plie)).strip()[:_EMPREINTE]


@dataclass(frozen=True)
class Configuration:
    """Un réglage d'ingestion et de recherche, et ce qu'il change par rapport à la référence."""

    nom: str
    pourquoi: str = ""
    modele: str = MODELE
    taille: int = 900
    recouvrement: int = 200
    decesure: bool = False
    sans_gardes: bool = False
    sans_texte_standard: bool = False
    #: Nombre d'occurrences à partir duquel une page est jugée standard.
    seuil_standard: int = 5
    titre_en_prefixe: bool = False


def _titre(document: str) -> str:
    """Titre lisible déduit du nom de fichier, faute de métadonnée à ce stade."""
    return document.replace("_", " ")


def indexer(
    racine: Path,
    config: Configuration,
    pages: list[Page] | None = None,
    titres: dict[str, str] | None = None,
) -> list[Passage]:
    """Découpe le corpus selon une configuration donnée."""
    if pages is None:
        pages = extraire_pages(racine)

    brutes = [
        (page.document, page.numero, page.texte)
        for page in pages
        if not (config.sans_gardes and est_page_de_garde(page.texte))
    ]

    if config.sans_texte_standard:
        vus = Counter(empreinte_de_page(t) for _, _, t in brutes)
        seuil = config.seuil_standard
        brutes = [(d, p, t) for d, p, t in brutes if vus[empreinte_de_page(t)] < seuil]

    passages: list[Passage] = []
    for document, numero, texte in brutes:
        if config.decesure:
            texte = decesurer(texte)
        morceaux = decouper(document, numero, texte, config.taille, config.recouvrement)
        if config.titre_en_prefixe:
            # Le titre replace le fragment dans son document.
            intitule = (titres or {}).get(document) or _titre(document)
            morceaux = [Passage(m.document, m.page, f"{intitule}. {m.texte}") for m in morceaux]
        passages.extend(morceaux)
    return passages


#: Poids du bras vectoriel : 1,0 ignore le lexical, 0,0 n'écoute que lui.
POIDS: tuple[float, ...] = (0.0, 0.3, 1.0)


@dataclass
class Resultat:
    """Ce qu'une configuration a donné."""

    config: Configuration
    n_passages: int
    par_variante: dict[str, dict[int, float]] = field(default_factory=dict)
    #: Rang réciproque moyen par variante — le critère de classement.
    rang: dict[str, float] = field(default_factory=dict)

    def meilleur(self) -> tuple[str, float]:
        """Variante la mieux classée, jugée sur le rang réciproque moyen."""
        nom = max(self.rang, key=lambda v: self.rang[v])
        return nom, self.rang[nom]


def evaluer(
    racine: Path,
    config: Configuration,
    questions: list[Question],
    k: int,
    pages: list[Page] | None = None,
) -> Resultat:
    """Indexe, vectorise une fois, puis évalue tous les poids de fusion."""
    passages = indexer(racine, config, pages)
    vecteurs = vectoriser(passages, config.modele, cache=racine / "fixtures" / "vecteurs")
    modele = charger_modele(config.modele)
    requetes = modele.encode(["query: " + q.question for q in questions], normalize_embeddings=True)
    scores_v = requetes @ vecteurs.T
    lexical = Bm25([p.texte for p in passages])

    profondeur = k * 5
    rangs_v, rangs_l = [], []
    for question, ligne in zip(questions, scores_v, strict=True):
        rangs_v.append([int(i) for i in np.argsort(ligne)[::-1][:profondeur]])
        scores_l = np.array(lexical.scores(question.question))
        rangs_l.append([int(i) for i in np.argsort(scores_l)[::-1][:profondeur]])

    resultat = Resultat(config, len(passages))
    for poids in POIDS:
        trouves = []
        for v, x in zip(rangs_v, rangs_l, strict=True):
            if poids >= 1.0:
                ordre = v
            elif poids <= 0.0:
                ordre = x
            else:
                ordre = fusion_rrf(v, x, poids=(poids, 1.0 - poids))
            trouves.append([passages[i] for i in ordre[:k]])
        nom = {0.0: "lexical", 1.0: "vectoriel"}.get(poids, f"hybride {poids:.1f}")
        resultat.par_variante[nom] = rappels(questions, trouves, k)
        resultat.rang[nom] = rang_reciproque_moyen(questions, trouves)
    return resultat


def variantes(reference: Configuration) -> list[Configuration]:
    """La référence, un seul facteur modifié à chaque fois, puis leur cumul."""
    return [
        reference,
        replace(reference, nom="decesurage", pourquoi="recolle les mots coupés", decesure=True),
        replace(
            reference,
            nom="sans_pages_de_garde",
            pourquoi="écarte les pages de titre, détectées par leur contenu",
            sans_gardes=True,
        ),
        replace(
            reference,
            nom="sans_texte_standard",
            pourquoi="écarte les pages vues 5 fois ou plus",
            sans_texte_standard=True,
        ),
        replace(
            reference,
            nom="texte_standard_seuil_2",
            pourquoi="écarte toute page vue deux fois — vide 235 documents",
            sans_texte_standard=True,
            seuil_standard=2,
        ),
        replace(
            reference,
            nom="texte_standard_seuil_3",
            pourquoi="écarte toute page vue trois fois — vide 43 documents",
            sans_texte_standard=True,
            seuil_standard=3,
        ),
        replace(
            reference,
            nom="titre_en_prefixe",
            pourquoi="replace chaque passage dans son document",
            titre_en_prefixe=True,
        ),
        replace(
            reference, nom="fenetres_courtes", pourquoi="500/100", taille=500, recouvrement=100
        ),
        replace(
            reference, nom="fenetres_longues", pourquoi="1400/300", taille=1400, recouvrement=300
        ),
        replace(
            reference,
            nom="cumul",
            pourquoi="tous les traitements retenus, ensemble",
            decesure=True,
            sans_gardes=True,
            sans_texte_standard=True,
            titre_en_prefixe=True,
        ),
    ]


#: Configurations écartées du lot courant, rappelables par `--seulement`.
def variantes_lourdes(reference: Configuration) -> list[Configuration]:
    return [
        replace(
            reference,
            nom="modele_base",
            pourquoi="e5-base au lieu de e5-small",
            modele=MODELE_BASE,
        ),
    ]


def rapporter(resultats: list[Resultat], k: int, chemin: Path, etalon: str = "?") -> None:
    reference = resultats[0]
    _, base = reference.meilleur()
    seuils = sorted(next(iter(reference.par_variante.values())))

    lignes = [
        "# Balayage — un facteur à la fois",
        "",
        f"Étalon : `{etalon}` · configurations mesurées : "
        + ", ".join(r.config.nom for r in resultats),
        "",
        f"Référence : découpage {reference.config.taille}/{reference.config.recouvrement}, "
        f"`{reference.config.modele}`, aucune option.",
        f"Profondeur k = {k} · {len(seuils)} seuils · poids de fusion balayés : "
        + ", ".join(f"{p:.1f}" for p in POIDS),
        "",
        "Classement sur le **rang réciproque moyen** : 0,50 se lit « la bonne réponse est",
        "en moyenne au deuxième rang ». L'écart est donné par rapport à la référence.",
        "",
        "| configuration | ce qu'elle change | passages | meilleure variante | rang | "
        + " | ".join(f"@{s}" for s in seuils)
        + " | écart |",
        "| --- | --- | ---: | --- | ---: | " + " | ".join("---:" for _ in seuils) + " | ---: |",
    ]
    for r in resultats:
        nom, rang = r.meilleur()
        valeurs = r.par_variante[nom]
        lignes.append(
            f"| {r.config.nom} | {r.config.pourquoi or '—'} | {r.n_passages} | {nom} | "
            f"{rang:.3f} | "
            + " | ".join(f"{valeurs[s]:.0%}" for s in seuils)
            + f" | {rang - base:+.3f} |"
        )

    lignes += ["", "## Détail par poids de fusion", ""]
    for r in resultats:
        entete = "| variante | rang | " + " | ".join(f"@{s}" for s in seuils) + " |"
        separateur = "| --- | ---: | " + " | ".join("---:" for _ in seuils) + " |"
        lignes += [f"### {r.config.nom}", "", entete, separateur]
        for nom, valeurs in r.par_variante.items():
            lignes.append(
                f"| {nom} | {r.rang[nom]:.3f} | "
                + " | ".join(f"{valeurs[s]:.0%}" for s in seuils)
                + " |"
            )
        lignes.append("")

    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"\nRapport écrit -> {chemin}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sonde-balayage")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument("--questions", type=Path, default=Path("fixtures/questions.json"))
    parser.add_argument(
        "--seulement",
        help="noms de configurations séparés par des virgules ; seul moyen d'appeler "
        "les variantes lourdes, « modele_base » notamment",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None, help="processus d'extraction des PDF")
    args = parser.parse_args()

    questions = charger_questions(args.racine / args.questions)
    reference = Configuration("reference", "aucune")
    configs = variantes(reference)
    if args.seulement:
        voulus = {n.strip() for n in args.seulement.split(",")}
        connues = configs + variantes_lourdes(reference)
        configs = [c for c in connues if c.nom in voulus]
        if not configs:
            noms = ", ".join(c.nom for c in connues)
            raise SystemExit(f"Configuration inconnue. Disponibles : {noms}.")

    pages = extraire_pages(args.racine, args.workers)
    resultats = []
    for index, config in enumerate(configs, 1):
        print(f"\n[{index}/{len(configs)}] {config.nom} — {config.pourquoi or 'référence'}")
        resultat = evaluer(args.racine, config, questions, args.k, pages)
        nom, rang = resultat.meilleur()
        valeurs = resultat.par_variante[nom]
        print(
            f"    {resultat.n_passages} passages · « {nom} » · rang {rang:.3f} · "
            + " ".join(f"@{s} {v:.0%}" for s, v in valeurs.items())
        )
        resultats.append(resultat)

    # Le nom du rapport porte l'étalon et la sélection, sans quoi deux rapports s'écrasent.
    portee = "_".join(sorted(c.nom for c in configs)) if args.seulement else "complet"
    defaut = args.racine / "reports" / f"balayage_{args.questions.stem}_{portee}.md"
    rapporter(resultats, args.k, args.out or defaut, args.questions.name)


if __name__ == "__main__":
    main()
