"""Sonde 3 : quelle recherche ramène le bon passage — vectorielle, lexicale, ou les deux ?

uv sync --extra embeddings
uv run sonde-recherche
uv run sonde-recherche --decesurer --sans-liminaires
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from documentaliste.encodage import (
    MODELE,
    MOTEURS,
    PREFIXE_PASSAGE,
    PREFIXE_REQUETE,
    charger_modele,
)
from documentaliste.probes.lexical import Bm25, fusion_rrf
from documentaliste.probes.metrics import (
    _pourcent,
    part_lexicale,
    part_litterale,
    plafond_de_rappel,
    rang_du_bon_document,
    rang_reciproque_moyen,
    rappels,
)

#: Découpage : fenêtres se recouvrant, à l'intérieur d'une page.
_TAILLE_FENETRE = 900
_RECOUVREMENT = 200

_ESPACES = re.compile(r"[ \t ]+")

#: Césure de fin de ligne : un mot coupé par la mise en page, pas par l'auteur.
_CESURE = re.compile(r"([a-zà-öø-ÿ])-\s*\n\s*([a-zà-öø-ÿ]+)")

#: Pronoms enclitiques : leur trait d'union est restauré, jamais supprimé.
_ENCLITIQUES = frozenset(
    "il ils elle elles on je tu nous vous ce le la les moi toi lui leur y en là ci".split()
)

#: Seuils de détection d'une page de garde, **par son contenu et non par son rang**.
_GARDE_MOTS_MAX = 60
_FIN_DE_PHRASE = re.compile(r"[.!?](?:\s|$)")


def est_page_de_garde(texte: str) -> bool:
    """Page de titre : peu de mots, et pas une seule phrase achevée."""
    return len(texte.split()) < _GARDE_MOTS_MAX and not _FIN_DE_PHRASE.search(texte)


def decesurer(texte: str) -> str:
    """Recolle les mots coupés en fin de ligne."""

    def recoller(m: re.Match[str]) -> str:
        liaison = "-" if m.group(2) in _ENCLITIQUES else ""
        return f"{m.group(1)}{liaison}{m.group(2)}"

    return _CESURE.sub(recoller, texte)


@dataclass(frozen=True)
class Passage:
    """Un fragment indexé, et de quoi le citer."""

    document: str
    page: int
    texte: str


@dataclass(frozen=True)
class Question:
    """Une question de référence et les documents qui y répondent légitimement."""

    question: str
    documents: frozenset[str]
    mots_cles: tuple[str, ...] = ()
    #: Document et page d'où la question a été tirée, quand ils sont connus. L'étalon
    #: négatif n'en a pas : ses questions ne viennent d'aucun passage.
    source: tuple[str, int] | None = None


def _nettoyer(texte: str) -> str:
    return _ESPACES.sub(" ", texte).strip()


def decouper(
    document: str,
    page: int,
    texte: str,
    taille: int = _TAILLE_FENETRE,
    recouvrement: int = _RECOUVREMENT,
) -> list[Passage]:
    """Fenêtres se recouvrant à l'intérieur d'une page."""
    propre = _nettoyer(texte)
    if len(propre) < 120:
        return []
    passages: list[Passage] = []
    pas = max(taille - recouvrement, 1)
    for debut in range(0, max(len(propre) - recouvrement, 1), pas):
        fragment = propre[debut : debut + taille]
        if len(fragment) >= 120:
            passages.append(Passage(document, page, fragment))
    return passages


def indexer(
    racine: Path,
    decesure: bool = False,
    sans_liminaires: bool = False,
    workers: int | None = None,
) -> list[Passage]:
    """Découpe le corpus en passages citables, à partir des pages mises en cache."""
    from documentaliste.probes.pages import extraire

    passages: list[Passage] = []
    for page in tqdm(extraire(racine, workers), desc="découpage", unit="page"):
        # Détection par le contenu, jamais par le rang.
        if sans_liminaires and est_page_de_garde(page.texte):
            continue
        texte = decesurer(page.texte) if decesure else page.texte
        passages.extend(decouper(page.document, page.numero, texte))
    return passages


#: Taille de lot pour l'encodage sur processeur.
_LOT = 64


def empreinte_texte(texte: str, modele_nom: str, moteur: str = "torch") -> str:
    """Clé d'un passage : son texte, le modèle qui l'encode, et le moteur d'exécution."""
    return hashlib.sha256(f"{moteur}\x00{modele_nom}\x00{texte}".encode()).hexdigest()[:20]


def _fichier_cache(cache: Path, modele_nom: str, moteur: str) -> Path:
    """Un dépôt par modèle et par moteur : les vecteurs ne s'y mélangent jamais."""
    slug = re.sub(r"[^a-z0-9]+", "_", f"{moteur}_{modele_nom}".lower()).strip("_")
    return cache / f"{slug}.npz"


def _lire_cache(fichier: Path) -> dict[str, np.ndarray]:
    if not fichier.exists():
        return {}
    try:
        depot = np.load(fichier, allow_pickle=False)
        return dict(zip(depot["cles"].tolist(), depot["vecteurs"], strict=True))
    except (OSError, ValueError, KeyError) as erreur:
        # Un cache illisible se reconstruit ; il ne doit jamais arrêter une mesure.
        print(f"Cache de vecteurs ignoré ({erreur}).")
        return {}


def _ecrire_cache(fichier: Path, connus: dict[str, np.ndarray]) -> None:
    fichier.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fichier,
        cles=np.array(list(connus), dtype=object).astype(str),
        vecteurs=np.stack(list(connus.values())),
    )


def vectoriser(
    passages: list[Passage],
    modele_nom: str,
    lot: int = _LOT,
    cache: Path | None = None,
    moteur: str = "torch",
) -> np.ndarray:
    """Vecteurs normalisés des passages, calculés sur processeur."""
    if cache is None:
        modele = charger_modele(modele_nom, moteur)
        return modele.encode(
            [PREFIXE_PASSAGE + p.texte for p in passages],
            batch_size=lot,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    fichier = _fichier_cache(cache, modele_nom, moteur)
    connus = _lire_cache(fichier)
    cles = [empreinte_texte(p.texte, modele_nom, moteur) for p in passages]

    # Dédoublonné : un même texte peut apparaître deux fois dans un corpus découpé.
    manquantes = list(dict.fromkeys(c for c in cles if c not in connus))
    print(
        f"{len(passages)} passages · {len(passages) - len(manquantes)} relus du cache · "
        f"{len(manquantes)} à encoder"
    )
    if manquantes:
        par_cle = {c: p.texte for c, p in zip(cles, passages, strict=True)}
        modele = charger_modele(modele_nom, moteur)
        calcules = modele.encode(
            [PREFIXE_PASSAGE + par_cle[c] for c in manquantes],
            batch_size=lot,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        connus.update(zip(manquantes, calcules, strict=True))
        _ecrire_cache(fichier, connus)

    return np.stack([connus[c] for c in cles])


def classements(
    questions: list[Question],
    passages: list[Passage],
    vecteurs: np.ndarray,
    modele_nom: str,
    k: int,
    moteur: str = "torch",
) -> dict[str, list[list[Passage]]]:
    """Les k premiers passages de chaque bras, pour chaque question."""
    modele = charger_modele(modele_nom, moteur)
    requetes = modele.encode(
        [PREFIXE_REQUETE + q.question for q in questions], normalize_embeddings=True
    )
    # Vecteurs normalisés : le produit scalaire est la similarité cosinus.
    scores_vecteurs = requetes @ vecteurs.T
    index_lexical = Bm25([p.texte for p in passages])

    resultats: dict[str, list[list[Passage]]] = {"vectoriel": [], "lexical": [], "hybride": []}
    for question, ligne in zip(questions, scores_vecteurs, strict=True):
        rangs_v = list(np.argsort(ligne)[::-1][: k * 5])
        scores_l = np.array(index_lexical.scores(question.question))
        rangs_l = list(np.argsort(scores_l)[::-1][: k * 5])
        rangs_h = fusion_rrf([int(i) for i in rangs_v], [int(i) for i in rangs_l])
        resultats["vectoriel"].append([passages[int(i)] for i in rangs_v[:k]])
        resultats["lexical"].append([passages[int(i)] for i in rangs_l[:k]])
        resultats["hybride"].append([passages[i] for i in rangs_h[:k]])
    return resultats


def charger_questions(chemin: Path) -> list[Question]:
    """Accepte `documents` (liste) ou `document` (chaîne), pour ne pas casser l'existant."""
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    questions = []
    for d in donnees:
        attendus = d.get("documents") or [d["document"]]
        questions.append(
            Question(d["question"], frozenset(attendus), tuple(d.get("mots_cles", [])))
        )
    return questions


def rapporter(questions: list[Question], par_bras: dict[str, list[list[Passage]]], k: int) -> None:
    """Compare les bras, puis expose ce que le meilleur rate encore."""
    entetes = sorted(rappels(questions, par_bras["vectoriel"], k))
    print(
        f"\n{'bras':>12} {'rang moy.':>10} "
        + " ".join(f"{'@' + str(s):>6}" for s in entetes)
        + f" {'termes':>8}"
    )
    for nom, trouves in par_bras.items():
        r = rappels(questions, trouves, k)
        print(
            f"{nom:>12} {rang_reciproque_moyen(questions, trouves):10.3f} "
            + " ".join(f"{r[s]:6.0%}" for s in entetes)
            + f" {_pourcent(part_lexicale(questions, trouves)):>8}"
        )

    meilleur = max(par_bras, key=lambda n: rang_reciproque_moyen(questions, par_bras[n]))
    plafond = plafond_de_rappel(questions, par_bras)
    atteint = rappels(questions, par_bras[meilleur], k)[k]
    print(
        f"\nMeilleur bras : {meilleur} · plafond des bras réunis {plafond:.0%} "
        f"contre {atteint:.0%} atteint — une fusion parfaite rapporterait "
        f"{plafond - atteint:+.0%}"
    )
    echecs = [
        q
        for q, r in zip(questions, par_bras[meilleur], strict=True)
        if rang_du_bon_document(r, q.documents) is None
    ]
    if echecs:
        print(f"\n{len(echecs)} question(s) sans document attendu dans les {k} premiers :")
        for q in echecs:
            print(f"  · {q.question}  → attendu : {' ou '.join(sorted(q.documents))}")


def ecrire_rapport(
    chemin: Path,
    questions: list[Question],
    par_bras: dict[str, list[list[Passage]]],
    k: int,
    modele_nom: str,
    options: str,
    etalon: str = "?",
    recouvrement: float | None = None,
) -> None:
    """Consigne la mesure sur le disque : ce qui n'existe qu'au terminal ne se compare pas."""
    entetes = sorted(rappels(questions, par_bras["vectoriel"], k))
    lignes = [
        "# Sonde 3 — recherche vectorielle, lexicale et hybride",
        "",
        f"Étalon : `{etalon}` · {len(questions)} questions",
        f"Modèle : `{modele_nom}` · profondeur k = {k}",
        f"Options : {options}",
        "",
        "",
        f"Recouvrement littéral avec le corpus : **{_pourcent(recouvrement)}** — part des"
        " questions présentes mot pour mot dans les documents indexés. Élevé, le rappel"
        " mesure une correspondance de chaînes plutôt qu'une recherche.",
        "",
        "## Rappel par bras",
        "",
        "Le classement se fait sur le **rang réciproque moyen** : 0,50 se lit « la bonne",
        "réponse est en moyenne au deuxième rang ». Les rappels restent affichés, mais ils",
        "ne départagent plus — ils ne distinguaient pas un résultat en tête d'un résultat",
        "en fin de liste.",
        "",
        "| bras | rang réciproque | "
        + " | ".join(f"@{s}" for s in entetes)
        + " | termes attendus |",
        "| --- | ---: | " + " | ".join("---:" for _ in entetes) + " | ---: |",
    ]
    for nom, trouves in par_bras.items():
        r = rappels(questions, trouves, k)
        lignes.append(
            f"| {nom} | {rang_reciproque_moyen(questions, trouves):.3f} | "
            + " | ".join(f"{r[s]:.0%}" for s in entetes)
            + f" | {_pourcent(part_lexicale(questions, trouves))} |"
        )
    plafond = plafond_de_rappel(questions, par_bras)
    lignes += [
        "",
        f"Plafond des bras réunis : **{plafond:.0%}** — part des questions qu'au moins un",
        "bras ramène. Une fusion parfaite ne dépasserait pas ce chiffre ; ce qui manque",
        "au-delà relève de l'indexation, pas du reclassement.",
    ]

    meilleur = max(par_bras, key=lambda n: rang_reciproque_moyen(questions, par_bras[n]))
    lignes += [
        "",
        f"## Détail du bras « {meilleur} »",
        "",
        "| question | rang | 1er résultat |",
        "| --- | ---: | --- |",
    ]
    for q, resultats in zip(questions, par_bras[meilleur], strict=True):
        rang = rang_du_bon_document(resultats, q.documents)
        tete = f"{resultats[0].document} p.{resultats[0].page}" if resultats else "—"
        lignes.append(f"| {q.question[:66]} | {rang if rang else 'absent'} | {tete} |")

    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print(f"\nRapport écrit -> {chemin}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sonde-recherche")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument("--questions", type=Path, default=Path("fixtures/questions.json"))
    parser.add_argument("--modele", default=MODELE)
    parser.add_argument("--decesurer", action="store_true", help="recolle les mots coupés")
    parser.add_argument(
        "--sans-liminaires", action="store_true", help="écarte les pages de garde de l'index"
    )
    parser.add_argument("--out", type=Path, default=None, help="rapport Markdown")
    parser.add_argument("-k", type=int, default=10, help="profondeur de la recherche")
    parser.add_argument("--lot", type=int, default=_LOT, help="taille de lot pour l'encodage")
    parser.add_argument(
        "--moteur", choices=MOTEURS, default="torch", help="moteur d'inférence local"
    )
    parser.add_argument("--workers", type=int, default=None, help="processus d'extraction des PDF")
    args = parser.parse_args()

    questions = charger_questions(args.racine / args.questions)
    print(f"Étalon : {args.questions} · {len(questions)} questions")
    passages = indexer(args.racine, args.decesurer, args.sans_liminaires, args.workers)
    print(f"{len(passages)} passages indexés, {len({p.document for p in passages})} documents")

    vecteurs = vectoriser(
        passages, args.modele, args.lot, args.racine / "fixtures" / "vecteurs", args.moteur
    )
    recouvrement = part_litterale(questions, passages)
    print(f"Recouvrement littéral de l'étalon avec le corpus : {recouvrement:.0%}")
    par_bras = classements(questions, passages, vecteurs, args.modele, args.k, args.moteur)
    rapporter(questions, par_bras, args.k)

    options = ", ".join(
        [
            "décésurage " + ("activé" if args.decesurer else "désactivé"),
            "pages liminaires " + ("écartées" if args.sans_liminaires else "conservées"),
            f"moteur {args.moteur}",
        ]
    )
    # Le nom du fichier porte l'étalon, sans quoi deux rapports s'écrasent.
    suffixe = ("d" if args.decesurer else "") + ("l" if args.sans_liminaires else "") or "brut"
    suffixe = f"{args.moteur}_{suffixe}"
    sortie = args.out or args.racine / "reports" / f"recherche_{args.questions.stem}_{suffixe}.md"
    ecrire_rapport(
        sortie, questions, par_bras, args.k, args.modele, options, args.questions.name, recouvrement
    )


if __name__ == "__main__":
    main()
