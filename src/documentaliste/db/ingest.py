"""Étape 2 de l'ingestion : des pages en base aux passages vectorisés.

docker compose up -d
uv run documentaliste-pages
uv run documentaliste-ingerer
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from documentaliste.db.manifeste import Document, lire_manifeste, titres
from documentaliste.db.pool import appliquer_schema, connexion, dsn, sans_secret
from documentaliste.probes.pages import Page
from documentaliste.probes.retrieval import Passage, vectoriser
from documentaliste.probes.sweep import Configuration, indexer, variantes

#: Nombre de documents traités entre deux validations.
LOT_DOCUMENTS = 50


def configuration_retenue() -> Configuration:
    """La configuration `cumul`, seule à avoir été mesurée de bout en bout."""
    return next(c for c in variantes(Configuration("reference", "aucune")) if c.nom == "cumul")


def numeroter(passages: list[Passage]) -> list[tuple[Passage, int]]:
    """Associe à chaque passage son rang dans sa page."""
    compteurs: dict[tuple[str, int], int] = defaultdict(int)
    numerotes = []
    for passage in passages:
        cle = (passage.document, passage.page)
        numerotes.append((passage, compteurs[cle]))
        compteurs[cle] += 1
    return numerotes


def empreintes_repetees(cnx: object, seuil: int) -> set[str]:
    """Empreintes vues au moins `seuil` fois dans le corpus.

    Le seuil vaut 5. À 2 — la valeur d'origine — les deux exemplaires d'une fiche présente
    dans deux dossiers disparaissaient ensemble : 235 documents devenaient inatteignables
    par toute recherche, pour 11 624 pages écartées contre 1 398 au seuil 5.
    """
    lignes = cnx.execute(  # type: ignore[attr-defined]
        "SELECT empreinte FROM page GROUP BY empreinte HAVING count(*) >= %s", (seuil,)
    ).fetchall()
    return {empreinte for (empreinte,) in lignes}


_DOCUMENTS_AVEC_PAGES = "SELECT DISTINCT document FROM page"
_DOCUMENTS_AVEC_PASSAGES = "SELECT DISTINCT document FROM passage"


def documents_a_traiter(cnx: object, connus: set[str], reprendre: bool) -> list[str]:
    """Documents ayant des pages mais pas encore de passages."""
    lignes = cnx.execute(_DOCUMENTS_AVEC_PAGES).fetchall()  # type: ignore[attr-defined]
    avec_pages = {nom for (nom,) in lignes} & connus
    if not reprendre:
        return sorted(avec_pages)
    faits = cnx.execute(_DOCUMENTS_AVEC_PASSAGES).fetchall()  # type: ignore[attr-defined]
    return sorted(avec_pages - {nom for (nom,) in faits})


def lire_pages(cnx: object, noms: list[str], repetees: set[str]) -> list[Page]:
    """Pages de ces documents, privées du texte standard répété."""
    lignes = cnx.execute(  # type: ignore[attr-defined]
        "SELECT document, numero, texte FROM page"
        " WHERE document = ANY(%s) AND NOT (empreinte = ANY(%s))"
        " ORDER BY document, numero",
        (noms, list(repetees)),
    ).fetchall()
    return [Page(document=d, numero=n, texte=t) for d, n, t in lignes]


def inserer_passages(  # noqa: ANN001
    cnx: object, numerotes: list[tuple[Passage, int]], vecteurs
) -> None:
    """Écrit les passages et leurs vecteurs par flux `COPY`."""
    ordre = "COPY passage (document, page, rang, texte, vecteur) FROM STDIN"
    with cnx.cursor() as curseur, curseur.copy(ordre) as flux:  # type: ignore[attr-defined]
        for (passage, rang), vecteur in zip(numerotes, vecteurs, strict=True):
            flux.write_row(
                (
                    passage.document,
                    passage.page,
                    rang,
                    passage.texte,
                    "[" + ",".join(f"{v:.6f}" for v in vecteur) + "]",
                )
            )


def traiter_lot(
    cnx: object,
    racine: Path,
    config: Configuration,
    noms: list[str],
    repetees: set[str],
    intitules: dict[str, str],
    moteur: str,
) -> int:
    """Découpe, vectorise et écrit un lot de documents. Rend le nombre de passages."""
    pages = lire_pages(cnx, noms, repetees)
    if not pages:
        return 0
    passages = indexer(racine, config, pages=pages, titres=intitules)
    if not passages:
        return 0
    vecteurs = vectoriser(passages, config.modele, cache=None, moteur=moteur)
    with cnx.cursor() as curseur:  # type: ignore[attr-defined]
        curseur.execute("DELETE FROM passage WHERE document = ANY(%s)", (noms,))
    inserer_passages(cnx, numeroter(passages), vecteurs)
    return len(passages)


def ingerer(racine: Path, moteur: str, reprendre: bool, lot: int) -> None:
    """Découpe et vectorise ce qui reste à faire, puis reconstruit l'index."""
    metadonnees: dict[str, Document] = lire_manifeste(racine)
    intitules = titres(metadonnees)
    # Retrait du texte standard déjà appliqué en amont, sur le corpus entier.
    config = replace(configuration_retenue(), sans_texte_standard=False)
    appliquer_schema()

    with connexion() as cnx:
        repetees = empreintes_repetees(cnx, config.seuil_standard)
        restants = documents_a_traiter(cnx, set(metadonnees), reprendre)
        total_pages = cnx.execute("SELECT count(*) FROM page").fetchone()[0]

    if not total_pages:
        raise SystemExit("Aucune page en base. Lancer d'abord « uv run documentaliste-pages ».")
    print(
        f"{total_pages} pages en base · {len(repetees)} empreinte(s) de texte standard "
        f"écartée(s) · {len(restants)} document(s) à traiter · moteur {moteur}"
    )
    if not restants:
        print("Rien à ingérer.")
        return

    # L'index vectoriel est reconstruit après le chargement.
    with connexion(autocommit=True) as cnx:
        cnx.execute("DROP INDEX IF EXISTS passage_vecteur_idx")

    ecrits = 0
    lots = [restants[i : i + lot] for i in range(0, len(restants), lot)]
    with connexion() as cnx:
        barre = tqdm(lots, desc="ingestion", unit="lot")
        for noms in barre:
            ecrits += traiter_lot(cnx, racine, config, noms, repetees, intitules, moteur)
            cnx.commit()
            barre.set_postfix(passages=ecrits)

    print("\nConstruction de l'index vectoriel…")
    with connexion(autocommit=True) as cnx:
        cnx.execute(
            "CREATE INDEX passage_vecteur_idx ON passage USING hnsw (vecteur vector_ip_ops)"
        )
        cnx.execute("ANALYZE passage")

    with connexion() as cnx:
        n_doc = cnx.execute("SELECT count(DISTINCT document) FROM passage").fetchone()[0]
        n_pas = cnx.execute("SELECT count(*) FROM passage").fetchone()[0]
    print(f"\n{n_pas} passages sur {n_doc} documents — {sans_secret(dsn())}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="documentaliste-ingerer")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument("--moteur", choices=("torch", "onnx"), default="onnx")
    parser.add_argument("--lot", type=int, default=LOT_DOCUMENTS)
    parser.add_argument(
        "--tout-refaire",
        action="store_true",
        help="réindexe les documents déjà pourvus de passages au lieu de les sauter",
    )
    args = parser.parse_args()
    ingerer(args.racine, args.moteur, reprendre=not args.tout_refaire, lot=args.lot)


if __name__ == "__main__":
    main()
