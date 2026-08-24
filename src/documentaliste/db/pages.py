"""Étape 1 de l'ingestion : le texte des PDF, page par page, vers la base.

docker compose up -d
uv run documentaliste-pages
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from documentaliste.db.manifeste import Document, lire_manifeste
from documentaliste.db.pool import appliquer_schema, connexion, dsn, sans_secret
from documentaliste.probes.pages import lire_pdf
from documentaliste.probes.sweep import empreinte_de_page

#: Parallélisme par défaut, borné pour ne pas ouvrir trop de PDF simultanément.
WORKERS = 8

#: Documents traités entre deux validations.
LOT_VALIDATION = 100


def _extraire_un(travail: tuple[str, str]) -> tuple[str, list[tuple[int, str]], str]:
    """Nom du document, ses pages, et le motif d'erreur si la lecture a échoué."""
    nom, chemin = travail
    try:
        return nom, [(numero, texte) for _, numero, texte in lire_pdf(Path(chemin))], ""
    except Exception as erreur:  # noqa: BLE001
        return nom, [], f"{type(erreur).__name__}: {erreur}"


def purger_absents(cnx: object, connus: set[str]) -> list[str]:
    """Retire de la base les documents absents du manifeste, et rend leurs noms.

    L'ingestion précédente vidait les tables avant de les remplir ; celle-ci procède par
    lots pour être reprenable, et ne peut donc pas retirer ce qui a disparu du manifeste.
    Sans cette purge, un document retiré du corpus resterait citable alors que son PDF
    n'est plus là pour vérifier la citation.
    """
    lignes = cnx.execute("SELECT nom FROM document").fetchall()  # type: ignore[attr-defined]
    absents = sorted({nom for (nom,) in lignes} - connus)
    if absents:
        with cnx.cursor() as curseur:  # type: ignore[attr-defined]
            curseur.execute("DELETE FROM document WHERE nom = ANY(%s)", (absents,))
    return absents


def deja_extraits(cnx: object) -> set[str]:
    """Documents ayant déjà au moins une page en base."""
    requete = "SELECT DISTINCT document FROM page"
    return {nom for (nom,) in cnx.execute(requete).fetchall()}  # type: ignore[attr-defined]


def inserer_documents(cnx: object, documents: list[Document]) -> None:
    """Écrit les métadonnées, en rafraîchissant celles d'une ingestion précédente."""
    with cnx.cursor() as curseur:  # type: ignore[attr-defined]
        curseur.executemany(
            """
            INSERT INTO document (nom, fiche, titre, type_publication, mise_en_ligne,
                                  validation, url_pdf, url_fiche, themes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (nom) DO UPDATE SET
                fiche = EXCLUDED.fiche, titre = EXCLUDED.titre,
                type_publication = EXCLUDED.type_publication,
                mise_en_ligne = EXCLUDED.mise_en_ligne, validation = EXCLUDED.validation,
                url_pdf = EXCLUDED.url_pdf, url_fiche = EXCLUDED.url_fiche,
                themes = EXCLUDED.themes
            """,
            [
                (
                    d.nom,
                    d.fiche,
                    d.titre,
                    d.type_publication,
                    d.mise_en_ligne,
                    d.validation,
                    d.url_pdf,
                    d.url_fiche,
                    list(d.themes),
                )
                for d in documents
            ],
        )


def inserer_pages(cnx: object, nom: str, pages: list[tuple[int, str]]) -> int:
    """Remplace les pages d'un document et rend le nombre de pages écrites."""
    utiles = [(numero, texte) for numero, texte in pages if texte.strip()]
    with cnx.cursor() as curseur:  # type: ignore[attr-defined]
        curseur.execute("DELETE FROM page WHERE document = %s", (nom,))
        ordre = "COPY page (document, numero, texte, empreinte) FROM STDIN"
        with curseur.copy(ordre) as flux:
            for numero, texte in utiles:
                flux.write_row((nom, numero, texte, empreinte_de_page(texte)))
    return len(utiles)


def extraire(racine: Path, workers: int, reprendre: bool) -> None:
    """Parcourt le manifeste et remplit la table des pages."""
    documents = lire_manifeste(racine)
    appliquer_schema()

    with connexion() as cnx:
        purges = purger_absents(cnx, set(documents))
        if purges:
            print(f"{len(purges)} document(s) hors manifeste retiré(s) : {purges[:3]}")
        inserer_documents(cnx, list(documents.values()))
        cnx.commit()
        faits = deja_extraits(cnx) if reprendre else set()

    dossier = racine / "fixtures" / "pdf"
    travaux = [
        (nom, str(dossier / f"{nom}.pdf"))
        for nom in sorted(documents)
        if nom not in faits and (dossier / f"{nom}.pdf").exists()
    ]
    manquants = [n for n in documents if not (dossier / f"{n}.pdf").exists()]
    if faits:
        print(f"{len(faits)} document(s) déjà extrait(s), repris là où l'on s'était arrêté")
    if manquants:
        print(f"{len(manquants)} PDF absent(s) du dossier, ignoré(s) : {sorted(manquants)[:3]}")
    if not travaux:
        print("Rien à extraire.")
        return

    total_pages, illisibles = 0, []
    with connexion() as cnx, ProcessPoolExecutor(max_workers=workers) as pool:
        futurs = [pool.submit(_extraire_un, t) for t in travaux]
        barre = tqdm(as_completed(futurs), total=len(futurs), desc="extraction", unit="doc")
        for numero, futur in enumerate(barre, start=1):
            nom, pages, erreur = futur.result()
            if erreur:
                illisibles.append((nom, erreur))
                continue
            total_pages += inserer_pages(cnx, nom, pages)
            if numero % LOT_VALIDATION == 0:
                cnx.commit()
        cnx.commit()

    if illisibles:
        print(f"\n{len(illisibles)} PDF illisible(s) :")
        for nom, motif in illisibles[:10]:
            print(f"  {nom[:56]:58} {motif}")

    with connexion() as cnx:
        n_doc = cnx.execute("SELECT count(DISTINCT document) FROM page").fetchone()[0]
        n_pag = cnx.execute("SELECT count(*) FROM page").fetchone()[0]
    print(f"\n{n_pag} pages sur {n_doc} documents en base ({total_pages} écrites ici)")
    print(f"Base : {sans_secret(dsn())}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="documentaliste-pages")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--tout-refaire",
        action="store_true",
        help="réextrait les documents déjà en base au lieu de les sauter",
    )
    args = parser.parse_args()
    extraire(args.racine, args.workers, reprendre=not args.tout_refaire)


if __name__ == "__main__":
    main()
