"""État du corpus en base : volumes, couverture du manifeste, anomalies.

uv run documentaliste-etat
"""

from __future__ import annotations

import argparse
from pathlib import Path

from documentaliste.db.ingest import configuration_retenue
from documentaliste.db.manifeste import lire_manifeste
from documentaliste.db.pool import connexion, dsn, sans_secret

#: Requêtes de comptage, dans l'ordre d'affichage.
VOLUMES: tuple[tuple[str, str], ...] = (
    ("documents au manifeste", "SELECT count(*) FROM document"),
    ("documents avec pages", "SELECT count(DISTINCT document) FROM page"),
    ("documents avec passages", "SELECT count(DISTINCT document) FROM passage"),
    ("pages", "SELECT count(*) FROM page"),
    ("passages", "SELECT count(*) FROM passage"),
)

#: Anomalies : intitulé, requête, et ce que le nombre attendu vaut.
ANOMALIES: tuple[tuple[str, str], ...] = (
    (
        "passages sans document",
        "SELECT count(*) FROM passage p"
        " LEFT JOIN document d ON d.nom = p.document WHERE d.nom IS NULL",
    ),
    (
        "pages sans document",
        "SELECT count(*) FROM page p"
        " LEFT JOIN document d ON d.nom = p.document WHERE d.nom IS NULL",
    ),
    ("passages vides", "SELECT count(*) FROM passage WHERE btrim(texte) = ''"),
    (
        "documents avec passages mais sans pages",
        "SELECT count(*) FROM (SELECT DISTINCT document FROM passage) s"
        " WHERE NOT EXISTS (SELECT 1 FROM page WHERE page.document = s.document)",
    ),
)

#: Pourquoi un document reste sans passage : soit son PDF n'a rendu aucune page, soit
#: toutes ses pages ont été écartées par les filtres d'indexation.
SANS_PASSAGE = """
SELECT CASE WHEN p.document IS NULL THEN 'aucune page extraite'
            ELSE 'toutes ses pages écartées' END, count(*)
FROM document d
LEFT JOIN (SELECT DISTINCT document FROM page) p ON p.document = d.nom
WHERE NOT EXISTS (SELECT 1 FROM passage WHERE passage.document = d.nom)
GROUP BY 1
"""

#: Pages écartées **au seuil en vigueur**. Le compte doit suivre la configuration : une
#: version antérieure comptait toujours au seuil 2 et annonçait 11 624 pages écartées là
#: où l'ingestion en avait retiré 1 398.
PAGES_ECARTEES = """
SELECT count(*) FROM page WHERE empreinte IN (
    SELECT empreinte FROM page GROUP BY empreinte HAVING count(*) >= %s
)
"""

#: Effet du seuil de répétition : combien de pages seraient écartées, et combien de
#: documents y perdraient la totalité de leur contenu.
EFFET_DU_SEUIL = """
WITH repetees AS (
    SELECT empreinte FROM page GROUP BY empreinte HAVING count(*) >= %s
),
restant AS (
    SELECT document, count(*) AS gardees
    FROM page WHERE empreinte NOT IN (SELECT empreinte FROM repetees)
    GROUP BY document
)
SELECT
    (SELECT count(*) FROM page WHERE empreinte IN (SELECT empreinte FROM repetees)),
    (SELECT count(DISTINCT document) FROM page) - (SELECT count(*) FROM restant)
"""

TAILLES = """
SELECT relname, pg_size_pretty(pg_total_relation_size(c.oid))
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public' AND relkind = 'r'
ORDER BY pg_total_relation_size(c.oid) DESC
"""


#: Seuils comparés : à partir de combien d'occurrences une page est jugée standard.
SEUILS: tuple[int, ...] = (2, 3, 5, 10)


def effet_du_seuil(racine: Path) -> None:
    """Compare les seuils de répétition, sans rien modifier."""
    en_vigueur = configuration_retenue().seuil_standard
    with connexion() as cnx:
        documents = cnx.execute("SELECT count(DISTINCT document) FROM page").fetchone()[0]
        pages = cnx.execute("SELECT count(*) FROM page").fetchone()[0]
        mesures = [(s, *cnx.execute(EFFET_DU_SEUIL, (s,)).fetchone()) for s in SEUILS]

    print(f"\nEffet du seuil de répétition, sur {pages} pages et {documents} documents :\n")
    print(f"  {'seuil':>5}  {'pages écartées':>16}  {'documents vidés':>17}")
    for seuil, ecartees, vides in mesures:
        marque = " <- en vigueur" if seuil == en_vigueur else ""
        print(f"  {seuil:>5}  {ecartees:>9} ({ecartees / pages:5.1%})  {vides:>17}{marque}")
    print(
        "\nUn document vidé n'est plus atteignable par aucune recherche. Choisir un seuil\n"
        "est un arbitrage à mesurer sur l'étalon, pas à trancher sur ce tableau seul."
    )


def rapporter(racine: Path) -> int:
    """Affiche l'état et rend le nombre d'anomalies trouvées."""
    attendus = len(lire_manifeste(racine))
    seuil = configuration_retenue().seuil_standard
    print(f"Base : {sans_secret(dsn())}\n")

    with connexion() as cnx:
        mesures = {nom: cnx.execute(requete).fetchone()[0] for nom, requete in VOLUMES}
        anomalies = {nom: cnx.execute(requete).fetchone()[0] for nom, requete in ANOMALIES}
        tailles = cnx.execute(TAILLES).fetchall()
        index = cnx.execute(
            "SELECT count(*) FROM pg_indexes WHERE indexname = 'passage_vecteur_idx'"
        ).fetchone()[0]
        pages_par_document = cnx.execute(
            "SELECT round(avg(n), 1) FROM (SELECT count(*) n FROM page GROUP BY document) s"
        ).fetchone()[0]
        motifs = cnx.execute(SANS_PASSAGE).fetchall()
        ecartees = cnx.execute(PAGES_ECARTEES, (seuil,)).fetchone()[0]

    for nom, valeur in mesures.items():
        part = f"  ({valeur / attendus:.1%} du manifeste)" if "documents" in nom else ""
        print(f"  {valeur:>9,}".replace(",", " ") + f"  {nom}{part}")
    print(f"\n  {pages_par_document} pages par document en moyenne")
    print(f"  index vectoriel : {'présent' if index else 'ABSENT'}")

    pages = mesures["pages"] or 1
    print(
        f"\nTexte standard écarté au seuil {seuil} : {ecartees} pages "
        f"sur {pages} ({ecartees / pages:.1%})"
    )
    if motifs:
        print("\nDocuments sans passage :")
        for motif, nombre in sorted(motifs, key=lambda m: -m[1]):
            print(f"  {nombre:>6}  {motif}")

    print("\nTailles :")
    for table, taille in tailles:
        print(f"  {table:12} {taille}")

    #: Un PDF absent du disque rend ses citations invérifiables. Le manifeste étant écrit
    #: avant l'extraction, il peut décrire des documents qui n'ont jamais été écrits.
    dossier = racine / "fixtures" / "pdf"
    sur_disque = {chemin.stem for chemin in dossier.glob("*.pdf")}
    absents = sorted(set(lire_manifeste(racine)) - sur_disque)
    if absents:
        print(f"\n!! {len(absents)} PDF du manifeste absent(s) de {dossier} :")
        for nom in absents[:5]:
            print(f"     {nom[:76]}")
        print("   Relancer « uv run corpus-extraire ».")

    total = sum(anomalies.values()) + (0 if index else 1) + len(absents)
    print("\nAnomalies :")
    for nom, valeur in anomalies.items():
        marque = "  " if valeur == 0 else "!!"
        print(f"{marque} {valeur:>6}  {nom}")
    if not index:
        print("!!          index vectoriel absent — reconstruction interrompue ?")
    manquants = attendus - mesures["documents avec passages"]
    if manquants:
        print(f"   {manquants:>6}  document(s) du manifeste sans passage")
    return total


#: Connexions ouvertes pour mesurer ce que coûte l'établissement d'une connexion. Assez
#: pour distinguer un coût fixe d'un accident, assez peu pour rester instantané.
_CONNEXIONS = 10


def chronometrer_la_base() -> None:
    """Sépare ce que coûte une connexion de ce que coûte un transfert.

    Deux causes très différentes se confondent sous « la base est lente ». Un coût par
    connexion se corrige dans les tests, en partageant une connexion ; un coût au débit
    tient à la couche réseau — sous Windows, le proxy de Docker Desktop — et se corrige en
    changeant d'environnement ou en transférant moins.
    """
    import time

    debut = time.perf_counter()
    for _ in range(_CONNEXIONS):
        with connexion() as cnx:
            cnx.execute("SELECT 1").fetchone()
    par_connexion = (time.perf_counter() - debut) / _CONNEXIONS * 1000

    with connexion() as cnx:
        debut = time.perf_counter()
        cnx.execute("SELECT count(*) FROM passage").fetchone()
        comptage = (time.perf_counter() - debut) * 1000

        debut = time.perf_counter()
        lignes = cnx.execute("SELECT terme, passages FROM statistique_terme").fetchall()
        transfert = (time.perf_counter() - debut) * 1000

    print("\nCoût d'accès à la base")
    print(f"  connexion         : {par_connexion:8.1f} ms  (moyenne sur {_CONNEXIONS})")
    print(f"  comptage passages : {comptage:8.1f} ms  (parcours complet, côté serveur)")
    print(f"  lecture lexique   : {transfert:8.1f} ms  ({len(lignes)} lignes transférées)")
    if par_connexion > 100:
        print("\n  La connexion coûte cher. Dans les tests, la partager plutôt que d'en")
        print("  ouvrir une par cas ; c'est indépendant de l'environnement.")
    if transfert > 5 * max(comptage, 1):
        print("\n  Le transfert coûte bien plus que le calcul : le goulot est la couche")
        print("  réseau, non PostgreSQL. Sous Windows, le proxy de Docker Desktop en est")
        print("  la cause habituelle, et un client lancé depuis WSL le contourne.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="documentaliste-etat")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument(
        "--texte-standard",
        action="store_true",
        help="compare les seuils de répétition, sans rien modifier",
    )
    parser.add_argument(
        "--chronometre",
        action="store_true",
        help="sépare le coût d'une connexion de celui d'un transfert",
    )
    args = parser.parse_args()
    anomalies = rapporter(args.racine)
    if args.texte_standard:
        effet_du_seuil(args.racine)
    if args.chronometre:
        chronometrer_la_base()
    raise SystemExit(1 if anomalies else 0)


if __name__ == "__main__":
    main()
