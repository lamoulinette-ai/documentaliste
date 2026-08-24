"""Construction du corpus à partir des archives open data de la HAS.

    uv run corpus-extraire --archive TextesPublicationsHAS.zip \
                           --metadonnees has-publications-split.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

#: Jeux de données d'origine, pour que personne n'ait à les rechercher.
JEU_TEXTES = "https://www.data.gouv.fr/datasets/textes-des-publications-de-la-has-7"
JEU_METADONNEES = "https://www.data.gouv.fr/datasets/metadonnees-des-publications-de-la-has-1"

#: Types de publication retenus. Les avis sur les médicaments et les produits de santé
#: en sont exclus : ce sont des décisions de remboursement, pas des recommandations.
PERIMETRE: tuple[str, ...] = (
    "RecommandationsProfessionnelles",
    "RecommandationVaccinale",
    "GuideMedecinALD",
    "GuideMethodologique",
    "EvaluationDesPratiques",
    "EvaluationDesTechnologiesDeSante",
    "EtudeEtEnquete",
    "GuidePatient",
)

#: Marques de retrait inscrites par la HAS dans l'intitulé de ses publications.
RETRAIT = re.compile(
    r"(?i)\b(?:recommandation|fiche|guide|texte)?\s*"
    r"(?:retir[ée]e?|abrog[ée]e?|suspendue?|annul[ée]e?|caduque?"
    r"|n['’ ]est plus (?:en vigueur|applicable)|remplac[ée]e? par)\b"
)

#: Date en tête du nom de fichier : « 2020-09-16_avis_ct18742.pdf ».
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_")


@dataclass(frozen=True)
class Ressource:
    """Un PDF retenu, avec de quoi le citer."""

    nom: str
    chemin_archive: str
    publication: str
    thematique: str
    titre: str = ""
    date_publication: str = ""
    date_validation: str = ""
    url_fiche: str = ""
    themes: tuple[str, ...] = ()
    octets: int = 0


@dataclass
class Bilan:
    """Ce que la sélection a retenu, et ce qu'elle a écarté — avec le motif."""

    retenus: list[Ressource] = field(default_factory=list)
    doublons: int = 0
    retires: list[str] = field(default_factory=list)
    sans_metadonnees: list[str] = field(default_factory=list)


def lire_metadonnees(archive: Path) -> dict[str, dict]:
    """Publications du périmètre, indexées par identifiant HAS."""
    if not archive.exists():
        raise SystemExit(
            f"Métadonnées introuvables : {archive}\n"
            f"Télécharger « has-publications-split.zip » depuis {JEU_METADONNEES}."
        )
    publications: dict[str, dict] = {}
    with zipfile.ZipFile(archive) as zip_:
        for type_ in PERIMETRE:
            nom = f"json/{type_}.json"
            if nom not in zip_.namelist():
                continue
            for publication in json.loads(zip_.read(nom)):
                publications[publication["id"]] = publication
    return publications


def est_retiree(publication: dict) -> bool:
    """La HAS a-t-elle inscrit un retrait dans l'intitulé de cette publication ?"""
    texte = f"{publication.get('title') or ''} {publication.get('soustitre') or ''}"
    return bool(RETRAIT.search(texte))


def date_du_nom(nom: str) -> str:
    """Date de publication portée par le nom de fichier, au format ISO."""
    marque = _DATE.match(nom)
    return "-".join(marque.groups()) if marque else ""


#: Longueur maximale d'un nom de document. Le nom sert de nom de fichier ; Windows borne
#: le chemin complet à 260 caractères, et un dossier de projet plus profond que le nôtre
#: doit rester possible. Un nom écourté reste lisible, et le manifeste porte le titre.
NOM_MAX = 120


def raccourcir(souche: str, chemin_archive: str) -> str:
    """Écourte un nom trop long en lui laissant de quoi rester unique et stable.

    Le suffixe dérive du chemin dans l'archive : deux noms écourtés au même préfixe
    restent distincts, et le même document garde son nom d'une extraction à l'autre —
    ce qu'un compteur d'ordre ne garantirait pas.
    """
    if len(souche) <= NOM_MAX:
        return souche
    marque = hashlib.sha256(chemin_archive.encode()).hexdigest()[:8]
    return f"{souche[: NOM_MAX - 9].rstrip('_-')}_{marque}"


def nom_unique(souche: str, chemin_archive: str, pris: set[str]) -> str:
    """Nom de document garanti unique, en désambiguïsant par le dossier d'origine."""
    souche = raccourcir(souche, chemin_archive)
    if souche not in pris:
        pris.add(souche)
        return souche
    parties = chemin_archive.split("/")
    candidat = f"{souche}__{parties[-2]}" if len(parties) >= 2 else souche
    rang = 2
    while candidat in pris:
        candidat = f"{souche}__{parties[-2]}_{rang}"
        rang += 1
    pris.add(candidat)
    return candidat


def selectionner(archive: Path, publications: dict[str, dict]) -> Bilan:
    """Choisit les PDF à indexer, en consignant chaque exclusion et son motif."""
    if not archive.exists():
        raise SystemExit(
            f"Archive introuvable : {archive}\n"
            f"Télécharger « TextesPublicationsHAS.zip » depuis {JEU_TEXTES}."
        )
    bilan = Bilan()
    vus: set[tuple[int, int]] = set()
    noms: set[str] = set()
    with zipfile.ZipFile(archive) as zip_:
        entrees = [
            i
            for i in zip_.infolist()
            if i.filename.lower().endswith(".pdf") and i.filename.split("/")[0] in PERIMETRE
        ]
        for entree in sorted(entrees, key=lambda i: i.filename):
            thematique, publication_id, _, fichier = entree.filename.split("/")
            publication = publications.get(publication_id)
            if publication is None:
                bilan.sans_metadonnees.append(entree.filename)
                continue
            if est_retiree(publication):
                bilan.retires.append(f"{entree.filename} — {publication.get('title', '')[:70]}")
                continue
            # Déduplication par somme de contrôle : le nom de fichier n'est pas un critère sûr.
            empreinte = (entree.CRC, entree.file_size)
            if empreinte in vus:
                bilan.doublons += 1
                continue
            vus.add(empreinte)
            bilan.retenus.append(
                Ressource(
                    nom=nom_unique(Path(fichier).stem, entree.filename, noms),
                    chemin_archive=entree.filename,
                    publication=publication_id,
                    thematique=thematique,
                    titre=publication.get("title") or "",
                    date_publication=(publication.get("publicationDate") or "")[:10]
                    or date_du_nom(fichier),
                    date_validation=(publication.get("dateDeValidation") or "")[:10],
                    url_fiche=publication.get("pageURL") or "",
                    themes=tuple(publication.get("categoriesThematiques") or ()),
                    octets=entree.file_size,
                )
            )
    return bilan


def extraire(archive: Path, bilan: Bilan, destination: Path) -> list[tuple[str, str]]:
    """Écrit les PDF retenus, en sautant ceux qui sont déjà présents et intacts.

    Un fichier qu'on ne peut pas écrire est consigné et l'extraction continue. La version
    précédente laissait remonter l'erreur : un seul chemin trop long pour Windows a
    interrompu l'extraction au rang 1994 sur 6 504, et comme le manifeste est écrit avant,
    il décrivait ensuite un corpus qui n'existait qu'à moitié.
    """
    destination.mkdir(parents=True, exist_ok=True)
    echecs: list[tuple[str, str]] = []
    with zipfile.ZipFile(archive) as zip_:
        for ressource in tqdm(bilan.retenus, desc="extraction", unit="pdf"):
            cible = destination / f"{ressource.nom}.pdf"
            if cible.exists() and cible.stat().st_size == ressource.octets:
                continue
            try:
                cible.write_bytes(zip_.read(ressource.chemin_archive))
            except (OSError, KeyError) as erreur:
                echecs.append((ressource.nom, f"{type(erreur).__name__}: {erreur}"))
    return echecs


def date_de_l_archive(archive: Path) -> str:
    """Date du fichier le plus récent de l'archive, au format ISO.

    Le corpus s'arrête à cette date. C'est une information de produit : une réponse ne
    doit pas laisser croire qu'elle couvre des publications postérieures.
    """
    if not archive.exists():
        return ""
    with zipfile.ZipFile(archive) as zip_:
        dates = [i.date_time for i in zip_.infolist() if not i.is_dir()]
    if not dates:
        return ""
    annee, mois, jour = max(dates)[:3]
    return f"{annee:04d}-{mois:02d}-{jour:02d}"


def ecrire_manifeste(bilan: Bilan, chemin: Path, archive: Path | None = None) -> None:
    """Consigne la sélection **et les exclusions**, motif par motif."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    publications = sorted(r.date_publication for r in bilan.retenus if r.date_publication)
    chemin.write_text(
        json.dumps(
            {
                "source": JEU_TEXTES,
                "licence": "Licence Ouverte 2.0",
                "perimetre": list(PERIMETRE),
                #: Date de l'archive téléchargée : le corpus ne contient rien après elle.
                "archive_datee_du": date_de_l_archive(archive) if archive else "",
                #: Publication la plus récente effectivement retenue.
                "derniere_publication": publications[-1] if publications else "",
                "retenus": [asdict(r) for r in bilan.retenus],
                "ecartes": {
                    "doublons_de_contenu": bilan.doublons,
                    "publications_retirees": bilan.retires,
                    "sans_metadonnees": bilan.sans_metadonnees,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def resumer(bilan: Bilan) -> None:
    """Affiche ce qui a été retenu et ce qui ne l'a pas été."""
    par_theme = Counter(r.thematique for r in bilan.retenus)
    par_decennie: Counter[str] = Counter()
    for ressource in bilan.retenus:
        annee = ressource.date_publication[:4]
        par_decennie[f"{annee[:3]}0" if annee else "?"] += 1

    print(f"\n{len(bilan.retenus)} documents retenus")
    for theme, nombre in par_theme.most_common():
        print(f"    {nombre:>5}  {theme}")
    print("\n  par décennie :")
    for decennie, nombre in sorted(par_decennie.items()):
        print(f"    {decennie}s  {nombre:>5}")
    print(
        f"\n  écartés : {bilan.doublons} doublons de contenu, "
        f"{len(bilan.retires)} publication(s) retirée(s), "
        f"{len(bilan.sans_metadonnees)} sans métadonnées"
    )
    for ligne in bilan.retires:
        print(f"      retiré : {ligne}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="corpus-extraire")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument("--archive", type=Path, default=Path("TextesPublicationsHAS.zip"))
    parser.add_argument("--metadonnees", type=Path, default=Path("has-publications-split.zip"))
    parser.add_argument(
        "--inventaire", action="store_true", help="décrit la sélection sans écrire les PDF"
    )
    args = parser.parse_args()

    publications = lire_metadonnees(args.metadonnees)
    print(f"{len(publications)} publications du périmètre dans les métadonnées")
    bilan = selectionner(args.archive, publications)
    resumer(bilan)
    print(
        f"\nArchive datée du {date_de_l_archive(args.archive) or '?'}"
        " — le corpus ne contient rien de postérieur."
    )

    ecrire_manifeste(bilan, args.racine / "fixtures" / "corpus.json", args.archive)
    print(f"\nManifeste écrit -> {args.racine / 'fixtures' / 'corpus.json'}")
    if args.inventaire:
        print("Inventaire seul : aucun PDF n'a été écrit.")
        return
    echecs = extraire(args.archive, bilan, args.racine / "fixtures" / "pdf")
    volume = sum(r.octets for r in bilan.retenus)
    ecrits = len(bilan.retenus) - len(echecs)
    print(f"\n{ecrits} PDF dans fixtures/pdf ({volume / 1e9:.2f} Go)")
    if echecs:
        print(f"\n{len(echecs)} fichier(s) non écrit(s) :")
        for nom, motif in echecs[:10]:
            print(f"  {nom[:56]:58} {motif[:60]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
