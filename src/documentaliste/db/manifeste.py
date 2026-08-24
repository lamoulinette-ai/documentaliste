"""Métadonnées du corpus : ce qu'une citation doit pouvoir afficher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Un PDF de la HAS, avec ce que sa publication en dit."""

    nom: str
    #: Identifiant de publication (« c_1239410 »), commun aux PDF d'un même dossier.
    fiche: str
    titre: str
    type_publication: str = ""
    mise_en_ligne: str = ""
    validation: str = ""
    url_pdf: str = ""
    url_fiche: str = ""
    themes: tuple[str, ...] = ()


def lire_manifeste(racine: Path) -> dict[str, Document]:
    """Métadonnées par nom de document, depuis le manifeste du corpus."""
    chemin = racine / "fixtures" / "corpus.json"
    if not chemin.exists():
        raise SystemExit(f"Manifeste absent ({chemin}). Lancer d'abord « uv run corpus-extraire ».")
    brut = json.loads(chemin.read_text(encoding="utf-8"))
    return {
        entree["nom"]: Document(
            nom=entree["nom"],
            fiche=entree.get("publication", ""),
            titre=entree.get("titre") or entree["nom"],
            type_publication=entree.get("thematique", ""),
            mise_en_ligne=entree.get("date_publication", ""),
            validation=entree.get("date_validation", ""),
            url_fiche=entree.get("url_fiche", ""),
            themes=tuple(entree.get("themes", ())),
        )
        for entree in brut["retenus"]
    }


def titres(documents: dict[str, Document]) -> dict[str, str]:
    """Titre par nom de document, pour le préfixe appliqué avant vectorisation."""
    return {nom: doc.titre for nom, doc in documents.items()}
