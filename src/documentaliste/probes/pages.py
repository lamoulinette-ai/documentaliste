"""Extraction du texte des PDF, faite une fois pour toutes."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

#: Emplacement du cache, à côté des PDF dont il dérive.
CACHE = Path("fixtures") / "pages.json.gz"

#: Parallélisme par défaut, plafonné pour borner la mémoire.
WORKERS_MAX = 8


@dataclass(frozen=True)
class Page:
    """Une page de PDF, telle que le lecteur la rend."""

    document: str
    numero: int
    texte: str


def empreinte(fichiers: list[Path]) -> str:
    """Signature du corpus : nom et taille de chaque PDF, dans l'ordre."""
    signature = "|".join(f"{f.name}:{f.stat().st_size}" for f in sorted(fichiers))
    return hashlib.sha256(signature.encode()).hexdigest()[:16]


def lire_pdf(chemin: Path) -> list[tuple[str, int, str]]:
    """Texte page à page d'un seul PDF. Exécuté dans un processus séparé."""
    import pymupdf

    with pymupdf.open(chemin) as pdf:
        return [(chemin.stem, n, page.get_text("text", sort=True)) for n, page in enumerate(pdf, 1)]


def workers_effectifs(demandes: int | None) -> int:
    """Parallélisme retenu, borné par les cœurs disponibles et par `WORKERS_MAX`."""
    if demandes is not None and demandes > 0:
        return min(demandes, WORKERS_MAX)
    return max(1, min(os.cpu_count() or 1, WORKERS_MAX))


def extraire(racine: Path, workers: int | None = None, cache: bool = True) -> list[Page]:
    """Toutes les pages du corpus, depuis le cache s'il correspond au corpus présent."""
    fichiers = sorted((racine / "fixtures" / "pdf").glob("*.pdf"))
    if not fichiers:
        raise SystemExit("Aucun PDF : lancer d'abord « sonde-collecte telecharger ».")

    chemin = racine / CACHE
    signature = empreinte(fichiers)
    if cache and chemin.exists():
        depot = json.loads(gzip.decompress(chemin.read_bytes()).decode("utf-8"))
        if depot.get("empreinte") == signature:
            return [Page(*p) for p in depot["pages"]]
        print(f"Corpus modifié depuis le cache ({len(fichiers)} PDF) : réextraction.")

    pages: list[Page] = []
    n = workers_effectifs(workers)
    with ProcessPoolExecutor(max_workers=n) as pool:
        travaux = pool.map(lire_pdf, fichiers)
        for lot in tqdm(travaux, total=len(fichiers), desc=f"extraction ({n} processus)"):
            pages.extend(Page(*p) for p in lot)

    if cache:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        charge = {"empreinte": signature, "pages": [[p.document, p.numero, p.texte] for p in pages]}
        brut = json.dumps(charge, ensure_ascii=False).encode()
        chemin.write_bytes(gzip.compress(brut))
    return pages


def par_document(pages: list[Page]) -> dict[str, list[str]]:
    """Regroupe les pages par document, dans l'ordre de pagination."""
    groupes: dict[str, list[str]] = {}
    for page in sorted(pages, key=lambda p: (p.document, p.numero)):
        groupes.setdefault(page.document, []).append(page.texte)
    return groupes
