"""Outils partagés par les tests qui portent sur la forme du code."""

from __future__ import annotations

import ast
from pathlib import Path

PORTEURS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def code_seul(fichier: Path) -> str:
    """Source privée de ses commentaires et de ses docstrings.

    Un test qui cherche une chaîne dans le fichier entier confond ce que le code fait avec
    ce que sa documentation discute : « ts_rank » est nommé dans l'explication de son
    propre rejet, « pdftotext » dans celle de son abandon.
    """
    arbre = ast.parse(fichier.read_text(encoding="utf-8"))
    for noeud in ast.walk(arbre):
        if isinstance(noeud, PORTEURS) and ast.get_docstring(noeud, clean=False) is not None:
            noeud.body = noeud.body[1:] or [ast.Pass()]
    return ast.unparse(arbre)


def corps(fichier: Path, fonction: str) -> str:
    """Code d'une fonction, docstring et commentaires retirés."""
    arbre = ast.parse(code_seul(fichier))
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.FunctionDef) and noeud.name == fonction:
            return ast.unparse(noeud)
    raise AssertionError(f"fonction introuvable : {fonction}")
