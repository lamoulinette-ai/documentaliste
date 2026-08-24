"""Statistiques lexicales du corpus, sur lesquelles BM25 s'appuie.

BM25 pondère un terme par sa rareté **dans la collection entière**. Les calculer sur les
seuls candidats d'une requête donnerait d'autres scores que la version en mémoire, et la
promesse de rangs identiques tomberait.

Trois tables, reconstruites d'un bloc. Le nombre de passages contenant chaque terme et la
longueur moyenne d'un passage, en mots normalisés par `probes.lexical` — c'est ce que BM25
consomme. Puis le nombre de passages contenant chaque **lexème PostgreSQL**, qui sert à
écarter de la requête plein texte les racines trop répandues pour discriminer.

Les deux dénombrements portent sur des unités différentes et ne sont pas interchangeables :
la racinisation de PostgreSQL ne découpe pas comme la nôtre.

    uv run documentaliste-lexique
    uv run documentaliste-lexique --lexemes-seulement   # sans reparcourir le corpus
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from documentaliste.db.pool import appliquer_schema, connexion
from documentaliste.probes.lexical import normaliser

#: Passages lus par aller-retour. Assez pour amortir la latence, assez peu pour que la
#: mémoire ne suive pas la taille du corpus.
LOT = 20_000


def parcourir(cnx: object, lot: int) -> tuple[Counter[str], int, int]:
    """Compte les passages par terme, leur nombre et le total des longueurs."""
    presence: Counter[str] = Counter()
    passages = total_mots = 0
    dernier = 0
    with tqdm(desc="lexique", unit="passage") as barre:
        while True:
            lignes = cnx.execute(  # type: ignore[attr-defined]
                "SELECT id, texte FROM passage WHERE id > %s ORDER BY id LIMIT %s",
                (dernier, lot),
            ).fetchall()
            if not lignes:
                return presence, passages, total_mots
            for identifiant, texte in lignes:
                jetons = normaliser(texte)
                presence.update(set(jetons))
                total_mots += len(jetons)
                passages += 1
                dernier = identifiant
            barre.update(len(lignes))


def ecrire(cnx: object, presence: Counter[str], passages: int, total_mots: int) -> None:
    """Remplace les statistiques par flux `COPY`."""
    moyenne = total_mots / max(passages, 1)
    with cnx.cursor() as curseur:  # type: ignore[attr-defined]
        curseur.execute("TRUNCATE statistique_terme")
        ordre = "COPY statistique_terme (terme, passages) FROM STDIN"
        with curseur.copy(ordre) as flux:
            for terme, compte in presence.items():
                flux.write_row((terme, compte))
        curseur.execute("TRUNCATE statistique_corpus")
        curseur.execute(
            "INSERT INTO statistique_corpus (passages, longueur_moyenne) VALUES (%s, %s)",
            (passages, moyenne),
        )


#: `ts_stat` parcourt les 796 000 `tsvector` déjà calculés et rend, pour chaque lexème, le
#: nombre de passages qui le contiennent. Le recalculer côté Python demanderait de relire
#: tout le corpus pour retrouver ce que la colonne générée contient déjà.
_LEXEMES = """
INSERT INTO statistique_lexeme (lexeme, passages)
SELECT word, ndoc FROM ts_stat('SELECT tsv FROM passage')
"""


def ecrire_lexemes(cnx: object) -> int:
    """Remplace les statistiques par lexème PostgreSQL. Rend leur nombre."""
    with cnx.cursor() as curseur:  # type: ignore[attr-defined]
        curseur.execute("TRUNCATE statistique_lexeme")
        curseur.execute(_LEXEMES)
        return curseur.rowcount


def construire(lot: int, lexemes_seulement: bool = False) -> None:
    """Reconstruit le lexique depuis les passages en base."""
    appliquer_schema()
    with connexion() as cnx:
        if lexemes_seulement:
            print("statistiques par lexème PostgreSQL…")
            lexemes = ecrire_lexemes(cnx)
            cnx.commit()
            print(f"\n{lexemes} lexèmes PostgreSQL distincts")
            return
        presence, passages, total_mots = parcourir(cnx, lot)
        if not passages:
            raise SystemExit("Aucun passage. Lancer d'abord « uv run documentaliste-ingerer ».")
        ecrire(cnx, presence, passages, total_mots)
        print("statistiques par lexème PostgreSQL…")
        lexemes = ecrire_lexemes(cnx)
        cnx.commit()
    print(f"\n{len(presence)} termes distincts sur {passages} passages")
    print(f"{lexemes} lexèmes PostgreSQL distincts")
    print(f"longueur moyenne : {total_mots / passages:.1f} mots normalisés")


def main() -> None:
    parser = argparse.ArgumentParser(prog="documentaliste-lexique")
    parser.add_argument("--racine", type=Path, default=Path("."))
    parser.add_argument("--lot", type=int, default=LOT)
    parser.add_argument(
        "--lexemes-seulement",
        action="store_true",
        help="ne recalcule que les lexèmes PostgreSQL, sans reparcourir le corpus",
    )
    args = parser.parse_args()
    construire(args.lot, args.lexemes_seulement)


if __name__ == "__main__":
    main()
