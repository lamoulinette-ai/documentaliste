"""Recherche en base : les deux bras et leur fusion, sur 796 000 passages.

La version en mémoire recalcule le corpus entier à chaque lancement. À cette échelle elle
demande des heures et une dizaine de gigaoctets, pour refaire ce que la base contient.

**Le bras vectoriel est exact.** Les vecteurs sont normalisés, donc leur produit scalaire
est la similarité cosinus, et l'index HNSW rend les mêmes voisins que le produit matriciel.

**Le bras lexical se fait en deux temps, et il faut dire pourquoi.** PostgreSQL racinise et
filtre les mots vides à sa façon ; son `ts_rank` n'est pas BM25. Sa recherche plein texte
sert donc à **trouver** des candidats, jamais à les classer : le classement reste calculé
par `probes.lexical.contribution`, avec les statistiques du corpus entier lues dans le
lexique. Les rangs sont alors identiques à ceux de la version en mémoire, sur les candidats
ramenés — et ce que la génération de candidats laisse dehors est mesuré séparément.

**La requête est disjonctive.** Un passage candidat doit contenir *au moins un* terme de la
question, jamais tous : exiger la conjonction reviendrait à ne servir que les questions
recopiées du corpus. C'est BM25 qui départage ensuite, et lui seul.
"""

from __future__ import annotations

from dataclasses import dataclass

from documentaliste.probes.lexical import (
    contribution,
    entrelacer,
    fusion_rrf,
    normaliser,
    rarete,
)

#: Candidats ramenés par la recherche plein texte avant reclassement exact. Le reclassement
#: en paie chacun — 0,107 ms mesurée par candidat — quand la base, elle, est indifférente à
#: la largeur. 500 est ce que le budget interactif laisse au bras lexical.
CANDIDATS = 500

#: Poids du bras vectoriel dans la fusion. Centre d'un plateau de douze réglages contigus
#: (0,10 à 0,32) choisi sur une moitié de l'étalon, et qui rend 76 % de rappel@10 sur
#: l'autre contre 71 % pour le bras lexical seul.
POIDS_VECTORIEL = 0.22

_VOISINS = """
SELECT id, document, page, texte
FROM passage
ORDER BY vecteur <#> %s::vector
LIMIT %s
"""

#: Part maximale du corpus qu'un lexème peut occuper pour entrer dans la requête. À 1.0 le
#: filtre est inactif. Écarter le vocabulaire répandu coûte du plafond — 91,4 % sans filtre,
#: 88,3 % à 3 % — et divise la génération par neuf, seule façon de tenir dans le budget.
PART_MAX = 0.03

#: Lexèmes de la question, tels que l'analyseur français les produit, avec le nombre de
#: passages qui les contiennent. `tsvector_to_array` les rend déjà racinisés et débarrassés
#: des mots vides ; `quote_literal` met chacun à l'abri d'une apostrophe ou d'un tiret qui
#: casserait `to_tsquery`.
_LEXEMES = """
SELECT quote_literal(l.lexeme), coalesce(s.passages, 0)
FROM unnest(tsvector_to_array(to_tsvector('french', %s))) AS l(lexeme)
LEFT JOIN statistique_lexeme s ON s.lexeme = l.lexeme
"""

#: La requête est bâtie à part et passée en paramètre : dans une seule instruction, elle
#: serait référencée deux fois — filtre puis tri — et le planificateur, la matérialisant,
#: renoncerait à l'index GIN. L'aller-retour supplémentaire coûte moins qu'un parcours
#: séquentiel de 796 000 passages.
_CANDIDATS = """
SELECT id, document, page, texte
FROM passage
WHERE tsv @@ to_tsquery('french', %s)
ORDER BY ts_rank(tsv, to_tsquery('french', %s)) DESC
LIMIT %s
"""

_STATISTIQUES = "SELECT passages, longueur_moyenne FROM statistique_corpus"

_MEILLEURE_SIMILARITE = """
SELECT -(vecteur <#> %s::vector)
FROM passage
ORDER BY vecteur <#> %s::vector
LIMIT 1
"""


@dataclass(frozen=True)
class Resultat:
    """Un passage rendu par la recherche, avec de quoi le citer."""

    identifiant: int
    document: str
    page: int
    texte: str


def _resultats(lignes: list[tuple]) -> list[Resultat]:
    return [Resultat(i, d, p, t) for i, d, p, t in lignes]


def statistiques(cnx: object) -> tuple[int, float]:
    """Nombre de passages et longueur moyenne, depuis le lexique."""
    ligne = cnx.execute(_STATISTIQUES).fetchone()  # type: ignore[attr-defined]
    if ligne is None:
        raise SystemExit("Lexique absent. Lancer d'abord « uv run documentaliste-lexique ».")
    return ligne[0], ligne[1]


def raretes(cnx: object, termes: list[str], total: int) -> dict[str, float]:
    """Poids de rareté des termes de la requête, calculés sur le corpus entier."""
    if not termes:
        return {}
    lignes = cnx.execute(  # type: ignore[attr-defined]
        "SELECT terme, passages FROM statistique_terme WHERE terme = ANY(%s)", (termes,)
    ).fetchall()
    return {terme: rarete(passages, total) for terme, passages in lignes}


def _forme(vecteur) -> str:  # noqa: ANN001
    return "[" + ",".join(f"{v:.6f}" for v in vecteur) + "]"


def meilleure_similarite(cnx: object, vecteur) -> float:  # noqa: ANN001
    """Cosinus du passage le plus proche de la question.

    Les vecteurs étant normalisés, l'opposé du produit scalaire que pgvector minimise est
    la similarité cosinus. C'est la seule grandeur du système qui soit bornée et comparable
    d'une question à l'autre : un score BM25 dépend de la longueur de la requête et de la
    rareté de ses termes, un rang de fusion ne dit rien de la qualité de ce qu'il classe.
    """
    forme = _forme(vecteur)
    curseur = cnx.execute(_MEILLEURE_SIMILARITE, (forme, forme))  # type: ignore[attr-defined]
    ligne = curseur.fetchone()
    return float(ligne[0]) if ligne else 0.0


def disjonction(lexemes: list[tuple[str, int]], plafond: float) -> str:
    """Requête `to_tsquery` bâtie sur les lexèmes assez rares pour discriminer.

    `ts_rank` ignore la rareté : il note la fréquence d'un terme dans le passage et sa
    position, jamais son pouvoir de distinguer. Une racine présente dans un tiers du corpus
    y pèse autant qu'un nom de molécule, et les candidats s'en trouvent choisis sur du
    vocabulaire commun.

    Si le filtre ne laisse rien — question entièrement bâtie sur des mots répandus — les
    lexèmes les plus rares de la question sont conservés : mieux vaut des candidats
    médiocres que pas de candidats.
    """
    if not lexemes:
        return ""
    retenus = [terme for terme, passages in lexemes if passages <= plafond]
    if not retenus:
        rarete_minimale = min(passages for _, passages in lexemes)
        retenus = [terme for terme, passages in lexemes if passages == rarete_minimale]
    return " | ".join(retenus)


def candidats_lexicaux(
    cnx: object, question: str, combien: int = CANDIDATS, plafond: float = float("inf")
) -> list[tuple]:
    """Passages contenant au moins un lexème retenu, les mieux notés par `ts_rank`."""
    lexemes = cnx.execute(_LEXEMES, (question,)).fetchall()  # type: ignore[attr-defined]
    requete = disjonction(lexemes, plafond)
    if not requete:
        return []
    curseur = cnx.execute(_CANDIDATS, (requete, requete, combien))  # type: ignore[attr-defined]
    return curseur.fetchall()


#: Titre et adresses d'une poignée de documents, pour les citer autrement que par leur nom
#: de fichier. Lecture séparée et non jointure : les requêtes des deux bras sont mesurées,
#: et les alourdir d'une jointure changerait ce qu'on a mesuré pour un besoin d'affichage.
_DOCUMENTS = """
    SELECT nom, titre, url_fiche, url_pdf, type_publication, mise_en_ligne
    FROM document
    WHERE nom = ANY(%s)
"""


def documents(cnx: object, noms: list[str]) -> dict[str, dict[str, str]]:
    """Métadonnées de publication, par nom de document.

    Un nom de fichier comme `2007-05-03_rpc_sftg_insomnie_-_argumentaire_mel` est
    vérifiable mais illisible : devant un professionnel de santé, il dessert la traçabilité
    qu'il est censé établir.
    """
    if not noms:
        return {}
    lignes = cnx.execute(_DOCUMENTS, (list(set(noms)),)).fetchall()  # type: ignore[attr-defined]
    return {
        nom: {
            "titre": titre,
            "url_fiche": fiche,
            "url_pdf": pdf,
            "type_publication": type_publication,
            "mise_en_ligne": mise_en_ligne,
        }
        for nom, titre, fiche, pdf, type_publication, mise_en_ligne in lignes
    }


def bras_vectoriel(cnx: object, vecteur, k: int) -> list[Resultat]:  # noqa: ANN001
    """Les k passages les plus proches, par l'index HNSW."""
    lignes = cnx.execute(_VOISINS, (_forme(vecteur), k)).fetchall()  # type: ignore[attr-defined]
    return _resultats(lignes)


def classer_par_bm25(
    lignes: list[tuple], question: str, total: int, moyenne: float, poids: dict[str, float]
) -> list[tuple[float, Resultat]]:
    """Score BM25 de chaque passage, décroissant, à statistiques données.

    Les statistiques sont reçues plutôt que lues : la sonde de portage peut alors appeler
    ce classement avec celles d'un sous-corpus et le confronter à la version en mémoire,
    sur exactement la même collection.
    """
    moyenne = max(moyenne, 1e-9)
    # La liste, pas l'ensemble : un terme répété dans la question pèse deux fois, comme
    # dans `Bm25.scores`. La version par ensemble faisait diverger neuf questions sur 163.
    termes = normaliser(question)
    scores: list[tuple[float, Resultat]] = []
    for identifiant, document, page, texte in lignes:
        jetons = normaliser(texte)
        frequences: dict[str, int] = {}
        for jeton in jetons:
            frequences[jeton] = frequences.get(jeton, 0) + 1
        relative = len(jetons) / moyenne
        score = sum(
            contribution(frequences[terme], relative, poids[terme])
            for terme in termes
            if terme in frequences and terme in poids
        )
        scores.append((score, Resultat(identifiant, document, page, texte)))
    scores.sort(key=lambda paire: -paire[0])
    return scores


def bras_lexical(
    cnx: object,
    question: str,
    k: int,
    candidats: int = CANDIDATS,
    part_max: float = PART_MAX,
) -> list[Resultat]:
    """Les k passages les mieux classés par BM25, parmi les candidats du plein texte.

    Le classement n'utilise pas `ts_rank` : celui-ci ne pondère pas comme BM25, et deux
    classements différents ne se comparent pas aux chiffres établis en mémoire.
    """
    total, moyenne = statistiques(cnx)
    lignes = candidats_lexicaux(cnx, question, candidats, part_max * total)
    if not lignes:
        return []
    termes = list(dict.fromkeys(normaliser(question)))
    poids = raretes(cnx, termes, total)
    return [r for _, r in classer_par_bm25(lignes, question, total, moyenne, poids)[:k]]


#: Constante de la fusion par rangs réciproques. Elle décide de ce que le rang pèse face au
#: poids ; à 60, valeur de l'article d'origine, le poids cessait de doser pour choisir un
#: bras. Zéro laisse au rang tout son poids, et c'est la seule valeur qui offre un plateau.
CONSTANTE_RRF = 0


def fusionner(
    vectoriels: list[Resultat],
    lexicaux: list[Resultat],
    poids_vectoriel: float,
    k: int,
    constante: int = CONSTANTE_RRF,
) -> list[Resultat]:
    """Mêle deux classements par rangs réciproques pondérés.

    Séparée de `chercher` pour qu'un balayage puisse la rejouer sur des classements déjà
    obtenus : la fusion est un calcul pur, et deux cents réglages ne justifient pas deux
    cents interrogations de la base.
    """
    par_identifiant = {r.identifiant: r for r in vectoriels + lexicaux}
    if poids_vectoriel >= 1.0:
        ordre = [r.identifiant for r in vectoriels]
    elif poids_vectoriel <= 0.0:
        ordre = [r.identifiant for r in lexicaux]
    else:
        ordre = fusion_rrf(
            [r.identifiant for r in vectoriels],
            [r.identifiant for r in lexicaux],
            k=constante,
            poids=(poids_vectoriel, 1.0 - poids_vectoriel),
        )
    return [par_identifiant[i] for i in ordre[:k]]


def entrelacer_les_bras(
    vectoriels: list[Resultat], lexicaux: list[Resultat], k: int
) -> list[Resultat]:
    """Alterne les deux bras un à un, sans pondération ni score commun."""
    par_identifiant = {r.identifiant: r for r in vectoriels + lexicaux}
    ordre = entrelacer([r.identifiant for r in vectoriels], [r.identifiant for r in lexicaux])
    return [par_identifiant[i] for i in ordre[:k]]


def chercher(
    cnx: object,
    question: str,
    vecteur,  # noqa: ANN001
    k: int,
    poids_vectoriel: float = POIDS_VECTORIEL,
    candidats: int = CANDIDATS,
) -> list[Resultat]:
    """Fusionne les deux bras par rangs réciproques, comme la version en mémoire."""
    profondeur = k * 5
    vectoriels = bras_vectoriel(cnx, vecteur, profondeur)
    lexicaux = bras_lexical(cnx, question, profondeur, candidats)
    return fusionner(vectoriels, lexicaux, poids_vectoriel, k, CONSTANTE_RRF)
