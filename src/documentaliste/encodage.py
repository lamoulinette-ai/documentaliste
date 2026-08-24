"""Le modèle d'embedding et l'encodage des requêtes.

Hors du paquet `probes`, et c'est tout l'objet de ce module : encoder une question est un
besoin d'exécution, pas de mesure. Tant que l'encodeur habitait le module de mesure,
l'application ne pouvait pas être livrée sans l'appareil de mesure qui l'entoure.

Le nom `MODELE` désigne ici le modèle d'embedding. `probes.mistral` définit un `MODELE`
homonyme qui désigne le modèle de langue : les deux ne se croisent jamais dans un même
fichier, et les importer tous deux demanderait un alias explicite.
"""

from __future__ import annotations

#: Modèle multilingue de petite taille, choisi pour tourner sur processeur sans peine.
MODELE = "intfloat/multilingual-e5-small"

#: Préfixes attendus par la famille E5, qui distingue une requête d'un passage. Les omettre
#: ne lève rien et dégrade le classement en silence.
PREFIXE_REQUETE = "query: "
PREFIXE_PASSAGE = "passage: "

#: Moteurs d'exécution admis ; « onnx » est quantifié en 8 bits et ses vecteurs diffèrent.
#: Le moteur entre pour cette raison dans la clé du cache vectoriel : mélanger les deux
#: donnerait un classement calculé sur des vecteurs hétérogènes, sans que rien ne le signale.
MOTEURS: tuple[str, ...] = ("torch", "onnx")

#: Fichier de poids quantifiés attendu dans les dépôts exportés par sentence-transformers.
_POIDS_QUANTIFIES = "onnx/model_qint8_avx512_vnni.onnx"

#: Modèles déjà chargés, par moteur et par nom, pour ne lire les poids qu'une fois.
_MODELES: dict[str, object] = {}


def charger_modele(nom: str, moteur: str = "torch"):  # noqa: ANN201
    """Le modèle demandé, construit au plus une fois par processus et par moteur."""
    if moteur not in MOTEURS:
        raise SystemExit(f"Moteur inconnu : {moteur}. Attendu : {', '.join(MOTEURS)}.")
    cle = f"{moteur}:{nom}"
    if cle not in _MODELES:
        from sentence_transformers import SentenceTransformer

        if moteur == "onnx":
            _MODELES[cle] = SentenceTransformer(
                nom,
                device="cpu",
                backend="onnx",
                model_kwargs={"file_name": _POIDS_QUANTIFIES},
            )
        else:
            _MODELES[cle] = SentenceTransformer(nom, device="cpu")
    return _MODELES[cle]


def encoder(questions: list[str], moteur: str = "torch", nom: str = MODELE):  # noqa: ANN201
    """Vecteurs de requête, avec le préfixe que le modèle attend."""
    modele = charger_modele(nom, moteur)
    return modele.encode(
        [PREFIXE_REQUETE + q for q in questions],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
