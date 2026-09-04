# syntax=docker/dockerfile:1
# Image du Documentaliste : FastAPI devant PostgreSQL/pgvector et l'API Mistral.
#
# Le modèle d'embedding est CUIT DANS L'IMAGE, pas téléchargé au démarrage. Deux raisons,
# et la seconde est dirimante : le premier appel ne doit pas dépendre de la disponibilité
# de Hugging Face, et le conteneur tourne en `read_only` — un téléchargement à chaud
# n'aurait nulle part où écrire. L'image grossit d'environ 120 Mo, ce qui est le bon échange.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (gestion des dépendances, lock figé) — même version que l'auditeur.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# 1) Couche dépendances (cache tant que le lock ne bouge pas).
COPY pyproject.toml uv.lock ./
# Les extras du service, et eux seuls : « api » pour FastAPI, « bdd » pour psycopg,
# « embeddings » et « onnx » pour l'encodage de la question. L'appareil de mesure n'entre
# pas dans l'image de production — c'est le tri décidé pour le dépôt vitrine, appliqué ici.
ARG EXTRAS="--extra api --extra bdd --extra embeddings --extra onnx"
RUN uv sync --frozen --no-dev --no-install-project ${EXTRAS}

# 2) Poids du modèle, dans une couche à part : ils ne changent jamais, et les remettre
#    en cache à chaque modification du code coûterait 120 Mo de transfert par déploiement.
#
#    `MOTEUR_EMBEDDING` vaut « onnx » par défaut : quantifié en 8 bits, plus léger et plus
#    rapide sur processeur. Le fichier de poids porte « avx512_vnni » dans son nom, mais
#    onnxruntime retombe sur un chemin générique quand le processeur n'a pas ces
#    instructions — plus lent, jamais faux. Vérifier `lscpu | grep -o avx512_vnni` sur la
#    cible : si l'instruction manque et que la latence gêne, reconstruire avec `torch`.
ARG MOTEUR_EMBEDDING=onnx
ENV HF_HOME=/app/modeles
#    `/app/.venv/bin/python` et non `uv run` : à ce stade les dépendances sont installées
#    mais le projet ne l'est pas — `src/` n'est copié qu'à la couche suivante. `uv run`
#    tenterait de le construire et échouerait sur les fichiers déclarés au `pyproject`.
RUN MOTEUR="${MOTEUR_EMBEDDING}" /app/.venv/bin/python -c "\
import os; \
from sentence_transformers import SentenceTransformer; \
onnx = os.environ['MOTEUR'] == 'onnx'; \
SentenceTransformer('intfloat/multilingual-e5-small', device='cpu', \
    **({'backend': 'onnx', \
        'model_kwargs': {'file_name': 'onnx/model_qint8_avx512_vnni.onnx'}} if onnx else {}))" \
    && find /app/modeles -name "*.lock" -delete

# 3) Code. README.md requis : pyproject le déclare comme readme (build hatchling).
COPY src/ ./src/
COPY README.md ./
RUN uv sync --frozen --no-dev ${EXTRAS}

# Utilisateur non-root. Aucun dossier d'état inscriptible : tout ce que le service écrit
# — le cache des questions déjà posées — va sur le tmpfs monté par docker-compose.
RUN useradd -m -u 10001 documentaliste && chown -R documentaliste:documentaliste /app

# Le dossier du cache est créé ici, avec son propriétaire, et non par le compose. Docker
# initialise un volume nommé vide en recopiant le contenu ET les droits du chemin de
# l'image : sans cette ligne, le volume appartiendrait à root et le service, qui tourne en
# 10001, n'y écrirait jamais — sans le dire, ses erreurs d'écriture étant avalées.
RUN mkdir -p /var/cache/documentaliste \
    && chown documentaliste:documentaliste /var/cache/documentaliste
USER documentaliste

# `HF_HUB_OFFLINE` n'est PAS activé, et c'est un constat plutôt qu'un choix : en backend
# ONNX, sentence-transformers interroge l'index du dépôt pour résoudre le nom du fichier de
# poids AVANT de consulter le cache local. Hors ligne, cette résolution échoue et le service
# refuse de démarrer — poids présents ou non.
#
# Les poids restent cuits dans l'image : rien n'est retéléchargé, seule la liste des
# fichiers est demandée. Le démarrage dépend donc d'un aller-retour vers Hugging Face,
# ce qui reste à corriger en chargeant le modèle depuis un chemin local explicite.
ENV PATH="/app/.venv/bin:${PATH}" \
    HOME=/tmp \
    DOCUMENTALISTE_MOTEUR=${MOTEUR_EMBEDDING} \
    DOCUMENTALISTE_CACHE=/tmp/cache

EXPOSE 8005
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8005/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "documentaliste.api.app:app", "--host", "0.0.0.0", "--port", "8005"]
