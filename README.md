# Documentaliste — recherche documentée sur le corpus de la HAS

Une question de pratique clinique en français. Des passages retrouvés dans les publications
de la Haute Autorité de Santé. Une réponse dont **chaque affirmation cite l'extrait qui la
fonde** — ou un refus, quand rien dans le corpus ne répond.

**Démonstration : [documentaliste.lamoulinette.ai](https://documentaliste.lamoulinette.ai)**

> Démonstrateur technique. Les réponses sont produites automatiquement à partir de
> publications sous Licence Ouverte 2.0 et ne constituent ni un avis médical, ni une
> position de la Haute Autorité de Santé.

---

## Le pari

Un système qui **se tait quand il ne sait pas** vaut mieux qu'un système qui répond
toujours. Tout le reste en découle, y compris les difficultés : un refus est facile à
produire et très difficile à justifier.

Trois issues, et non deux :

| issue | ce qu'elle signifie |
| --- | --- |
| `reponse` | un passage servi soutient un énoncé portant sur la question posée |
| `constat_absence` | **la source elle-même écrit qu'il n'existe pas de donnée établie** |
| `refus` | aucun passage ne porte sur la question |

La distinction du milieu n'est pas une subtilité. Sans elle, « la HAS n'a pas tranché » se
compterait comme une réponse ordinaire, et rapporter fidèlement une absence passerait pour
une régression.

## Ce que le système vaut

Mesuré sur un étalon de 162 questions **reformulées** — les questions tirées mot pour mot du
corpus mesuraient la correspondance de chaînes, pas la recherche.

| mesure | valeur |
| --- | ---: |
| rappel@10, fusion | 80 % |
| rappel@10, hors échantillon | 76 % contre 71 % pour le meilleur bras seul |
| rang réciproque moyen, fusion | 0,666 |
| **nDCG@10** | **0,322 à 0,693** |
| plafond de rappel | 81 % |
| latence, à chaud | 16 ms |

**Le nDCG se lit comme une fourchette, et c'est délibéré.** Le chiffre dépend de ce qu'on
accepte comme succès :

| la réponse compte si le passage vient… | rappel | nDCG@10 | statut |
| --- | ---: | ---: | --- |
| de la page exacte dont la question est tirée | 52 % | **0,322** | **minorant** |
| du document dont elle est tirée | 73 % | 0,576 | intermédiaire |
| de n'importe quel document du dossier HAS | 81 % | **0,693** | **majorant** |

Aucune des trois n'est la vérité. Publier la dernière seule inviterait à la comparer aux
0,314 du meilleur modèle standard sur R2MED — en taisant que ce banc d'essai est conçu pour
exiger un raisonnement, là où nos questions sont tirées des documents qu'elles doivent
retrouver. **En lecture stricte, nous sommes à 0,322 sur une tâche plus facile.** Ces deux
bornes vont ensemble partout.

## Ce que le système ne mesure pas

À lire avant de se faire une opinion favorable.

- **Le taux de citations réellement fondées est inconnu.** Un juge automatique devait
  l'établir ; il s'est révélé non recevable — cinq de ses propres appuis n'étaient pas
  recopiés depuis le texte, et son accord avec une référence humaine arbitrée plafonne à
  56 %. La littérature rapporte 50 à 90 % de citations non pleinement soutenues sur ce type
  de tâche. Nous ne savons pas où nous sommes sur cette échelle.
- **La décision de se taire ne se délègue pas à la recherche.** L'aire sous la courbe
  d'abstention vaut 0,731 : la similarité porte un signal, pas une décision. Les quatre
  questions que la HAS ne tranche pas franchissent *toutes* le seuil de refus — leur sujet
  est traité, seule la conclusion manque.
- **La vérité de terrain au document reste imparfaite** pour 162 questions positives.
- **La stabilité sous paraphrase** n'a jamais été mesurée.

## Architecture

```
question ─▶ encodage e5-small ─┬─▶ bras vectoriel  (pgvector, HNSW)  ─┐
                               └─▶ bras lexical    (tsvector + BM25) ─┴─▶ fusion RRF
                                                                            │
                          réponse citée ◀─ contrôles mécaniques ◀─ Mistral ─┘
```

**Deux bras, parce que la fusion l'emporte sur les trois mesures à la fois** — rang
réciproque, rappel@10 et nDCG. C'est le seul argument solide en sa faveur : une seule
métrique gagnante n'aurait pas suffi.

| réglage | valeur | ce qui l'a décidé |
| --- | --- | --- |
| `POIDS_VECTORIEL` | 0,22 | centre d'un plateau de douze réglages, vérifié hors échantillon |
| `CONSTANTE_RRF` | 0 | à 60, le poids cessait de doser pour choisir un bras |
| `CANDIDATS` | 500 | le reclassement coûte 0,107 ms par candidat |
| `PART_MAX` | 0,03 | seul seuil dont la génération tienne sous 300 ms |
| `hnsw.ef_search` | 200 | — |

Corpus : archive HAS du 18 juin 2026, huit thématiques, **6 504 documents**, 214 629 pages,
**796 019 passages**, index vectoriel de 1 551 Mo.

## Mise en route

Prérequis : Python 3.11, [uv](https://docs.astral.sh/uv/), Docker, et l'archive HAS.

```bash
docker compose up -d base
uv sync --extra api --extra bdd --extra embeddings --extra onnx

uv run corpus-extraire          # archive HAS -> fixtures/corpus.json + PDF
uv run documentaliste-pages     # extraction du texte page à page
uv run documentaliste-ingerer   # découpage, encodage, insertion
uv run documentaliste-lexique   # statistiques pour BM25
uv run documentaliste-etat      # vérification

uv run uvicorn documentaliste.api.app:app --port 8005
```

L'archive n'est pas dans le dépôt : 15 Go, retéléchargeables depuis l'open data de la HAS.
`fixtures/` n'y est pas non plus — tout y est dérivé et reconstruit par les commandes
ci-dessus.

**Vérification** — les trois, jamais l'une sans les autres :

```bash
uv run ruff check src/ tests/ ; uv run ruff format --check src/ tests/ ; uv run pytest
```

## API

| route | rôle |
| --- | --- |
| `GET /health` | le processus répond, et l'état du plafond de dépense |
| `GET /perimetre` | ce que le corpus couvre, pour que l'interface l'affiche |
| `POST /question` | une question, des passages, une réponse citée |

**`POST /question` rend toujours les passages retrouvés, refus compris.** Ce n'est pas un
détail d'affichage : on a mesuré 19 refus excessifs sur 53, et un refus nu ressemble à une
panne. Un refus qui montre ce qu'il a trouvé ressemble à ce qu'il est — de la prudence.

Le service **dégrade au lieu de tomber**. Budget épuisé, clé absente ou fournisseur en
panne : la recherche continue, seule la rédaction se coupe, et le journal nomme la cause.

## Déploiement

FastAPI derrière un reverse-proxy Caddy, PostgreSQL/pgvector dans le même `docker compose`,
image publiée sur GHCR par GitHub Actions à chaque poussée sur `main`.

Le service tourne en `read_only`, sans capacités, avec un plafond mémoire et un limiteur de
débit par adresse. Les poids du modèle d'embeddings sont **cuits dans l'image** : rien n'est
téléchargé au démarrage, et le conteneur en lecture seule n'aurait de toute façon nulle part
où écrire.

La procédure détaillée n'est pas publiée — elle nomme l'hôte, le compte de service et
l'emplacement des secrets.

## Note sur ce dépôt

Il contient **l'application**, pas l'appareil de mesure. Les trente et une sondes qui ont
produit les chiffres ci-dessus — balayages de réglage, matrices de confusion, juge de
fondement, relectures humaines — vivent hors du dépôt.

Le paquet s'appelle encore `probes/` par héritage ; il ne contient plus que du code
d'exécution.

## Licence et sources

Corpus HAS sous [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
Modèle d'embeddings : [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small).
Rédaction : API Mistral.
