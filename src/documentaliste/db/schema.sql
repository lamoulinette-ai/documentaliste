-- Schéma du documentaliste : les documents de la HAS, leurs pages et leurs passages.

CREATE EXTENSION IF NOT EXISTS vector;

-- Un PDF publié par la HAS, avec ce que sa publication en dit.
CREATE TABLE IF NOT EXISTS document (
    nom               text PRIMARY KEY,
    fiche             text NOT NULL,
    titre             text NOT NULL,
    type_publication  text NOT NULL DEFAULT '',
    -- Dates conservées en texte, telles que la HAS les publie.
    mise_en_ligne     text NOT NULL DEFAULT '',
    validation        text NOT NULL DEFAULT '',
    url_pdf           text NOT NULL DEFAULT '',
    url_fiche         text NOT NULL DEFAULT '',
    themes            text[] NOT NULL DEFAULT '{}'
);

-- Le texte d'une page, tel que PyMuPDF le lit.
CREATE TABLE IF NOT EXISTS page (
    document   text NOT NULL REFERENCES document (nom) ON DELETE CASCADE,
    numero     integer NOT NULL,
    texte      text NOT NULL,
    -- Forme repliée du début du texte, sur laquelle se compte la répétition.
    empreinte  text NOT NULL,
    PRIMARY KEY (document, numero)
);

CREATE INDEX IF NOT EXISTS page_empreinte_idx ON page (empreinte);

-- Un fragment citable, rattaché à sa page.
CREATE TABLE IF NOT EXISTS passage (
    id         bigserial PRIMARY KEY,
    document   text NOT NULL REFERENCES document (nom) ON DELETE CASCADE,
    page       integer NOT NULL,
    -- Rang du passage dans la page : le recouvrement en produit plusieurs.
    rang       integer NOT NULL,
    texte      text NOT NULL,
    -- 384 dimensions : « intfloat/multilingual-e5-small ».
    vecteur    vector(384) NOT NULL,
    UNIQUE (document, page, rang)
);

CREATE INDEX IF NOT EXISTS passage_document_idx ON passage (document);

-- Recherche plein texte : sert à **trouver** des candidats, jamais à les classer.
-- PostgreSQL racinise et filtre les mots vides à sa façon ; le classement reste calculé
-- par notre BM25, sur les candidats que cette colonne ramène.
ALTER TABLE passage
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('french', texte)) STORED;

CREATE INDEX IF NOT EXISTS passage_tsv_idx ON passage USING gin (tsv);

-- Nombre de passages contenant chaque terme, selon **notre** découpage en mots.
-- BM25 a besoin de cette statistique sur le corpus entier : la calculer sur les seuls
-- candidats donnerait d'autres scores que la version en mémoire.
CREATE TABLE IF NOT EXISTS statistique_terme (
    terme     text PRIMARY KEY,
    passages  integer NOT NULL
);

-- Nombre de passages contenant chaque lexème, selon la racinisation de **PostgreSQL**.
-- Distincte de `statistique_terme`, qui suit notre découpage : les deux ne dénombrent pas
-- les mêmes unités et ne sont pas interchangeables. Celle-ci sert à écarter de la requête
-- disjonctive les racines trop répandues pour discriminer quoi que ce soit.
CREATE TABLE IF NOT EXISTS statistique_lexeme (
    lexeme    text PRIMARY KEY,
    passages  integer NOT NULL
);

-- Une seule ligne : nombre de passages et longueur moyenne, en mots normalisés.
CREATE TABLE IF NOT EXISTS statistique_corpus (
    unique_    boolean PRIMARY KEY DEFAULT true CHECK (unique_),
    passages   integer NOT NULL,
    longueur_moyenne double precision NOT NULL
);

-- Index vectoriel HNSW sur le produit scalaire : les vecteurs sont normalisés.
CREATE INDEX IF NOT EXISTS passage_vecteur_idx
    ON passage USING hnsw (vecteur vector_ip_ops);
