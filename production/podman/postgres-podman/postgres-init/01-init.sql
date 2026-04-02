-- file di inizializzazione che si avvia solo quando la directory postgres-data è ancora vuota 
-- Crea l'estensione pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Verifica l'installazione
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Crea il database msgdb se non esiste 
CREATE DATABASE msgdb;



-- Crea l'enum dei ruoli e l'estensione nel database msgdb
-- connetto al db
\c msgdb
CREATE TYPE message_role AS ENUM('user', 'assistant', 'system', 'tool');
-- Crea l'estensione pgvector
CREATE EXTENSION IF NOT EXISTS vector;
-- Verifica l'installazione
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';