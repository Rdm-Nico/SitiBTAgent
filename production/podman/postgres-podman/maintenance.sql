-- connect to db
\c msgdb

-- drop table commesse 
DROP TABLE IF EXISTS commesse CASCADE;
-- drop table commerciali
DROP TABLE IF EXISTS followups CASCADE;

\echo 'Tabella commesse droppata con successo';