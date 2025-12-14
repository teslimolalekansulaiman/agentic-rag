DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_database WHERE datname = 'crew_agent'
   ) THEN
      CREATE DATABASE crew_agent;
   END IF;
END
$do$;

\c crew_agent

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS procurement_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1024)
);

CREATE TABLE IF NOT EXISTS procurement_docs (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    metadata JSONB
);
