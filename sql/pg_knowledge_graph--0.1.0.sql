-- pg_knowledge_graph schema
-- Requires: pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Entities table
CREATE TABLE IF NOT EXISTS kg_entities (
    id          BIGSERIAL PRIMARY KEY,
    entity_type TEXT      NOT NULL,
    name        TEXT      NOT NULL,
    properties  JSONB     NOT NULL DEFAULT '{}',
    embedding   vector(1536),  -- OpenAI text-embedding-ada-002/003 dimension
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities (name);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type_name ON kg_entities (entity_type, name);
CREATE INDEX IF NOT EXISTS idx_kg_entities_embedding ON kg_entities USING hnsw (embedding vector_cosine_ops);

-- Relations table
CREATE TABLE IF NOT EXISTS kg_relations (
    id          BIGSERIAL PRIMARY KEY,
    source_id   BIGINT    NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    target_id   BIGINT    NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
    rel_type    TEXT      NOT NULL,
    weight      FLOAT8    NOT NULL DEFAULT 1.0,
    properties  JSONB     NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations (source_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations (target_id);
CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations (rel_type);
CREATE INDEX IF NOT EXISTS idx_kg_relations_source_type ON kg_relations (source_id, rel_type);
