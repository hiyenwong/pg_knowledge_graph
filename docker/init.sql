-- ============================================================
-- pg_knowledge_graph — test initialization
-- ============================================================

-- Load required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_knowledge_graph;

-- ── Schema DDL ───────────────────────────────────────────────
-- pgrx generates function stubs only; create tables explicitly
CREATE TABLE IF NOT EXISTS kg_entities (
    id          BIGSERIAL PRIMARY KEY,
    entity_type TEXT      NOT NULL,
    name        TEXT      NOT NULL,
    properties  JSONB     NOT NULL DEFAULT '{}',
    embedding   vector(1536),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities (name);
CREATE INDEX IF NOT EXISTS idx_kg_entities_type_name ON kg_entities (entity_type, name);
CREATE INDEX IF NOT EXISTS idx_kg_entities_embedding ON kg_entities USING hnsw (embedding vector_cosine_ops);

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

-- ── Sanity checks ────────────────────────────────────────────
SELECT kg_version();
SELECT kg_stats();
SELECT kg_quantize_info();

-- ── Seed test entities ───────────────────────────────────────
INSERT INTO kg_entities (entity_type, name, properties) VALUES
    ('person',  'Alice',   '{"role": "engineer"}'),
    ('person',  'Bob',     '{"role": "manager"}'),
    ('person',  'Charlie', '{"role": "designer"}'),
    ('company', 'Acme',    '{"industry": "tech"}'),
    ('company', 'Globex',  '{"industry": "finance"}');

-- ── Seed test relations ──────────────────────────────────────
INSERT INTO kg_relations (source_id, target_id, rel_type, weight) VALUES
    (1, 2, 'reports_to',  1.0),
    (3, 2, 'reports_to',  1.0),
    (1, 4, 'works_at',    1.0),
    (2, 4, 'works_at',    1.0),
    (3, 5, 'works_at',    1.0),
    (4, 5, 'partners_with', 0.5);

-- ── Graph algorithm smoke tests ───────────────────────────────
-- BFS from Alice (id=1) up to depth 3
SELECT * FROM kg_bfs(1, 3);

-- DFS from Alice
SELECT * FROM kg_dfs(1, 3);

-- Shortest path Alice → Charlie
SELECT kg_shortest_path(1, 3, 5);

-- PageRank
SELECT * FROM kg_pagerank(0.85, 100) ORDER BY score DESC;

-- Louvain communities
SELECT * FROM kg_louvain();

-- Connected components
SELECT * FROM kg_connected_components();

-- Stats after seeding
SELECT kg_stats();
