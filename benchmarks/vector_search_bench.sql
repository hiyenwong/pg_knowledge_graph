-- Benchmark: Vector Search Performance
-- Usage: psql -f benchmarks/vector_search_bench.sql

-- Setup: Create benchmark tables if not exist
\echo 'Setting up benchmark data...'

-- Generate test embeddings (1536 dimensions)
-- Insert 10000 entities with random embeddings for benchmark
INSERT INTO kg_entities (entity_type, name, embedding)
SELECT
    'benchmark',
    'entity_' || i,
    array_agg(random())::vector(1536)
FROM generate_series(1, 10000) i
CROSS JOIN generate_series(1, 1536) j
GROUP BY i
ON CONFLICT DO NOTHING;

-- Generate some relations for hybrid search benchmark
INSERT INTO kg_relations (source_id, target_id, rel_type, weight)
SELECT
    (i % 10000) + 1,
    ((i + floor(random() * 100)) % 10000) + 1,
    'relates_to',
    random()
FROM generate_series(1, 50000) i
ON CONFLICT DO NOTHING;

-- Create a query vector
\set query_vec '[' || repeat('0.1,', 1535) || '0.1]'

\echo 'Running benchmarks...'
\echo ''

-- Benchmark 1: Vector search latency
\echo '=== Benchmark 1: Vector Search (k=10) ==='
\timing on
SELECT COUNT(*) FROM kg_vector_search(:query_vec::vector, 10);
\timing off

-- Benchmark 2: Vector search with larger k
\echo ''
\echo '=== Benchmark 2: Vector Search (k=100) ==='
\timing on
SELECT COUNT(*) FROM kg_vector_search(:query_vec::vector, 100);
\timing off

-- Benchmark 3: Hybrid search
\echo ''
\echo '=== Benchmark 3: Hybrid Search (k=10, depth=2) ==='
\timing on
SELECT COUNT(*) FROM kg_hybrid_search(:query_vec::vector, 10, 2, 0.7, 0.3);
\timing off

-- Benchmark 4: Hybrid search with deeper graph
\echo ''
\echo '=== Benchmark 4: Hybrid Search (k=10, depth=4) ==='
\timing on
SELECT COUNT(*) FROM kg_hybrid_search(:query_vec::vector, 10, 4, 0.7, 0.3);
\timing off

-- Benchmark 5: Context extraction
\echo ''
\echo '=== Benchmark 5: Context Extraction (depth=2) ==='
\timing on
SELECT kg_get_context(1, 2);
\timing off

-- Benchmark 6: Raw vector distance (baseline)
\echo ''
\echo '=== Benchmark 6: Raw Vector Distance (baseline, k=10) ==='
\timing on
SELECT id, 1 - (embedding <=> :query_vec::vector) AS similarity
FROM kg_entities
WHERE embedding IS NOT NULL
ORDER BY embedding <=> :query_vec::vector
LIMIT 10;
\timing off

-- Memory stats
\echo ''
\echo '=== Memory Statistics ==='
SELECT
    pg_size_pretty(pg_total_relation_size('kg_entities')) AS entities_size,
    pg_size_pretty(pg_total_relation_size('kg_relations')) AS relations_size,
    (SELECT COUNT(*) FROM kg_entities) AS entity_count,
    (SELECT COUNT(*) FROM kg_relations) AS relation_count;

-- Cleanup (optional - uncomment to remove benchmark data)
-- DELETE FROM kg_entities WHERE entity_type = 'benchmark';