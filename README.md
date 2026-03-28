# pg_knowledge_graph

PostgreSQL extension that adds **graph algorithm capabilities** to complement pgvector. Think of it as "pgvector with graph traversal".

## Features

### Graph Operations

| Function | Description |
|---|---|
| `kg_version()` | Returns extension version |
| `kg_stats()` | Returns entity count, relation count, graph density |
| `kg_bfs(start_id, max_depth)` | Breadth-first traversal, returns `SETOF json` |
| `kg_dfs(start_id, max_depth)` | Depth-first traversal, returns `SETOF json` |
| `kg_shortest_path(from_id, to_id, max_depth)` | BFS shortest path as `json` |
| `kg_pagerank(damping, max_iter)` | PageRank scores, returns `TABLE(entity_id, score)` |
| `kg_louvain()` | Louvain community detection, returns `TABLE(entity_id, community_id, modularity)` |
| `kg_connected_components()` | Weakly connected components |
| `kg_strongly_connected_components()` | Kosaraju's SCC |

### Vector Search (Phase 3)

| Function | Description |
|---|---|
| `kg_vector_search(query_vector, k)` | Semantic search using pgvector cosine similarity |
| `kg_hybrid_search(query_vector, k, graph_depth, alpha, beta)` | Hybrid search combining vector similarity + graph structure |
| `kg_get_context(entity_id, depth)` | Extract N-hop neighborhood for RAG context enrichment |

### Vector Quantization (Phase 4 - TurboQuant)

| Function | Description |
|---|---|
| `kg_quantized_search(query_vector, k)` | Fast approximate search with Int8 quantization |
| `kg_quantize_info()` | Returns available quantization levels and compression ratios |

**Quantization Levels:**

| Level | Compression | Speedup | Recall Loss |
|-------|-------------|---------|-------------|
| Int8  | 4x          | 2-3x    | ~0%         |
| Int4  | 8x          | 4-6x    | ~2%         |
| Binary| 32x         | 8-10x   | ~5%         |

## Requirements

- PostgreSQL 16, 17, or 18
- Rust 1.75+
- [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx) 0.17.0
- [pgvector](https://github.com/pgvector/pgvector) (for vector search)

## Installation

```bash
cargo install cargo-pgrx --version "=0.17.0" --locked
cargo pgrx init --pg18 /Applications/Postgres.app/Contents/Versions/18/bin/pg_config
cargo pgrx install --no-default-features --features pg18
```

Then in `psql`:

```sql
CREATE EXTENSION pgvector;  -- Required for vector search
CREATE EXTENSION pg_knowledge_graph;
```

## Quick Start

```sql
-- Create entities with embeddings
INSERT INTO kg_entities (entity_type, name, properties, embedding)
VALUES ('person', 'Alice', '{"age": 30}', '[0.1, 0.2, ...]'::vector),
       ('person', 'Bob',   '{"age": 25}', '[0.3, 0.4, ...]'::vector),
       ('person', 'Carol', '{"age": 28}', '[0.5, 0.6, ...]'::vector);

-- Create relations
INSERT INTO kg_relations (source_id, target_id, rel_type, weight)
VALUES (1, 2, 'knows', 1.0),
       (2, 3, 'knows', 0.8);

-- BFS from Alice with depth 2
SELECT * FROM kg_bfs(1, 2);

-- PageRank
SELECT * FROM kg_pagerank(0.85, 100) ORDER BY score DESC;

-- Community detection
SELECT * FROM kg_louvain();

-- Shortest path Alice -> Carol
SELECT kg_shortest_path(1, 3, 5);

-- Vector search (find similar entities)
SELECT * FROM kg_vector_search('[0.1, 0.2, ...]'::vector, 10);

-- Hybrid search (vector + graph structure)
SELECT * FROM kg_hybrid_search('[0.1, 0.2, ...]'::vector, 10, 2, 0.7, 0.3);

-- Quantized search (faster, approximate)
SELECT * FROM kg_quantized_search('[0.1, 0.2, ...]'::vector, 10);
```

## Development

```bash
# Set up environment (macOS)
export SDKROOT=$(xcrun --show-sdk-path)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(xcrun --show-sdk-path)/usr/include"

# Run tests against PG18
cargo pgrx test pg18

# Run a single test
cargo pgrx test pg18 -- test_kg_version

# Lint and format
cargo clippy --no-default-features --features pg18 -- -D warnings
cargo fmt
```

## Architecture

```
src/
├── lib.rs                  # #[pg_extern] entry points for all SQL functions
├── quantize.rs             # TurboQuant scalar quantization (Int8/Int4/Binary)
├── graph/
│   ├── mod.rs              # Shared SPI helpers (load_edges, load_entity_ids)
│   ├── traversal.rs        # BFS, DFS, shortest path
│   ├── pagerank.rs         # Iterative PageRank with dangling node handling
│   ├── louvain.rs          # Greedy Louvain community detection
│   └── components.rs       # Weakly/strongly connected components (Kosaraju)
├── vector.rs               # pgvector integration, semantic search
└── rag.rs                  # Hybrid search, context extraction for RAG
sql/
└── pg_knowledge_graph--0.1.0.sql  # DDL: kg_entities, kg_relations, indexes
```

Data layer is accessed entirely via `pgrx::Spi` — no external database drivers.

## Development Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Schema DDL, version, stats, CI setup |
| Phase 2 | ✅ Complete | Graph algorithms (BFS/DFS, PageRank, Louvain, SCC) |
| Phase 3 | ✅ Complete | pgvector integration, hybrid search, RAG context |
| Phase 4 | ✅ Complete | TurboQuant vector quantization (Int8/Int4/Binary) |

## License

MIT
