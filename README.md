# pg_knowledge_graph

PostgreSQL extension that adds **graph algorithm capabilities** to complement pgvector. Think of it as "pgvector with graph traversal".

## Features

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

## Requirements

- PostgreSQL 16, 17, or 18
- Rust 1.75+
- [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx) 0.12.9

## Installation

```bash
cargo install cargo-pgrx --version "=0.12.9"
cargo pgrx init --pg17 $(which pg_config)   # or --pg16 / --pg18
cargo pgrx install
```

Then in `psql`:

```sql
CREATE EXTENSION pg_knowledge_graph;
```

## Quick Start

```sql
-- Create entities
INSERT INTO kg_entities (entity_type, name, properties)
VALUES ('person', 'Alice', '{"age": 30}'),
       ('person', 'Bob',   '{"age": 25}'),
       ('person', 'Carol', '{"age": 28}');

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
```

## Development

```bash
# Run tests against PG17
cargo pgrx test pg17

# Run a single test
cargo pgrx test pg17 -- test_kg_version

# Lint
cargo clippy --features pg17 -- -D warnings
cargo fmt
```

## Architecture

```
src/
├── lib.rs                  # #[pg_extern] entry points for all SQL functions
└── graph/
    ├── mod.rs              # Shared SPI helpers (load_edges, load_entity_ids)
    ├── traversal.rs        # BFS, DFS, shortest path
    ├── pagerank.rs         # Iterative PageRank with dangling node handling
    ├── louvain.rs          # Greedy Louvain community detection
    └── components.rs       # Weakly/strongly connected components (Kosaraju)
sql/
└── kg--0.1.0.sql           # DDL: kg_entities, kg_relations, indexes
```

Data layer is accessed entirely via `pgrx::Spi` — no external database drivers.
