# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pg_knowledge_graph` is a PostgreSQL extension built with **Rust + pgrx** framework. It adds graph algorithm capabilities on top of pgvector — think "pgvector with graph traversal". It does **not** re-implement vector storage; it reuses pgvector's HNSW/IVF indexes.

Reference implementation (SQLite version to port from): https://github.com/hiyenwong/sqlite-knowledge-graph

## Core Commands

### Build & Test
```bash
# Install pgrx CLI (version must match Cargo.toml exactly)
cargo install cargo-pgrx --version "=0.17.0" --locked

# Initialize pgrx using existing Postgres.app PG18
cargo pgrx init --pg18 /Applications/Postgres.app/Contents/Versions/18/bin/pg_config

# On macOS: bindgen needs SDK headers (required for cargo check / cargo pgrx test)
export SDKROOT=$(xcrun --show-sdk-path)
export BINDGEN_EXTRA_CLANG_ARGS="-I$(xcrun --show-sdk-path)/usr/include"

# Type-check only (fast)
cargo check --no-default-features --features pg18

# Run all tests
cargo pgrx test pg18

# Run a single test function
cargo pgrx test pg18 -- test_kg_version

# Package the extension for installation
cargo pgrx package --no-default-features --features pg18
```

### Lint & Format
```bash
cargo clippy --no-default-features --features pg18 -- -D warnings
cargo fmt
```

### Install Extension Locally
```bash
cargo pgrx install --no-default-features --features pg18
# Then in psql:
# CREATE EXTENSION pg_knowledge_graph;
```

## Architecture

### Key Files

| File | Purpose |
|------|---------|
| `src/lib.rs` | Entry point — all `#[pg_extern]` SQL functions registered here |
| `src/graph/mod.rs` | Shared SPI helpers: `load_edges()`, `load_entity_ids()` |
| `src/graph/traversal.rs` | BFS, DFS, shortest path (Dijkstra-BFS hybrid) |
| `src/graph/pagerank.rs` | Iterative PageRank with dangling node correction |
| `src/graph/louvain.rs` | Greedy single-level Louvain community detection |
| `src/graph/components.rs` | Weakly connected (BFS) + strongly connected (Kosaraju) |
| `sql/kg--0.1.0.sql` | DDL for `kg_entities` and `kg_relations` tables |
| `pg_knowledge_graph.control` | Extension metadata (version, requires, relocatable) |
| `Cargo.toml` | pgrx 0.17.0, serde_json, thiserror. Supports PG16/17/18. |
| `.github/workflows/ci.yml` | CI matrix: PG 16/17/18 via `cargo pgrx test` |

### Data Model

```
kg_entities (id BIGSERIAL, entity_type TEXT, name TEXT, properties JSONB, embedding VECTOR, created_at, updated_at)
kg_relations (id BIGSERIAL, source_id → kg_entities, target_id → kg_entities, rel_type TEXT, weight FLOAT8, properties JSONB, created_at)
```

Both tables use `ON DELETE CASCADE` foreign keys. Indexes on `entity_type`, `name`, `source_id`, `target_id`.

### Graph Algorithm Modules (planned under `src/algorithms/`)

- `pagerank.rs` — iterative PageRank on `kg_relations` weight graph
- `community.rs` — Louvain community detection
- `traversal.rs` — BFS/DFS, shortest path (Dijkstra), connected components

### Vector/RAG Integration

- Hybrid search: graph traversal + cosine similarity via pgvector
- `pg_vector` crate used to handle `VECTOR` type in Rust
- Exposed as SQL functions: `kg_graph_vector_search(query_vec VECTOR, depth INT)`

### pgrx Conventions

- All SQL-callable functions use `#[pg_extern]` attribute
- Use `pgrx::Json` for JSONB return types
- Use `pgrx::pg_sys::Oid` for type OIDs when needed
- Test functions live in `#[cfg(any(test, feature = "pg_test"))]` blocks
- `pg_test` feature is required for `cargo pgrx test` to discover tests

## Development Phases

- **Phase 1** (current): Schema DDL, `kg_version()`, `kg_stats()`, CI setup
- **Phase 2**: Graph algorithm implementations (PageRank, Louvain, BFS/DFS, shortest path)
- **Phase 3**: pgvector integration, hybrid graph+vector search
