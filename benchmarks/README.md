# Benchmark Suite

This directory contains benchmarks for measuring pg_knowledge_graph performance.

## Quick Start

```bash
# Run SQL benchmarks (requires extension to be installed)
./benchmarks/run_benchmarks.sh

# Run Rust unit benchmarks
cargo test --features pg_test bench_ -- --nocapture
```

## Benchmark Types

### 1. SQL Benchmarks (`vector_search_bench.sql`)

Measures end-to-end performance of:
- `kg_vector_search()` — Vector similarity search
- `kg_hybrid_search()` — Vector + graph hybrid search
- `kg_get_context()` — Context extraction

### 2. Rust Benchmarks (`src/bench.rs`)

Measures algorithmic performance without PostgreSQL overhead:
- Brute-force k-NN search
- Cosine distance computation
- Vector generation

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--entities` | 10000 | Number of test entities |
| `--dimensions` | 1536 | Vector dimensions |
| `--iterations` | 5 | Iterations per benchmark |
| `--pg` | pg18 | PostgreSQL version |

## Output

Results are written to `benchmark_output.txt` and include:
- Latency (ms) per operation
- Throughput (ops/sec)
- Memory usage

## Baseline Metrics (for Phase 4 comparison)

| Metric | Baseline (float32) | Target (TurboQuant) |
|--------|-------------------|---------------------|
| Vector size | 6144 bytes | ~3 bits = ~576 bytes |
| k-NN latency | TBD | TBD |
| Recall@10 | 100% | ≥99% |

Run benchmarks after CI passes to establish baseline values.