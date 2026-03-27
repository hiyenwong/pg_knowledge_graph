# Phase 3 Benchmark Results

**Date**: 2026-03-27
**Platform**: macOS (Apple Silicon)
**Rust**: release profile (opt-level=3)

## Configuration

| Parameter | Value |
|-----------|-------|
| Vector Dimensions | 1536 (OpenAI embedding size) |
| Number of Vectors | 10,000 |
| k (top-k) | 10 |
| Iterations | 5 |

## Results

### 1. Vector Generation

| Metric | Value |
|--------|-------|
| Time | 142 ms |
| Memory | 58.59 MB |

### 2. Cosine Distance (Single Operation)

| Metric | Value |
|--------|-------|
| Average Time | 4,147.58 ns (4.15 µs) |
| Throughput | 241,104 distances/sec |

### 3. Brute-Force k-NN Search

| Metric | Value |
|--------|-------|
| Average Time | 42.66 ms |
| Throughput | 234,430 vectors/sec |
| Latency (k=10) | 42.66 ms |

| Iteration | Time (ms) |
|-----------|-----------|
| 1 | 43.24 |
| 2 | 42.56 |
| 3 | 41.85 |
| 4 | 43.69 |
| 5 | 41.94 |

## Memory Estimates

| Format | Size | Compression Ratio |
|--------|------|-------------------|
| Raw (float32) | 58.59 MB | 1x |
| 8-bit quantized | 14.65 MB | 4x |
| 4-bit quantized | 7.32 MB | 8x |
| Binary quantized | 1.83 MB | 32x |

## Expected Improvements with TurboQuant

| Quantization | Search Speed | Recall Impact |
|--------------|--------------|---------------|
| 8-bit | 2-3x faster | ~0% loss |
| 4-bit | 4-6x faster | ~2% loss |
| Binary | 8-10x faster | ~5% loss |

## Baseline for Phase 4

These results serve as the **baseline** for comparing TurboQuant implementation:

- **Search Latency**: 42.66 ms (brute-force, 10k vectors)
- **Memory Usage**: 58.59 MB (10k vectors × 1536 dimensions)
- **Throughput**: 234,430 vectors/sec

### Target Improvements (Phase 4)

With TurboQuant:
- Latency: < 15 ms (3x improvement)
- Memory: < 15 MB (4x compression)
- Recall: > 95% (vs exact search)

## Notes

1. These benchmarks measure **pure computation** without PostgreSQL overhead
2. Real-world performance with pgvector HNSW will differ
3. Actual TurboQuant performance depends on dataset characteristics