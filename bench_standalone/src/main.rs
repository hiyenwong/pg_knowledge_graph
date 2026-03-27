//! Standalone benchmark for vector operations.
//!
//! This binary can be run without PostgreSQL to measure baseline performance.
//!
//! Run with: cargo run --release

use std::time::Instant;

fn main() {
    println!("==============================================");
    println!("pg_knowledge_graph Benchmark Suite");
    println!("Phase 3+4 (Vector + Quantization)");
    println!("==============================================\n");

    // Benchmark parameters
    let dimensions = 1536;
    let num_vectors = 10000;
    let k = 10;
    let iterations = 5;
    let distance_iterations = 100000;

    bench_vector_generation(dimensions, num_vectors);
    bench_cosine_distance(dimensions, distance_iterations);
    bench_brute_force_knn(dimensions, num_vectors, k, iterations);
    bench_memory_estimates(dimensions, num_vectors);

    // Phase 4: Quantization benchmarks
    println!("\n==============================================");
    println!("Phase 4: Quantization Benchmarks");
    println!("==============================================\n");

    bench_quantize_vectors(dimensions, num_vectors);
    bench_quantized_distance(dimensions);
    bench_recall_comparison(dimensions, num_vectors, k);

    println!("\n==============================================");
    println!("Benchmark Complete!");
    println!("==============================================");
}

fn random_vector(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::random::<f32>()).collect()
}

fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| random_vector(dim)).collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (mag_a * mag_b + 1e-10)
}

fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_distance(query, v)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

fn bench_vector_generation(dimensions: usize, count: usize) {
    println!("=== Vector Generation Benchmark ===");
    println!("Count: {}", count);
    println!("Dimensions: {}", dimensions);

    let start = Instant::now();
    let vectors = generate_vectors(count, dimensions);
    let duration = start.elapsed().as_millis();

    println!("Generated {} vectors in {} ms", vectors.len(), duration);
    println!(
        "Memory estimate: {:.2} MB",
        (count * dimensions * 4) as f64 / 1024.0 / 1024.0
    );
    println!();
}

fn bench_cosine_distance(dimensions: usize, iterations: usize) {
    println!("=== Cosine Distance Benchmark ===");
    println!("Dimensions: {}", dimensions);
    println!("Iterations: {}", iterations);

    let a = random_vector(dimensions);
    let b = random_vector(dimensions);

    // Warm-up
    for _ in 0..100 {
        let _ = cosine_distance(&a, &b);
    }

    let mut sum = 0.0f32;
    let start = Instant::now();
    for _ in 0..iterations {
        sum += cosine_distance(&a, &b);
    }
    let duration = start.elapsed().as_micros();
    let avg_ns = duration as f64 / iterations as f64 * 1000.0;

    // Prevent optimization
    if sum < 0.0 {
        println!("Unexpected result");
    }

    println!("Total time: {} µs", duration);
    println!("Average: {:.2} ns per distance", avg_ns);
    println!(
        "Throughput: {:.0} distances/sec",
        iterations as f64 / (duration as f64 / 1_000_000.0)
    );
    println!();
}

fn bench_brute_force_knn(dimensions: usize, num_vectors: usize, k: usize, iterations: usize) {
    println!("=== Brute-Force k-NN Benchmark ===");
    println!("Vectors: {}", num_vectors);
    println!("Dimensions: {}", dimensions);
    println!("k: {}", k);
    println!("Iterations: {}", iterations);
    println!();

    let vectors = generate_vectors(num_vectors, dimensions);
    let query = random_vector(dimensions);

    let mut total_time = 0u128;

    for i in 0..iterations {
        let start = Instant::now();
        let r = brute_force_knn(&vectors, &query, k);
        let duration = start.elapsed().as_micros();
        total_time += duration;
        println!(
            "Iteration {}: {:.2} ms (found {} results)",
            i + 1,
            duration as f64 / 1000.0,
            r.len()
        );
    }

    let avg_time = total_time as f64 / iterations as f64 / 1000.0;
    println!("\nAverage time: {:.2} ms", avg_time);
    println!(
        "Throughput: {:.0} vectors/sec",
        num_vectors as f64 / (avg_time / 1000.0)
    );
    println!();
}

fn bench_memory_estimates(dimensions: usize, num_vectors: usize) {
    println!("=== Memory Estimates ===");

    let raw_size_bytes = num_vectors * dimensions * 4;
    let raw_size_mb = raw_size_bytes as f64 / 1024.0 / 1024.0;

    // TurboQuant targets (from paper):
    // - 8-bit quantization: ~4x compression
    // - 4-bit quantization: ~8x compression
    // - Binary quantization: ~32x compression

    println!("Raw vector data: {:.2} MB", raw_size_mb);
    println!("With 8-bit quantization: {:.2} MB (4x compression)", raw_size_mb / 4.0);
    println!("With 4-bit quantization: {:.2} MB (8x compression)", raw_size_mb / 8.0);
    println!("With binary quantization: {:.2} MB (32x compression)", raw_size_mb / 32.0);
    println!();

    // Estimated query throughput improvement
    println!("Expected improvements with quantization:");
    println!("  - 8-bit: ~2-3x faster search");
    println!("  - 4-bit: ~4-6x faster search");
    println!("  - Binary: ~8-10x faster search (with ~5% recall drop)");
    println!();
}

// ============================================================================
// Phase 4: Quantization Benchmarks
// ============================================================================

/// Simple scalar quantizer for benchmarking
struct SimpleQuantizer {
    mins: Vec<f32>,
    maxs: Vec<f32>,
    dims: usize,
}

impl SimpleQuantizer {
    fn train(vectors: &[Vec<f32>]) -> Self {
        let dims = vectors[0].len();
        let mut mins = vec![f32::MAX; dims];
        let mut maxs = vec![f32::MIN; dims];

        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                if val < mins[i] { mins[i] = val; }
                if val > maxs[i] { maxs[i] = val; }
            }
        }

        Self { mins, maxs, dims }
    }

    fn quantize_int8(&self, vector: &[f32]) -> Vec<u8> {
        vector.iter().enumerate().map(|(i, &val)| {
            let normalized = (val - self.mins[i]) / (self.maxs[i] - self.mins[i]);
            (normalized * 255.0).round().clamp(0.0, 255.0) as u8
        }).collect()
    }

    fn cosine_distance_int8(&self, quantized: &[u8], query: &[f32]) -> f32 {
        let dot: f32 = quantized.iter().enumerate().map(|(i, &q)| {
            let dequant = self.mins[i] + (q as f32 / 255.0) * (self.maxs[i] - self.mins[i]);
            dequant * query[i]
        }).sum();

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let quant_norm: f32 = quantized.iter().enumerate().map(|(i, &q)| {
            let dequant = self.mins[i] + (q as f32 / 255.0) * (self.maxs[i] - self.mins[i]);
            dequant * dequant
        }).sum::<f32>().sqrt();

        1.0 - dot / (query_norm * quant_norm + 1e-10)
    }
}

fn bench_quantize_vectors(dimensions: usize, num_vectors: usize) {
    println!("=== Quantization Benchmark ===");
    println!("Vectors: {}", num_vectors);
    println!("Dimensions: {}", dimensions);

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dimensions).map(|_| rand::random::<f32>()).collect())
        .collect();

    // Train quantizer
    let start = Instant::now();
    let quantizer = SimpleQuantizer::train(&vectors);
    let train_time = start.elapsed();

    // Quantize all vectors
    let start = Instant::now();
    let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| quantizer.quantize_int8(v)).collect();
    let quantize_time = start.elapsed();

    println!("Training time: {:?}", train_time);
    println!("Quantize {} vectors: {:?}", num_vectors, quantize_time);
    println!("Per vector: {:.2} µs", quantize_time.as_micros() as f64 / num_vectors as f64);

    // Memory comparison
    let original_size = num_vectors * dimensions * 4;
    let quantized_size = num_vectors * dimensions;
    println!("Original size: {:.2} MB", original_size as f64 / 1024.0 / 1024.0);
    println!("Quantized size: {:.2} MB", quantized_size as f64 / 1024.0 / 1024.0);
    println!("Compression: {:.1}x", original_size as f64 / quantized_size as f64);
    println!();
}

fn bench_quantized_distance(dimensions: usize) {
    println!("=== Quantized Distance Benchmark ===");
    println!("Dimensions: {}", dimensions);

    let iterations = 10000;

    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..dimensions).map(|_| rand::random::<f32>()).collect())
        .collect();

    let quantizer = SimpleQuantizer::train(&vectors);
    let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| quantizer.quantize_int8(v)).collect();

    // Warm-up
    for _ in 0..100 {
        let _ = quantizer.cosine_distance_int8(&quantized[0], &vectors[0]);
    }

    // Benchmark quantized distance
    let start = Instant::now();
    for _ in 0..iterations {
        for q in &quantized {
            let _ = quantizer.cosine_distance_int8(q, &vectors[0]);
        }
    }
    let quantized_time = start.elapsed();

    // Benchmark exact distance
    let start = Instant::now();
    for _ in 0..iterations {
        for v in &vectors {
            let _ = cosine_distance(&vectors[0], v);
        }
    }
    let exact_time = start.elapsed();

    println!("Exact distance ({} iterations): {:?}", iterations, exact_time);
    println!("Quantized distance ({} iterations): {:?}", iterations, quantized_time);
    println!("Speedup: {:.2}x", exact_time.as_nanos() as f64 / quantized_time.as_nanos() as f64);
    println!();
}

fn bench_recall_comparison(dimensions: usize, num_vectors: usize, k: usize) {
    println!("=== Recall Comparison ===");
    println!("Vectors: {}", num_vectors);
    println!("k: {}", k);

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dimensions).map(|_| rand::random::<f32>()).collect())
        .collect();

    let quantizer = SimpleQuantizer::train(&vectors);
    let quantized: Vec<Vec<u8>> = vectors.iter().map(|v| quantizer.quantize_int8(v)).collect();

    let mut total_recall = 0.0;
    let num_queries = 100;

    for q in 0..num_queries {
        let query = &vectors[q];

        // Exact k-NN
        let mut exact_distances: Vec<(usize, f32)> = vectors.iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(query, v)))
            .collect();
        exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_neighbors: std::collections::HashSet<usize> = exact_distances.iter()
            .take(k)
            .map(|(i, _)| *i)
            .collect();

        // Quantized k-NN
        let mut quant_distances: Vec<(usize, f32)> = quantized.iter()
            .enumerate()
            .map(|(i, q)| (i, quantizer.cosine_distance_int8(q, query)))
            .collect();
        quant_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let quant_neighbors: std::collections::HashSet<usize> = quant_distances.iter()
            .take(k)
            .map(|(i, _)| *i)
            .collect();

        // Recall = intersection / k
        let intersection = exact_neighbors.intersection(&quant_neighbors).count();
        total_recall += intersection as f64 / k as f64;
    }

    let avg_recall = total_recall / num_queries as f64;
    println!("Average recall@{}: {:.2}%", k, avg_recall * 100.0);
    println!();
}