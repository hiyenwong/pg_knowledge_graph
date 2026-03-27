//! Standalone benchmark for vector operations.
//!
//! This binary can be run without PostgreSQL to measure baseline performance.
//!
//! Run with: cargo run --release

use std::time::Instant;

fn main() {
    println!("==============================================");
    println!("pg_knowledge_graph Benchmark Suite");
    println!("Phase 3 Baseline (No Compression)");
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