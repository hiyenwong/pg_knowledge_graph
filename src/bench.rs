//! Benchmark module for vector search and graph algorithms.
//!
//! Run with: cargo bench --features pg_test

#[cfg(test)]
mod bench_tests {
    use std::time::Instant;

    /// Generate a random vector of given dimensions
    fn random_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rand::random::<f32>()).collect()
    }

    /// Generate multiple random vectors
    fn generate_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..count).map(|_| random_vector(dim)).collect()
    }

    /// Cosine distance between two vectors
    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - dot / (mag_a * mag_b + 1e-10)
    }

    /// Brute-force k-NN search
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

    #[test]
    fn bench_brute_force_knn() {
        let dimensions = 1536;
        let num_vectors = 10000;
        let k = 10;
        let iterations = 5;

        println!("\n=== Brute-Force k-NN Benchmark ===");
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
            let results = brute_force_knn(&vectors, &query, k);
            let duration = start.elapsed().as_micros();
            total_time += duration;

            println!(
                "Iteration {}: {:.2} ms (found {} results)",
                i + 1,
                duration as f64 / 1000.0,
                results.len()
            );
        }

        let avg_time = total_time as f64 / iterations as f64 / 1000.0;
        println!("\nAverage time: {:.2} ms", avg_time);
        println!(
            "Throughput: {:.0} vectors/sec",
            num_vectors as f64 / (avg_time / 1000.0)
        );
    }

    #[test]
    fn bench_vector_generation() {
        let dimensions = 1536;
        let count = 10000;

        println!("\n=== Vector Generation Benchmark ===");
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
    }

    #[test]
    fn bench_cosine_distance() {
        let dimensions = 1536;
        let iterations = 10000;

        let a = random_vector(dimensions);
        let b = random_vector(dimensions);

        println!("\n=== Cosine Distance Benchmark ===");
        println!("Dimensions: {}", dimensions);
        println!("Iterations: {}", iterations);

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = cosine_distance(&a, &b);
        }
        let duration = start.elapsed().as_micros();
        let avg_ns = duration as f64 / iterations as f64 * 1000.0;

        println!("Total time: {} µs", duration);
        println!("Average: {:.2} ns per distance", avg_ns);
        println!(
            "Throughput: {:.0} distances/sec",
            iterations as f64 / (duration as f64 / 1_000_000.0)
        );
    }
}
