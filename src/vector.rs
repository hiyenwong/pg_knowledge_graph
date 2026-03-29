//! Vector semantic search module.
//!
//! Provides pgvector-based semantic search functionality.

use crate::quantize::QuantLevel;
use pgrx::prelude::*;
use serde::Serialize;

/// Vector search result.
#[derive(Debug, Serialize)]
pub struct VectorSearchResult {
    pub entity_id: i64,
    pub entity_name: String,
    pub entity_type: String,
    pub similarity: f64,
}

/// Raw entity embedding row fetched from DB.
struct EntityEmbedding {
    id: i64,
    name: String,
    entity_type: String,
    embedding: Vec<f32>,
}

/// Check if a table exists in the current schema
fn table_exists(table_name: &str) -> bool {
    pgrx::Spi::connect(|client| {
        let query = format!(
            "SELECT EXISTS(SELECT 1 FROM pg_class WHERE relname = '{}' AND relkind = 'r')",
            table_name
        );
        match client.select(&query, None, &[]) {
            Ok(tup_table) => Ok::<bool, pgrx::spi::SpiError>(
                tup_table
                    .first()
                    .get_one::<bool>()
                    .unwrap_or(Some(false))
                    .unwrap_or(false),
            ),
            Err(_) => Ok(false),
        }
    })
    .unwrap_or(false)
}

/// Execute vector semantic search.
///
/// Returns the top-k entities most similar to the query vector.
///
/// # Arguments
/// * `query_vector` - Query embedding vector
/// * `k` - Number of results to return
///
/// # Returns
/// Vector of search results sorted by similarity (descending)
pub fn semantic_search(query_vector: Vec<f32>, k: i32) -> Vec<VectorSearchResult> {
    if query_vector.is_empty() || k <= 0 {
        return Vec::new();
    }

    if !table_exists("kg_entities") {
        return Vec::new();
    }

    // Format vector as PostgreSQL array literal
    let vector_str = format!(
        "[{}]",
        query_vector
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    // Use string interpolation for the query (safe since vector_str is controlled format)
    let query = format!(
        r#"
        SELECT id, name, entity_type, 1 - (embedding <=> '{}') AS similarity
        FROM kg_entities
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> '{}'
        LIMIT {}
        "#,
        vector_str, vector_str, k
    );

    Spi::connect(|client| {
        let mut results: Vec<VectorSearchResult> = Vec::new();
        let result = client.select(&query, None, &[]);

        match result {
            Ok(tup_table) => {
                for row in tup_table {
                    let entity_id: i64 = row["id"].value::<i64>()?.unwrap_or(0);
                    let entity_name: String = row["name"].value::<String>()?.unwrap_or_default();
                    let entity_type: String =
                        row["entity_type"].value::<String>()?.unwrap_or_default();
                    let similarity: f64 = row["similarity"].value::<f64>()?.unwrap_or(0.0);

                    results.push(VectorSearchResult {
                        entity_id,
                        entity_name,
                        entity_type,
                        similarity,
                    });
                }
                Ok::<Vec<VectorSearchResult>, pgrx::spi::SpiError>(results)
            }
            Err(_) => Ok::<Vec<VectorSearchResult>, pgrx::spi::SpiError>(Vec::new()), // Table doesn't exist
        }
    })
    .unwrap_or_default()
}

/// Load all entity embeddings from the database.
fn load_all_embeddings() -> Vec<EntityEmbedding> {
    if !table_exists("kg_entities") {
        return Vec::new();
    }

    let query = r#"
        SELECT id, name, entity_type, embedding::text
        FROM kg_entities
        WHERE embedding IS NOT NULL
    "#;

    Spi::connect(|client| {
        let mut results: Vec<EntityEmbedding> = Vec::new();
        match client.select(query, None, &[]) {
            Ok(tup_table) => {
                for row in tup_table {
                    let id: i64 = row["id"].value::<i64>()?.unwrap_or(0);
                    let name: String = row["name"].value::<String>()?.unwrap_or_default();
                    let entity_type: String =
                        row["entity_type"].value::<String>()?.unwrap_or_default();
                    let emb_text: String =
                        row["embedding"].value::<String>()?.unwrap_or_default();

                    // Parse pgvector text format "[v1,v2,...,vn]"
                    if let Some(embedding) = parse_vector_text(&emb_text) {
                        results.push(EntityEmbedding {
                            id,
                            name,
                            entity_type,
                            embedding,
                        });
                    }
                }
                Ok::<Vec<EntityEmbedding>, pgrx::spi::SpiError>(results)
            }
            Err(_) => Ok(Vec::new()),
        }
    })
    .unwrap_or_default()
}

/// Parse pgvector text format "[v1,v2,...,vn]" into Vec<f32>.
fn parse_vector_text(s: &str) -> Option<Vec<f32>> {
    let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return None;
    }
    let values: Result<Vec<f32>, _> = trimmed.split(',').map(|v| v.trim().parse::<f32>()).collect();
    values.ok().filter(|v| !v.is_empty())
}

/// Execute quantized vector search using TurboQuant-inspired algorithm.
///
/// # Algorithm
/// 1. Load all entity embeddings from `kg_entities`
/// 2. Create a `TurboQuantizer` (no training needed — uses Gaussian-optimal codebook)
/// 3. For each stored embedding: L2-normalize → sign-flip → Lloyd-Max encode
/// 4. For the query: keep as float32 (Asymmetric Distance Computation / ADC)
/// 5. Rank by approximate cosine similarity, return top-k
///
/// # Arguments
/// * `query_vector` - Query embedding vector (float32, stays unquantized)
/// * `k` - Number of results to return
/// * `level` - Quantization precision (Int8=4x, Int4=8x, Binary=32x)
pub fn quantized_search(
    query_vector: Vec<f32>,
    k: i32,
    level: QuantLevel,
) -> Vec<VectorSearchResult> {
    if query_vector.is_empty() || k <= 0 {
        return Vec::new();
    }

    let entities = load_all_embeddings();
    if entities.is_empty() {
        return Vec::new();
    }

    let query_dims = query_vector.len();

    // Build TurboQuantizer — calibration-free, works immediately
    let quantizer = crate::quantize::TurboQuantizer::new(query_dims, level, 0);

    // Quantize each stored embedding and compute ADC similarity
    let mut scored: Vec<(f64, usize)> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| e.embedding.len() == query_dims)
        .filter_map(|(idx, e)| {
            // Quantize stored vector using Lloyd-Max codebook
            let qv = quantizer.quantize(&e.embedding).ok()?;
            // Asymmetric similarity: query stays float32
            let sim = quantizer.cosine_similarity(&qv, &query_vector) as f64;
            Some((sim, idx))
        })
        .collect();

    // Sort by similarity descending, return top-k
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .take(k as usize)
        .map(|(sim, idx)| VectorSearchResult {
            entity_id: entities[idx].id,
            entity_name: entities[idx].name.clone(),
            entity_type: entities[idx].entity_type.clone(),
            similarity: sim,
        })
        .collect()
}
