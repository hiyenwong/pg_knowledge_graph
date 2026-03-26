//! Vector semantic search module.
//!
//! Provides pgvector-based semantic search functionality.

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
        let tup_table = client.select(&query, None, &[])?;

        for row in tup_table {
            let entity_id: i64 = row["id"].value::<i64>()?.unwrap_or(0);
            let entity_name: String = row["name"].value::<String>()?.unwrap_or_default();
            let entity_type: String = row["entity_type"].value::<String>()?.unwrap_or_default();
            let similarity: f64 = row["similarity"].value::<f64>()?.unwrap_or(0.0);

            results.push(VectorSearchResult {
                entity_id,
                entity_name,
                entity_type,
                similarity,
            });
        }

        Ok::<Vec<VectorSearchResult>, pgrx::spi::SpiError>(results)
    })
    .unwrap_or_default()
}
