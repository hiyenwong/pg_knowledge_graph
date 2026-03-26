//! Hybrid search module combining vector similarity and graph structure.
//!
//! Provides RAG (Retrieval-Augmented Generation) support through hybrid search.

use crate::graph::traversal::bfs;
use serde::Serialize;

/// Hybrid search result combining vector and graph scores.
#[derive(Debug, Serialize)]
pub struct HybridSearchResult {
    pub entity_id: i64,
    pub entity_name: String,
    pub entity_type: String,
    pub vector_score: f64,
    pub graph_score: f64,
    pub combined_score: f64,
    pub neighbors_count: i64,
}

/// Execute hybrid search: vector similarity + graph structure weighting.
///
/// # Arguments
/// * `query_vector` - Query embedding vector
/// * `k` - Number of results to return
/// * `graph_depth` - Depth for graph traversal (BFS)
/// * `alpha` - Weight for vector score (default 0.7)
/// * `beta` - Weight for graph score (default 0.3)
///
/// # Returns
/// Vector of hybrid search results sorted by combined score (descending)
pub fn hybrid_search(
    query_vector: Vec<f32>,
    k: i32,
    graph_depth: i32,
    alpha: f64,
    beta: f64,
) -> Vec<HybridSearchResult> {
    use crate::vector::semantic_search;

    // Fetch more candidates for re-ranking
    let vector_results = semantic_search(query_vector, k * 2);

    let mut results: Vec<HybridSearchResult> = vector_results
        .iter()
        .take(k as usize)
        .map(|vr| {
            // Use BFS to compute local graph structure
            let neighbors = bfs(vr.entity_id, graph_depth as i64);
            let neighbors_count = (neighbors.len() as i64).saturating_sub(1); // Exclude self

            // Graph score: normalized by connection density
            // Higher connectivity = lower graph score (penalize hubs)
            let graph_score = if neighbors_count > 0 {
                1.0 / (1.0 + neighbors_count as f64 * 0.1)
            } else {
                0.5 // Default score for isolated nodes
            };

            let combined_score = alpha * vr.similarity + beta * graph_score;

            HybridSearchResult {
                entity_id: vr.entity_id,
                entity_name: vr.entity_name.clone(),
                entity_type: vr.entity_type.clone(),
                vector_score: vr.similarity,
                graph_score,
                combined_score,
                neighbors_count,
            }
        })
        .collect();

    // Sort by combined score (descending)
    results.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results
}

/// Get N-hop context for an entity (for RAG prompt construction).
///
/// # Arguments
/// * `entity_id` - Center entity ID
/// * `depth` - Traversal depth
///
/// # Returns
/// JSON structure containing context nodes and metadata
pub fn get_context(entity_id: i64, depth: i32) -> serde_json::Value {
    let nodes = bfs(entity_id, depth as i64);

    serde_json::json!({
        "center_entity_id": entity_id,
        "depth": depth,
        "context_nodes": nodes.iter().map(|n| {
            serde_json::json!({
                "entity_id": n.entity_id,
                "depth": n.depth,
                "parent_id": n.parent_id,
                "rel_type": n.rel_type,
                "weight": n.weight,
            })
        }).collect::<Vec<_>>(),
        "total_nodes": nodes.len(),
    })
}
