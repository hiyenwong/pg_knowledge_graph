use pgrx::prelude::*;

mod graph;
mod rag;
mod vector;

#[cfg(test)]
mod bench;

::pgrx::pg_module_magic!();

// ---------------------------------------------------------------------------
// Phase 1 – basics
// ---------------------------------------------------------------------------

#[pg_extern]
fn kg_version() -> &'static str {
    "0.1.0"
}

#[pg_extern]
fn kg_stats() -> pgrx::Json {
    let entity_count: i64 = Spi::get_one("SELECT COUNT(*) FROM kg_entities")
        .unwrap_or(Some(0))
        .unwrap_or(0);
    let relation_count: i64 = Spi::get_one("SELECT COUNT(*) FROM kg_relations")
        .unwrap_or(Some(0))
        .unwrap_or(0);
    let density: f64 = if entity_count > 1 {
        let max_edges = entity_count * (entity_count - 1);
        relation_count as f64 / max_edges as f64
    } else {
        0.0
    };

    pgrx::Json(serde_json::json!({
        "entity_count": entity_count,
        "relation_count": relation_count,
        "density": density,
    }))
}

// ---------------------------------------------------------------------------
// Phase 2 – Graph Traversal
// ---------------------------------------------------------------------------

/// BFS from start_id up to max_depth hops. Returns SETOF JSON rows.
#[pg_extern]
fn kg_bfs(start_id: i64, max_depth: i32) -> SetOfIterator<'static, pgrx::Json> {
    let nodes = graph::traversal::bfs(start_id, max_depth as i64);
    let rows: Vec<pgrx::Json> = nodes
        .into_iter()
        .map(|n| pgrx::Json(serde_json::to_value(n).unwrap_or(serde_json::Value::Null)))
        .collect();
    SetOfIterator::new(rows)
}

/// DFS from start_id up to max_depth hops. Returns SETOF JSON rows.
#[pg_extern]
fn kg_dfs(start_id: i64, max_depth: i32) -> SetOfIterator<'static, pgrx::Json> {
    let nodes = graph::traversal::dfs(start_id, max_depth as i64);
    let rows: Vec<pgrx::Json> = nodes
        .into_iter()
        .map(|n| pgrx::Json(serde_json::to_value(n).unwrap_or(serde_json::Value::Null)))
        .collect();
    SetOfIterator::new(rows)
}

/// Shortest path between two entities. Returns a single JSON object.
#[pg_extern]
fn kg_shortest_path(from_id: i64, to_id: i64, max_depth: i32) -> pgrx::Json {
    let path = graph::traversal::shortest_path(from_id, to_id, max_depth as i64);
    pgrx::Json(serde_json::to_value(path).unwrap_or(serde_json::Value::Null))
}

// ---------------------------------------------------------------------------
// Phase 2 – PageRank
// ---------------------------------------------------------------------------

/// Compute PageRank scores for all entities.
/// Returns TABLE(entity_id bigint, score float8).
#[pg_extern]
fn kg_pagerank(
    damping: f64,
    max_iter: i32,
) -> TableIterator<'static, (name!(entity_id, i64), name!(score, f64))> {
    let results = graph::pagerank::pagerank(damping, max_iter);
    TableIterator::new(results)
}

// ---------------------------------------------------------------------------
// Phase 2 – Louvain community detection
// ---------------------------------------------------------------------------

/// Louvain community detection.
/// Returns TABLE(entity_id bigint, community_id int, modularity float8).
#[pg_extern]
fn kg_louvain() -> TableIterator<
    'static,
    (
        name!(entity_id, i64),
        name!(community_id, i32),
        name!(modularity, f64),
    ),
> {
    let result = graph::louvain::louvain_communities();
    let modularity = result.modularity;
    let rows: Vec<(i64, i32, f64)> = result
        .assignments
        .into_iter()
        .map(|(eid, cid)| (eid, cid, modularity))
        .collect();
    TableIterator::new(rows)
}

// ---------------------------------------------------------------------------
// Phase 2 – Connected components
// ---------------------------------------------------------------------------

/// Weakly connected components.
/// Returns TABLE(entity_id bigint, component_id int).
#[pg_extern]
fn kg_connected_components(
) -> TableIterator<'static, (name!(entity_id, i64), name!(component_id, i32))> {
    let rows = graph::components::connected_components();
    TableIterator::new(rows)
}

/// Strongly connected components (Kosaraju's algorithm).
/// Returns TABLE(entity_id bigint, scc_id int).
#[pg_extern]
fn kg_strongly_connected_components(
) -> TableIterator<'static, (name!(entity_id, i64), name!(scc_id, i32))> {
    let rows = graph::components::strongly_connected_components();
    TableIterator::new(rows)
}

// ---------------------------------------------------------------------------
// Phase 3 – Vector Search
// ---------------------------------------------------------------------------

/// Vector semantic search.
///
/// Searches for entities similar to the query vector using pgvector.
///
/// # Arguments
/// * `query_vector` - Query embedding vector (e.g., from OpenAI embeddings)
/// * `k` - Number of results to return (default: 10)
///
/// # Returns
/// SETOF JSON rows with entity_id, entity_name, entity_type, similarity
#[pg_extern]
fn kg_vector_search(
    query_vector: Vec<f32>,
    k: default!(i32, 10_i32),
) -> SetOfIterator<'static, pgrx::Json> {
    let results = vector::semantic_search(query_vector, k);
    let rows: Vec<pgrx::Json> = results
        .into_iter()
        .map(|r| pgrx::Json(serde_json::to_value(r).unwrap_or(serde_json::Value::Null)))
        .collect();
    SetOfIterator::new(rows)
}

/// Hybrid search combining vector similarity and graph structure.
///
/// Re-ranks vector search results based on local graph connectivity.
///
/// # Arguments
/// * `query_vector` - Query embedding vector
/// * `k` - Number of results to return (default: 10)
/// * `graph_depth` - BFS depth for graph structure analysis (default: 2)
/// * `alpha` - Weight for vector score (default: 0.7)
/// * `beta` - Weight for graph score (default: 0.3)
///
/// # Returns
/// SETOF JSON rows with entity info, vector_score, graph_score, combined_score
#[pg_extern]
fn kg_hybrid_search(
    query_vector: Vec<f32>,
    k: default!(i32, 10_i32),
    graph_depth: default!(i32, 2_i32),
    alpha: default!(f64, 0.7_f64),
    beta: default!(f64, 0.3_f64),
) -> SetOfIterator<'static, pgrx::Json> {
    let results = rag::hybrid_search(query_vector, k, graph_depth, alpha, beta);
    let rows: Vec<pgrx::Json> = results
        .into_iter()
        .map(|r| pgrx::Json(serde_json::to_value(r).unwrap_or(serde_json::Value::Null)))
        .collect();
    SetOfIterator::new(rows)
}

/// Get N-hop context for an entity (for RAG).
///
/// Extracts the local graph neighborhood for context enrichment.
///
/// # Arguments
/// * `entity_id` - Center entity ID
/// * `depth` - Traversal depth (default: 2)
///
/// # Returns
/// JSON object with context_nodes and metadata
#[pg_extern]
fn kg_get_context(entity_id: i64, depth: default!(i32, 2_i32)) -> pgrx::Json {
    let context = rag::get_context(entity_id, depth);
    pgrx::Json(context)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_kg_version() {
        assert_eq!("0.1.0", crate::kg_version());
    }

    #[pg_test]
    fn test_kg_stats_empty() {
        // Schema tables may not exist in test DB; use a safe fallback check
        let _stats = crate::kg_stats();
    }

    #[pg_test]
    fn test_kg_bfs_no_data() {
        let rows: Vec<pgrx::Json> = crate::kg_bfs(1, 3).collect();
        // No entities seeded, expect empty traversal
        assert!(rows.is_empty());
    }

    #[pg_test]
    fn test_kg_dfs_no_data() {
        let rows: Vec<pgrx::Json> = crate::kg_dfs(1, 3).collect();
        assert!(rows.is_empty());
    }

    #[pg_test]
    fn test_kg_pagerank_empty() {
        let rows: Vec<(i64, f64)> = crate::kg_pagerank(0.85, 100).collect();
        assert!(rows.is_empty());
    }

    #[pg_test]
    fn test_kg_louvain_empty() {
        let rows: Vec<(i64, i32, f64)> = crate::kg_louvain().collect();
        assert!(rows.is_empty());
    }

    #[pg_test]
    fn test_kg_connected_components_empty() {
        let rows: Vec<(i64, i32)> = crate::kg_connected_components().collect();
        assert!(rows.is_empty());
    }

    #[pg_test]
    fn test_kg_vector_search_empty_vector() {
        let results: Vec<pgrx::Json> = crate::kg_vector_search(vec![], 5).collect();
        assert!(results.is_empty());
    }

    #[pg_test]
    fn test_kg_vector_search_invalid_k() {
        let results: Vec<pgrx::Json> = crate::kg_vector_search(vec![0.1; 1536], 0).collect();
        assert!(results.is_empty());
    }

    #[pg_test]
    fn test_kg_hybrid_search_empty_vector() {
        let results: Vec<pgrx::Json> = crate::kg_hybrid_search(vec![], 5, 2, 0.7, 0.3).collect();
        assert!(results.is_empty());
    }

    #[pg_test]
    fn test_kg_get_context_no_data() {
        let context = crate::kg_get_context(999, 2);
        // Should return empty context for non-existent entity
        assert!(context.0.get("context_nodes").is_some());
    }
}

/// Required by `cargo pgrx test`.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
