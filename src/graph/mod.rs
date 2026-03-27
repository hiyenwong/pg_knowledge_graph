pub mod components;
pub mod louvain;
pub mod pagerank;
pub mod traversal;

use pgrx::prelude::*;

/// Load all edges from kg_relations.
/// Returns (source_id, target_id, weight).
/// Returns empty Vec if table doesn't exist.
pub(crate) fn load_edges() -> Vec<(i64, i64, f64)> {
    Spi::connect(|client| {
        let mut edges: Vec<(i64, i64, f64)> = Vec::new();
        // Use a subtransaction to handle missing table gracefully
        let result = client.select(
            "SELECT source_id, target_id, weight FROM kg_relations",
            None,
            &[],
        );
        match result {
            Ok(tup_table) => {
                for row in tup_table {
                    let src: i64 = row["source_id"].value::<i64>()?.unwrap_or(0);
                    let tgt: i64 = row["target_id"].value::<i64>()?.unwrap_or(0);
                    let wgt: f64 = row["weight"].value::<f64>()?.unwrap_or(1.0);
                    edges.push((src, tgt, wgt));
                }
                Ok::<Vec<(i64, i64, f64)>, pgrx::spi::SpiError>(edges)
            }
            Err(_) => Ok::<Vec<(i64, i64, f64)>, pgrx::spi::SpiError>(Vec::new()), // Table doesn't exist, return empty
        }
    })
    .unwrap_or_default()
}

/// Load all entity IDs from kg_entities.
/// Returns empty Vec if table doesn't exist.
pub(crate) fn load_entity_ids() -> Vec<i64> {
    Spi::connect(|client| {
        let mut ids: Vec<i64> = Vec::new();
        let result = client.select("SELECT id FROM kg_entities", None, &[]);
        match result {
            Ok(tup_table) => {
                for row in tup_table {
                    let id: i64 = row["id"].value::<i64>()?.unwrap_or(0);
                    ids.push(id);
                }
                Ok::<Vec<i64>, pgrx::spi::SpiError>(ids)
            }
            Err(_) => Ok::<Vec<i64>, pgrx::spi::SpiError>(Vec::new()), // Table doesn't exist, return empty
        }
    })
    .unwrap_or_default()
}
