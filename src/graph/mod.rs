pub mod components;
pub mod louvain;
pub mod pagerank;
pub mod traversal;

use pgrx::prelude::*;

/// Check if a table exists in the current schema
pub(crate) fn table_exists(table_name: &str) -> bool {
    Spi::connect(|client| {
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

/// Load all edges from kg_relations.
/// Returns (source_id, target_id, weight).
/// Returns empty Vec if table doesn't exist.
pub(crate) fn load_edges() -> Vec<(i64, i64, f64)> {
    if !table_exists("kg_relations") {
        return Vec::new();
    }

    Spi::connect(|client| {
        let mut edges: Vec<(i64, i64, f64)> = Vec::new();
        let tup_table = client.select(
            "SELECT source_id, target_id, weight FROM kg_relations",
            None,
            &[],
        )?;
        for row in tup_table {
            let src: i64 = row["source_id"].value::<i64>()?.unwrap_or(0);
            let tgt: i64 = row["target_id"].value::<i64>()?.unwrap_or(0);
            let wgt: f64 = row["weight"].value::<f64>()?.unwrap_or(1.0);
            edges.push((src, tgt, wgt));
        }
        Ok::<Vec<(i64, i64, f64)>, pgrx::spi::SpiError>(edges)
    })
    .unwrap_or_default()
}

/// Load all entity IDs from kg_entities.
/// Returns empty Vec if table doesn't exist.
pub(crate) fn load_entity_ids() -> Vec<i64> {
    if !table_exists("kg_entities") {
        return Vec::new();
    }

    Spi::connect(|client| {
        let mut ids: Vec<i64> = Vec::new();
        let tup_table = client.select("SELECT id FROM kg_entities", None, &[])?;
        for row in tup_table {
            let id: i64 = row["id"].value::<i64>()?.unwrap_or(0);
            ids.push(id);
        }
        Ok::<Vec<i64>, pgrx::spi::SpiError>(ids)
    })
    .unwrap_or_default()
}
