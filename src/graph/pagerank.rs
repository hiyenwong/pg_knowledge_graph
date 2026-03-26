use std::collections::{HashMap, HashSet};

use super::{load_edges, load_entity_ids};

/// PageRank scores, sorted by score descending.
/// Returns (entity_id, score).
pub fn pagerank(damping: f64, max_iter: i32) -> Vec<(i64, f64)> {
    let edges = load_edges();
    let all_node_ids = load_entity_ids();

    let mut all_nodes: HashSet<i64> = HashSet::new();
    for id in &all_node_ids {
        all_nodes.insert(*id);
    }
    for (src, tgt, _) in &edges {
        all_nodes.insert(*src);
        all_nodes.insert(*tgt);
    }

    if all_nodes.is_empty() {
        return Vec::new();
    }

    let mut out_edges: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut in_edges: HashMap<i64, Vec<i64>> = HashMap::new();

    for (src, tgt, _) in &edges {
        out_edges.entry(*src).or_default().push(*tgt);
        in_edges.entry(*tgt).or_default().push(*src);
    }

    let n = all_nodes.len() as f64;
    let initial_score = 1.0 / n;
    let tolerance = 1e-6;

    let mut scores: HashMap<i64, f64> = all_nodes.iter().map(|&id| (id, initial_score)).collect();
    let mut new_scores: HashMap<i64, f64> = HashMap::with_capacity(all_nodes.len());

    for _ in 0..max_iter {
        // Dangling nodes contribute their score uniformly
        let dangling_sum: f64 = all_nodes
            .iter()
            .filter(|&&id| out_edges.get(&id).is_none_or(|e| e.is_empty()))
            .map(|&id| scores[&id])
            .sum();

        for &node in &all_nodes {
            let incoming: f64 = in_edges
                .get(&node)
                .map(|sources| {
                    sources
                        .iter()
                        .map(|&from| {
                            let out_deg = out_edges.get(&from).map_or(1, |e| e.len().max(1)) as f64;
                            scores[&from] / out_deg
                        })
                        .sum()
                })
                .unwrap_or(0.0);

            new_scores.insert(
                node,
                (1.0 - damping) / n + damping * (incoming + dangling_sum / n),
            );
        }

        let diff: f64 = all_nodes
            .iter()
            .map(|&id| (scores[&id] - new_scores[&id]).abs())
            .sum();

        std::mem::swap(&mut scores, &mut new_scores);

        if diff < tolerance {
            break;
        }
    }

    let mut result: Vec<(i64, f64)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}
