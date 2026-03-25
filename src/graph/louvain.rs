use std::collections::{HashMap, HashSet};

use super::load_edges;

/// Result: (entity_id, community_id, modularity)
pub struct LouvainResult {
    pub assignments: Vec<(i64, i32)>,
    pub modularity: f64,
}

/// Louvain community detection (single-level greedy phase).
pub fn louvain_communities() -> LouvainResult {
    let edges = load_edges();

    if edges.is_empty() {
        return LouvainResult {
            assignments: Vec::new(),
            modularity: 0.0,
        };
    }

    // Build weighted adjacency and collect all nodes
    let mut nodes: HashSet<i64> = HashSet::new();
    let mut adj: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
    let mut total_weight = 0.0f64;

    for (src, tgt, w) in &edges {
        nodes.insert(*src);
        nodes.insert(*tgt);
        adj.entry(*src).or_default().push((*tgt, *w));
        adj.entry(*tgt).or_default().push((*src, *w)); // undirected
        total_weight += w;
    }

    // node_degree[v] = sum of edge weights incident to v
    let mut node_degree: HashMap<i64, f64> = HashMap::new();
    for &node in &nodes {
        let deg: f64 = adj.get(&node).map(|ns| ns.iter().map(|(_, w)| w).sum()).unwrap_or(0.0);
        node_degree.insert(node, deg);
    }

    // Initialize: each node in its own community
    let node_list: Vec<i64> = nodes.iter().cloned().collect();
    let mut node_to_comm: HashMap<i64, i64> = node_list
        .iter()
        .enumerate()
        .map(|(_i, &id)| (id, id)) // community id = node id initially
        .collect();

    // community_weight[c] = sum of all edge weights inside community c (including boundary)
    let mut comm_weight: HashMap<i64, f64> = node_list
        .iter()
        .map(|&id| (id, *node_degree.get(&id).unwrap_or(&0.0)))
        .collect();

    let m2 = 2.0 * total_weight; // 2m

    // Greedy phase: iterate until no improvement
    let max_iter = 100;
    for _ in 0..max_iter {
        let mut improved = false;

        for &node in &node_list {
            let cur_comm = node_to_comm[&node];

            // Compute weights to each neighboring community
            let mut comm_weights: HashMap<i64, f64> = HashMap::new();
            if let Some(neighbors) = adj.get(&node) {
                for (neighbor, w) in neighbors {
                    let nc = node_to_comm[neighbor];
                    *comm_weights.entry(nc).or_insert(0.0) += w;
                }
            }

            let ki = node_degree[&node];
            let w_in_cur = comm_weights.get(&cur_comm).copied().unwrap_or(0.0);
            let sigma_tot_cur = comm_weight[&cur_comm];

            // Temporarily remove node from current community
            let sigma_tot_minus = sigma_tot_cur - ki;
            let gain_remove = -(w_in_cur / total_weight)
                + (sigma_tot_minus * ki) / (m2 * total_weight);

            let mut best_gain = gain_remove; // gain from staying = 0 vs gain_remove
            let mut best_comm = cur_comm;

            for (&cand_comm, &w_in_cand) in &comm_weights {
                if cand_comm == cur_comm {
                    continue;
                }
                let sigma_tot_cand = comm_weight[&cand_comm];
                let gain_add = (w_in_cand / total_weight)
                    - (sigma_tot_cand * ki) / (m2 * total_weight);

                let delta = gain_add - gain_remove;
                if delta > best_gain {
                    best_gain = delta;
                    best_comm = cand_comm;
                }
            }

            if best_comm != cur_comm {
                // Move node to best_comm
                *comm_weight.entry(cur_comm).or_insert(0.0) -= ki;
                *comm_weight.entry(best_comm).or_insert(0.0) += ki;
                node_to_comm.insert(node, best_comm);
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Renumber communities 0..N
    let mut comm_remap: HashMap<i64, i32> = HashMap::new();
    let mut next_id: i32 = 0;
    let assignments: Vec<(i64, i32)> = node_list
        .iter()
        .map(|&node| {
            let raw_comm = node_to_comm[&node];
            let new_id = *comm_remap.entry(raw_comm).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            (node, new_id)
        })
        .collect();

    let modularity = calculate_modularity(&edges, &node_to_comm, total_weight);

    LouvainResult {
        assignments,
        modularity,
    }
}

fn calculate_modularity(
    edges: &[(i64, i64, f64)],
    node_to_comm: &HashMap<i64, i64>,
    total_weight: f64,
) -> f64 {
    if total_weight == 0.0 {
        return 0.0;
    }

    let mut node_degree: HashMap<i64, f64> = HashMap::new();
    for (src, tgt, w) in edges {
        *node_degree.entry(*src).or_insert(0.0) += w;
        *node_degree.entry(*tgt).or_insert(0.0) += w;
    }

    let m = total_weight;
    let mut q = 0.0f64;

    for (src, tgt, w) in edges {
        if node_to_comm.get(src) == node_to_comm.get(tgt) {
            let ki = node_degree.get(src).copied().unwrap_or(0.0);
            let kj = node_degree.get(tgt).copied().unwrap_or(0.0);
            q += w - (ki * kj) / (2.0 * m);
        }
    }

    q / m
}
