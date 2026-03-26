use std::collections::{HashMap, HashSet, VecDeque};

use super::{load_edges, load_entity_ids};

/// Weakly connected components (treats graph as undirected).
/// Returns (entity_id, component_id), component IDs start from 0.
pub fn connected_components() -> Vec<(i64, i32)> {
    let edges = load_edges();
    let all_ids = load_entity_ids();

    // Build undirected adjacency
    let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();
    for &id in &all_ids {
        adj.entry(id).or_default();
    }
    for (src, tgt, _) in &edges {
        adj.entry(*src).or_default().push(*tgt);
        adj.entry(*tgt).or_default().push(*src);
    }

    let mut visited: HashSet<i64> = HashSet::new();
    let mut result: Vec<(i64, i32)> = Vec::new();
    let mut component_id: i32 = 0;

    let mut all_nodes: Vec<i64> = adj.keys().cloned().collect();
    all_nodes.sort();

    for &start in &all_nodes {
        if visited.contains(&start) {
            continue;
        }

        // BFS for this component
        let mut queue: VecDeque<i64> = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            result.push((node, component_id));
            if let Some(neighbors) = adj.get(&node) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        component_id += 1;
    }

    result
}

/// Strongly connected components using Kosaraju's algorithm.
/// Returns (entity_id, scc_id), sorted by scc size descending.
pub fn strongly_connected_components() -> Vec<(i64, i32)> {
    let edges = load_edges();
    let all_ids = load_entity_ids();

    let mut all_nodes: HashSet<i64> = HashSet::new();
    for &id in &all_ids {
        all_nodes.insert(id);
    }

    let mut out_adj: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut rev_adj: HashMap<i64, Vec<i64>> = HashMap::new();

    for &id in &all_nodes {
        out_adj.entry(id).or_default();
        rev_adj.entry(id).or_default();
    }
    for (src, tgt, _) in &edges {
        all_nodes.insert(*src);
        all_nodes.insert(*tgt);
        out_adj.entry(*src).or_default().push(*tgt);
        rev_adj.entry(*tgt).or_default().push(*src);
    }

    // Pass 1: compute finish order via iterative DFS
    let mut visited: HashSet<i64> = HashSet::new();
    let mut finish_order: Vec<i64> = Vec::new();

    for &node in &all_nodes {
        if !visited.contains(&node) {
            dfs_finish(&out_adj, node, &mut visited, &mut finish_order);
        }
    }

    // Pass 2: traverse reverse graph in reverse finish order
    let mut visited2: HashSet<i64> = HashSet::new();
    let mut sccs: Vec<Vec<i64>> = Vec::new();

    for &node in finish_order.iter().rev() {
        if !visited2.contains(&node) {
            let mut component: Vec<i64> = Vec::new();
            dfs_collect(&rev_adj, node, &mut visited2, &mut component);
            sccs.push(component);
        }
    }

    // Sort SCCs by size descending, then assign IDs
    sccs.sort_by_key(|b| std::cmp::Reverse(b.len()));

    let mut result: Vec<(i64, i32)> = Vec::new();
    for (scc_id, scc) in sccs.iter().enumerate() {
        for &node in scc {
            result.push((node, scc_id as i32));
        }
    }

    result
}

fn dfs_finish(
    adj: &HashMap<i64, Vec<i64>>,
    start: i64,
    visited: &mut HashSet<i64>,
    finish_order: &mut Vec<i64>,
) {
    let mut stack: Vec<(i64, usize)> = vec![(start, 0)];
    visited.insert(start);

    while let Some((node, idx)) = stack.last_mut() {
        let node = *node;
        let neighbors = adj.get(&node).map(|v| v.as_slice()).unwrap_or(&[]);
        if *idx < neighbors.len() {
            let next = neighbors[*idx];
            *idx += 1;
            if visited.insert(next) {
                stack.push((next, 0));
            }
        } else {
            finish_order.push(node);
            stack.pop();
        }
    }
}

fn dfs_collect(
    adj: &HashMap<i64, Vec<i64>>,
    start: i64,
    visited: &mut HashSet<i64>,
    component: &mut Vec<i64>,
) {
    let mut stack: Vec<i64> = vec![start];
    visited.insert(start);
    component.push(start);

    while let Some(node) = stack.pop() {
        if let Some(neighbors) = adj.get(&node) {
            for &neighbor in neighbors {
                if visited.insert(neighbor) {
                    component.push(neighbor);
                    stack.push(neighbor);
                }
            }
        }
    }
}
