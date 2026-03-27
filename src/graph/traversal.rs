use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};

/// Queue/stack item for traversal: (node_id, depth, parent_id, rel_type, weight)
type TraversalItem = (i64, i64, Option<i64>, Option<String>, f64);

#[derive(Debug, Serialize)]
pub struct TraversalNode {
    pub entity_id: i64,
    pub depth: i64,
    pub parent_id: Option<i64>,
    pub rel_type: Option<String>,
    pub weight: f64,
}

#[derive(Debug, Serialize)]
pub struct PathStep {
    pub entity_id: i64,
    pub rel_type: String,
    pub weight: f64,
}

#[derive(Debug, Serialize)]
pub struct ShortestPath {
    pub found: bool,
    pub from_id: i64,
    pub to_id: i64,
    pub total_cost: f64,
    pub hops: i64,
    pub path: Vec<PathStep>,
}

/// Build a directed adjacency list: node -> Vec<(neighbor, weight, rel_type)>
fn build_adjacency(edges: &[(i64, i64, f64, String)]) -> HashMap<i64, Vec<(i64, f64, String)>> {
    let mut adj: HashMap<i64, Vec<(i64, f64, String)>> = HashMap::new();
    for (src, tgt, w, rt) in edges {
        adj.entry(*src).or_default().push((*tgt, *w, rt.clone()));
    }
    adj
}

/// Load edges with rel_type for traversal queries.
/// Returns empty Vec if table doesn't exist.
fn load_edges_with_type() -> Vec<(i64, i64, f64, String)> {
    pgrx::Spi::connect(|client| {
        let mut edges: Vec<(i64, i64, f64, String)> = Vec::new();
        let result = client.select(
            "SELECT source_id, target_id, weight, rel_type FROM kg_relations",
            None,
            &[],
        );
        match result {
            Ok(tup_table) => {
                for row in tup_table {
                    let src: i64 = row["source_id"].value::<i64>()?.unwrap_or(0);
                    let tgt: i64 = row["target_id"].value::<i64>()?.unwrap_or(0);
                    let wgt: f64 = row["weight"].value::<f64>()?.unwrap_or(1.0);
                    let rt: String = row["rel_type"].value::<String>()?.unwrap_or_default();
                    edges.push((src, tgt, wgt, rt));
                }
                Ok::<Vec<(i64, i64, f64, String)>, pgrx::spi::SpiError>(edges)
            }
            Err(_) => Ok::<Vec<(i64, i64, f64, String)>, pgrx::spi::SpiError>(Vec::new()), // Table doesn't exist
        }
    })
    .unwrap_or_default()
}

/// Breadth-first search from start_id up to max_depth hops.
pub fn bfs(start_id: i64, max_depth: i64) -> Vec<TraversalNode> {
    let edges = load_edges_with_type();
    let adj = build_adjacency(&edges);

    let mut result: Vec<TraversalNode> = Vec::new();
    let mut visited: HashSet<i64> = HashSet::new();
    let mut queue: VecDeque<TraversalItem> = VecDeque::new();

    visited.insert(start_id);
    queue.push_back((start_id, 0, None, None, 0.0));

    while let Some((node, depth, parent, rel, weight)) = queue.pop_front() {
        result.push(TraversalNode {
            entity_id: node,
            depth,
            parent_id: parent,
            rel_type: rel,
            weight,
        });

        if depth >= max_depth {
            continue;
        }

        if let Some(neighbors) = adj.get(&node) {
            for (neighbor, w, rt) in neighbors {
                if visited.insert(*neighbor) {
                    queue.push_back((*neighbor, depth + 1, Some(node), Some(rt.clone()), *w));
                }
            }
        }
    }

    result
}

/// Depth-first search from start_id up to max_depth hops.
pub fn dfs(start_id: i64, max_depth: i64) -> Vec<TraversalNode> {
    let edges = load_edges_with_type();
    let adj = build_adjacency(&edges);

    let mut result: Vec<TraversalNode> = Vec::new();
    let mut visited: HashSet<i64> = HashSet::new();
    let mut stack: Vec<TraversalItem> = Vec::new();

    stack.push((start_id, 0, None, None, 0.0));

    while let Some((node, depth, parent, rel, weight)) = stack.pop() {
        if visited.contains(&node) {
            continue;
        }
        visited.insert(node);

        result.push(TraversalNode {
            entity_id: node,
            depth,
            parent_id: parent,
            rel_type: rel,
            weight,
        });

        if depth >= max_depth {
            continue;
        }

        if let Some(neighbors) = adj.get(&node) {
            // Reverse so we push in natural order (stack pops last first)
            for (neighbor, w, rt) in neighbors.iter().rev() {
                if !visited.contains(neighbor) {
                    stack.push((*neighbor, depth + 1, Some(node), Some(rt.clone()), *w));
                }
            }
        }
    }

    result
}

/// Find shortest path using BFS (by hops) with cost accumulation.
pub fn shortest_path(from_id: i64, to_id: i64, max_depth: i64) -> ShortestPath {
    let edges = load_edges_with_type();
    let adj = build_adjacency(&edges);

    // BFS: track parent and edge info for path reconstruction
    let mut visited: HashSet<i64> = HashSet::new();
    // (node, depth, accumulated_cost)
    let mut queue: VecDeque<(i64, i64, f64)> = VecDeque::new();
    // parent_map: node -> (parent_node, rel_type, weight)
    let mut parent_map: HashMap<i64, (i64, String, f64)> = HashMap::new();

    visited.insert(from_id);
    queue.push_back((from_id, 0, 0.0));

    let mut found = false;
    let mut total_cost = 0.0;
    let mut hops = 0i64;

    'outer: while let Some((node, depth, cost)) = queue.pop_front() {
        if node == to_id {
            found = true;
            total_cost = cost;
            hops = depth;
            break;
        }
        if depth >= max_depth {
            continue;
        }
        if let Some(neighbors) = adj.get(&node) {
            for (neighbor, w, rt) in neighbors {
                if visited.insert(*neighbor) {
                    parent_map.insert(*neighbor, (node, rt.clone(), *w));
                    queue.push_back((*neighbor, depth + 1, cost + w));
                    if *neighbor == to_id {
                        found = true;
                        total_cost = cost + w;
                        hops = depth + 1;
                        break 'outer;
                    }
                }
            }
        }
    }

    if !found {
        return ShortestPath {
            found: false,
            from_id,
            to_id,
            total_cost: 0.0,
            hops: 0,
            path: vec![],
        };
    }

    // Reconstruct path
    let mut path: Vec<PathStep> = Vec::new();
    let mut cur = to_id;
    while let Some((parent, rt, w)) = parent_map.get(&cur) {
        path.push(PathStep {
            entity_id: cur,
            rel_type: rt.clone(),
            weight: *w,
        });
        cur = *parent;
    }
    path.push(PathStep {
        entity_id: from_id,
        rel_type: String::new(),
        weight: 0.0,
    });
    path.reverse();

    ShortestPath {
        found: true,
        from_id,
        to_id,
        total_cost,
        hops,
        path,
    }
}
