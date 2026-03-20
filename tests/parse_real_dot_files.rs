use std::fs;

#[test]
fn parse_consensus_task_dot() {
    let content = fs::read_to_string("../consensus_task.dot")
        .expect("consensus_task.dot not found at project root");
    match attractor::parser::parse_dot(&content) {
        Ok(graph) => {
            println!(
                "consensus_task.dot: {} nodes, {} edges",
                graph.nodes.len(),
                graph.edges.len()
            );
            println!("  Goal: {:?}", graph.graph_attrs.goal);
            for (id, node) in &graph.nodes {
                println!(
                    "  Node: {} (shape={:?}, node_type={:?})",
                    id, node.shape, node.node_type
                );
            }
            for edge in &graph.edges {
                println!(
                    "  Edge: {} -> {} (cond={:?}, label={:?})",
                    edge.from, edge.to, edge.condition, edge.label
                );
            }
        }
        Err(e) => {
            panic!("Failed to parse consensus_task.dot: {}", e);
        }
    }
}

#[test]
fn parse_semport_dot() {
    let content =
        fs::read_to_string("../semport.dot").expect("semport.dot not found at project root");
    match attractor::parser::parse_dot(&content) {
        Ok(graph) => {
            println!(
                "semport.dot: {} nodes, {} edges",
                graph.nodes.len(),
                graph.edges.len()
            );
            println!("  Goal: {:?}", graph.graph_attrs.goal);
            for (id, node) in &graph.nodes {
                println!(
                    "  Node: {} (shape={:?}, node_type={:?})",
                    id, node.shape, node.node_type
                );
            }
            for edge in &graph.edges {
                println!(
                    "  Edge: {} -> {} (cond={:?}, label={:?})",
                    edge.from, edge.to, edge.condition, edge.label
                );
            }
        }
        Err(e) => {
            panic!("Failed to parse semport.dot: {}", e);
        }
    }
}

#[test]
fn validate_consensus_task_dot() {
    let content = fs::read_to_string("../consensus_task.dot").unwrap();
    let graph = attractor::parser::parse_dot(&content).expect("parse failed");
    let diagnostics = attractor::validation::validate(&graph, &[]);
    println!("Validation diagnostics for consensus_task.dot:");
    for d in &diagnostics {
        println!("  {:?}: {}", d.severity, d.message);
    }
}

#[test]
fn validate_semport_dot() {
    let content = fs::read_to_string("../semport.dot").unwrap();
    let graph = attractor::parser::parse_dot(&content).expect("parse failed");
    let diagnostics = attractor::validation::validate(&graph, &[]);
    println!("Validation diagnostics for semport.dot:");
    for d in &diagnostics {
        println!("  {:?}: {}", d.severity, d.message);
    }
}
