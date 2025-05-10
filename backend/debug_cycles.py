import os
import sys
import logging
import json
from pathlib import Path
import time
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_cycles")

# Add the parent directory to path if running directly
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if parent_dir not in sys.path:
        sys.path.append(str(parent_dir))

from backend.enhanced_graph_retriever import EnhancedGraphRetriever

def load_config(vault_path=None):
    """Load configuration from config file or environment"""
    # Try to load from config.yaml in the backend directory
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            # Read the file as a string for environment variable substitution
            config_str = f.read()
            # Replace environment variables in the YAML string
            for key, value in os.environ.items():
                if value:  # Only replace if the environment variable has a value
                    config_str = config_str.replace(f"${{{key}}}", value)
            
            # Parse the YAML with substitutions
            config = yaml.safe_load(config_str)
            
            # Override vault path if provided explicitly
            if vault_path and not config.get("vault", {}).get("path"):
                if not config.get("vault"):
                    config["vault"] = {}
                config["vault"]["path"] = vault_path
            return config
    
    # Get vault path from argument, environment variable, or default
    if not vault_path:
        vault_path = os.environ.get("OBSIDIAN_VAULT_PATH", ".")
    
    # Fallback to default config
    return {
        "vault": {
            "path": vault_path
        }
    }

def detect_cycles_in_graph(vault_path):
    """Build the knowledge graph and detect cycles"""
    logger.info(f"Building knowledge graph for vault: {vault_path}")
    
    try:
        # Initialize the graph retriever
        retriever = EnhancedGraphRetriever(vault_path=vault_path)
        
        # Build the graph
        logger.info("Building graph...")
        start_time = time.time()
        retriever.build_graph()
        build_time = time.time() - start_time
        logger.info(f"Graph built in {build_time:.2f} seconds")
        
        # Print graph statistics
        num_nodes = retriever.graph.number_of_nodes()
        num_edges = retriever.graph.number_of_edges()
        num_docs = sum(1 for _, data in retriever.graph.nodes(data=True) if data.get("type") == "document")
        num_tags = sum(1 for _, data in retriever.graph.nodes(data=True) if data.get("type") == "tag")
        
        logger.info(f"Graph statistics:")
        logger.info(f"  - Total nodes: {num_nodes}")
        logger.info(f"  - Total edges: {num_edges}")
        logger.info(f"  - Document nodes: {num_docs}")
        logger.info(f"  - Tag nodes: {num_tags}")
        
        # Detect cycles in the graph
        logger.info("Detecting cycles in the graph...")
        
        # Method 1: Using NetworkX cycles finding algorithms
        cycles = list(nx.simple_cycles(retriever.graph))
        
        logger.info(f"Found {len(cycles)} cycles in the graph")
        
        # Log the first 10 cycles
        for i, cycle in enumerate(cycles[:10]):
            logger.info(f"Cycle {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
            
            # Log node types in the cycle
            cycle_node_types = [retriever.graph.nodes[node].get("type", "unknown") for node in cycle]
            logger.info(f"Node types: {cycle_node_types}")
            
            # Log edge types in the cycle
            edge_types = []
            for j in range(len(cycle)):
                source = cycle[j]
                target = cycle[(j+1) % len(cycle)]
                edge_type = retriever.graph.edges[source, target].get("type", "unknown")
                edge_types.append(edge_type)
            logger.info(f"Edge types: {edge_types}")
        
        # Check for bidirectional edges (immediate cycles of length 2)
        bidirectional_edges = []
        for u, v in retriever.graph.edges():
            if retriever.graph.has_edge(v, u):
                edge_type_forward = retriever.graph.edges[u, v].get("type", "unknown")
                edge_type_backward = retriever.graph.edges[v, u].get("type", "unknown")
                bidirectional_edges.append((u, v, edge_type_forward, edge_type_backward))
        
        logger.info(f"Found {len(bidirectional_edges)} bidirectional edges")
        
        # Log some bidirectional edges for inspection
        for i, (u, v, type_uv, type_vu) in enumerate(bidirectional_edges[:10]):
            logger.info(f"Bidirectional edge {i+1}: {u} <-({type_uv})-> {v} <-({type_vu})-> {u}")
            
            # Log node types
            u_type = retriever.graph.nodes[u].get("type", "unknown")
            v_type = retriever.graph.nodes[v].get("type", "unknown")
            logger.info(f"Node types: {u} is {u_type}, {v} is {v_type}")
        
        # Identify long cycles (potentially problematic)
        long_cycles = [cycle for cycle in cycles if len(cycle) > 5]
        logger.info(f"Found {len(long_cycles)} long cycles (length > 5)")
        
        # Find nodes with many connections (high degree)
        high_degree_nodes = [(node, retriever.graph.degree(node)) for node in retriever.graph.nodes()]
        high_degree_nodes.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Nodes with highest degree (most connections):")
        for i, (node, degree) in enumerate(high_degree_nodes[:10]):
            node_type = retriever.graph.nodes[node].get("type", "unknown")
            logger.info(f"  {i+1}. {node} ({node_type}): {degree} connections")
            
            # Count connection types
            in_edges = retriever.graph.in_edges(node, data=True)
            out_edges = retriever.graph.out_edges(node, data=True)
            
            in_edge_types = {}
            for _, _, data in in_edges:
                edge_type = data.get("type", "unknown")
                in_edge_types[edge_type] = in_edge_types.get(edge_type, 0) + 1
                
            out_edge_types = {}
            for _, _, data in out_edges:
                edge_type = data.get("type", "unknown")
                out_edge_types[edge_type] = out_edge_types.get(edge_type, 0) + 1
                
            logger.info(f"    In edges: {dict(in_edge_types)}")
            logger.info(f"    Out edges: {dict(out_edge_types)}")
        
        # Check the search space size starting from different entry points
        logger.info("Testing traversal from high-degree nodes...")
        for i, (node, degree) in enumerate(high_degree_nodes[:5]):
            if retriever.graph.nodes[node].get("type") != "document":
                continue
                
            logger.info(f"Testing traversal from {node}...")
            visited = set()
            to_visit = [(node, 0)]  # (node, distance)
            max_hops = 2
            
            while to_visit:
                current, distance = to_visit.pop(0)
                
                if current in visited:
                    continue
                    
                visited.add(current)
                
                if distance >= max_hops:
                    continue
                    
                for neighbor in retriever.graph.neighbors(current):
                    if neighbor not in visited:
                        to_visit.append((neighbor, distance + 1))
            
            logger.info(f"  From {node}: visited {len(visited)} nodes within {max_hops} hops")
                
        return {
            "success": True,
            "graph_stats": {
                "nodes": num_nodes,
                "edges": num_edges,
                "documents": num_docs,
                "tags": num_tags
            },
            "cycles": {
                "total_cycles": len(cycles),
                "bidirectional_edges": len(bidirectional_edges),
                "long_cycles": len(long_cycles),
                "example_cycles": [
                    {"nodes": cycle, "length": len(cycle)} 
                    for cycle in cycles[:10]
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting cycles: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Parse command line arguments
    vault_path = None
    
    # Check if vault path is provided
    if len(sys.argv) > 1:
        # If it looks like a path
        if os.path.exists(sys.argv[1]) or sys.argv[1].startswith("/") or sys.argv[1].startswith("./") or ":" in sys.argv[1]:
            vault_path = sys.argv[1]
    
    # Load config to get vault path (if not provided via command line)
    config = load_config(vault_path)
    vault_path = config.get("vault", {}).get("path", ".")
    logger.info(f"Using vault path: {vault_path}")
    
    # Verify vault path exists
    if not os.path.exists(vault_path):
        logger.error(f"Vault path does not exist: {vault_path}")
        sys.exit(1)
    
    # Detect cycles
    results = detect_cycles_in_graph(vault_path)
    
    # Write results to file
    output_path = Path(__file__).parent.parent / "cycle_detection_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Cycle detection results written to {output_path}")
    logger.info("Done!") 