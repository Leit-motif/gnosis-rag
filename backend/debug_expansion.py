import os
import sys
import logging
import json
from pathlib import Path
import time
import networkx as nx
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_expansion")

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

class DebugGraphRetriever(EnhancedGraphRetriever):
    """Extended version of EnhancedGraphRetriever with debugging capabilities"""
    
    def debug_expand_neighborhood(self, doc_id, entry_score=0.9, max_distance=2):
        """Debug version of _expand_neighborhood"""
        logger.info(f"Testing neighborhood expansion from document: {doc_id}")
        
        # Check if document exists
        if not self.graph.has_node(doc_id):
            logger.error(f"Document {doc_id} does not exist in the graph")
            return None
        
        logger.info(f"Document node type: {self.graph.nodes[doc_id].get('type')}")
        logger.info(f"Document connections: {self.graph.degree(doc_id)}")
        
        # Prepare for expansion
        discovered_docs = {}
        visited = set()
        
        # Add initial document as entry point
        discovered_docs[doc_id] = {
            "document": doc_id,
            "score": entry_score,
            "distance": 0,
            "entry_point": True,
            "connections": [],
            "tags": self.graph.nodes[doc_id].get("tags", [])
        }
        
        # Start expansion
        logger.info(f"Starting expansion from {doc_id} with max_distance={max_distance}")
        
        # Log each step of the expansion process
        def debug_expand(current_doc, current_distance, visited):
            """Debug version of expansion with detailed logging"""
            logger.info(f"[{current_distance}] Expanding from: {current_doc}")
            
            # Log if already visited
            if current_doc in visited:
                logger.info(f"  - Already visited {current_doc}, skipping")
                return

            # Check if node is a document
            node_type = self.graph.nodes[current_doc].get("type")
            if node_type != "document":
                logger.info(f"  - Node {current_doc} is a {node_type}, not a document, skipping expansion")
                return
            
            # Mark as visited to prevent cycles
            visited.add(current_doc)
            
            # Examine all neighbors to collect 1-hop away nodes
            logger.info(f"  - Examining neighbors of {current_doc}")
            for neighbor in self.graph.neighbors(current_doc):
                edge_data = self.graph.edges[current_doc, neighbor]
                edge_type = edge_data.get("type", "unknown")
                neighbor_type = self.graph.nodes[neighbor].get("type", "unknown")
                
                logger.info(f"    - Neighbor: {neighbor} (Type: {neighbor_type}), Edge Type: {edge_type}")
                
                # If this is a tag node, we can expand it to find more documents
                if neighbor_type == "tag" and self.config["traversal"]["tag_expansion_enabled"]:
                    # Check if we've already visited this tag
                    if neighbor in visited:
                        logger.info(f"      - Tag already visited, skipping")
                        continue
                    
                    # Mark tag as visited
                    visited.add(neighbor)
                    
                    # Find documents with this tag
                    tag_docs = []
                    tag_edges = 0
                    for tag_neighbor in self.graph.neighbors(neighbor):
                        tag_edge_data = self.graph.edges[neighbor, tag_neighbor]
                        tag_edge_type = tag_edge_data.get("type", "unknown")
                        tag_neighbor_type = self.graph.nodes[tag_neighbor].get("type", "unknown")
                        
                        logger.info(f"      - Tag neighbor: {tag_neighbor} (Type: {tag_neighbor_type}), Edge Type: {tag_edge_type}")
                        
                        if tag_neighbor_type == "document" and tag_edge_type == "tagged_document":
                            tag_docs.append(tag_neighbor)
                            tag_edges += 1
                    
                    logger.info(f"      - Tag {neighbor} connects to {tag_edges} documents")
                    logger.info(f"      - First few connected docs: {tag_docs[:5]}")
                    
                # Skip to next neighbor if this isn't a document node
                if neighbor_type != "document":
                    logger.info(f"      - Not a document node, skipping")
                    continue
                
                # Calculate new distance for this neighbor
                next_distance = current_distance + 1
                
                # If we're at max distance, don't go further
                if next_distance > max_distance:
                    logger.info(f"      - Max distance reached ({next_distance} > {max_distance}), stopping expansion")
                    continue
                
                # Calculate score based on distance and edge type
                neighbor_score = entry_score * (1.0 - (0.3 * next_distance))
                if edge_type == "linked_from":
                    neighbor_score *= 0.9  # Slightly lower weight for backlinks
                    logger.info(f"      - Backlink score adjustment: {neighbor_score}")
                    
                # Create connection info
                connection = {
                    "from": current_doc,
                    "to": neighbor,
                    "type": edge_type,
                    "distance": next_distance
                }
                
                # Record or update this document
                if neighbor in discovered_docs:
                    # Already found through another path, update if better score
                    logger.info(f"      - Document {neighbor} already discovered, checking for better score")
                    if neighbor_score > discovered_docs[neighbor]["score"]:
                        logger.info(f"      - Updating score: {discovered_docs[neighbor]['score']} -> {neighbor_score}")
                        discovered_docs[neighbor]["score"] = neighbor_score
                        discovered_docs[neighbor]["distance"] = min(discovered_docs[neighbor]["distance"], next_distance)
                        
                    # Add connection if not duplicate
                    existing_connections = [c["from"] for c in discovered_docs[neighbor]["connections"]]
                    if current_doc not in existing_connections:
                        logger.info(f"      - Adding new connection from {current_doc}")
                        discovered_docs[neighbor]["connections"].append(connection)
                else:
                    # New document discovered
                    logger.info(f"      - New document discovered: {neighbor} (Score: {neighbor_score}, Distance: {next_distance})")
                    discovered_docs[neighbor] = {
                        "document": neighbor,
                        "score": neighbor_score,
                        "distance": next_distance,
                        "entry_point": False,
                        "connections": [connection],
                        "tags": self.graph.nodes[neighbor].get("tags", [])
                    }
                    
                    # Continue expansion from this node
                    logger.info(f"      - Continuing expansion from {neighbor}")
                    debug_expand(neighbor, next_distance, visited)
            
            logger.info(f"[{current_distance}] Completed expansion from: {current_doc}")
        
        # Start expansion from the entry point
        debug_expand(doc_id, 0, visited)
        
        # Log expansion results
        logger.info(f"Expansion complete. Discovered {len(discovered_docs)} documents:")
        for i, (doc, data) in enumerate(discovered_docs.items()):
            logger.info(f"  {i+1}. {doc} (Score: {data['score']:.3f}, Distance: {data['distance']})")
            
        # Create a subgraph for visualization
        subgraph_nodes = list(discovered_docs.keys())
        for data in discovered_docs.values():
            for connection in data.get("connections", []):
                if connection["from"] not in subgraph_nodes:
                    subgraph_nodes.append(connection["from"])
                if connection["to"] not in subgraph_nodes:
                    subgraph_nodes.append(connection["to"])
        
        # Get the subgraph with these nodes
        subgraph = self.graph.subgraph(subgraph_nodes)
        logger.info(f"Created subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        
        # Export subgraph to JSON for visualization
        subgraph_data = {
            "nodes": [],
            "links": []
        }
        
        for node in subgraph.nodes():
            node_type = subgraph.nodes[node].get("type", "unknown")
            node_data = {
                "id": node,
                "type": node_type,
                "is_entry": node == doc_id,
                "distance": discovered_docs.get(node, {}).get("distance", -1) if node_type == "document" else -1,
                "score": discovered_docs.get(node, {}).get("score", 0) if node_type == "document" else 0
            }
            subgraph_data["nodes"].append(node_data)
            
        for u, v, data in subgraph.edges(data=True):
            edge_data = {
                "source": u,
                "target": v,
                "type": data.get("type", "unknown")
            }
            subgraph_data["links"].append(edge_data)
            
        return {
            "discovered_docs": discovered_docs,
            "subgraph": subgraph_data,
            "visited": list(visited)
        }

def debug_expansion(vault_path):
    """Debug the neighborhood expansion mechanism"""
    logger.info(f"Debugging neighborhood expansion for vault: {vault_path}")
    
    try:
        # Initialize the debug graph retriever
        retriever = DebugGraphRetriever(vault_path=vault_path)
        
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
        
        # Select some documents for testing expansion
        doc_nodes = [n for n, data in retriever.graph.nodes(data=True) if data.get("type") == "document"]
        
        if not doc_nodes:
            logger.error("No document nodes found in the graph!")
            return {"success": False, "error": "No document nodes found"}
        
        # Try a few different entry points
        results = {}
        
        # Start with some high-connectivity nodes
        high_degree_nodes = [(node, retriever.graph.degree(node)) for node in doc_nodes]
        high_degree_nodes.sort(key=lambda x: x[1], reverse=True)
        high_conn_docs = [node for node, degree in high_degree_nodes[:3] if degree > 2]
        
        # Also try some random documents 
        random_docs = random.sample(doc_nodes, min(3, len(doc_nodes)))
        
        # Combine both sets
        test_docs = high_conn_docs + [d for d in random_docs if d not in high_conn_docs]
        test_docs = test_docs[:5]  # Limit to 5 tests
        
        for i, doc_id in enumerate(test_docs):
            logger.info(f"=== Test {i+1}: Expanding from {doc_id} ===")
            expansion_result = retriever.debug_expand_neighborhood(doc_id)
            
            if expansion_result:
                results[doc_id] = {
                    "num_discovered": len(expansion_result["discovered_docs"]),
                    "num_visited": len(expansion_result["visited"]),
                    "expansion_result": expansion_result
                }
        
        return {
            "success": True,
            "graph_stats": {
                "nodes": num_nodes,
                "edges": num_edges,
                "documents": num_docs,
                "tags": num_tags
            },
            "expansion_tests": results
        }
        
    except Exception as e:
        logger.error(f"Error debugging expansion: {str(e)}", exc_info=True)
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
    
    # Debug expansion
    results = debug_expansion(vault_path)
    
    # Write results to file
    output_path = Path(__file__).parent.parent / "expansion_debug_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Expansion debug results written to {output_path}")
    logger.info("Done!") 