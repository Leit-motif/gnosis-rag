import os
import sys
import logging
import json
from pathlib import Path
import time
import pprint

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("debug_graph_rag")

# Add the parent directory to path if running directly
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    if parent_dir not in sys.path:
        sys.path.append(str(parent_dir))

from backend.enhanced_graph_retriever import EnhancedGraphRetriever
from backend.graph_rag_integration import GraphRAGIntegration
from backend.rag_pipeline import RAGPipeline

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

def test_graph_retriever(vault_path, query="What are the main concepts?"):
    """Test the EnhancedGraphRetriever directly"""
    logger.info(f"Initializing EnhancedGraphRetriever with vault path: {vault_path}")
    
    # Optional custom config - you can modify these parameters for testing
    custom_config = {
        "entry_points": {
            "vector_entry_count": 3,
            "tag_entry_enabled": True,
            "entry_weight_vector": 0.7,
            "entry_weight_tags": 0.3
        },
        "traversal": {
            "max_hops": 2,
            "max_documents": 15,
            "tag_expansion_enabled": True,
            "path_expansion_enabled": True,
            "min_similarity": 0.5
        },
        "hybrid": {
            "enabled": True,
            "graph_weight": 0.6,
            "vector_weight": 0.4,
            "recency_bonus": 0.1
        }
    }
    
    try:
        # Initialize the graph retriever
        retriever = EnhancedGraphRetriever(vault_path=vault_path, config=custom_config)
        
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
        
        # Sample some nodes for inspection
        logger.info("Sample document nodes:")
        doc_nodes = [n for n, data in retriever.graph.nodes(data=True) if data.get("type") == "document"]
        for node in doc_nodes[:5]:  # Show first 5 document nodes
            logger.info(f"  - {node}")
            
        # Query the graph
        logger.info(f"Querying graph with: '{query}'")
        start_time = time.time()
        results = retriever.query(query)
        query_time = time.time() - start_time
        logger.info(f"Query completed in {query_time:.2f} seconds")
        
        # Print query results
        logger.info(f"Query results:")
        logger.info(f"  - Number of results: {len(results.get('results', []))}")
        
        # Print the first 2 results in detail
        for i, result in enumerate(results.get("results", [])[:2]):
            logger.info(f"Result {i+1}:")
            logger.info(f"  - Document: {result.get('id')}")
            logger.info(f"  - Title: {result.get('metadata', {}).get('title', 'Untitled')}")
            logger.info(f"  - Score: {result.get('score', 0)}")
            if 'distance' in result:
                logger.info(f"  - Graph distance: {result.get('distance')}")
            if 'entry_point' in result:
                logger.info(f"  - Entry point: {result.get('entry_point')}")
            if 'tags' in result:
                logger.info(f"  - Tags: {', '.join(result.get('tags', []))}")
        
        # Format context for LLM
        logger.info("Formatting context for LLM...")
        context = retriever.format_context_for_llm(results)
        logger.info(f"Context length: {len(context)} characters")
        logger.info("Context preview (first 500 chars):")
        logger.info(context[:500] + "...")
        
        return {
            "success": True,
            "graph_stats": {
                "nodes": num_nodes,
                "edges": num_edges,
                "documents": num_docs,
                "tags": num_tags
            },
            "query_results": results,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Error testing graph retriever: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

def test_rag_integration(vault_path, query="What are the main concepts?"):
    """Test the GraphRAGIntegration with RAGPipeline"""
    logger.info(f"Testing RAG integration with vault path: {vault_path}")
    
    try:
        # Load the full config (with all required sections)
        config = load_config(vault_path)
        rag_pipeline = RAGPipeline(config)
        
        # Initialize the Graph RAG integration
        graph_config = {
            "enabled": True,
            "default_mode": "hybrid",
            "graph_retriever": {
                "traversal": {
                    "max_hops": 2,
                    "max_documents": 15,
                }
            }
        }
        
        logger.info("Initializing GraphRAGIntegration...")
        graph_rag = GraphRAGIntegration(rag_pipeline, config=graph_config)
        
        # Test vector mode query
        logger.info(f"Testing vector mode query: '{query}'")
        start_time = time.time()
        vector_results = rag_pipeline.query(query, retrieval_mode="vector")
        vector_time = time.time() - start_time
        logger.info(f"Vector query completed in {vector_time:.2f} seconds")
        
        # Test graph mode query
        logger.info(f"Testing graph mode query: '{query}'")
        start_time = time.time()
        graph_results = rag_pipeline.query(query, retrieval_mode="graph")
        graph_time = time.time() - start_time
        logger.info(f"Graph query completed in {graph_time:.2f} seconds")
        
        # Test hybrid mode query
        logger.info(f"Testing hybrid mode query: '{query}'")
        start_time = time.time()
        hybrid_results = rag_pipeline.query(query, retrieval_mode="hybrid")
        hybrid_time = time.time() - start_time
        logger.info(f"Hybrid query completed in {hybrid_time:.2f} seconds")
        
        return {
            "success": True,
            "vector_results": {
                "time": vector_time,
                "num_results": len(vector_results.get("results", [])),
                "mode": vector_results.get("mode")
            },
            "graph_results": {
                "time": graph_time,
                "num_results": len(graph_results.get("results", [])),
                "mode": graph_results.get("mode")
            },
            "hybrid_results": {
                "time": hybrid_time,
                "num_results": len(hybrid_results.get("results", [])),
                "mode": hybrid_results.get("mode")
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing RAG integration: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    # Parse command line arguments
    vault_path = None
    query = "What are the main concepts?"
    mode = "both"
    
    # Check if vault path is provided
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # First argument is either vault path or query
        if os.path.exists(sys.argv[1]) or sys.argv[1].startswith("/") or sys.argv[1].startswith("./") or ":" in sys.argv[1]:
            # Looks like a path
            vault_path = sys.argv[1]
            # Shift other arguments
            if len(sys.argv) > 2:
                query = sys.argv[2]
            if len(sys.argv) > 3:
                mode = sys.argv[3]
        else:
            # First argument is the query
            query = sys.argv[1]
            if len(sys.argv) > 2:
                mode = sys.argv[2]
    
    # Load config to get vault path (if not provided via command line)
    config = load_config(vault_path)
    vault_path = config.get("vault", {}).get("path", ".")
    logger.info(f"Using vault path: {vault_path}")
    
    # Verify vault path exists
    if not os.path.exists(vault_path):
        logger.error(f"Vault path does not exist: {vault_path}")
        sys.exit(1)
    
    results = {}
    
    if mode in ["direct", "both"]:
        # Test EnhancedGraphRetriever directly
        logger.info("===== TESTING DIRECT GRAPH RETRIEVER =====")
        direct_results = test_graph_retriever(vault_path, query)
        results["direct"] = direct_results
    
    if mode in ["rag", "both"]:
        # Test GraphRAGIntegration
        logger.info("===== TESTING RAG INTEGRATION =====")
        rag_results = test_rag_integration(vault_path, query)
        results["rag"] = rag_results
    
    # Write results to file
    output_path = Path(__file__).parent.parent / "debug_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Debug results written to {output_path}")
    logger.info("Done!") 