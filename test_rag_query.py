#!/usr/bin/env python
import os
import json
import logging
from pathlib import Path
import yaml
from backend.rag_pipeline import RAGPipeline
from backend.enhanced_graph_retriever import EnhancedGraphRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml"""
    try:
        config_path = Path("backend/config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                # Replace environment variables with actual values or defaults
                for section in config:
                    if isinstance(config[section], dict):
                        for key, value in config[section].items():
                            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                                env_var = value[2:-1]
                                config[section][key] = os.environ.get(env_var, "")
                return config
        
        # Fall back to a basic configuration
        return {
            "vector_store": {
                "dimension": 1536,
                "index_path": "data/vector_store/faiss.index"
            },
            "embeddings": {
                "model": "text-embedding-ada-002" 
            },
            "vault": {
                "path": "data/test"  # Adjust to your test vault path
            },
            "chat_model": "gpt-4",
            "conversation_memory": {
                "storage_dir": "data/conversations",
                "max_history": 10,
                "context_window": 5
            }
        }
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def test_graph_retriever(query, vault_path):
    """Test the EnhancedGraphRetriever directly"""
    logger.info(f"Testing EnhancedGraphRetriever with query: {query}")
    
    # Initialize and build graph
    retriever = EnhancedGraphRetriever(vault_path)
    retriever.build_graph()
    
    # Create mock vector results as entry points
    mock_results = []  # We'll let the retriever find entry points based on tags/context
    
    # Query the graph
    results = retriever.query(query, vector_results=mock_results)
    
    # Print formatted context
    formatted_context = retriever.format_context_for_llm(results)
    
    print("\n" + "="*50)
    print("GRAPH RETRIEVER RESULTS")
    print("="*50)
    print("\nQuery:", query)
    print(f"\nFound {len(results.get('results', []))} results")
    
    # Print document info
    for i, doc in enumerate(results.get("results", []), 1):
        print(f"\n[{i}] Document: {doc['id']}")
        print(f"  Score: {doc.get('score', 0):.4f}")
        print(f"  Title: {doc['metadata'].get('title', 'N/A')}")
        
        # Print only essential metadata
        metadata = doc.get("metadata", {})
        if "tags" in metadata and metadata["tags"]:
            print(f"  Tags: {', '.join(metadata['tags'][:3])}")
        
        # Print metadata size to verify optimization
        metadata_size = len(json.dumps(metadata))
        print(f"  Metadata size: {metadata_size} bytes")
    
    print("\n" + "="*50)
    print("FORMATTED CONTEXT FOR LLM:")
    print("="*50)
    print(formatted_context[:1000] + "..." if len(formatted_context) > 1000 else formatted_context)
    
    return results

def test_full_rag_pipeline(query, config):
    """Test the full RAG pipeline including graph and vector"""
    logger.info(f"Testing RAG pipeline with query: {query}")
    
    try:
        # Initialize the pipeline 
        pipeline = RAGPipeline(config)
        
        # Run the query
        result = pipeline.query(query, k=3)
        
        # Print the response
        print("\n" + "="*50)
        print("FULL RAG PIPELINE RESULTS")
        print("="*50)
        print("\nQuery:", query)
        print("\nResponse from LLM:")
        print("-"*50)
        print(result.get("response", "No response generated"))
        print("-"*50)
        
        # Print source information with metadata
        print("\nSources used:")
        sources = result.get("sources", [])
        for i, source in enumerate(sources, 1):
            print(f"\n[{i}] Score: {source.get('score', 'N/A'):.4f}")
            
            # Print essential metadata only
            metadata = source.get("metadata", {})
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Path: {metadata.get('path', 'N/A')}")
            
            # Only print tags if present
            if "tags" in metadata and metadata["tags"]:
                print(f"  Tags: {', '.join(metadata['tags'][:3])}")
            
            # Only print timestamps if present
            if "modified" in metadata:
                print(f"  Modified: {metadata['modified']}")
            
            # Print metadata size to verify optimization
            metadata_size = len(json.dumps(metadata))
            print(f"  Metadata size: {metadata_size} bytes")
            
            # Print a snippet of content
            content = source.get('content', '')
            if content:
                print(f"  Content snippet: {content[:100]}...")
        
        return result
    except Exception as e:
        logger.error(f"Error during pipeline test: {e}")
        raise

def main():
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    # Load config
    config = load_config()
    
    # Get vault path from config or use default
    vault_path = config.get("vault", {}).get("path", "data/test")
    print(f"Using vault path: {vault_path}")
    
    # Test query - more general to find matches in our test data
    query = "What is important in my notes?"
    
    # Test just the graph retriever first
    print("\nTesting the EnhancedGraphRetriever with optimized metadata...\n")
    test_graph_retriever(query, vault_path)
    
    # Test the full RAG pipeline
    print("\nTesting the full RAG pipeline...\n")
    try:
        test_full_rag_pipeline(query, config)
    except Exception as e:
        print(f"Full pipeline test failed: {e}")
        print("Full pipeline test failed, but graph retriever test was completed.")

if __name__ == "__main__":
    main() 