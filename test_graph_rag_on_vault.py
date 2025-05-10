import os
import sys
import logging
import json
from pathlib import Path
import time
import re
from dotenv import load_dotenv

"""
SETUP INSTRUCTIONS:
------------------
1. Create a .env file in the root directory with the following variables:
   
   # OpenAI API Key (Required)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Path to your Obsidian vault (Required)
   # Use forward slashes even on Windows (e.g., C:/Users/YourName/Documents/ObsidianVault)
   OBSIDIAN_VAULT_PATH=your_obsidian_vault_path_here
   
2. Install required dependencies if you haven't already:
   pip install python-dotenv openai faiss-cpu networkx pyyaml tqdm langchain_community unstructured langchain-text-splitters

3. Run this script:
   python test_graph_rag_on_vault.py
"""

# Add the current directory to the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from backend.rag_pipeline import RAGPipeline
from backend.graph_rag_integration import GraphRAGIntegration
from backend.obsidian_loader_v2 import ObsidianLoaderV2

def sanitize_text(text):
    """Sanitize text to make it safe for embedding API calls"""
    if not text:
        return "Empty content"
    
    # Remove null bytes and other potential problematic characters
    text = re.sub(r'[\x00-\x09\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Limit length if needed
    if len(text) > 100000:  # Arbitrary limit
        text = text[:100000]
        
    return text

def main():
    """
    Test the enhanced graph RAG on your actual vault.
    
    This script:
    1. Loads configuration from .env file
    2. Initializes the regular RAG pipeline
    3. Enhances it with graph RAG capabilities
    4. Performs test queries using different retrieval modes
    """
    # Make sure environment variables are set
    required_vars = [
        "OPENAI_API_KEY",
        "OBSIDIAN_VAULT_PATH"
    ]
    
    for var in required_vars:
        if not os.environ.get(var):
            logger.error(f"Missing required environment variable: {var}")
            logger.error("Please create a .env file with the required variables. See instructions at the top of this script.")
            sys.exit(1)
    
    # Define config manually (simplified version of config.yaml)
    config = {
        "vault": {
            "path": os.environ.get("OBSIDIAN_VAULT_PATH"),
            "exclude_folders": ["templates", ".trash", ".git"],
            "file_extensions": [".md"]
        },
        "embeddings": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        },
        "vector_store": {
            "type": "faiss",
            "dimension": 1536,
            "index_path": "data/vector_store/faiss.index"
        },
        "chunking": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "split_by": "paragraph"
        },
        "logging": {
            "level": "INFO",
            "file": "logs/gnosis.log"
        }
    }
    
    # Ensure directories exist
    Path("data/vector_store").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using vault path: {config['vault']['path']}")
    
    try:
        # Initialize the regular RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(config)
        
        # Ask if the user wants to reindex the vault
        reindex = input("Would you like to reindex your vault? This may take some time. (y/n): ").lower().strip() == 'y'
        
        if reindex:
            # Load documents from the vault
            logger.info("Loading documents from vault...")
            loader = ObsidianLoaderV2(config["vault"]["path"])
            documents = loader.load_vault(config)
            logger.info(f"Loaded {len(documents)} documents from vault")
            
            # Convert to the format expected by RAGPipeline.index_documents
            docs_for_indexing = []
            for doc in documents:
                # Sanitize content to prevent API errors
                sanitized_content = sanitize_text(doc.page_content)
                docs_for_indexing.append({
                    'content': sanitized_content,
                    'metadata': doc.metadata
                })
            
            # Index the documents
            logger.info("Indexing documents...")
            start_time = time.time()
            rag_pipeline.index_documents(docs_for_indexing)
            elapsed = time.time() - start_time
            logger.info(f"Indexing completed in {elapsed:.2f} seconds")
        
        # Patch RAGPipeline's embed method to handle invalid content
        original_embed = rag_pipeline.embed
        def safe_embed(texts):
            sanitized_texts = [sanitize_text(text) if isinstance(text, str) else "Invalid content" for text in texts]
            return original_embed(sanitized_texts)
        rag_pipeline.embed = safe_embed
        
        # Enhance with Graph RAG
        logger.info("Initializing Graph RAG integration...")
        graph_rag = GraphRAGIntegration(
            rag_pipeline,
            config={
                "default_mode": "hybrid",
                "graph_retriever": {
                    "traversal": {
                        "max_hops": 2,
                        "max_documents": 10
                    }
                }
            }
        )
        
        # Get graph statistics
        logger.info("Graph statistics:")
        stats = graph_rag.get_graph_statistics()
        print(json.dumps(stats, indent=2))
        
        # Ask the user for a custom query
        custom_query = input("\nEnter your test query (or press Enter to use default): ")
        test_query = custom_query if custom_query else "What are the key concepts in my vault?"
        print(f"Using query: {test_query}")
        
        # Test vector-only mode (original RAG)
        logger.info("\n\n=== Testing Vector-Only Retrieval ===")
        try:
            vector_result = rag_pipeline.query(
                query=test_query,
                k=5,
                retrieval_mode="vector"
            )
            print("\nVector Results:")
            print(f"Found {len(vector_result.get('results', []))} documents")
            if vector_result.get('results'):
                print("First result:")
                print(f"  Path: {vector_result['results'][0]['metadata'].get('path', 'N/A')}")
                print(f"  Title: {vector_result['results'][0]['metadata'].get('title', 'N/A')}")
                print(f"  Score: {vector_result['results'][0].get('score', 0)}")
        except Exception as e:
            logger.error(f"Vector retrieval failed: {str(e)}")
            print("\nVector retrieval failed due to an error")
        
        # Test hybrid mode (combines vector and graph)
        logger.info("\n\n=== Testing Hybrid Retrieval ===")
        try:
            hybrid_result = rag_pipeline.query(
                query=test_query,
                k=5,
                retrieval_mode="hybrid"
            )
            print("\nHybrid Results:")
            print(f"Found {len(hybrid_result.get('results', []))} documents")
            if hybrid_result.get('results'):
                print("First result:")
                print(f"  ID: {hybrid_result['results'][0].get('id', 'N/A')}")
                if 'metadata' in hybrid_result['results'][0]:
                    print(f"  Path: {hybrid_result['results'][0]['metadata'].get('path', 'N/A')}")
                    print(f"  Title: {hybrid_result['results'][0]['metadata'].get('title', 'N/A')}")
                print(f"  Score: {hybrid_result['results'][0].get('score', 0)}")
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            print("\nHybrid retrieval failed due to an error")
        
        # Test graph-only mode
        logger.info("\n\n=== Testing Graph-Only Retrieval ===")
        try:
            graph_result = rag_pipeline.query(
                query=test_query,
                k=5,
                retrieval_mode="graph"
            )
            print("\nGraph Results:")
            print(f"Found {len(graph_result.get('results', []))} documents")
            if graph_result.get('results'):
                print("First result:")
                print(f"  ID: {graph_result['results'][0].get('id', 'N/A')}")
                if 'metadata' in graph_result['results'][0]:
                    print(f"  Path: {graph_result['results'][0]['metadata'].get('path', 'N/A')}")
                    print(f"  Title: {graph_result['results'][0]['metadata'].get('title', 'N/A')}")
                print(f"  Score: {graph_result['results'][0].get('score', 0)}")
        except Exception as e:
            logger.error(f"Graph retrieval failed: {str(e)}")
            print("\nGraph retrieval failed due to an error")
        
        # Print formatted context
        logger.info("\n\n=== Formatted Context (for LLM) ===")
        context = hybrid_result.get("context", "No context available")
        print(context[:1000] + "..." if len(context) > 1000 else context)
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 