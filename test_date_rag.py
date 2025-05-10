#!/usr/bin/env python
import os
import json
import logging
from pathlib import Path
import yaml
from datetime import datetime
from backend.rag_pipeline import RAGPipeline
from backend.enhanced_graph_retriever import EnhancedGraphRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config file"""
    try:
        # Try loading from config.json first
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info("Loaded configuration from config.json")
                return config
                
        # Fall back to YAML if it exists
        yaml_path = Path("backend/config.yaml")
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                # Replace environment variables with actual values or defaults
                for section in config:
                    for key, value in config[section].items():
                        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                            env_var = value[2:-1]  # Extract environment variable name
                            config[section][key] = os.environ.get(env_var, "")
                logger.info("Loaded configuration from config.yaml")
                return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
    
    # Fall back to a basic configuration
    logger.info("Using default configuration")
    return {
        "vector_store": {
            "dimension": 1536,  # Make sure dimension is defined
            "index_path": "data/vector_store/faiss.index"
        },
        "embeddings": {
            "model": "text-embedding-ada-002",
            "api_key": os.environ.get("OPENAI_API_KEY", "")
        },
        "vault": {
            "path": "data/test"  # Adjust to your test vault path
        },
        "chat_model": "gpt-4-turbo-preview",  # Updated to latest model
        "conversation_memory": {
            "storage_dir": "data/conversations",
            "max_history": 10,
            "context_window": 5
        }
    }

def test_year_based_query():
    """Test RAG performance with year-based queries"""
    logger.info("Testing year-based queries...")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration")
        return
        
    # Initialize RAG pipeline
    rag = RAGPipeline(config)
    
    # Test queries
    test_queries = [
        "What happened in my vault in 2023?",
        "Show me entries from 2022",
        "Find documents from July 2023",
        "What did I write about in January 2024?",
        "Find information from 2023 about mindfulness"
    ]
    
    for query_str in test_queries:
        logger.info(f"\nTesting query: {query_str}")
        
        try:
            # Process query
            response = rag.query(query_str)
            
            # Print response
            logger.info("Response:")
            logger.info(response["response"])
            
            # Print source details for analysis
            logger.info("\nSources:")
            for i, source in enumerate(response["sources"], 1):
                # Extract relevant metadata
                metadata = source.get("metadata", {})
                title = metadata.get("title", "Untitled")
                date = metadata.get("date", "No date")
                path = metadata.get("path", "Unknown path")
                
                # Print source information focusing on date
                logger.info(f"Source {i}: {title}")
                logger.info(f"  - Date: {date}")
                logger.info(f"  - Path: {path}")
                
                # Get year from file path as a double-check
                year_from_path = "Not found"
                import re
                year_match = re.search(r'/(\d{4})/', path) or re.search(r'\\(\d{4})\\', path)
                if year_match:
                    year_from_path = year_match.group(1)
                logger.info(f"  - Year from path: {year_from_path}")
                
            logger.info("-" * 80)
            
        except Exception as e:
            logger.error(f"Error processing query '{query_str}': {e}")
    
    logger.info("Year-based query testing completed")

if __name__ == "__main__":
    test_year_based_query() 