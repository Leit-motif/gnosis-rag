from backend.rag_pipeline import RAGPipeline
import json
import logging
from pathlib import Path
import shutil
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Test configuration
config = {
    "vector_store": {
        "index_path": "C:/Users/Rando/Workspaces/obsidian-rag-chatgpt-plugin/data/vector_store/faiss.index",
        "dimension": 1536
    },
    "vault": {
        "path": "test_vault"
    },
    "embeddings": {
        "model": "text-embedding-3-small"
    },
    "conversation_memory": {
        "storage_dir": "test_data/conversations",
        "max_history": 5,
        "context_window": 3
    }
}

def setup_test_environment():
    """Setup test environment with sample documents"""
    # Create test vault directory
    test_vault = Path("test_vault")
    test_vault.mkdir(exist_ok=True)
    
    # Create some test markdown files
    test_docs = [
        {
            "filename": "note1.md",
            "content": "# Test Note 1\nThis is a test note about artificial intelligence and machine learning.",
            "metadata": {"tags": ["AI", "ML"], "created": datetime.now().isoformat()}
        },
        {
            "filename": "note2.md",
            "content": "# Test Note 2\nRAG (Retrieval Augmented Generation) combines search and language models.",
            "metadata": {"tags": ["RAG", "LLM"], "created": datetime.now().isoformat()}
        }
    ]
    
    for doc in test_docs:
        file_path = test_vault / doc["filename"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"])
    
    return test_docs

def cleanup_test_environment():
    """Clean up test files and directories"""
    paths_to_cleanup = [
        Path("test_vault"),
        Path("test_data"),
        Path("data/temp")
    ]
    
    for path in paths_to_cleanup:
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: Could not clean up {path}: {e}")

def test_initialization():
    print("\n=== Testing RAGPipeline Initialization ===")
    try:
        # Initialize RAGPipeline
        rag = RAGPipeline(config)
        
        # Print initialization results
        print(f"\nInitialization successful!")
        print(f"Index path: {rag.index_path}")
        print(f"Index dimension: {rag.dimension}")
        print(f"Number of vectors in index: {rag.index.ntotal}")
        print(f"Document store size: {len(rag.document_store)}")
        
        # Test if directories were created
        print(f"\nDirectory exists: {rag.index_path.parent.exists()}")
        return rag
        
    except Exception as e:
        print(f"\nError during initialization: {str(e)}")
        return None

def test_document_indexing(rag: RAGPipeline, test_docs: list):
    print("\n=== Testing Document Indexing ===")
    try:
        # Prepare documents for indexing
        documents = []
        for doc in test_docs:
            documents.append({
                "content": doc["content"],
                "metadata": doc["metadata"]
            })
        
        # Index documents
        rag.index_documents(documents)
        
        print(f"\nIndexing successful!")
        print(f"Number of vectors after indexing: {rag.index.ntotal}")
        print(f"Document store size after indexing: {len(rag.document_store)}")
        return True
        
    except Exception as e:
        print(f"\nError during indexing: {str(e)}")
        return False

def test_querying(rag: RAGPipeline):
    print("\n=== Testing Query Functionality ===")
    try:
        # Test queries
        test_queries = [
            "What is RAG?",
            "Tell me about artificial intelligence"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            result = rag.query(query, k=2)
            
            print("Results:")
            for i, source in enumerate(result["sources"]):
                print(f"\nSource {i+1}:")
                print(f"Content: {source['content'][:100]}...")
                print(f"Score: {source['score']:.4f}")
            
            print(f"\nResponse: {result['response'][:200]}...")
        
        return True
            
    except Exception as e:
        print(f"\nError during querying: {str(e)}")
        return False

def test_conversation_memory(rag: RAGPipeline):
    print("\n=== Testing Conversation Memory ===")
    try:
        session_id = "test_session"
        
        # Test multiple interactions
        conversations = [
            "What is artificial intelligence?",
            "How does it relate to RAG?",
            "Can you summarize what we discussed?"
        ]
        
        for query in conversations:
            print(f"\nUser: {query}")
            result = rag.query(query, session_id=session_id)
            print(f"Assistant: {result['response'][:200]}...")
        
        return True
            
    except Exception as e:
        print(f"\nError during conversation memory test: {str(e)}")
        return False

def run_all_tests():
    try:
        # Setup
        print("\nSetting up test environment...")
        test_docs = setup_test_environment()
        
        # Run tests
        rag = test_initialization()
        if not rag:
            return False
            
        if not test_document_indexing(rag, test_docs):
            return False
            
        if not test_querying(rag):
            return False
            
        if not test_conversation_memory(rag):
            return False
            
        print("\n=== All tests completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"\nTest suite failed: {str(e)}")
        return False
        
    finally:
        # Cleanup
        print("\nCleaning up test environment...")
        cleanup_test_environment()

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 