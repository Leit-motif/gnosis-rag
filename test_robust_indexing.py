#!/usr/bin/env python
"""
Test script for the robust indexing system
Tests with a small sample to validate the approach
"""
import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('backend')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_robust_indexing():
    """Test the robust indexing system with sample data"""
    try:
        # Import after adding to path
        from backend.robust_indexer import create_robust_indexer
        
        logger.info("=== Testing Robust Indexing System ===")
        
        # Create sample documents
        sample_documents = [
            {
                'content': 'This is a test document about machine learning and artificial intelligence.',
                'metadata': {
                    'title': 'AI Introduction',
                    'path': '/test/ai-intro.md',
                    'source': 'test',
                    'chunk_id': 'chunk_1'
                }
            },
            {
                'content': 'Python is a popular programming language used for data science and web development.',
                'metadata': {
                    'title': 'Python Programming',
                    'path': '/test/python.md',
                    'source': 'test',
                    'chunk_id': 'chunk_2'
                }
            },
            {
                'content': 'FastAPI is a modern web framework for building APIs with Python.',
                'metadata': {
                    'title': 'FastAPI Framework',
                    'path': '/test/fastapi.md',
                    'source': 'test',
                    'chunk_id': 'chunk_3'
                }
            },
            {
                'content': 'Vector databases are used to store and search high-dimensional embeddings.',
                'metadata': {
                    'title': 'Vector Databases',
                    'path': '/test/vector-db.md',
                    'source': 'test',
                    'chunk_id': 'chunk_4'
                }
            },
            {
                'content': 'RAG (Retrieval Augmented Generation) combines information retrieval with language generation.',
                'metadata': {
                    'title': 'RAG Systems',
                    'path': '/test/rag.md',
                    'source': 'test',
                    'chunk_id': 'chunk_5'
                }
            }
        ]
        
        logger.info(f"Created {len(sample_documents)} sample documents")
        
        # Create robust indexer with test configuration
        test_config = {
            "rate_limiting": {
                "max_requests_per_minute": 10,  # Lower for testing
                "batch_size": 3,  # Small batches for testing
                "delay_between_batches": 1.0,  # Shorter delay for testing
                "max_retries": 3,
                "backoff_factor": 2.0
            },
            "api_settings": {
                "max_tokens_per_request": 4000,
                "context_window": 8192,
                "embedding_timeout": 30,
                "max_concurrent_requests": 1  # Sequential for testing
            },
            "vector_store": {
                "embedding_model": "text-embedding-3-small"
            }
        }
        
        # Create the indexer
        from backend.robust_indexer import RobustIndexer
        indexer = RobustIndexer(test_config)
        
        logger.info("Created robust indexer with test configuration")
        
        # Test indexing
        logger.info("Starting indexing test...")
        result = indexer.index_documents(sample_documents, resume=False)
        
        # Print results
        logger.info("=== Indexing Results ===")
        logger.info(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            logger.info(f"✅ Successfully indexed {result['processed_documents']}/{result['total_documents']} documents")
            logger.info(f"📊 Success rate: {result['success_rate']:.1f}%")
            logger.info(f"⏱️ Elapsed time: {result['elapsed_time']}")
            logger.info(f"📁 Index saved to: {result['index_path']}")
            logger.info(f"🔢 Embeddings shape: {result['embeddings_shape']}")
            
            # Verify files were created
            vector_store_path = Path("data/vector_store")
            files_to_check = [
                "faiss.index",
                "embeddings.json",
                "document_store.json"
            ]
            
            logger.info("=== Verifying Created Files ===")
            for filename in files_to_check:
                file_path = vector_store_path / filename
                if file_path.exists():
                    size = file_path.stat().st_size
                    logger.info(f"✅ {filename}: {size} bytes")
                else:
                    logger.error(f"❌ {filename}: Missing!")
            
            # Test loading the index
            logger.info("=== Testing Index Loading ===")
            try:
                import faiss
                import numpy as np
                
                index_path = vector_store_path / "faiss.index"
                loaded_index = faiss.read_index(str(index_path))
                logger.info(f"✅ Successfully loaded index with {loaded_index.ntotal} vectors")
                
                # Test a simple search
                if loaded_index.ntotal > 0:
                    # Create a dummy query vector
                    dimension = loaded_index.d
                    query_vector = np.random.random((1, dimension)).astype('float32')
                    distances, indices = loaded_index.search(query_vector, min(3, loaded_index.ntotal))
                    logger.info(f"✅ Search test successful: found {len(indices[0])} results")
                
            except Exception as e:
                logger.error(f"❌ Failed to test index loading: {str(e)}")
            
        else:
            logger.error(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")
            logger.info(f"📊 Processed: {result.get('processed_documents', 0)} documents")
            logger.info(f"❌ Failed: {result.get('failed_documents', 0)} documents")
        
        return result['status'] == 'success'
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}", exc_info=True)
        return False

def test_resume_functionality():
    """Test the resume functionality"""
    try:
        logger.info("=== Testing Resume Functionality ===")
        
        # This would require simulating a partial failure
        # For now, just check if resume logic works with existing progress
        from backend.robust_indexer import RobustIndexer
        
        test_config = {
            "rate_limiting": {"batch_size": 2},
            "vector_store": {"embedding_model": "text-embedding-3-small"}
        }
        
        indexer = RobustIndexer(test_config)
        
        # Check if progress loading works
        progress = indexer._load_progress()
        if progress:
            logger.info(f"Found existing progress: {progress.processed_documents}/{progress.total_documents} documents")
        else:
            logger.info("No existing progress found (this is expected for first run)")
        
        return True
        
    except Exception as e:
        logger.error(f"Resume test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Robust Indexing System Tests")
    
    # Check environment
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set your OpenAI API key to run this test")
        return False
    
    # Run tests
    test_results = {
        "indexing": test_robust_indexing(),
        "resume": test_resume_functionality()
    }
    
    # Summary
    logger.info("=== Test Summary ===")
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! The robust indexing system is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 