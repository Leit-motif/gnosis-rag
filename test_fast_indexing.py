#!/usr/bin/env python
"""
Test script for the fast indexing system
Tests speed optimizations and validates performance
"""
import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))
sys.path.append('backend')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_fast_indexing():
    """Test the fast indexing system with sample data"""
    try:
        from backend.fast_indexer import FastIndexer, FAST_CONFIG
        
        logger.info("=== Testing Fast Indexing System ===")
        
        # Create a larger set of sample documents for speed testing
        sample_documents = []
        
        # Generate 200 sample documents to test speed
        base_texts = [
            "This is a test document about machine learning and artificial intelligence. Machine learning algorithms are used to build mathematical models based on training data.",
            "Python is a popular programming language used for data science and web development. It has a simple syntax and powerful libraries for various applications.",
            "FastAPI is a modern web framework for building APIs with Python. It's fast, easy to use, and includes automatic API documentation generation.",
            "Vector databases are used to store and search high-dimensional embeddings. They enable semantic search and similarity matching for AI applications.",
            "RAG (Retrieval Augmented Generation) combines information retrieval with language generation. This approach improves AI responses by grounding them in relevant context.",
            "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It includes tasks like text classification and sentiment analysis.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It has revolutionized fields like computer vision and natural language understanding.",
            "Cloud computing provides on-demand access to computing resources over the internet. It enables scalable and flexible infrastructure for modern applications.",
            "DevOps is a set of practices that combines software development and IT operations. It aims to shorten the systems development life cycle and provide continuous delivery.",
            "Microservices architecture is an approach to building software applications as a collection of loosely coupled services. Each service is independently deployable and scalable."
        ]
        
        for i in range(200):  # Create 200 test documents
            text_idx = i % len(base_texts)
            sample_documents.append({
                'content': f"{base_texts[text_idx]} Document ID: {i+1}. This adds uniqueness to each document for testing purposes.",
                'metadata': {
                    'title': f'Test Document {i+1}',
                    'path': f'/test/doc_{i+1}.md',
                    'source': 'test',
                    'chunk_id': f'chunk_{i+1}',
                    'category': f'category_{(i % 5) + 1}'
                }
            })
        
        logger.info(f"Created {len(sample_documents)} sample documents for speed testing")
        
        # Test different configurations
        test_configs = [
            {
                "name": "Conservative",
                "config": {
                    "batch_size": 50,
                    "max_concurrent_requests": 5,
                    "embedding_timeout": 60,
                    "checkpoint_interval": 100,
                    "use_streaming": True,
                    "embedding_model": "text-embedding-3-small"
                }
            },
            {
                "name": "Balanced",
                "config": {
                    "batch_size": 100,
                    "max_concurrent_requests": 10,
                    "embedding_timeout": 60,
                    "checkpoint_interval": 200,
                    "use_streaming": True,
                    "embedding_model": "text-embedding-3-small"
                }
            },
            {
                "name": "Aggressive",
                "config": {
                    "batch_size": 150,
                    "max_concurrent_requests": 15,
                    "embedding_timeout": 90,
                    "checkpoint_interval": 300,
                    "use_streaming": True,
                    "embedding_model": "text-embedding-3-small"
                }
            }
        ]
        
        results = []
        
        for test_config in test_configs:
            logger.info(f"\n=== Testing {test_config['name']} Configuration ===")
            
            # Clean up any existing progress
            progress_path = Path("data/vector_store/indexing_progress.pkl")
            if progress_path.exists():
                progress_path.unlink()
            
            try:
                # Create the indexer
                indexer = FastIndexer(test_config['config'])
                
                # Measure indexing time
                start_time = time.time()
                result = await indexer.index_documents_fast(sample_documents, resume=False)
                end_time = time.time()
                
                elapsed = end_time - start_time
                docs_per_sec = len(sample_documents) / elapsed if elapsed > 0 else 0
                
                logger.info(f"=== {test_config['name']} Results ===")
                logger.info(f"Status: {result['status']}")
                
                if result['status'] == 'success':
                    logger.info(f"✅ Processed {result['processed_documents']}/{result['total_documents']} documents")
                    logger.info(f"⏱️ Total time: {elapsed:.2f}s")
                    logger.info(f"🚀 Speed: {docs_per_sec:.1f} docs/sec")
                    logger.info(f"💾 Index shape: {result.get('embeddings_shape', 'N/A')}")
                    
                    results.append({
                        "config": test_config['name'],
                        "status": "success",
                        "processed": result['processed_documents'],
                        "total": result['total_documents'],
                        "elapsed": elapsed,
                        "docs_per_sec": docs_per_sec,
                        "failed": result.get('failed_documents', 0)
                    })
                    
                else:
                    logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    results.append({
                        "config": test_config['name'],
                        "status": "error",
                        "error": result.get('error', 'Unknown error'),
                        "processed": result.get('processed_documents', 0)
                    })
                
                # Brief pause between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Test failed for {test_config['name']}: {str(e)}", exc_info=True)
                results.append({
                    "config": test_config['name'],
                    "status": "error",
                    "error": str(e)
                })
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['docs_per_sec'])
            
            logger.info(f"🏆 Best Performance: {best_result['config']}")
            logger.info(f"   Speed: {best_result['docs_per_sec']:.1f} docs/sec")
            logger.info(f"   Time: {best_result['elapsed']:.2f}s")
            logger.info(f"   Success Rate: {((best_result['processed'] - best_result.get('failed', 0)) / best_result['total'] * 100):.1f}%")
            
            logger.info("\nAll Results:")
            for result in successful_results:
                logger.info(f"  {result['config']}: {result['docs_per_sec']:.1f} docs/sec ({result['elapsed']:.2f}s)")
        
        # Test status checking
        logger.info("\n=== Testing Status Checking ===")
        indexer = FastIndexer(FAST_CONFIG)
        status = indexer.get_indexing_status()
        logger.info(f"Current status: {status}")
        
        return len(successful_results) > 0
        
    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}", exc_info=True)
        return False

async def test_resume_functionality():
    """Test the resume functionality"""
    try:
        logger.info("\n=== Testing Resume Functionality ===")
        
        from backend.fast_indexer import FastIndexer, FAST_CONFIG
        
        # Create some test documents
        test_documents = [
            {
                'content': f'Resume test document {i}',
                'metadata': {'title': f'Resume Test {i}', 'source': 'resume_test'}
            }
            for i in range(50)
        ]
        
        # Create indexer
        config = FAST_CONFIG.copy()
        config['checkpoint_interval'] = 10  # Checkpoint every 10 docs for testing
        indexer = FastIndexer(config)
        
        # Start indexing (this should create progress file)
        logger.info("Starting partial indexing to test resume...")
        result = await indexer.index_documents_fast(test_documents[:30], resume=False)
        
        if result['status'] == 'success':
            logger.info(f"✅ Partial indexing successful: {result['processed_documents']} documents")
            
            # Check status
            status = indexer.get_indexing_status()
            logger.info(f"Status after partial: {status}")
            
            # Test resume with remaining documents
            logger.info("Testing resume with remaining documents...")
            resume_result = await indexer.index_documents_fast(test_documents, resume=True)
            
            if resume_result['status'] == 'success':
                logger.info(f"✅ Resume successful: {resume_result['processed_documents']} total documents")
                return True
            else:
                logger.error(f"❌ Resume failed: {resume_result.get('error')}")
                return False
        else:
            logger.error(f"❌ Initial indexing failed: {result.get('error')}")
            return False
        
    except Exception as e:
        logger.error(f"Resume test failed: {str(e)}", exc_info=True)
        return False

async def main():
    """Main test function"""
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set the OPENAI_API_KEY environment variable")
        print("Create a .env file with: OPENAI_API_KEY=your_key_here")
        return False
    
    logger.info("🎯 Fast Indexing Performance Testing Starting...")
    print("\n" + "="*70)
    print("🚀 FAST INDEXING PERFORMANCE TEST")
    print("="*70)
    print("This will test the optimized fast indexing system with different")
    print("configurations to find the best settings for your environment.")
    print("="*70)
    
    # Run main speed test
    speed_test_success = await test_fast_indexing()
    
    # Run resume test
    resume_test_success = await test_resume_functionality()
    
    print("\n" + "="*70)
    if speed_test_success and resume_test_success:
        print("🎉 ALL TESTS SUCCESSFUL!")
        print("✅ Speed optimization test: PASSED")
        print("✅ Resume functionality test: PASSED")
        print("\nRecommendations:")
        print("- Use the 'Aggressive' configuration for large vaults")
        print("- Monitor API rate limits and adjust if needed")
        print("- Use /index_fast endpoint for maximum speed")
    else:
        print("❌ SOME TESTS FAILED!")
        if not speed_test_success:
            print("❌ Speed optimization test: FAILED")
        if not resume_test_success:
            print("❌ Resume functionality test: FAILED")
    print("="*70)
    
    return speed_test_success and resume_test_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 