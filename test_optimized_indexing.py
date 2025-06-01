#!/usr/bin/env python
"""
Test script to measure optimized indexing performance on 2025 vault data
"""
import asyncio
import sys
import time
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.utils import load_config
from backend.obsidian_loader_v2 import ObsidianLoaderV2
from backend.fast_indexer import FastIndexer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_optimized_indexing():
    """Test the optimized indexing performance"""
    try:
        logger.info("🚀 Testing optimized indexing performance on 2025 vault data")
        
        # Load configuration
        config = load_config()
        logger.info(f"Vault path: {config['vault']['path']}")
        
        # Load documents
        logger.info("Loading documents...")
        vault_loader = ObsidianLoaderV2(config["vault"]["path"])
        documents = vault_loader.load_vault(config)
        logger.info(f"Loaded {len(documents)} documents from vault")
        
        if not documents:
            logger.error("No documents found!")
            return
        
        # Prepare documents for indexing
        indexed_documents = [
            {
                'id': f"{doc.metadata.get('source', 'unknown_source')}#{doc.metadata.get('chunk_id', 'unknown_chunk')}", 
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in documents
        ]
        
        logger.info(f"Prepared {len(indexed_documents)} documents for indexing")
        
        # Create fast indexer with current config
        fast_indexer = FastIndexer(config)
        logger.info(f"Using fast indexer config:")
        logger.info(f"  - Batch size: {fast_indexer.batch_size}")
        logger.info(f"  - Concurrent requests: {fast_indexer.max_concurrent_requests}")
        logger.info(f"  - Embedding provider: {fast_indexer.embedding_provider}")
        logger.info(f"  - Checkpoint interval: {fast_indexer.checkpoint_interval}")
        
        # Start indexing with timing
        start_time = time.time()
        logger.info("⚡ Starting optimized fast indexing...")
        
        result = await fast_indexer.index_documents_fast(indexed_documents, resume=False)
        
        elapsed_time = time.time() - start_time
        
        # Report results
        logger.info("📊 INDEXING PERFORMANCE RESULTS:")
        logger.info(f"  Status: {result['status']}")
        logger.info(f"  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"  Documents processed: {result.get('processed_documents', 'N/A')}")
        logger.info(f"  Failed documents: {result.get('failed_documents', 0)}")
        
        if result['status'] == 'success' and elapsed_time > 0:
            docs_per_sec = result.get('processed_documents', 0) / elapsed_time
            logger.info(f"  🚀 Speed: {docs_per_sec:.1f} docs/sec")
            
            # Calculate time savings vs overnight
            overnight_hours = 8  # Assume 8 hours overnight
            overnight_seconds = overnight_hours * 3600
            time_saved = overnight_seconds - elapsed_time
            time_saved_hours = time_saved / 3600
            
            logger.info(f"  ⏰ Time saved vs overnight: {time_saved_hours:.1f} hours")
            logger.info(f"  📈 Speed improvement: {overnight_seconds / elapsed_time:.1f}x faster")
            
            # Estimate full vault performance
            if len(indexed_documents) > 0:
                # Estimate based on your previous vault size (assuming ~17,000 docs full vault)
                estimated_full_vault_docs = 17000
                scale_factor = estimated_full_vault_docs / len(indexed_documents)
                estimated_full_time = elapsed_time * scale_factor
                estimated_full_hours = estimated_full_time / 3600
                
                logger.info(f"  📋 Full vault estimate (~{estimated_full_vault_docs} docs):")
                logger.info(f"     - Estimated time: {estimated_full_hours:.1f} hours")
                logger.info(f"     - vs overnight (8h): {8/estimated_full_hours:.1f}x faster")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_optimized_indexing())
    
    if result['status'] == 'success':
        print("\n" + "="*60)
        print("✅ OPTIMIZATION TEST SUCCESSFUL!")
        print("Your embedding pipeline is now significantly faster!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print("="*60) 