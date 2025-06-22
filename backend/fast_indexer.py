#!/usr/bin/env python
"""
Fast Indexer - Optimized for speed and large vault processing
Prioritizes throughput while maintaining reliability
"""
import os
import json
import time
import asyncio
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from tqdm import tqdm
import math
from openai import AsyncOpenAI
import aiohttp
import pickle

logger = logging.getLogger(__name__)

@dataclass
class IndexingProgress:
    total_documents: int
    processed_documents: int
    failed_documents: int
    start_time: float
    last_checkpoint: float
    current_batch: int
    embeddings_generated: int
    
    @property
    def progress_percent(self) -> float:
        return (self.processed_documents / self.total_documents) * 100 if self.total_documents > 0 else 0
    
    @property
    def estimated_time_remaining(self) -> float:
        if self.processed_documents == 0:
            return 0
        elapsed = time.time() - self.start_time
        rate = self.processed_documents / elapsed
        remaining = self.total_documents - self.processed_documents
        return remaining / rate if rate > 0 else 0

class FastIndexer:
    """
    Optimized indexer for fast embedding generation and FAISS index creation.
    Supports aggressive parallelization, streaming, and checkpointing for large vaults.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Default configuration
        default_config = {
            "batch_size": 100,
            "max_concurrent_requests": 10,
            "embedding_timeout": 60,
            "checkpoint_interval": 500,
            "use_streaming": True,
            "max_memory_mb": 2000,
            "embedding_model": "text-embedding-3-small"
        }
        
        # Merge with provided config
        merged_config = {**default_config, **config}
        
        # Configuration
        self.batch_size = merged_config["batch_size"]
        self.max_concurrent_requests = merged_config["max_concurrent_requests"]
        self.embedding_timeout = merged_config["embedding_timeout"]
        self.checkpoint_interval = merged_config["checkpoint_interval"]
        self.use_streaming = merged_config["use_streaming"]
        self.max_memory_mb = merged_config["max_memory_mb"]
        
        # Apply preset if specified
        preset = merged_config.get("preset")
        if preset:
            preset_config = self._get_preset_config(preset)
            for key, value in preset_config.items():
                setattr(self, key, value)
        
        # Initialize OpenAI client (only supported provider)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.embedding_model = merged_config.get("embedding_model", "text-embedding-3-small")
        
        # Paths
        self.vector_store_path = Path("data/vector_store")
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.vector_store_path / "faiss.index"
        self.embeddings_path = self.vector_store_path / "embeddings.pkl"  # Use pickle for speed
        self.document_store_path = self.vector_store_path / "document_store.pkl"
        self.progress_path = self.vector_store_path / "indexing_progress.pkl"
        self.checkpoint_path = self.vector_store_path / "checkpoints"
        self.checkpoint_path.mkdir(exist_ok=True)
        
        # Initialize progress tracking
        self.progress: Optional[IndexingProgress] = None
    
    def _get_preset_config(self, preset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific preset"""
        presets = {
            "conservative": {
                "batch_size": 25,
                "max_concurrent_requests": 3,
                "embedding_timeout": 60,
                "checkpoint_interval": 50,
                "use_streaming": True,
                "max_memory_mb": 1000
            },
            "small_vault": {
                "batch_size": 50,
                "max_concurrent_requests": 5,
                "embedding_timeout": 60,
                "checkpoint_interval": 100,
                "use_streaming": False,
                "max_memory_mb": 1000
            },
            "medium_vault": {
                "batch_size": 100,
                "max_concurrent_requests": 10,
                "embedding_timeout": 60,
                "checkpoint_interval": 250,
                "use_streaming": True,
                "max_memory_mb": 2000
            },
            "large_vault": {
                "batch_size": 150,
                "max_concurrent_requests": 15,
                "embedding_timeout": 90,
                "checkpoint_interval": 500,
                "use_streaming": True,
                "max_memory_mb": 3000
            },
            "massive_vault": {
                "batch_size": 200,
                "max_concurrent_requests": 20,
                "embedding_timeout": 120,
                "checkpoint_interval": 1000,
                "use_streaming": True,
                "max_memory_mb": 4000
            }
        }
        
        return presets.get(preset_name, presets["large_vault"])
        
    async def index_documents_fast(self, documents: List[Dict[str, Any]], resume: bool = True) -> Dict[str, Any]:
        """
        Fast indexing with aggressive optimization for speed
        """
        try:
            # Load existing progress if resuming
            if resume and self.progress_path.exists():
                self.progress = self._load_progress()
                logger.info(f"Resuming indexing from {self.progress.processed_documents}/{self.progress.total_documents} documents")
                documents = documents[self.progress.processed_documents:]
            else:
                self.progress = IndexingProgress(
                    total_documents=len(documents),
                    processed_documents=0,
                    failed_documents=0,
                    start_time=time.time(),
                    last_checkpoint=time.time(),
                    current_batch=0,
                    embeddings_generated=0
                )
            
            if not documents:
                return {"status": "success", "message": "All documents already processed"}
            
            logger.info(f"[FAST-INDEX] Starting fast indexing of {len(documents)} documents")
            logger.info(f"[CONFIG] Using {self.max_concurrent_requests} concurrent requests, batch size {self.batch_size}")
            
            start_time = time.time()
            
            if self.use_streaming:
                result = await self._index_with_streaming(documents)
            else:
                result = await self._index_with_batches(documents)
            
            elapsed_time = time.time() - start_time
            
            # Clean up progress file on successful completion
            if result["status"] == "success" and self.progress_path.exists():
                self.progress_path.unlink()
            
            result.update({
                "elapsed_time": f"{elapsed_time:.2f}s",
                "documents_per_second": f"{len(documents) / elapsed_time:.1f}",
                "total_documents": self.progress.total_documents,
                "processed_documents": self.progress.processed_documents,
                "failed_documents": self.progress.failed_documents
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Fast indexing failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "processed_documents": self.progress.processed_documents if self.progress else 0
            }
    
    async def _index_with_streaming(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stream-based processing for memory efficiency"""
        logger.info("[STREAMING] Using streaming mode for memory-efficient processing")
        
        all_embeddings = []
        processed_docs = []
        
        # Process documents in chunks to manage memory
        chunk_size = min(self.batch_size * 5, 1000)  # Process 1000 docs at a time max
        
        for chunk_start in range(0, len(documents), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(documents))
            chunk = documents[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {chunk_start}-{chunk_end} ({len(chunk)} documents)")
            
            # Generate embeddings for this chunk
            chunk_embeddings, chunk_docs = await self._process_chunk_fast(chunk)
            
            if chunk_embeddings is not None:
                all_embeddings.append(chunk_embeddings)
                processed_docs.extend(chunk_docs)
                
                # Update progress
                self.progress.processed_documents += len(chunk_docs)
                self.progress.embeddings_generated += len(chunk_embeddings)
                
                # Save checkpoint
                if self.progress.processed_documents % self.checkpoint_interval == 0:
                    await self._save_checkpoint(all_embeddings, processed_docs)
            
            # Log progress
            progress_pct = (self.progress.processed_documents / self.progress.total_documents) * 100
            eta_minutes = self.progress.estimated_time_remaining / 60
            logger.info(f"[PROGRESS] {progress_pct:.1f}% complete - ETA: {eta_minutes:.1f} min")
        
        # Create final index
        if all_embeddings:
            return await self._finalize_index(all_embeddings, processed_docs)
        else:
            return {"status": "error", "error": "No embeddings generated"}
    
    async def _process_chunk_fast(self, chunk: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Process a chunk of documents with maximum speed"""
        
        # Split chunk into batches for API calls
        batches = [chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        
        # Process all batches concurrently
        tasks = []
        for batch_idx, batch in enumerate(batches):
            task = self._generate_embeddings_batch(batch, batch_idx)
            tasks.append(task)
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*[process_with_semaphore(task) for task in tasks], return_exceptions=True)
        
        # Combine results
        all_embeddings = []
        all_docs = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {str(result)}")
                self.progress.failed_documents += self.batch_size
                continue
                
            embeddings, docs = result
            if embeddings is not None:
                all_embeddings.append(embeddings)
                all_docs.extend(docs)
        
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
            return combined_embeddings, all_docs
        else:
            return None, []
    
    async def _generate_embeddings_batch(self, batch: List[Dict[str, Any]], batch_idx: int) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Generate embeddings for a single batch using OpenAI API or local model"""
        try:
            texts = [doc['content'][:8191] for doc in batch]  # Truncate to avoid token limits
            
            # Make async OpenAI API call
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                timeout=self.embedding_timeout
            )
            
            # Extract embeddings
            embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
            
            return embeddings, batch
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {batch_idx}: {str(e)}")
            return None, []
    
    async def _finalize_index(self, all_embeddings: List[np.ndarray], processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create and save the final FAISS index"""
        try:
            logger.info("[INDEX] Creating final FAISS index...")
            
            # Combine all embeddings
            combined_embeddings = np.vstack(all_embeddings)
            logger.info(f"[INDEX] Combined embeddings shape: {combined_embeddings.shape}")
            
            # Create FAISS index
            dimension = combined_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(combined_embeddings)
            index.add(combined_embeddings)
            
            # Save index
            faiss.write_index(index, str(self.index_path))
            logger.info(f"[SAVE] Saved FAISS index to {self.index_path}")
            
            # Prepare document store
            document_store = {}
            doc_embeddings = {}
            
            for i, doc in enumerate(processed_docs):
                doc_id = str(i)
                document_store[doc_id] = {
                    'content': doc['content'],
                    'metadata': doc.get('metadata', {})
                }
                doc_embeddings[doc_id] = combined_embeddings[i].tolist()
            
            # Save using pickle for speed
            with open(self.document_store_path, 'wb') as f:
                pickle.dump(document_store, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(doc_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"[SAVE] Saved document store and embeddings")
            
            return {
                "status": "success",
                "message": f"Successfully indexed {len(processed_docs)} documents",
                "index_path": str(self.index_path),
                "embeddings_shape": combined_embeddings.shape,
                "dimension": dimension
            }
            
        except Exception as e:
            logger.error(f"Failed to finalize index: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def _save_checkpoint(self, embeddings: List[np.ndarray], docs: List[Dict[str, Any]]):
        """Save a checkpoint to resume from"""
        try:
            checkpoint_file = self.checkpoint_path / f"checkpoint_{self.progress.processed_documents}.pkl"
            
            checkpoint_data = {
                'embeddings': embeddings,
                'documents': docs,
                'progress': self.progress
            }
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save progress separately for quick loading
            with open(self.progress_path, 'wb') as f:
                pickle.dump(self.progress, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"[CHECKPOINT] Saved checkpoint at {self.progress.processed_documents} documents")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def _load_progress(self) -> Optional[IndexingProgress]:
        """Load existing progress"""
        try:
            with open(self.progress_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load progress: {str(e)}")
            return None
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """Get current indexing status"""
        if not self.progress_path.exists():
            return {"status": "not_started"}
        
        progress = self._load_progress()
        if not progress:
            return {"status": "error", "error": "Could not load progress"}
        
        if progress.processed_documents >= progress.total_documents:
            return {
                "status": "completed",
                "total_documents": progress.total_documents,
                "processed_documents": progress.processed_documents,
                "success_rate": ((progress.total_documents - progress.failed_documents) / progress.total_documents) * 100
            }
        else:
            return {
                "status": "in_progress",
                "total_documents": progress.total_documents,
                "processed_documents": progress.processed_documents,
                "failed_documents": progress.failed_documents,
                "progress_percent": progress.progress_percent,
                "estimated_time_remaining": f"{progress.estimated_time_remaining / 60:.1f} minutes"
            }

# Optimized configuration for large vaults
FAST_CONFIG = {
    "batch_size": 100,  # Large batches for efficiency
    "max_concurrent_requests": 15,  # High concurrency
    "embedding_timeout": 60,  # Longer timeout
    "checkpoint_interval": 500,  # Frequent checkpoints
    "use_streaming": True,  # Memory efficient
    "max_memory_mb": 2000,  # 2GB memory limit
    "embedding_model": "text-embedding-3-small"  # Fast model
} 