#!/usr/bin/env python
"""
Fast Indexer - Optimized for speed and large vault processing
Prioritizes throughput while maintaining reliability
"""
import os
import time
import asyncio
import numpy as np
import faiss
from pathlib import Path
import tarfile
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from openai import AsyncOpenAI
import pickle

from backend.storage.base import VaultStorage

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
    
    def __init__(self, config: Dict[str, Any], storage: VaultStorage):
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
        
        # Paths are now managed within a temporary directory during indexing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vector_store_path = Path(self.temp_dir.name)
        
        self.index_path = self.vector_store_path / "faiss.index"
        self.embeddings_path = self.vector_store_path / "embeddings.pkl"
        self.document_store_path = self.vector_store_path / "document_store.pkl"
        self.progress_path = self.vector_store_path / "indexing_progress.pkl"
        
        # Initialize progress tracking
        self.progress: Optional[IndexingProgress] = None
        self.storage = storage
    
    def __del__(self):
        # Clean up the temporary directory when the object is destroyed
        self.temp_dir.cleanup()
    
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
            # Attempt to load an existing index from storage
            try:
                tarball_name = "vector_store.tar.zst"
                local_tarball_path = self.vector_store_path / tarball_name
                logger.info("Attempting to load existing vector store from storage...")
                self.storage.load_vector_store(str(local_tarball_path))
                self._load_index_from_tarball(str(local_tarball_path))
                logger.info("Successfully loaded and extracted existing vector store.")
            except FileNotFoundError:
                logger.info("No existing vector store found. Starting fresh index.")
            except Exception as e:
                logger.warning(f"Could not load existing vector store, will re-index. Error: {e}")

            if not documents:
                return {"status": "success", "message": "No new documents to process."}
            
            self.progress = IndexingProgress(
                total_documents=len(documents),
                processed_documents=0,
                failed_documents=0,
                start_time=time.time(),
                last_checkpoint=time.time(),
                current_batch=0,
                embeddings_generated=0
            )
            
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
            return {"status": "success", "message": "No embeddings were generated."}
    
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
        """
        Builds the final FAISS index, saves all components, and uploads to storage.
        """
        try:
            logger.info("Finalizing index...")
            final_embeddings = np.vstack(all_embeddings)
            
            # Check if an index file already exists from a previous run
            if self.index_path.exists():
                logger.info("Loading existing FAISS index to merge.")
                index = faiss.read_index(str(self.index_path))
            else:
                logger.info("Creating new FAISS index.")
                dimension = final_embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
            
            logger.info(f"Adding {len(final_embeddings)} new embeddings to FAISS index.")
            index.add(final_embeddings)
            
            # Save the updated index and document store
            faiss.write_index(index, str(self.index_path))
            with open(self.document_store_path, "wb") as f:
                pickle.dump(processed_docs, f)
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(final_embeddings, f)
            
            logger.info(f"Index updated with {index.ntotal} total vectors.")
            
            # Create a tarball of the vector store and upload it
            tarball_path = self._save_index_to_tarball()
            self.storage.save_vector_store(tarball_path)
            logger.info(f"Vector store successfully saved to storage.")

            return {"status": "success", "total_indexed": len(final_embeddings)}
        
        except Exception as e:
            logger.error(f"Failed to finalize index: {str(e)}", exc_info=True)
            return {"status": "error", "error": f"Failed to finalize index: {str(e)}"}

    def _save_index_to_tarball(self) -> str:
        """Creates a compressed tarball from the vector store files."""
        tarball_name = "vector_store.tar.zst"
        local_tarball_path = self.vector_store_path.parent / tarball_name
        
        files_to_archive = [
            self.index_path,
            self.document_store_path,
            self.embeddings_path,
        ]
        
        logger.info(f"Creating tarball at {local_tarball_path}...")
        with tarfile.open(local_tarball_path, "w:zst") as tar:
            for file_path in files_to_archive:
                if file_path.exists():
                    tar.add(file_path, arcname=file_path.name)
                    logger.debug(f"Added {file_path.name} to tarball.")
                else:
                    logger.warning(f"File not found, skipping: {file_path}")
        return str(local_tarball_path)

    def _load_index_from_tarball(self, tarball_path: str):
        """Extracts a tarball to the vector store path."""
        logger.info(f"Extracting tarball {tarball_path} to {self.vector_store_path}...")
        with tarfile.open(tarball_path, "r:zst") as tar:
            tar.extractall(path=self.vector_store_path)
        logger.info("Extraction complete.")

    async def _save_checkpoint(self, embeddings: List[np.ndarray], docs: List[Dict[str, Any]]):
        # This method is now obsolete with the new save/load mechanism.
        # Checkpointing will be handled by the atomicity of the final upload.
        pass

    def _load_progress(self) -> Optional[IndexingProgress]:
        # Also obsolete
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