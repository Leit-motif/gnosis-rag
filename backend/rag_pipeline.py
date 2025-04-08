from typing import List, Optional, Dict, Any, Tuple
import os
import faiss
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from qdrant_client import QdrantClient
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from fastapi import HTTPException
import time
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import tempfile
import shutil
from .graph_retriever import GraphRetriever

class RAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = OpenAI()
        self.logger = logging.getLogger(__name__)
        self.document_store = {}  # Initialize empty document store
        
        # Initialize embeddings
        if config["embeddings"]["provider"] == "openai":
            self.embed = self._embed_openai
            self.batch_size = 500  # Increased batch size for OpenAI API
        else:
            self.model = SentenceTransformer(config["embeddings"]["local_model"])
            self.embed = self._embed_local
            self.batch_size = config["embeddings"]["batch_size"]
            
        # Initialize vector store
        if config["vector_store"]["type"] == "faiss":
            # Check if we have embeddings
            if self._have_temp_embeddings():
                self.logger.info("Found existing embeddings, loading them...")
                try:
                    embeddings, documents = self._load_temp_embeddings()
                    self.logger.info(f"Loaded {len(embeddings)} embeddings")
                    
                    # Store documents
                    for i, doc in enumerate(documents):
                        self.document_store[str(i)] = {
                            'content': doc['content'],
                            'metadata': doc['metadata']
                        }
                    
                    # Create index
                    self.index = self._create_faiss_index(embeddings)
                    self.logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
                except Exception as e:
                    self.logger.error(f"Failed to load embeddings: {str(e)}")
                    self.logger.info("Creating empty index instead")
                    self.index = self._init_faiss()
            else:
                self.index = self._init_faiss()
        elif config["vector_store"]["type"] == "chroma":
            self.vector_store = chromadb.Client()
        else:  # qdrant
            self.vector_store = QdrantClient("localhost")
            
        # Initialize graph retriever
        self.graph_retriever = GraphRetriever(config["vault"]["path"])
        self.graph_retriever.build_graph()
        self.logger.info("Graph retriever initialized")
        
    def _init_faiss(self) -> faiss.Index:
        """Initialize FAISS index"""
        try:
            dimension = self.config["vector_store"]["dimension"]
            index_path = self.config["vector_store"]["index_path"]
            
            # Convert to Path object and resolve
            index_path = Path(index_path).resolve()
            self.logger.info(f"Using index path: {index_path}")
            
            # Create directory if it doesn't exist
            index_dir = index_path.parent
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure directory is writable
            try:
                test_file = index_dir / "test_write"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                self.logger.error(f"Directory {index_dir} is not writable: {str(e)}")
                raise ValueError(f"Directory {index_dir} is not writable: {str(e)}")
            
            # Create a new index first
            self.logger.info(f"Creating new FAISS index with dimension {dimension}")
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            if index_path.exists():
                try:
                    self.logger.info(f"Found existing index at {index_path}, attempting to load...")
                    loaded_index = faiss.read_index(str(index_path))
                    return loaded_index
                except Exception as e:
                    self.logger.warning(f"Could not load existing index: {str(e)}")
                    self.logger.info("Will create a new index instead")
            
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {str(e)}")
            # Create an in-memory index as fallback
            self.logger.info("Creating in-memory index as fallback")
            return faiss.IndexFlatIP(self.config["vector_store"]["dimension"])
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.config["embeddings"]["model"]
            )
            return np.array([e.embedding for e in response.data])
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using local model"""
        return self.model.encode(texts, batch_size=self.batch_size)
    
    def _process_batch(self, batch: List[Dict[str, Any]], batch_idx: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process a single batch of documents"""
        try:
            # Get embeddings for batch
            texts = [doc['content'] for doc in batch]
            embeddings = self.embed(texts)
            
            # Store document content and metadata
            processed_docs = []
            for j, doc in enumerate(batch):
                doc_id = str(len(self.document_store) + (batch_idx * self.batch_size) + j)
                self.document_store[doc_id] = {
                    'content': doc['content'],
                    'metadata': doc['metadata']
                }
                processed_docs.append(doc)
            
            return embeddings, processed_docs
        except Exception as e:
            self.logger.error(f"Failed to process batch {batch_idx}: {str(e)}")
            return None, []

    def _save_index(self, index_path: Path) -> None:
        """Save FAISS index to disk with proper Windows file handling"""
        try:
            # Create a temporary file in the same directory as the target
            temp_path = index_path.parent / f"{index_path.stem}_temp{index_path.suffix}"
            self.logger.info(f"Using temporary file: {temp_path}")
            
            # Remove temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            
            # Save to temporary file
            faiss.write_index(self.index, str(temp_path))
            
            # If target file exists, try to remove it
            try:
                if index_path.exists():
                    index_path.unlink()
            except Exception as e:
                self.logger.error(f"Could not remove existing index, trying to force: {str(e)}")
                try:
                    # Try to force remove using os.remove
                    os.remove(str(index_path))
                except Exception as e2:
                    self.logger.error(f"Could not force remove index: {str(e2)}")
                    # Clean up temp file
                    temp_path.unlink()
                    raise
            
            try:
                # Rename temporary file to target
                os.replace(str(temp_path), str(index_path))
                self.logger.info(f"Successfully saved index with {self.index.ntotal} vectors")
            except Exception as e:
                self.logger.error(f"Failed to rename temp file: {str(e)}")
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
                raise
            
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
            # Clean up temporary file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            raise

    def _save_embeddings_temp(self, embeddings: List[np.ndarray], documents: List[Dict[str, Any]]) -> Path:
        """Save embeddings and documents to temporary files"""
        temp_dir = Path("data/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        temp_embeddings = temp_dir / "temp_embeddings.npy"
        combined_embeddings = np.vstack(embeddings)
        np.save(str(temp_embeddings), combined_embeddings)
        
        # Save documents
        temp_docs = temp_dir / "temp_docs.npy"
        np.save(str(temp_docs), documents)
        
        self.logger.info(f"Saved {len(embeddings)} embedding batches to {temp_embeddings}")
        return temp_dir

    def _load_temp_embeddings(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and documents from temporary files"""
        temp_dir = Path("data/temp")
        temp_embeddings = temp_dir / "temp_embeddings.npy"
        temp_docs = temp_dir / "temp_docs.npy"
        
        embeddings = np.load(str(temp_embeddings))
        documents = np.load(str(temp_docs), allow_pickle=True)
        
        return embeddings, documents.tolist()

    def _have_temp_embeddings(self) -> bool:
        """Check if we have temporary embeddings saved"""
        temp_dir = Path("data/temp")
        temp_embeddings = temp_dir / "temp_embeddings.npy"
        temp_docs = temp_dir / "temp_docs.npy"
        return temp_embeddings.exists() and temp_docs.exists()

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create a new FAISS index in memory"""
        dimension = embeddings.shape[1]
        self.logger.info(f"Creating new FAISS index with dimension {dimension}")
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents in the vector store with parallel processing
        """
        try:
            if not documents:
                self.logger.warning("No documents to index")
                return
                
            self.logger.info(f"Indexing {len(documents)} documents")
            start_time = time.time()
            
            # STEP 1: Generate/Load Embeddings
            if self._have_temp_embeddings():
                self.logger.info("Found existing embeddings in temp storage")
                combined_embeddings, processed_docs = self._load_temp_embeddings()
                self.logger.info(f"Loaded {len(combined_embeddings)} embeddings from temp storage")
            else:
                self.logger.info("Generating new embeddings...")
                # Split documents into batches
                num_batches = math.ceil(len(documents) / self.batch_size)
                batches = [
                    documents[i:i + self.batch_size]
                    for i in range(0, len(documents), self.batch_size)
                ]
                
                # Process batches in parallel
                all_embeddings = []
                processed_docs = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for i, batch in enumerate(batches):
                        future = executor.submit(self._process_batch, batch, i)
                        futures.append(future)
                    
                    # Track progress with estimated time
                    with tqdm(total=len(batches), desc="Generating embeddings") as pbar:
                        for future in as_completed(futures):
                            embeddings, docs = future.result()
                            if embeddings is not None:
                                all_embeddings.append(embeddings)
                                processed_docs.extend(docs)
                            pbar.update(1)
                            
                            # Update ETA
                            elapsed = time.time() - start_time
                            docs_per_sec = len(processed_docs) / elapsed
                            remaining_docs = len(documents) - len(processed_docs)
                            eta_seconds = remaining_docs / docs_per_sec if docs_per_sec > 0 else 0
                            pbar.set_postfix({
                                'docs/s': f'{docs_per_sec:.1f}',
                                'eta': f'{timedelta(seconds=int(eta_seconds))}'
                            })
                
                # Save embeddings to temporary storage
                if all_embeddings:
                    self.logger.info("Saving embeddings to temporary storage...")
                    combined_embeddings = np.vstack(all_embeddings)
                    temp_dir = self._save_embeddings_temp([combined_embeddings], processed_docs)
            
            # STEP 2: Create FAISS Index
            if self.config["vector_store"]["type"] == "faiss":
                try:
                    self.logger.info("Creating FAISS index in memory...")
                    self.index = self._create_faiss_index(combined_embeddings)
                    self.logger.info(f"Successfully created index with {self.index.ntotal} vectors")
                    
                    # STEP 3: Save Index (optional)
                    try:
                        index_path = Path(self.config["vector_store"]["index_path"]).resolve()
                        self.logger.info(f"Attempting to save index to {index_path}")
                        
                        # Create directory if it doesn't exist
                        index_dir = index_path.parent
                        index_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Try to save index
                        faiss.write_index(self.index, str(index_path))
                        self.logger.info("Successfully saved index to disk")
                    except Exception as e:
                        self.logger.warning(f"Could not save index to disk: {str(e)}")
                        self.logger.info("Will continue with in-memory index")
                        
                except Exception as e:
                    self.logger.error(f"Failed to create FAISS index: {str(e)}")
                    raise
            
            elapsed = time.time() - start_time
            docs_per_sec = len(documents) / elapsed
            self.logger.info(f"Indexing complete in {timedelta(seconds=int(elapsed))} ({docs_per_sec:.1f} docs/s)")
            
        except Exception as e:
            self.logger.error(f"Indexing failed: {str(e)}")
            raise

    def query(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        k: int = 5,
        use_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Hybrid search combining semantic, metadata, and graph-based retrieval
        """
        try:
            # Get query embedding
            query_embedding = self.embed([query])[0]
            
            # Semantic search
            if self.config["vector_store"]["type"] == "faiss":
                if self.index.ntotal == 0:
                    return {
                        "response": "No documents have been indexed yet. Please index your Obsidian vault first.",
                        "sources": []
                    }
                
                # Get more results than needed for filtering
                D, I = self.index.search(query_embedding.reshape(1, -1), k * 2)
                
                # Filter results by metadata
                filtered_results = []
                for i, d in zip(I[0], D[0]):
                    doc_id = str(i)
                    if doc_id not in self.document_store:
                        continue
                        
                    doc = self.document_store[doc_id]
                    metadata = doc["metadata"]
                    
                    # Check date range if specified
                    if start_date or end_date:
                        if "date" not in metadata:
                            continue
                        doc_date = datetime.strptime(metadata["date"], "%Y-%m-%d")
                        if start_date and doc_date < start_date:
                            continue
                        if end_date and doc_date > end_date:
                            continue
                    
                    # Check tags if specified
                    if tags:
                        doc_tags = metadata.get("tags", [])
                        if not any(tag in doc_tags for tag in tags):
                            continue
                    
                    filtered_results.append({
                        "id": doc_id,
                        "score": float(d),
                        "excerpt": doc["content"],
                        "metadata": {
                            **metadata,
                            "graph_distance": 0,  # Direct semantic match has distance 0
                            "graph_similarity": 1.0  # Direct match has full similarity
                        }
                    })
                
                # Sort by score and take top k
                filtered_results.sort(key=lambda x: x["score"], reverse=True)
                semantic_results = filtered_results[:k]
                
                # If graph retrieval is enabled, enhance results with graph-based retrieval
                if use_graph and semantic_results:
                    # Get the top result's path
                    top_result = semantic_results[0]
                    source_path = top_result["metadata"].get("source", "")
                    
                    if source_path:
                        # Get related documents from graph with different distances
                        related_docs = []
                        for distance in range(1, 3):  # Try distances 1 and 2
                            docs = self.graph_retriever.get_related_documents(
                                source_path,
                                max_distance=distance,
                                min_similarity=0.3
                            )
                            related_docs.extend(docs)
                        
                        # Add graph-based results with distance-based scoring
                        for doc in related_docs:
                            doc_path = doc["document"]
                            metadata = self.graph_retriever.get_document_metadata(doc_path)
                            
                            # Skip if we already have this document
                            if any(r["metadata"].get("source") == doc_path for r in semantic_results):
                                continue
                                
                            # Calculate combined score based on graph distance and similarity
                            distance_weight = 1.0 / (1.0 + doc["distance"])  # Exponential decay
                            combined_score = doc["similarity"] * distance_weight
                            
                            # Add to results with adjusted score
                            semantic_results.append({
                                "id": f"graph_{len(semantic_results)}",
                                "score": combined_score,
                                "excerpt": metadata.get("content", ""),
                                "metadata": {
                                    **metadata,
                                    "source": doc_path,
                                    "graph_distance": doc["distance"],
                                    "graph_similarity": doc["similarity"]
                                }
                            })
                
                # Sort all results by score
                semantic_results.sort(key=lambda x: x["score"], reverse=True)
                results = semantic_results[:k]
                
                # Generate response using the top results
                context = "\n\n".join([
                    f"Source {i+1}:\n{r['excerpt']}"
                    for i, r in enumerate(results)
                ])
                
                # Add graph connections to the prompt
                graph_connections = []
                for i, r1 in enumerate(results):
                    for j, r2 in enumerate(results[i+1:], i+1):
                        if r1["metadata"].get("graph_distance") is not None and r2["metadata"].get("graph_distance") is not None:
                            common_tags = self.graph_retriever.get_common_tags(
                                r1["metadata"]["source"],
                                r2["metadata"]["source"]
                            )
                            if common_tags:
                                graph_connections.append(
                                    f"Sources {i+1} and {j+1} share these tags: {', '.join(common_tags)}"
                                )
                
                if graph_connections:
                    context += "\n\nGraph Connections:\n" + "\n".join(graph_connections)
                
                # Generate response
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that synthesizes information from multiple sources. Pay attention to the graph connections between sources and explain how they relate to each other."},
                        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nPlease provide a comprehensive answer based on the sources above, highlighting any connections between them."}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "sources": results
                }
                
            else:
                return {
                    "response": "Vector store type not supported",
                    "sources": []
                }
                
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
    def _symbolic_search(
        self,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using symbolic features (tags, dates, links)
        """
        # Implement symbolic graph traversal
        # This is a placeholder - implement actual logic
        return []
        
    def _combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        symbolic_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank semantic and symbolic search results
        """
        # Implement smart combination strategy
        # This is a placeholder - implement actual logic
        return semantic_results
        
    def _generate_response(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a natural language response using GPT-4
        """
        context = "\n".join([
            f"Source {i+1}: {result['excerpt']}"
            for i, result in enumerate(results)
        ])
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the user's personal notes and journal entries. Synthesize information from the provided sources and maintain a thoughtful, reflective tone."},
                {"role": "user", "content": f"Question: {query}\n\nRelevant sources:\n{context}"}
            ]
        )
        
        return response.choices[0].message.content
        
    def analyze_themes(self) -> List[Dict[str, Any]]:
        """
        Analyze recurring themes and patterns in the vault
        """
        # Implement theme analysis
        # This is a placeholder - implement actual logic
        return []
        
    def generate_reflection(
        self,
        mode: str,
        agent: str
    ) -> Dict[str, Any]:
        """
        Generate a reflection based on recent entries
        """
        # Get recent entries based on mode
        if mode == "weekly":
            days = 7
        else:  # monthly
            days = 30
            
        # Get entries
        start_date = datetime.now() - timedelta(days=days)
        entries = self._get_entries_in_range(start_date)
        
        # Generate reflection using GPT-4
        system_prompts = {
            "gnosis": "You are a wise mentor helping to extract deeper meaning and patterns from journal entries.",
            "anima": "You are the writer's inner voice, speaking from the heart about emotions and personal growth.",
            "archivist": "You are a careful observer, noting patterns, habits, and cycles in the writer's life."
        }
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompts[agent]},
                {"role": "user", "content": f"Generate a {mode} reflection based on these entries:\n\n{entries}"}
            ]
        )
        
        return {
            "reflection": response.choices[0].message.content,
            "time_period": f"Last {days} days",
            "insights": []  # Add extracted insights
        }
        
    def _get_entries_in_range(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        Get entries within a date range
        """
        # Implement entry retrieval
        # This is a placeholder - implement actual logic
        return "" 