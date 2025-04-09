from typing import List, Optional, Dict, Any, Tuple
import os
import faiss
import numpy as np
from datetime import datetime, timedelta
from openai import OpenAI
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
from .conversation_memory import ConversationMemory
import json

class RAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = OpenAI()
        self.logger = logging.getLogger(__name__)
        self.document_store = {}  # Initialize empty document store
        
        # Initialize embeddings - OpenAI only
        self.embed = self._embed_openai
        self.batch_size = 500  # Increased batch size for OpenAI API
            
        # Initialize FAISS index
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
            
        # Initialize graph retriever
        self.graph_retriever = GraphRetriever(config["vault"]["path"])
        self.graph_retriever.build_graph()
        self.logger.info("Graph retriever initialized")
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(
            storage_dir=config.get("conversation_storage_dir", "data/conversations"),
            max_history=config.get("max_conversation_history", 10),
            context_window=config.get("conversation_context_window", 5)
        )
        
        # Initialize document store
        self.docs_dir = Path(config["docs_dir"])
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector store
        self.index_path = Path(config["index_path"])
        self.embeddings_path = Path(config["embeddings_path"])
        
        if self.index_path.exists() and self.embeddings_path.exists():
            # Load existing index and embeddings
            self.index = faiss.read_index(str(self.index_path))
            with open(self.embeddings_path, 'r') as f:
                self.doc_embeddings = json.load(f)
        else:
            # Initialize new index and embeddings
            self.index = faiss.IndexFlatL2(1536)  # OpenAI embedding dimension
            self.doc_embeddings = {}

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
        k: int = 5,
        session_id: Optional[str] = None,
        conversation_memory: Optional[ConversationMemory] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline with conversation memory
        """
        try:
            # Use instance conversation memory if none provided
            conversation_memory = conversation_memory or self.conversation_memory
            
            # Generate session ID if none provided
            if not session_id:
                session_id = str(int(time.time()))
            
            # Get conversation context
            conversation_context = conversation_memory.get_context(session_id) if session_id else ""
            
            # Perform semantic search
            query_vector = self.embed([query])[0]
            D, I = self.index.search(query_vector.reshape(1, -1), k)
            
            # Get results
            results = []
            for i, idx in enumerate(I[0]):
                if idx == -1 or str(idx) not in self.document_store:
                    continue
                doc = self.document_store[str(idx)]
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(D[0][i])
                })
            
            # Get graph context
            graph_results = self.graph_retriever.get_context(results)
            
            # Combine results
            context = "\n\n".join([
                f"Source {i+1}:\n{result['content']}\nConnections: {', '.join(graph_results.get(result['metadata']['path'], []))}"
                for i, result in enumerate(results)
            ])
            
            # Build messages for chat completion
            messages = [
                {"role": "system", "content": """You are a helpful assistant that synthesizes information from multiple sources and maintains context across conversations. 
Pay attention to the graph connections between sources and explain how they relate to each other.
When referring to previous conversations, be natural and contextual in your responses."""},
            ]
            
            # Add conversation history if available
            if conversation_context:
                messages.append({"role": "system", "content": f"Previous conversation context:\n{conversation_context}"})
            
            # Add current query and context
            messages.append({"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nPlease provide a comprehensive answer based on the sources above, highlighting any connections between them and relating to our previous conversation where relevant."})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.config.get("chat_model", "gpt-4"),
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Store the interaction in conversation memory
            try:
                conversation_memory.add_interaction(
                    session_id=session_id,
                    user_message=query,
                    assistant_message=response.choices[0].message.content
                )
            except Exception as e:
                self.logger.warning(f"Failed to store conversation: {str(e)}")
            
            return {
                "response": response.choices[0].message.content,
                "sources": results,
                "session_id": session_id
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