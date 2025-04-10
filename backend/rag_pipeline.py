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
        self.doc_embeddings = {}  # Initialize empty embeddings store
        
        # Initialize embeddings - OpenAI only
        self.embed = self._embed_openai
        self.batch_size = 500  # Increased batch size for OpenAI API
        
        # Initialize vector store configuration
        vector_store_config = config.get("vector_store", {})
        self.dimension = vector_store_config.get("dimension", 1536)  # OpenAI's default dimension
        
        # Initialize vector store paths with proper Windows path handling
        self.index_path = Path(vector_store_config.get("index_path", "data/vector_store/faiss.index")).resolve()
        self.embeddings_path = Path(os.path.join(os.path.dirname(str(self.index_path)), "embeddings.json"))
        
        # Create directory if it doesn't exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try loading existing index first
        if self.index_path.exists() and self.embeddings_path.exists():
            try:
                self.logger.info(f"Loading existing index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                with open(self.embeddings_path, 'r') as f:
                    self.doc_embeddings = json.load(f)
                self.logger.info(f"Successfully loaded index with {self.index.ntotal} vectors")
                return
            except Exception as e:
                self.logger.error(f"Failed to load existing index: {str(e)}")
                self.logger.info("Will try loading from temp embeddings")
        
        # If no existing index or loading failed, try temp embeddings
        if self._have_temp_embeddings():
            self.logger.info("Found temporary embeddings, loading them...")
            try:
                embeddings, documents = self._load_temp_embeddings()
                self.logger.info(f"Loaded {len(embeddings)} embeddings")
                
                # Store documents
                for i, doc in enumerate(documents):
                    self.document_store[str(i)] = {
                        'content': doc['content'],
                        'metadata': doc['metadata']
                    }
                    self.doc_embeddings[str(i)] = embeddings[i].tolist()
                
                # Create index from embeddings
                self.index = self._create_faiss_index(embeddings)
                self.logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
                
                # Save the index and embeddings
                try:
                    self._save_index(self.index_path)
                    with open(self.embeddings_path, 'w') as f:
                        json.dump(self.doc_embeddings, f)
                    self.logger.info("Saved index and embeddings to disk")
                except Exception as e:
                    self.logger.warning(f"Failed to save index to disk: {str(e)}")
            except Exception as e:
                self.logger.error(f"Failed to load temp embeddings: {str(e)}")
                self.logger.info("Creating empty index")
                self.index = self._init_faiss()
        else:
            self.logger.info("No existing or temporary embeddings found, creating empty index")
            self.index = self._init_faiss()
            
        # Initialize graph retriever
        self.graph_retriever = GraphRetriever(config["vault"]["path"])
        self.graph_retriever.build_graph()
        self.logger.info("Graph retriever initialized")
        
        # Initialize conversation memory
        memory_config = config.get("conversation_memory", {})
        self.conversation_memory = ConversationMemory(
            storage_dir=memory_config.get("storage_dir", "data/conversations"),
            max_history=memory_config.get("max_history", 10),
            context_window=memory_config.get("context_window", 5)
        )

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
            # Create a temporary file in a temp directory
            with tempfile.NamedTemporaryFile(delete=False, suffix='.index') as temp_file:
                temp_path = Path(temp_file.name)
                self.logger.info(f"Using temporary file: {temp_path}")
                
                # Save to temporary file
                faiss.write_index(self.index, str(temp_path))
                
                # Close the file to ensure it's written
                temp_file.close()
                
                try:
                    # Try to move the temp file to the target location
                    # Using shutil.move which handles cross-device moves
                    shutil.move(str(temp_path), str(index_path))
                    self.logger.info(f"Successfully saved index with {self.index.ntotal} vectors")
                except Exception as e:
                    self.logger.error(f"Failed to move index file: {str(e)}")
                    # Try to copy instead of move as fallback
                    try:
                        shutil.copy2(str(temp_path), str(index_path))
                        self.logger.info(f"Successfully copied index with {self.index.ntotal} vectors")
                    except Exception as e2:
                        self.logger.error(f"Failed to copy index file: {str(e2)}")
                        raise
                finally:
                    # Clean up temp file if it still exists
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except:
                        pass
                        
        except Exception as e:
            self.logger.error(f"Failed to save index: {str(e)}")
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
            
            # Process documents in batches
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
                        docs_per_sec = len(processed_docs) / elapsed if elapsed > 0 else 0
                        remaining_docs = len(documents) - len(processed_docs)
                        eta_seconds = remaining_docs / docs_per_sec if docs_per_sec > 0 else 0
                        pbar.set_postfix({
                            'docs/s': f'{docs_per_sec:.1f}',
                            'eta': f'{timedelta(seconds=int(eta_seconds))}'
                        })
            
            if not all_embeddings:
                self.logger.error("No embeddings generated")
                return
                
            # Combine all embeddings
            combined_embeddings = np.vstack(all_embeddings)
            
            # Create new index
            self.index = self._create_faiss_index(combined_embeddings)
            
            # Update document store and embeddings
            self.document_store.clear()
            self.doc_embeddings.clear()
            for i, doc in enumerate(processed_docs):
                self.document_store[str(i)] = {
                    'content': doc['content'],
                    'metadata': doc['metadata']
                }
                self.doc_embeddings[str(i)] = combined_embeddings[i].tolist()
            
            # Save index and embeddings
            try:
                self._save_index(self.index_path)
                with open(self.embeddings_path, 'w') as f:
                    json.dump(self.doc_embeddings, f)
                self.logger.info("Saved index and embeddings to disk")
            except Exception as e:
                self.logger.warning(f"Failed to save to disk: {str(e)}")
            
            elapsed = time.time() - start_time
            docs_per_sec = len(documents) / elapsed if elapsed > 0 else 0
            self.logger.info(f"Indexing complete in {timedelta(seconds=int(elapsed))} ({docs_per_sec:.1f} docs/s)")
            
        except Exception as e:
            self.logger.error(f"Indexing failed: {str(e)}")
            raise

    def query(
        self,
        query: str,
        k: int = 5,
        session_id: Optional[str] = None,
        conversation_memory: Optional[ConversationMemory] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline with conversation memory and filtering options
        """
        try:
            # Use instance conversation memory if none provided
            conversation_memory = conversation_memory or self.conversation_memory
            
            # Generate session ID if none provided
            if not session_id:
                session_id = str(int(time.time()))
            
            # Get conversation context
            conversation_context = conversation_memory.get_context_window(session_id) if session_id else ""
            
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
            
            # Apply tag filtering if specified
            if tags:
                self.logger.info(f"Filtering results by tags: {tags}")
                results = [
                    result for result in results
                    if 'tags' in result['metadata'] and 
                    any(tag in result['metadata']['tags'] for tag in tags)
                ]
            
            # Apply date filtering if specified
            if start_date or end_date:
                self.logger.info(f"Filtering results by date range: {start_date} to {end_date}")
                filtered_results = []
                for result in results:
                    if 'created' in result['metadata']:
                        created_date = datetime.fromisoformat(result['metadata']['created'])
                        if start_date and created_date < start_date:
                            continue
                        if end_date and created_date > end_date:
                            continue
                        filtered_results.append(result)
                results = filtered_results
            
            # Get symbolic search results if tags or dates specified
            if tags or start_date or end_date:
                symbolic_results = self._symbolic_search(tags, start_date, end_date)
                # Combine semantic and symbolic results
                if symbolic_results:
                    results = self._combine_results(results, symbolic_results)
            
            # Get graph context
            graph_results = self.graph_retriever.get_context(results)
            
            # Combine results
            context = "\n\n".join([
                f"Source {i+1}:\n{result['content']}\nConnections: {', '.join(graph_results.get(result['metadata'].get('path', ''), []))}"
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