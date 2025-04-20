from typing import List, Optional, Dict, Any, Tuple
import os
import faiss
import numpy as np
from datetime import datetime, timedelta, timezone
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
        # Initialize OpenAI client with API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        # Simple initialization with only the API key
        self.client = OpenAI(api_key=api_key)
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
        
        index_loaded_from_disk = False
        # Try loading existing index first
        if self.index_path.exists() and self.embeddings_path.exists():
            try:
                self.logger.info(f"Loading existing index from {self.index_path}")
                self.index = faiss.read_index(str(self.index_path))
                with open(self.embeddings_path, 'r') as f:
                    self.doc_embeddings = json.load(f)
                
                # Populate document_store from loaded embeddings metadata
                # Assuming embeddings.json structure is { "id": {"content": "...", "metadata": {...}} }
                # If not, this needs adjustment based on actual embeddings.json structure.
                # Let's assume for now doc_embeddings stores { "id": embedding_vector_list } 
                # and we need a separate file or mechanism to get content/metadata.
                # 
                # *** Revision: Based on index_documents, embeddings.json looks like { "id": vector }
                # *** and document_store needs { "id": {"content": ..., "metadata": ...} }
                # *** The current code DOES NOT save content/metadata persistently alongside embeddings.json!
                # *** We need to save self.document_store to a file as well.
                # 
                # Let's save document_store to document_store.json during index_documents
                # and load it here.
                
                document_store_path = self.embeddings_path.parent / "document_store.json"
                if document_store_path.exists():
                    self.logger.info(f"Loading document store from {document_store_path}")
                    with open(document_store_path, 'r') as f:
                        self.document_store = json.load(f)
                        # Ensure keys are strings if Faiss IDs are used as keys directly
                        self.document_store = {str(k): v for k, v in self.document_store.items()} 
                else:
                     self.logger.warning(f"Document store file not found at {document_store_path}. Document content will be unavailable unless re-indexed.")
                     self.document_store = {} # Ensure it's initialized if file not found
                
                self.logger.info(f"Successfully loaded index with {self.index.ntotal} vectors and associated data.")
                index_loaded_from_disk = True # Mark as loaded
            except Exception as e:
                self.logger.error(f"Failed to load existing index or document store: {str(e)}")
                self.logger.info("Will try loading from temp embeddings")
        
        # If index wasn't loaded from disk, try temp or create new
        if not index_loaded_from_disk:
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
                        # Save the document store as well
                        document_store_path = self.embeddings_path.parent / "document_store.json"
                        with open(document_store_path, 'w') as f:
                            json.dump(self.document_store, f)
                        self.logger.info("Saved index, embeddings, and document store to disk")
                    except Exception as e:
                        self.logger.warning(f"Failed to save index to disk: {str(e)}")
                except Exception as e:
                    self.logger.error(f"Failed to load temp embeddings: {str(e)}")
                    self.logger.info("Creating empty index")
                    self.index = self._init_faiss()
            else:
                self.logger.info("No existing or temporary embeddings found, creating empty index")
                self.index = self._init_faiss()
            
        # Initialize graph retriever (ALWAYS RUN THIS)
        self.graph_retriever = GraphRetriever(config["vault"]["path"])
        self.graph_retriever.build_graph()
        self.logger.info("Graph retriever initialized")
        
        # Initialize conversation memory (ALWAYS RUN THIS)
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
                # Save the document store as well
                document_store_path = self.embeddings_path.parent / "document_store.json"
                with open(document_store_path, 'w') as f:
                    json.dump(self.document_store, f)
                self.logger.info("Saved index, embeddings, and document store to disk")
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
            self.logger.info(f"Session [{session_id}] Context window retrieved.") # Log context retrieval
            
            # Perform semantic search
            self.logger.info(f"Session [{session_id}] Performing semantic search for: '{query}'") # Log query
            query_vector = self.embed([query])[0]
            D, I = self.index.search(query_vector.reshape(1, -1), k)
            self.logger.info(f"Session [{session_id}] Raw semantic search IDs: {I.tolist()}, Scores: {D.tolist()}") # Log raw results
            
            # Get results
            semantic_results = []
            for i, idx in enumerate(I[0]):
                if idx == -1 or str(idx) not in self.document_store:
                    continue
                doc = self.document_store[str(idx)]
                semantic_results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(D[0][i])
                })
            self.logger.info(f"Session [{session_id}] Initial semantic results count: {len(semantic_results)}") # Log initial count
            
            # --- Filtering Logic (Tags and Dates) ---
            filtered_results = semantic_results
            # Apply tag filtering if specified
            if tags:
                self.logger.info(f"Session [{session_id}] Filtering results by tags: {tags}")
                filtered_results = [
                    result for result in filtered_results
                    if 'tags' in result['metadata'] and 
                    any(tag in result['metadata'].get('tags', []) for tag in tags)
                ]
                self.logger.info(f"Session [{session_id}] Results count after tag filter: {len(filtered_results)}") # Log after tag filter
            
            # Apply date filtering if specified
            if start_date or end_date:
                self.logger.info(f"Session [{session_id}] Filtering results by date range: {start_date} to {end_date}")
                temp_results = []
                for result in filtered_results:
                    if 'created' in result['metadata']:
                        try:
                            # Ensure created_date is offset-aware if start/end dates are
                            created_date_str = result['metadata']['created']
                            created_date = datetime.fromisoformat(created_date_str)
                            
                            # Make created_date timezone-aware if needed (assuming UTC if no tz)
                            if created_date.tzinfo is None:
                                created_date = created_date.replace(tzinfo=timezone.utc)
                            if start_date and start_date.tzinfo is None:
                                start_date = start_date.replace(tzinfo=timezone.utc)
                            if end_date and end_date.tzinfo is None:
                                end_date = end_date.replace(tzinfo=timezone.utc)

                            # Perform comparison
                            if start_date and created_date < start_date:
                                continue
                            if end_date and created_date > end_date:
                                continue
                            temp_results.append(result)
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Could not parse date {result['metadata'].get('created')}: {e}")
                            continue # Skip if date is invalid
                    else:
                         # Decide if docs without dates should be included or excluded
                         # Let's exclude them for now if a date range is specified
                         pass 
                filtered_results = temp_results
                self.logger.info(f"Session [{session_id}] Results count after date filter: {len(filtered_results)}") # Log after date filter

            # Get symbolic search results (if tags or dates specified)
            # Placeholder: Symbolic search currently returns empty
            symbolic_results = [] 
            if tags or start_date or end_date:
                 symbolic_results = self._symbolic_search(tags, start_date, end_date)
            
            # Combine results (Placeholder: currently just uses filtered semantic results)
            final_results = filtered_results # Replace/augment with combined results if symbolic search is implemented
            if symbolic_results: # Example basic combination (avoid duplicates)
                final_results_paths = {r['metadata'].get('path') for r in final_results}
                for sr in symbolic_results:
                    if sr['metadata'].get('path') not in final_results_paths:
                        final_results.append(sr)
            self.logger.info(f"Session [{session_id}] Final results count before graph context: {len(final_results)}") # Log final count before graph
            
            # --- Get graph context using the correct method ---
            graph_context_map = {}
            self.logger.info("Retrieving graph context for results...")
            for result in final_results:
                doc_path = result['metadata'].get('path') # Assuming 'path' is in metadata
                if doc_path:
                    related_docs = self.graph_retriever.get_related_documents(doc_path, max_distance=1)
                    # Store related document paths (excluding the source itself)
                    graph_context_map[doc_path] = [rel['document'] for rel in related_docs if rel['document'] != doc_path]
                else:
                    self.logger.warning(f"Document missing 'path' in metadata: {result.get('metadata')}")
            self.logger.info(f"Graph context map: {graph_context_map}")
            # --- End Graph Context Update ---
            
            # Combine results for the final context string
            context_parts = []
            for i, result in enumerate(final_results):
                doc_path = result['metadata'].get('path', f'doc_{i}')
                connections = graph_context_map.get(doc_path, [])
                connections_str = ', '.join(connections) if connections else 'None'
                context_parts.append(
                    f"Source {i+1} (Path: {doc_path}):\n{result['content']}\nConnections: {connections_str}"
                )
            context = "\n\n".join(context_parts)
            self.logger.info(f"Session [{session_id}] Context generated for LLM (length: {len(context)}):\n"
                            f"--- START CONTEXT ---\n"
                            f"{context[:500]}...\n"
                            f"--- END CONTEXT ---") # Log context sample
            
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
                "sources": final_results,
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