import json
import logging
import math
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from fastapi import HTTPException

# This must be before the local application imports to ensure correct module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conversation_memory import ConversationMemory  # noqa: E402
from graph_retriever import GraphRetriever  # noqa: E402
from obsidian_loader_v2 import ObsidianLoaderV2, ObsidianDocument  # noqa: E402


def sanitize_for_json(text: str) -> str:
    """
    Optimized sanitize text content for JSON serialization and OpenAI API.
    Reduced operations based on profiling analysis.
    """
    if not isinstance(text, str):
        return str(text)

    # Skip normalization for most common cases to save time
    if text.isascii():
        # Fast path for ASCII text - just remove null bytes
        return text.replace("\x00", "").replace("\ufffd", "")

    # Only normalize non-ASCII text
    text = unicodedata.normalize("NFKD", text)

    # Combined operation: remove problematic characters in single pass
    # Use translate for better performance than repeated character checking
    control_chars = {ord(c): None for c in ["\x00", "\ufffd"]}
    sanitized = text.translate(control_chars)

    # Only do expensive character validation if we suspect issues
    if any(ord(c) < 32 and c not in "\n\t\r" for c in sanitized[:100]):  # Sample check
        # Remove control characters except newlines, tabs, and carriage returns
        sanitized = "".join(
            char
            for char in sanitized
            if unicodedata.category(char)[0] != "C" or char in "\n\t\r"
        )

    return sanitized


# Add a custom JSON encoder to handle dates
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


class RAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Determine embedding provider
        embedding_config = config.get("embeddings", {})
        embedding_provider = embedding_config.get("provider", "openai")

        # Check environment variables first, then config
        env_provider = os.environ.get("EMBEDDING_PROVIDER")
        if env_provider:
            embedding_provider = env_provider

        if embedding_provider == "local":
            # For local embeddings, use LOCAL_MODEL env var or config model
            local_model = os.environ.get("LOCAL_MODEL") or embedding_config.get(
                "model", "all-MiniLM-L6-v2"
            )

            # Force faster model for speed optimization
            if "mpnet" in local_model:
                self.logger.warning(
                    f"Switching from {local_model} to all-MiniLM-L6-v2 for better speed"
                )
                local_model = "all-MiniLM-L6-v2"

            self.logger.info(f"Initializing local embedding model: {local_model}")
            self.sentence_transformer = SentenceTransformer(local_model)
            self.embed = self._embed_local

            # Set dimension based on model
            if "all-MiniLM-L6-v2" in local_model:
                self.dimension = 384
            elif "all-mpnet-base-v2" in local_model:
                self.dimension = 768
            else:
                self.dimension = 384  # Default for most sentence-transformers models
            self.batch_size = (
                150  # Optimized batch size for local processing (based on profiling)
            )
        else:
            # Initialize OpenAI embeddings
            embedding_model = os.environ.get("EMBEDDING_MODEL") or embedding_config.get(
                "model", "text-embedding-ada-002"
            )
            self.logger.info(f"Initializing OpenAI embedding model: {embedding_model}")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.client = OpenAI(api_key=api_key)
            self.embed = self._embed_openai
            self.dimension = 1536  # OpenAI dimension
            self.batch_size = 500  # Smaller batch for API

        # Initialize OpenAI client for chat completions (always needed regardless of embedding provider)
        chat_api_key = os.environ.get("OPENAI_API_KEY")
        if not chat_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for chat completions"
            )

        # If we haven't already initialized the client for embeddings, initialize it now for chat
        if not hasattr(self, "client"):
            self.client = OpenAI(api_key=chat_api_key)

        self.document_store = {}
        self.doc_embeddings = {}
        self._date_cache = {}

        vector_store_config = config.get("vector_store", {})
        self.dimension = vector_store_config.get("dimension", self.dimension)

        base_path = Path(
            vector_store_config.get("base_path", "data/vector_store")
        ).resolve()
        self.index_path = base_path / "faiss.index"
        self.doc_store_path = base_path / "document_store.json"
        self.embeddings_path = base_path / "embeddings.json"
        self.state_path = base_path / "index_state.json"

        base_path.mkdir(parents=True, exist_ok=True)

        self.is_ready = False
        try:
            self._load_or_initialize_store()
            self.is_ready = True
            self.logger.info("RAG pipeline is ready.")
        except Exception as e:
            self.logger.error(
                f"RAG pipeline initialization failed: {str(e)}", exc_info=True
            )
            # is_ready remains False

        # Initialize graph retriever (ALWAYS RUN THIS)
        self.graph_retriever = GraphRetriever(config["vault"]["path"])
        self.graph_retriever.build_graph()
        self.logger.info("Graph retriever initialized")

        # Initialize conversation memory (ALWAYS RUN THIS)
        memory_config = config.get("conversation_memory", {})
        self.conversation_memory = ConversationMemory(
            storage_dir=memory_config.get("storage_dir", "data/conversations"),
            max_history=memory_config.get("max_history", 10),
            context_window=memory_config.get("context_window", 5),
        )

    def check_health(self) -> Tuple[bool, str]:
        """
        Checks the health of the RAG pipeline.
        Returns a tuple of (is_healthy, message).
        """
        if not self.is_ready:
            return False, "RAG pipeline is not ready. Check initialization logs."
        if not self.index or self.index.ntotal == 0:
            return False, "FAISS index is not loaded or is empty."
        if not self.document_store:
            return False, "Document store is empty."
        if self.index.ntotal != len(self.document_store):
            return (
                False,
                f"Index and document store are out of sync. "
                f"Index size: {self.index.ntotal}, "
                f"Store size: {len(self.document_store)}",
            )
        return True, "RAG pipeline is healthy."

    def _atomic_json_save(self, data: Any, path: Path):
        """Saves a JSON file atomically."""
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
            os.replace(temp_path, path)
            self.logger.info(f"Successfully saved JSON to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {path}: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _load_state(self) -> Optional[Dict[str, float]]:
        """Loads the last indexed state from file."""
        if not self.state_path.exists():
            self.logger.info("No index state file found.")
            return None
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading index state: {e}")
            return None

    def _save_state(self, state: Dict[str, float]):
        """Saves the index state to file."""
        self._atomic_json_save(state, self.state_path)

    def _load_or_initialize_store(self):
        """Loads the index and document store or initializes a new one."""
        if (
            self.index_path.exists()
            and self.doc_store_path.exists()
            and self.embeddings_path.exists()
        ):
            try:
                self.logger.info(f"Loading assets from {self.index_path.parent}...")
                self.index = faiss.read_index(str(self.index_path))

                with open(self.doc_store_path, "r", encoding="utf-8") as f:
                    self.document_store = json.load(f)

                with open(self.embeddings_path, "r", encoding="utf-8") as f:
                    self.doc_embeddings = json.load(f)

                if self.index.ntotal != len(self.document_store):
                    raise ValueError(
                        "Index and document store are out of sync. " "Please re-index."
                    )
                self.logger.info(
                    f"Successfully loaded index with {self.index.ntotal} vectors."
                )
                self._populate_missing_dates_from_paths()
                return
            except Exception as e:
                self.logger.error(
                    f"Failed to load existing index/store: {e}. "
                    "Will attempt to re-initialize."
                )
                # Fall through to initialize a new store

        self.logger.info("Initializing new FAISS index and document store.")
        self._init_faiss()

    def _init_faiss(self) -> None:
        """Initializes an empty FAISS index."""
        try:
            # Ensure there's a valid dimension in the config
            dimension = self.config.get("vector_store", {}).get("dimension", 1536)
            index_path = self.config.get("vector_store", {}).get(
                "index_path", "data/vector_store/faiss.index"
            )

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
            self.index = faiss.IndexFlatL2(self.dimension)
            self.logger.info(
                f"Initialized new FAISS index with dimension {self.dimension}"
            )

            if index_path.exists():
                try:
                    self.logger.info(
                        f"Found existing index at {index_path}, attempting to load..."
                    )
                    loaded_index = faiss.read_index(str(index_path))
                    return loaded_index
                except Exception as e:
                    self.logger.warning(f"Could not load existing index: {str(e)}")
                    self.logger.info("Will create a new index instead")

            return self.index

        except Exception as e:
            self.logger.error(f"Error initializing FAISS index: {e}", exc_info=True)
            # Create an in-memory index as fallback with default dimension
            self.logger.info("Creating in-memory index as fallback with dimension 1536")
            return faiss.IndexFlatL2(1536)  # Default OpenAI embedding dimension

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using local sentence-transformers model - optimized for speed"""
        try:
            self.logger.info(
                f"Generating embeddings for {len(texts)} texts using local model (batch processing)"
            )

            # Process all texts at once for maximum speed
            embeddings = self.sentence_transformer.encode(
                texts,
                convert_to_numpy=True,
                batch_size=32,  # Optimal batch size for speed
                show_progress_bar=False,  # Disable progress bar for speed
                normalize_embeddings=True,  # Pre-normalize for better similarity search
            )

            return embeddings
        except Exception as e:
            self.logger.error(f"Local embedding error: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using OpenAI API"""
        try:
            # Get embedding model from config, with fallback
            model = self.config.get("embeddings", {}).get(
                "model", "text-embedding-ada-002"
            )
            self.logger.info(f"Using OpenAI embedding model: {model}")

            response = self.client.embeddings.create(input=texts, model=model)
            return np.array([e.embedding for e in response.data])
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            # If this is specifically a rate limit or quota error, log it clearly
            error_msg = str(e)
            if (
                "429" in error_msg
                or "rate limit" in error_msg.lower()
                or "quota" in error_msg.lower()
            ):
                self.logger.error(
                    "Hit OpenAI rate limit or quota - please check your API usage and billing details"
                )
                # If we have local embeddings stored, try to use them as a fallback
                if self._have_temp_embeddings():
                    self.logger.warning(
                        "Attempting to use local embeddings as fallback for rate-limited request"
                    )
                    try:
                        # Create a simple fallback embedding (not ideal but better than failing)
                        # This would work best with previously cached embeddings
                        # In a real system, you might use a local embeddings model as fallback
                        if len(texts) == 1 and texts[0]:
                            self.logger.info(
                                "Using fallback for single text embedding during rate limit"
                            )
                            # For single text, try a naive semantic search by direct string comparison
                            query_text = texts[0].lower()
                            # Return a fallback embedding that's all zeros except for a signature value
                            # This is not a real embedding but a placeholder to avoid breaking downstream code
                            fallback_dim = self.config.get("vector_store", {}).get(
                                "dimension", 1536
                            )
                            return np.zeros((1, fallback_dim))
                    except Exception as fallback_e:
                        self.logger.error(
                            f"Fallback embedding method failed: {str(fallback_e)}"
                        )
            raise

    def _extract_date_from_path(self, path: str) -> dict:
        """
        Extract date information from a file path with format YYYY-MM-DD
        Returns a dictionary with extracted date components or empty dict if no match
        Uses caching to avoid repeated processing of the same paths.
        """
        # Check cache first to avoid repeated processing
        if path in self._date_cache:
            return self._date_cache[path]

        date_info = {}

        # Try to extract from the last part of the path (filename)
        filename = os.path.basename(path)
        # Remove extension
        filename_without_ext = os.path.splitext(filename)[0]

        # Match YYYY-MM-DD pattern in filename
        date_pattern = r"(\d{4})-(\d{2})-(\d{2})"
        match = re.search(date_pattern, filename_without_ext)

        if not match:
            # If not found in filename, try to find in directory structure
            # Look for patterns like .../2023/04/... or .../2023-04/...
            dir_pattern1 = r"[/\\](\d{4})[/\\](\d{2})[/\\]"  # .../2023/04/...
            dir_pattern2 = r"[/\\](\d{4})-(\d{2})[/\\]"  # .../2023-04/...

            dir_match1 = re.search(dir_pattern1, path)
            if dir_match1:
                year, month = dir_match1.groups()

                # Try to extract day from filename if it's numeric
                day_match = re.search(r"^(\d{1,2})", filename_without_ext)
                day = (
                    day_match.group(1).zfill(2) if day_match else "01"
                )  # Default to first day if not found

                match = (year, month, day)
            else:
                dir_match2 = re.search(dir_pattern2, path)
                if dir_match2:
                    year, month = dir_match2.groups()
                    # Default to first day of month
                    match = (year, month, "01")

        if match:
            year, month, day = match if isinstance(match, tuple) else match.groups()
            # Create ISO format date string
            date_str = f"{year}-{month}-{day}"

            try:
                # Create datetime object to validate date is legit
                date_obj = datetime.fromisoformat(date_str)

                # Add date components to metadata
                date_info["date"] = date_str
                date_info["year"] = year
                date_info["month"] = month
                date_info["day"] = day
                date_info["created"] = date_str

                # Reduced logging frequency to avoid spam (only log at debug level)
                self.logger.debug(f"Extracted date {date_str} from path {path}")
            except ValueError:
                self.logger.warning(
                    f"Found date-like pattern in {path} but date is invalid"
                )

        # Cache the result (whether successful or not) to avoid repeated processing
        self._date_cache[path] = date_info
        return date_info

    def _process_batch(
        self, batch: List[Dict[str, Any]], batch_idx: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process a single batch of documents"""
        try:
            # Get embeddings for batch
            texts = [doc["content"] for doc in batch]
            embeddings = self.embed(texts)

            # Store document content and metadata
            processed_docs = []
            for j, doc in enumerate(batch):
                doc_id = str(
                    len(self.document_store) + (batch_idx * self.batch_size) + j
                )

                # If the document has a path, try to extract date information
                if "metadata" in doc and "path" in doc["metadata"]:
                    # Extract date from path and add to metadata
                    date_info = self._extract_date_from_path(doc["metadata"]["path"])
                    if date_info:
                        # Merge with existing metadata
                        doc["metadata"].update(date_info)

                self.document_store[doc_id] = {
                    "content": sanitize_for_json(doc["content"]),
                    "metadata": doc["metadata"],
                }
                processed_docs.append(doc)

            return embeddings, processed_docs
        except Exception as e:
            self.logger.error(f"Failed to process batch {batch_idx}: {str(e)}")
            return None, []

    def _save_index(self) -> None:
        """Saves the FAISS index and document/embedding stores atomically."""
        if not self.index:
            self.logger.warning("Attempted to save but index is not initialized.")
            return

        self.logger.info(f"Saving index and stores to {self.index_path.parent}...")
        faiss.write_index(self.index, str(self.index_path))
        self._atomic_json_save(self.document_store, self.doc_store_path)
        self._atomic_json_save(self.doc_embeddings, self.embeddings_path)
        self.logger.info("Successfully saved all data stores.")

    def _save_embeddings_temp(
        self, embeddings: List[np.ndarray], documents: List[Dict[str, Any]]
    ) -> Path:
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

        self.logger.info(
            f"Saved {len(embeddings)} embedding batches to {temp_embeddings}"
        )
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
                documents[i : i + self.batch_size]
                for i in range(0, len(documents), self.batch_size)
            ]

            # Process batches in parallel with optimized worker count (based on profiling)
            all_embeddings = []
            processed_docs = []
            with ThreadPoolExecutor(max_workers=15) as executor:
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
                        docs_per_sec = (
                            len(processed_docs) / elapsed if elapsed > 0 else 0
                        )
                        remaining_docs = len(documents) - len(processed_docs)
                        eta_seconds = (
                            remaining_docs / docs_per_sec if docs_per_sec > 0 else 0
                        )
                        pbar.set_postfix(
                            {
                                "docs/s": f"{docs_per_sec:.1f}",
                                "eta": f"{timedelta(seconds=int(eta_seconds))}",
                            }
                        )

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
                    "content": sanitize_for_json(doc["content"]),
                    "metadata": doc["metadata"],
                }
                self.doc_embeddings[str(i)] = combined_embeddings[i].tolist()

            # Save index and embeddings
            try:
                self._save_index()
            except Exception as e:
                self.logger.error(
                    f"Failed to save index after updates: {e}", exc_info=True
                )

            elapsed = time.time() - start_time
            docs_per_sec = len(documents) / elapsed if elapsed > 0 else 0
            self.logger.info(
                f"Indexing complete in {timedelta(seconds=int(elapsed))} ({docs_per_sec:.1f} docs/s)"
            )

        except Exception as e:
            self.logger.error(f"Indexing failed: {str(e)}")
            raise

    def _deduplicate_internal_links(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate internal_links in each search result's metadata
        Only process results that have metadata with internal_links
        """
        for result in results:
            # Skip if metadata or internal_links don't exist
            if not result.get("metadata") or "internal_links" not in result["metadata"]:
                continue

            internal_links = result["metadata"]["internal_links"]
            if not internal_links or not isinstance(internal_links, list):
                continue

            # Deduplicate case-insensitively
            links_lower_dict = {}
            for link in internal_links:
                link_lower = link.lower()
                # If we haven't seen this link yet, or this one has a better case format, keep it
                if link_lower not in links_lower_dict:
                    links_lower_dict[link_lower] = link

            # Replace with deduplicated list
            if links_lower_dict:
                result["metadata"]["internal_links"] = list(links_lower_dict.values())
            else:
                # Remove the key if empty to save space
                del result["metadata"]["internal_links"]

        return results

    def _preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocess the query to extract semantic meaning and special filters

        Returns:
            Dictionary containing extracted information and enhanced query
        """
        query_info = {
            "original_query": query,
            "enhanced_query": query,
            "extracted_year": None,
            "extracted_date": None,
            "date_range": {"start_date": None, "end_date": None},
            "path_filters": [],
        }

        # Extract year references (e.g., "in 2023", "from 2022", "during 2021")
        year_pattern = r"\b(in|from|during|for|about|of)\s+(\d{4})\b"
        year_match = re.search(year_pattern, query, re.IGNORECASE)

        if year_match:
            year = year_match.group(2)
            query_info["extracted_year"] = year
            self.logger.info(f"Extracted year from query: {year}")

            # Add path filter for folders containing this year
            query_info["path_filters"].append(f"/{year}/")
            query_info["path_filters"].append(f"\\{year}\\")  # Windows path format

            # Set date range for the entire year
            start_date = datetime(int(year), 1, 1, tzinfo=timezone.utc)
            end_date = datetime(int(year), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            query_info["date_range"]["start_date"] = start_date
            query_info["date_range"]["end_date"] = end_date

        return query_info

    def _apply_path_filtering(
        self, results: List[Dict[str, Any]], path_filters: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter or boost results based on path patterns
        """
        if not path_filters:
            return results

        # First pass: check if any results match the path filters
        matching_results = []
        for result in results:
            path = result.get("metadata", {}).get("path", "")
            if any(filter_pattern in path for filter_pattern in path_filters):
                # These are exact matches to our path pattern, give a boost
                result["score"] = result["score"] * 1.5  # 50% boost
                matching_results.append(result)

        # If we found matches, prioritize them but keep others
        if matching_results:
            # Keep other results but add them after matching results
            non_matching = [r for r in results if r not in matching_results]
            return matching_results + non_matching

        # If no exact matches, return original results
        return results

    def _find_documents_by_year(
        self, year: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Directly find documents from a specific year based on path and metadata
        Used as a fallback when semantic search doesn't find relevant content
        """
        results = []
        year_pattern = f"/{year}/"
        alt_year_pattern = f"\\{year}\\"  # Windows path alternative

        # Check each document in store for year pattern in path or year in metadata
        for doc_id, doc in self.document_store.items():
            path = doc.get("metadata", {}).get("path", "")

            # Check if year is in path
            year_in_path = year_pattern in path or alt_year_pattern in path

            # Check if year is in metadata fields
            year_in_metadata = False
            metadata = doc.get("metadata", {})

            # Check if the year matches explicitly stored year field
            if metadata.get("year") == year:
                year_in_metadata = True

            # Check for year in date string fields
            for date_field in ["created", "modified", "date"]:
                if date_field in metadata and metadata[date_field]:
                    date_str = str(metadata[date_field])
                    if year in date_str:
                        year_in_metadata = True
                        break

            # If year is found in path or metadata, add to results
            if year_in_path or year_in_metadata:
                results.append(
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": (
                            0.95 if year_in_path else 0.8
                        ),  # Higher score if in path
                    }
                )

        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _symbolic_search(
        self,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        year: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Symbolic (non-vector) search based on tags, dates, and paths"""

        # Placeholder for a proper symbolic search implementation
        results = []

        # Special case for year-based queries (direct path/metadata search)
        if year:
            year_results = self._find_documents_by_year(year)
            if year_results:
                results.extend(year_results)
                self.logger.info(
                    f"Found {len(year_results)} documents directly by year {year}"
                )

        return results

    def query(
        self,
        query: str,
        k: int = 5,
        session_id: Optional[str] = None,
        conversation_memory: Optional[ConversationMemory] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        retrieval_mode: str = "vector",
        date_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Queries the RAG pipeline with enhanced filtering and context.
        """
        health_ok, message = self.check_health()
        if not health_ok:
            self.logger.error(f"Query failed health check: {message}")
            raise HTTPException(
                status_code=503,
                detail=f"Service Unavailable: {message}. Please try re-indexing the vault.",
            )

        self.logger.info(f"Received query: '{query}' with mode '{retrieval_mode}'")
        start_time = time.time()
        try:
            # Use instance conversation memory if none provided
            conversation_memory = conversation_memory or self.conversation_memory

            # Generate session ID if none provided
            if not session_id:
                session_id = str(int(time.time()))

            # Preprocess query to extract semantic meaning, years, dates
            query_info = self._preprocess_query(query)

            # Override date parameters if extracted from query and not explicitly provided
            if not start_date and query_info["date_range"]["start_date"]:
                start_date = query_info["date_range"]["start_date"]
                self.logger.info(f"Using start_date from query: {start_date}")

            if not end_date and query_info["date_range"]["end_date"]:
                end_date = query_info["date_range"]["end_date"]
                self.logger.info(f"Using end_date from query: {end_date}")

            # Get conversation context
            conversation_context = (
                conversation_memory.get_context_window(session_id) if session_id else ""
            )
            self.logger.info(
                f"Session [{session_id}] Context window retrieved."
            )  # Log context retrieval

            # Perform semantic search
            self.logger.info(
                f"Session [{session_id}] Performing semantic search for: '{query}'"
            )  # Log query
            query_vector = self.embed([query])[0]
            D, I = self.index.search(query_vector.reshape(1, -1), k)
            self.logger.info(
                f"Session [{session_id}] Raw semantic search IDs: {I.tolist()}, Scores: {D.tolist()}"
            )  # Log raw results

            # Get results
            semantic_results = []
            for i, idx in enumerate(I[0]):
                if idx == -1 or str(idx) not in self.document_store:
                    continue
                doc = self.document_store[str(idx)]
                semantic_results.append(
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": float(D[0][i]),
                    }
                )
            self.logger.info(
                f"Session [{session_id}] Initial semantic results count: {len(semantic_results)}"
            )  # Log initial count

            # Apply path-based filtering from query preprocessing
            if query_info["path_filters"]:
                semantic_results = self._apply_path_filtering(
                    semantic_results, query_info["path_filters"]
                )
                self.logger.info(
                    f"Session [{session_id}] Results after path filtering: {len(semantic_results)}"
                )

            # --- Filtering Logic (Tags and Dates) ---
            filtered_results = semantic_results

            # Apply tag filtering if specified
            if tags:
                self.logger.info(
                    f"Session [{session_id}] Filtering results by tags: {tags}"
                )
                filtered_results = [
                    result
                    for result in filtered_results
                    if "tags" in result["metadata"]
                    and any(tag in result["metadata"].get("tags", []) for tag in tags)
                ]
                self.logger.info(
                    f"Session [{session_id}] Results count after tag filter: {len(filtered_results)}"
                )  # Log after tag filter

            # Apply date filtering if specified
            if start_date or end_date:
                self.logger.info(
                    f"Session [{session_id}] Filtering results by date range: {start_date} to {end_date}"
                )
                temp_results = []
                for result in filtered_results:
                    if "created" in result["metadata"]:
                        try:
                            # Ensure created_date is offset-aware if start/end dates are
                            created_date_str = result["metadata"]["created"]
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
                            self.logger.warning(
                                f"Could not parse date {result['metadata'].get('created')}: {e}"
                            )
                            continue  # Skip if date is invalid
                    else:
                        # Decide if docs without dates should be included or excluded
                        # Let's exclude them for now if a date range is specified
                        pass
                filtered_results = temp_results
                self.logger.info(
                    f"Session [{session_id}] Results count after date filter: {len(filtered_results)}"
                )  # Log after date filter

            # Get symbolic search results (if tags or dates specified)
            # Placeholder: Symbolic search currently returns empty
            symbolic_results = []
            if tags or start_date or end_date or query_info["extracted_year"]:
                symbolic_results = self._symbolic_search(
                    tags, start_date, end_date, query_info["extracted_year"]
                )

            # Combine results (Placeholder: currently just uses filtered semantic results)
            final_results = filtered_results  # Replace/augment with combined results if symbolic search is implemented
            if symbolic_results:  # Example basic combination (avoid duplicates)
                final_results_paths = {r["metadata"].get("path") for r in final_results}
                for sr in symbolic_results:
                    if sr["metadata"].get("path") not in final_results_paths:
                        final_results.append(sr)
            self.logger.info(
                f"Session [{session_id}] Final results count before graph context: {len(final_results)}"
            )  # Log final count before graph

            # --- Get graph context using the correct method ---
            graph_context_map = {}
            self.logger.info("Retrieving graph context for results...")
            for result in final_results:
                doc_path = result["metadata"].get(
                    "path"
                )  # Assuming 'path' is in metadata
                if doc_path:
                    related_docs = self.graph_retriever.get_related_documents(
                        doc_path, max_distance=1
                    )
                    # Store related document paths (excluding the source itself)
                    graph_context_map[doc_path] = [
                        rel["document"]
                        for rel in related_docs
                        if rel["document"] != doc_path
                    ]
                else:
                    self.logger.warning(
                        f"Document missing 'path' in metadata: {result.get('metadata')}"
                    )
            self.logger.info(f"Graph context map: {graph_context_map}")
            # --- End Graph Context Update ---

            # Combine results for the final context string
            context_parts = []
            for i, result in enumerate(final_results):
                doc_path = result["metadata"].get("path", f"doc_{i}")
                connections = graph_context_map.get(doc_path, [])
                connections_str = (
                    ", ".join([f'"{conn}"' for conn in connections])
                    if connections
                    else "None"
                )

                # Include more detailed metadata for date-based queries
                metadata = result["metadata"]
                metadata_parts = []

                # Add title
                metadata_parts.append(f"title: {metadata.get('title', 'Untitled')}")

                # Add tags if present
                if "tags" in metadata and metadata["tags"]:
                    metadata_parts.append(f"tags: {', '.join(metadata['tags'][:5])}")

                # Add date info if present
                for date_field in ["created", "modified", "date"]:
                    if date_field in metadata and metadata[date_field]:
                        metadata_parts.append(f"{date_field}: {metadata[date_field]}")

                # Add year/month/day if present (important for date queries)
                for time_field in ["year", "month", "day"]:
                    if time_field in metadata and metadata[time_field]:
                        metadata_parts.append(f"{time_field}: {metadata[time_field]}")

                # Format metadata
                metadata_str = ", ".join(metadata_parts)

                # Sanitize content before adding to context
                sanitized_content = sanitize_for_json(result["content"])

                # Create the full context entry
                context_parts.append(
                    f"Source {i+1} (Path: {doc_path}):\n{sanitized_content}\nMetadata: {metadata_str}\nConnections: {connections_str}"
                )
            context = "\n\n".join(context_parts)
            self.logger.info(
                f"Session [{session_id}] Context generated for LLM (length: {len(context)}):\n"
                f"--- START CONTEXT ---\n"
                f"{context[:500]}...\n"
                f"--- END CONTEXT ---"
            )  # Log context sample

            # Build messages for chat completion
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful assistant that synthesizes information from multiple sources and maintains context across conversations. 
When responding to date-based queries (e.g., "What happened in 2023?"), pay special attention to:
1. File paths that contain date information (e.g., "2023/04/file.md")
2. Date metadata fields (created, modified, date, year, month, day)
3. Content that mentions specific dates or timeframes

Pay attention to the graph connections between sources and explain how they relate to each other.
When referring to previous conversations, be natural and contextual in your responses.

For year-based queries, organize your response chronologically if possible, and explicitly mention timeframes.
Be specific about what information comes from which source files.""",
                },
            ]

            # Add conversation history if available
            if conversation_context:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Previous conversation context:\n{conversation_context}",
                    }
                )

            # Add current query and context
            user_content = f"Question: {query}\n\nContext:\n{context}\n\nPlease provide a comprehensive answer based on the sources above, highlighting any connections between them and relating to our previous conversation where relevant."

            # Check token count and truncate context if needed
            try:
                # Import tiktoken for token counting
                import tiktoken

                # Get the encoding for the model we're using
                model = self.config.get("chat_model", {}).get(
                    "name", "gpt-4-turbo-preview"
                )
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fallback to cl100k_base encoding if the model is not found
                    encoding = tiktoken.get_encoding("cl100k_base")

                # Count tokens in system message
                system_tokens = len(encoding.encode(messages[0]["content"]))

                # Count tokens in conversation context if present
                conv_context_tokens = 0
                if conversation_context:
                    conv_context_tokens = len(
                        encoding.encode(
                            f"Previous conversation context:\n{conversation_context}"
                        )
                    )

                # Count tokens in query
                query_tokens = len(encoding.encode(f"Question: {query}\n\n"))

                # Count tokens in final instruction
                final_instruction_tokens = len(
                    encoding.encode(
                        "\n\nPlease provide a comprehensive answer based on the sources above, highlighting any connections between them and relating to our previous conversation where relevant."
                    )
                )

                # Calculate how many tokens we have available for the context
                # Get model configuration
                chat_model_config = self.config.get("chat_model", {})
                model_name = chat_model_config.get("name", "gpt-4o-mini")
                max_response_tokens = chat_model_config.get("max_response_tokens", 4000)
                
                # Get context limit from config or auto-detect based on model
                max_total_tokens = chat_model_config.get("max_context_tokens")
                if not max_total_tokens:
                    # Auto-detect based on model if not configured
                    if "gpt-4o" in model_name:
                        max_total_tokens = 128000  # GPT-4o context limit
                    elif "gpt-4" in model_name and "turbo" in model_name:
                        max_total_tokens = 128000  # GPT-4 Turbo context limit  
                    else:
                        max_total_tokens = 16384   # Default for other models
                
                # Reserve tokens for response and add buffer
                available_context_tokens = (
                    max_total_tokens
                    - system_tokens
                    - conv_context_tokens
                    - query_tokens
                    - final_instruction_tokens
                    - max_response_tokens  # Use configured response token limit
                    - 1000  # Buffer for safety
                )

                # Count tokens in the context
                context_tokens = len(encoding.encode(f"Context:\n{context}"))

                # If context is too large, truncate it
                if (
                    context_tokens > available_context_tokens
                    and available_context_tokens > 0
                ):
                    self.logger.warning(
                        f"Context too large ({context_tokens} tokens). Truncating to {available_context_tokens} tokens."
                    )

                    # Split the context into individual documents
                    context_docs = context.split("\n\n[")
                    if len(context_docs) > 1:
                        # First item doesn't start with [, so handle it separately
                        truncated_context = [context_docs[0]]

                        remaining_tokens = available_context_tokens - len(
                            encoding.encode(context_docs[0])
                        )

                        # Add documents one by one until we reach the token limit
                        for doc in context_docs[1:]:
                            # Add back the "[" that was removed during splitting
                            doc_text = "[" + doc
                            doc_tokens = len(encoding.encode(doc_text))

                            if remaining_tokens >= doc_tokens:
                                truncated_context.append(doc_text)
                                remaining_tokens -= doc_tokens
                            else:
                                # No more room for complete documents
                                break

                        # Join the truncated context
                        context = "\n\n".join(truncated_context)

                        # Add a note about truncation
                        if len(truncated_context) < len(context_docs):
                            truncation_note = "\n\n[Note: Some documents were omitted due to context length limitations.]"
                            truncation_note_tokens = len(
                                encoding.encode(truncation_note)
                            )

                            if remaining_tokens >= truncation_note_tokens:
                                context += truncation_note

                        # Update the user content with truncated context
                        user_content = f"Question: {query}\n\nContext:\n{context}\n\nPlease provide a comprehensive answer based on the sources above, highlighting any connections between them and relating to our previous conversation where relevant."

                        self.logger.info(
                            f"Context truncated to {len(encoding.encode(context))} tokens ({len(truncated_context)} of {len(context_docs)} documents)."
                        )
            except ImportError:
                self.logger.warning(
                    "tiktoken not available, skipping token counting and truncation"
                )
            except Exception as e:
                self.logger.warning(
                    f"Error during token counting and truncation: {str(e)}"
                )

            # Sanitize all message content to ensure valid JSON
            for message in messages:
                message["content"] = sanitize_for_json(message["content"])

            # Add the user message with possibly truncated context
            messages.append(
                {"role": "user", "content": sanitize_for_json(user_content)}
            )

            # Log the final message structure for debugging
            self.logger.info(
                f"Session [{session_id}] Sending {len(messages)} messages to OpenAI"
            )
            for i, msg in enumerate(messages):
                content_preview = (
                    msg["content"][:100] + "..."
                    if len(msg["content"]) > 100
                    else msg["content"]
                )
                self.logger.info(f"  Message {i+1} ({msg['role']}): {content_preview}")

            # Generate response
            try:
                # Get configuration values with proper defaults
                chat_model_config = self.config.get("chat_model", {})
                model_name = chat_model_config.get("name", "gpt-4o-mini")
                temperature = chat_model_config.get("temperature", 0.7)
                max_response_tokens = chat_model_config.get("max_response_tokens", 4000)
                
                self.logger.info(f"Generating response with model: {model_name}, max_tokens: {max_response_tokens}")
                
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_response_tokens,
                )
            except Exception as api_error:
                self.logger.error(f"OpenAI API error: {str(api_error)}")
                # Log the exact request that failed for debugging
                self.logger.error(f"Failed request model: {model_name}")
                self.logger.error(f"Failed request messages count: {len(messages)}")
                for i, msg in enumerate(messages):
                    self.logger.error(
                        f"Message {i+1} role: {msg['role']}, length: {len(msg['content'])}"
                    )
                raise

            # Store the interaction in conversation memory
            try:
                conversation_memory.add_interaction(
                    session_id=session_id,
                    user_message=query,
                    assistant_message=response.choices[0].message.content,
                )
            except Exception as e:
                self.logger.warning(f"Failed to store conversation: {str(e)}")

            # Deduplicate internal_links in results before returning
            final_results = self._deduplicate_internal_links(final_results)

            return {
                "response": response.choices[0].message.content,
                "sources": final_results,
                "session_id": session_id,
                "context": context,  # Include the context for debugging
            }

        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        symbolic_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine and rerank semantic and symbolic search results
        """
        # Implement smart combination strategy
        # This is a placeholder - implement actual logic
        return semantic_results

    def _generate_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a natural language response using the configured model
        """
        context = "\n".join(
            [f"Source {i+1}: {result['excerpt']}" for i, result in enumerate(results)]
        )

        # Get configuration values
        chat_model_config = self.config.get("chat_model", {})
        model_name = chat_model_config.get("name", "gpt-4o-mini")
        temperature = chat_model_config.get("temperature", 0.7)
        max_response_tokens = chat_model_config.get("max_response_tokens", 4000)

        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the user's personal notes and journal entries. Synthesize information from the provided sources and maintain a thoughtful, reflective tone.",
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nRelevant sources:\n{context}",
                },
            ],
            temperature=temperature,
            max_tokens=max_response_tokens,
        )

        return response.choices[0].message.content

    def _populate_missing_dates_from_paths(self):
        for doc_id, doc in self.document_store.items():
            path = doc["metadata"].get("path", "")
            if path:
                date_info = self._extract_date_from_path(path)
                if date_info:
                    doc["metadata"].update(date_info)
                    self.logger.info(
                        f"Updated metadata for document {doc_id} with date information"
                    )
                else:
                    self.logger.warning(
                        f"No date information found for document {doc_id} with path: {path}"
                    )
            else:
                self.logger.warning(f"Document {doc_id} has no path in metadata")

    def sync_vault(self) -> Dict[str, Any]:
        """
        Performs an incremental sync of the vault.
        Detects changes and surgically updates the index.
        """
        self.logger.info("Starting vault synchronization...")

        # 1. Load the last state
        last_state = self._load_state()

        # 2. Use ObsidianLoaderV2 to get a diff of changes
        loader = ObsidianLoaderV2(self.config["vault"]["path"])
        changes = loader.load_vault(last_indexed_state=last_state, config=self.config)

        docs_to_add = changes["added"]
        docs_to_update = changes["updated"]
        paths_to_delete = changes["deleted"]
        new_state = changes["new_state"]

        if not docs_to_add and not docs_to_update and not paths_to_delete:
            self.logger.info("Vault is already up to date. No sync needed.")
            return {"status": "success", "message": "Vault is already up to date."}

        # 3. Determine which documents to remove from the current store
        paths_to_remove = set(paths_to_delete)
        for doc in docs_to_update:
            paths_to_remove.add(doc.source)

        # 4. Create the new document and embedding stores in memory
        new_document_store = {}

        # Keep documents that are not being removed
        for doc_id, doc in self.document_store.items():
            if doc["metadata"]["source"] not in paths_to_remove:
                new_document_store[doc_id] = doc

        # 5. Combine new/updated docs with the kept docs
        final_docs_for_index = list(new_document_store.values())

        newly_processed_docs = []
        for doc in docs_to_add + docs_to_update:
            # We assume ObsidianDocument has a to_dict() method
            # and that the 'id' can be constructed for index_documents
            newly_processed_docs.append(
                {
                    "id": f"{doc.source}#{doc.chunk_id}",
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        final_docs_for_index.extend(newly_processed_docs)

        # 6. Re-index with the final list of documents
        self.logger.info(f"Re-indexing {len(final_docs_for_index)} total documents.")
        if final_docs_for_index:
            self.index_documents(final_docs_for_index)
        else:  # Everything was deleted
            self._init_faiss()
            self.document_store = {}
            self.doc_embeddings = {}
            self._save_index()  # saves empty index and stores

        # 7. Save the new state
        self._save_state(new_state)

        self.logger.info("Vault synchronization complete.")
        return {
            "status": "success",
            "message": "Vault synchronized successfully.",
            "added": len(docs_to_add),
            "updated": len(docs_to_update),
            "deleted": len(paths_to_delete),
        }
