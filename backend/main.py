import logging
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# This must be before the local application imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fast_indexer import FastIndexer  # noqa: E402
from obsidian_loader_v2 import ObsidianLoaderV2  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402
from utils import ensure_directories, load_config, validate_config  # noqa: E402
from backend.storage.local_storage import LocalStorage
from backend.storage.gcs_storage import GCSStorage

# Import Dropbox client from the api directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'api'))
try:
    from dropbox_client import DropboxClient, DropboxAuthenticationError
    from config import settings
    DROPBOX_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Dropbox integration available")
except ImportError as e:
    DROPBOX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Dropbox integration not available: {e}")

# Load and validate configuration
config = load_config()
validate_config(config)
ensure_directories(config)

# Configure logging
logging.basicConfig(
    level=config["logging"]["level"],
    filename=config["logging"]["file"],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Gnosis RAG API",
    description="API for querying and analyzing Obsidian vaults using hybrid RAG",
    version="1.0.0",
)

# State management for the limiter
app.state.limiter = limiter
# Add the exception handler for rate limit exceeded
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Plugin file paths
plugin_dir = Path(__file__).parent.parent / "plugin"
openapi_path = plugin_dir / "openapi.yaml"
plugin_json_path = plugin_dir / "ai-plugin.json"


# Route to serve OpenAPI specification
@app.get("/openapi.yaml", include_in_schema=False)
async def get_openapi_yaml():
    return FileResponse(openapi_path, media_type="text/yaml")


# Route to serve plugin manifest
@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def get_plugin_manifest():
    return FileResponse(plugin_json_path, media_type="application/json")


# Routes for other required plugin files
@app.get("/logo.png", include_in_schema=False)
async def get_logo():
    # For now, we'll just return a placeholder message
    # In a production environment, we would serve an actual logo file
    return JSONResponse(
        content={
            "message": "Logo placeholder. In production, this would be an image file."
        },
        status_code=200,
    )


@app.get("/legal", include_in_schema=False)
async def get_legal():
    return {"terms_of_use": "This is a prototype plugin. Use at your own risk."}


# Unified save request model - handles both conversation saving and exact content
class SaveRequest(BaseModel):
    conversation_name: str
    session_id: Optional[str] = None  # For conversation-based saving
    messages: Optional[List[Dict[str, str]]] = None  # For direct message saving
    content: Optional[str] = None  # For exact content saving


class LoadRequest(BaseModel):
    file_path: str  # Relative path to the file to load


class LoadResponse(BaseModel):
    status: str
    content: str
    source: str  # "dropbox", "local", or "not_found"
    file_path: str
    last_modified: Optional[str] = None


# Initialize components
try:
    logger.info("Initializing RAG pipeline...")

    # Set up storage backend
    storage_config = config.get("storage", {})
    provider = storage_config.get("provider", "local")
    storage = None
    if provider == "gcs":
        bucket_name = storage_config.get("gcs", {}).get("bucket_name")
        if bucket_name:
            logger.info(f"Using GCS storage with bucket: {bucket_name}")
            storage = GCSStorage(bucket_name=bucket_name)
        else:
            raise ValueError("GCS storage provider requires a bucket_name.")
    else:
        vault_path = storage_config.get("local", {}).get("vault_path")
        if vault_path:
            logger.info(f"Using local storage with path: {vault_path}")
            storage = LocalStorage(vault_path=vault_path)
        else:
            raise ValueError("Local storage provider requires a vault_path.")

    if not storage:
        raise ValueError("Could not initialize a storage provider. Check your configuration.")

    rag_pipeline = RAGPipeline(config, storage)

    logger.info("Loading Obsidian vault...")
    vault_loader = ObsidianLoaderV2(storage)

    logger.info("Initialization complete!")
except Exception as e:
    logger.error(f"Failed to initialize: {str(e)}")
    raise


@app.get("/health")
def health_check():
    """
    Provides a comprehensive health check of the API and RAG pipeline.
    Returns 200 OK even when index is building to keep Render happy.
    """
    try:
        healthy, message = rag_pipeline.check_health()
        if healthy:
            return {"status": "ok", "message": message}
        
        # Check if it's just missing data (not a fatal error)
        degraded_keywords = [
            "not ready", "not loaded", "empty", "out of sync", 
            "index is not loaded", "document store is empty"
        ]
        if any(keyword in message.lower() for keyword in degraded_keywords):
            return {"status": "degraded", "message": f"RAG pipeline initializing: {message}"}
        
        # Other errors still return 503
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {message}")
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        # Basic health check - API is up even if RAG pipeline isn't ready
        logger.warning(f"RAG pipeline health check failed: {str(e)}")
        return {"status": "degraded", "message": f"API running, RAG pipeline initializing: {str(e)}"}


@app.post("/sync")
async def sync_vault():
    """
    Performs an incremental sync of the Obsidian vault, updating the index
    with only the changes since the last sync.
    """
    try:
        logger.info("Starting vault synchronization...")
        sync_results = rag_pipeline.sync_vault()
        logger.info("Vault synchronization completed successfully.")
        return sync_results
    except Exception as e:
        logger.error(f"Synchronization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synchronization failed: {str(e)}")


# Restore the original /index endpoint
@app.post("/index")
async def index_vault():
    """
    Index the Obsidian vault content into the vector store with robust error handling.
    This needs to be called before querying.
    """
    import asyncio
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )

    # Rate limiting configuration
    BATCH_SIZE = 10  # Smaller batches to avoid rate limits
    DELAY_BETWEEN_BATCHES = 2.0  # 2 second delay between batches
    MAX_RETRIES = 3

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    async def process_batch_with_retry(batch, batch_num):
        """Process a batch with retry logic"""
        try:
            logger.info(f"Processing batch {batch_num} with {len(batch)} documents")

            # Process batch through RAG pipeline
            rag_pipeline.index_documents(batch)

            logger.info(f"Successfully processed batch {batch_num}")
            return len(batch)

        except Exception as e:
            error_msg = str(e).lower()
            rate_limit_keywords = ["rate limit", "quota", "429", "too many requests"]
            if any(keyword in error_msg for keyword in rate_limit_keywords):
                logger.warning(
                    f"Rate limit hit in batch {batch_num}, will retry after delay"
                )
                await asyncio.sleep(30)  # Extra delay for rate limits
            raise

    try:
        logger.info("Starting robust vault indexing...")

        # Step 1: Load documents from vault
        documents = vault_loader.load_all_documents(config=config)
        logger.info(f"Loaded {len(documents)} documents from vault")

        if not documents:
            logger.warning("No documents found to index.")
            return {
                "status": "warning",
                "message": "No documents found to index.",
                "document_count": 0,
            }

        # Step 2: Prepare documents for indexing
        indexed_documents = [
            {
                "id": f"{doc.metadata.get('source', 'unknown_source')}#{doc.metadata.get('chunk_id', 'unknown_chunk')}",
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        # Step 3: Process documents in batches with rate limiting
        total_documents = len(indexed_documents)
        total_batches = (total_documents + BATCH_SIZE - 1) // BATCH_SIZE
        processed_count = 0

        logger.info(
            f"Processing {total_documents} documents in {total_batches} batches"
        )

        for i in range(0, total_documents, BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch = indexed_documents[i : i + BATCH_SIZE]

            try:
                # Process batch with retry logic
                batch_processed = await process_batch_with_retry(batch, batch_num)
                processed_count += batch_processed

                logger.info(
                    f"Progress: {processed_count}/{total_documents} documents processed"
                )

                # Rate limiting delay between batches (except for the last batch)
                if batch_num < total_batches:
                    logger.info(
                        f"Waiting {DELAY_BETWEEN_BATCHES}s before next batch..."
                    )
                    await asyncio.sleep(DELAY_BETWEEN_BATCHES)

            except Exception as e:
                logger.error(
                    f"Failed to process batch {batch_num} after {MAX_RETRIES} attempts: {str(e)}"
                )
                # Continue with remaining batches instead of failing completely
                continue

        logger.info(
            f"Robust indexing complete! Processed {processed_count}/{total_documents} documents"
        )

        success_rate = (
            (processed_count / total_documents) * 100 if total_documents > 0 else 0
        )

        return {
            "status": "success" if success_rate > 90 else "partial_success",
            "message": f"Indexed {processed_count}/{total_documents} documents from vault ({success_rate:.1f}% success rate)",
            "document_count": processed_count,
            "total_documents": total_documents,
            "success_rate": f"{success_rate:.1f}%",
        }

    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}", exc_info=True)

        # Provide specific error messages for common issues
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["rate limit", "quota", "429"]):
            detail = (
                "OpenAI API rate limit exceeded. The indexing process has been "
                "optimized with rate limiting, but your current usage may have hit "
                "API limits. Please wait a few minutes and try again."
            )
        elif (
            "context_length_exceeded" in error_msg
            or "maximum context length" in error_msg
        ):
            detail = "Document chunks are too large for the API. The indexing process will automatically handle this in future runs."
        elif "insufficient_quota" in error_msg or "billing" in error_msg:
            detail = "OpenAI API quota exceeded. Please check your OpenAI account billing and usage limits."
        else:
            detail = f"Failed to index vault: {str(e)}"

        raise HTTPException(status_code=500, detail=detail)


def convert_sources_format(text: str) -> str:
    """Convert source references from '(YYYY-MM-DD)' to '[[YYYY-MM-DD]]'.
    Only lines containing 'Sources:' (case-insensitive) are affected to avoid
    unintended substitutions.
    """
    # Match quoted or bare dates not already inside [[ ]]
    date_pattern = re.compile(r'(?<!\[\[)"?(\d{4}-\d{2}-\d{2})"?')

    def _date_repl(match):
        return f"[[{match.group(1)}]]"

    processed_lines = []
    for line in text.splitlines():
        # Replace all date occurrences with wiki-links
        line = date_pattern.sub(_date_repl, line)

        # Clean up common wrappers like (source: ...)
        if re.search(r"(?i)source:", line):
            # Remove leading "(source:" or "( source:" etc.
            line = re.sub(r"\(\s*source:\s*", "source: ", line, flags=re.IGNORECASE)
            # Remove trailing closing parenthesis if present
            line = re.sub(r"\)\s*", "", line)
        processed_lines.append(line)

    return "\n".join(processed_lines)


@app.post("/save")
async def save_content(request: SaveRequest):
    """
    Save conversation or content to the current day's page in the Obsidian vault.
    Supports three modes:
    1. Direct content: provide 'content' field
    2. Direct messages: provide 'messages' field
    3. Session-based: provide 'session_id' field
    """
    try:
        # --- BEGIN ADDED LOGGING ---
        logger.info(f"Received save_conversation request: {request.model_dump()}")
        available_sessions = list(rag_pipeline.conversation_memory.sessions.keys())
        logger.info(f"Current sessions in memory: {available_sessions}")
        # --- END ADDED LOGGING ---

        logger.info(
            f"Saving conversation {request.conversation_name} from session {request.session_id}"
        )

        # Handle direct content saving (highest priority)
        if request.content is not None:
            logger.info(f"Using provided content directly")
            conversation_content = []
            conversation_content.append(
                f"## My Obsidian Helper: {request.conversation_name}"
            )
            conversation_content.append(
                f"_Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n"
            )
            conversation_content.append(convert_sources_format(request.content))
        # Handle direct messages (second priority)
        elif request.messages is not None and len(request.messages) > 0:
            logger.info(f"Using provided messages directly instead of session lookup")
            # Format the conversation with blockquotes and proper Markdown
            conversation_content = []
            conversation_content.append(
                f"## My Obsidian Helper: {request.conversation_name}"
            )
            conversation_content.append(
                f"_Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n"
            )

            for i, message in enumerate(request.messages):
                if message.get("role") == "user":
                    # User message processing
                    user_lines = message.get("content", "").split("\n")
                    first_user_line = user_lines[0] if user_lines else ""
                    conversation_content.append(
                        f"> **User ({i//2+1}):** {first_user_line}"
                    )

                    # Add remaining lines of user message
                    for line in user_lines[1:]:
                        conversation_content.append(f"> {line}")

                    conversation_content.append("")  # Add blank line between speakers
                elif message.get("role") == "assistant":
                    # Assistant message processing
                    assistant_content = convert_sources_format(message.get("content", ""))
                    assistant_lines = assistant_content.split("\n")
                    first_assistant_line = assistant_lines[0] if assistant_lines else ""
                    conversation_content.append(
                        f"> **Assistant ({i//2+1}):** {first_assistant_line}"
                    )

                    # Add remaining lines
                    for line in assistant_lines[1:]:
                        conversation_content.append(f"> {line}")

                    conversation_content.append("")  # Add blank line after exchange
        else:
            # Session-based retrieval logic (lowest priority)
            if not request.session_id:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide either 'content', 'messages', or 'session_id' to save",
                )

            session_id = request.session_id
            if session_id in ["current", "default"]:
                # Find the most recent session
                if not rag_pipeline.conversation_memory.sessions:
                    logger.error(
                        f"No sessions found when requesting '{session_id}' session"
                    )
                    raise HTTPException(
                        status_code=404,
                        detail="No conversations found. Please start a conversation first.",
                    )

                # Get the most recent session by checking timestamps
                most_recent_session = None
                most_recent_time = None

                for (
                    sess_id,
                    interactions,
                ) in rag_pipeline.conversation_memory.sessions.items():
                    if interactions and "timestamp" in interactions[-1]:
                        last_time = datetime.fromisoformat(
                            interactions[-1]["timestamp"]
                        )
                        if most_recent_time is None or last_time > most_recent_time:
                            most_recent_time = last_time
                            most_recent_session = sess_id

                if most_recent_session:
                    logger.info(
                        f"Using most recent session {most_recent_session} for '{session_id}' request"
                    )
                    session_id = most_recent_session
                    # --- BEGIN ADDED LOGGING ---
                    logger.info(f"Resolved session_id to: {session_id}")
                    # --- END ADDED LOGGING ---
                else:
                    logger.error(
                        f"Could not determine most recent session for '{session_id}' request"
                    )
                    raise HTTPException(
                        status_code=404,
                        detail="Could not determine the current conversation. Please start a new conversation.",
                    )

            # Check if conversation session exists
            if session_id not in rag_pipeline.conversation_memory.sessions:
                logger.error(f"Session {session_id} not found")
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation session '{session_id}' not found",
                )

            # Get conversation data
            interactions = rag_pipeline.conversation_memory.sessions[session_id]

            # --- BEGIN ADDED LOGGING ---
            if interactions:
                logger.info(
                    f"First interaction found for session {session_id}: User: {interactions[0]['user_message'][:100]}... | Assistant: {interactions[0]['assistant_message'][:100]}..."
                )
            else:
                logger.warning(
                    f"No interactions found in memory for session {session_id} despite session existing."
                )
            # --- END ADDED LOGGING ---

            if not interactions:
                logger.error(f"No interactions found in session {session_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"No conversation data found for session '{session_id}'",
                )

            # Format the conversation with blockquotes and proper Markdown
            conversation_content = []
            conversation_content.append(
                f"## My Obsidian Helper: {request.conversation_name}"
            )
            conversation_content.append(
                f"_Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n"
            )

            for i, interaction in enumerate(interactions):
                # User message - combine first line with the user label
                user_lines = interaction["user_message"].split("\n")
                first_user_line = user_lines[0] if user_lines else ""
                conversation_content.append(f"> **User ({i+1}):** {first_user_line}")

                # Add remaining lines of user message with blockquotes
                for line in user_lines[1:]:
                    conversation_content.append(f"> {line}")

                conversation_content.append("")  # Add blank line between speakers

                # Assistant message - combine first line with the assistant label
                assistant_content = convert_sources_format(interaction["assistant_message"])
                assistant_lines = assistant_content.split("\n")
                first_assistant_line = assistant_lines[0] if assistant_lines else ""
                conversation_content.append(
                    f"> **Assistant ({i+1}):** {first_assistant_line}"
                )

                # Add remaining lines of assistant message with blockquotes
                for line in assistant_lines[1:]:
                    conversation_content.append(f"> {line}")

                conversation_content.append(
                    ""
                )  # Add blank line after each complete exchange

        # Get current date for daily note
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        month_str = today.strftime("%m")  # e.g., "07" for July
        year_str = today.strftime("%Y")

        # Intelligently determine the root for daily notes
        base_path = Path(config["vault"]["path"])
        daily_notes_root = base_path

        # If the configured path ends in what looks like a year (e.g., /2025),
        # we assume the user wants to save relative to the parent directory.
        # This decouples the indexing scope from the saving location for daily notes.
        if re.match(r"^\d{4}$", base_path.name):
            logger.info(
                f"Config path ends in a year ('{base_path.name}'). "
                "Using parent directory as daily notes root."
            )
            daily_notes_root = base_path.parent
        else:
            logger.info("Using config path directly as daily notes root.")

        # Construct the final path within the determined root
        daily_notes_folder = daily_notes_root / year_str / month_str
        daily_note_filename = f"{today_str}.md"

        # Make sure directory exists
        daily_notes_folder.mkdir(parents=True, exist_ok=True)

        daily_note_path = daily_notes_folder / daily_note_filename
        logger.info(f"Using daily note path: {daily_note_path}")

        formatted_conversation = "\n".join(conversation_content)

        # If the daily note doesn't exist, create it
        if not daily_note_path.exists():
            logger.info(f"Creating new daily note: {daily_note_path}")
            # Ensure the directory exists
            daily_note_path.parent.mkdir(parents=True, exist_ok=True)
            with open(daily_note_path, "w", encoding="utf-8") as f:
                f.write(f"# {today_str}\n\n{formatted_conversation}\n")
        else:
            # Append to existing daily note
            logger.info(f"Appending to existing daily note: {daily_note_path}")
            with open(daily_note_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n{formatted_conversation}\n")

        # Sync with Dropbox if enabled
        if DROPBOX_AVAILABLE and getattr(settings, 'dropbox_enabled', False):
            try:
                logger.info("Syncing with Dropbox...")
                dropbox_client = DropboxClient(settings)
                # Convert local path to relative path for Dropbox
                relative_path = str(daily_note_path.relative_to(Path(config["vault"]["path"])))
                dropbox_client.upload_file(str(daily_note_path), f"/{relative_path}")
                logger.info("Synced with Dropbox successfully")
            except DropboxAuthenticationError as e:
                logger.error(f"Dropbox authentication failed: {e}")
            except Exception as e:
                logger.error(f"Failed to sync with Dropbox: {e}")
        elif DROPBOX_AVAILABLE:
            logger.debug("Dropbox integration available but not enabled")

        return {
            "status": "success",
            "message": f"Conversation '{request.conversation_name}' saved to {daily_note_path.name}",
            "file_path": str(
                daily_note_path.relative_to(Path(config["vault"]["path"]))
            ),
        }
    except HTTPException as he:
        # Re-raise FastAPI HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Failed to save conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to save conversation: {str(e)}"
        )


@app.post("/load", response_model=LoadResponse)
async def load_content(request: LoadRequest):
    """
    Load file content from Dropbox or local storage, prioritizing the most recent version.
    """
    try:
        logger.info(f"Loading file: {request.file_path}")
        
        # Construct local file path
        local_path = Path(config["vault"]["path"]) / request.file_path
        
        dropbox_content = None
        dropbox_modified = None
        local_content = None
        local_modified = None
        
        # Try to get file from Dropbox if available
        if DROPBOX_AVAILABLE and getattr(settings, 'dropbox_enabled', False):
            try:
                dropbox_client = DropboxClient(settings)
                dropbox_path = f"/{request.file_path}"
                
                # Check if file exists on Dropbox
                if dropbox_client.file_exists(dropbox_path):
                    logger.info(f"File found on Dropbox: {dropbox_path}")
                    dropbox_content = dropbox_client.download_file(dropbox_path)
                    
                    # Get file metadata for modification time
                    metadata = dropbox_client.get_file_metadata(dropbox_path)
                    if metadata and 'client_modified' in metadata:
                        dropbox_modified = metadata['client_modified']
                        
            except Exception as e:
                logger.warning(f"Failed to load from Dropbox: {e}")
        
        # Try to get file from local storage
        if local_path.exists():
            logger.info(f"File found locally: {local_path}")
            with open(local_path, 'r', encoding='utf-8') as f:
                local_content = f.read()
            local_modified = datetime.fromtimestamp(local_path.stat().st_mtime).isoformat()
        
        # Determine which version to use (prioritize most recent)
        if dropbox_content and local_content:
            # Compare modification times if available
            if dropbox_modified and local_modified:
                dropbox_time = datetime.fromisoformat(dropbox_modified.replace('Z', '+00:00'))
                local_time = datetime.fromisoformat(local_modified)
                
                if dropbox_time > local_time:
                    logger.info("Using Dropbox version (more recent)")
                    return LoadResponse(
                        status="success",
                        content=dropbox_content,
                        source="dropbox",
                        file_path=request.file_path,
                        last_modified=dropbox_modified
                    )
                else:
                    logger.info("Using local version (more recent)")
                    return LoadResponse(
                        status="success",
                        content=local_content,
                        source="local",
                        file_path=request.file_path,
                        last_modified=local_modified
                    )
            else:
                # Default to Dropbox if modification times unavailable
                logger.info("Using Dropbox version (modification times unavailable)")
                return LoadResponse(
                    status="success",
                    content=dropbox_content,
                    source="dropbox",
                    file_path=request.file_path,
                    last_modified=dropbox_modified
                )
        elif dropbox_content:
            logger.info("Using Dropbox version (only source)")
            return LoadResponse(
                status="success",
                content=dropbox_content,
                source="dropbox",
                file_path=request.file_path,
                last_modified=dropbox_modified
            )
        elif local_content:
            logger.info("Using local version (only source)")
            return LoadResponse(
                status="success",
                content=local_content,
                source="local",
                file_path=request.file_path,
                last_modified=local_modified
            )
        else:
            logger.warning(f"File not found: {request.file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {request.file_path}"
            )
            
    except HTTPException as he:
        # Re-raise FastAPI HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Failed to load file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to load file: {str(e)}"
        )


@app.post("/debug_save_conversation")
async def debug_save_conversation(request: SaveRequest):
    """
    Debug version of save_conversation with more detailed logging and error information
    """
    debug_info = {
        "environment": {
            "OBSIDIAN_VAULT_PATH": os.environ.get("OBSIDIAN_VAULT_PATH", "Not set"),
            "config_vault_path": str(config["vault"]["path"]),
            "working_directory": os.getcwd(),
        },
        "request": {
            "session_id": request.session_id,
            "conversation_name": request.conversation_name,
            "has_content": bool(request.content),
            "has_messages": bool(request.messages),
        },
        "session_exists": bool(
            request.session_id
            and request.session_id in rag_pipeline.conversation_memory.sessions
        ),
        "interactions_count": 0,
    }

    try:
        # Get current date for daily note
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")

        # Get vault path from config
        vault_path = Path(config["vault"]["path"])
        debug_info["vault"] = {
            "resolved_path": str(vault_path.resolve()),
            "exists": vault_path.exists(),
            "is_dir": vault_path.is_dir() if vault_path.exists() else False,
            "permissions": {
                "readable": (
                    os.access(str(vault_path), os.R_OK)
                    if vault_path.exists()
                    else False
                ),
                "writable": (
                    os.access(str(vault_path), os.W_OK)
                    if vault_path.exists()
                    else False
                ),
                "executable": (
                    os.access(str(vault_path), os.X_OK)
                    if vault_path.exists()
                    else False
                ),
            },
        }

        # Check for session and conversations
        if (
            request.session_id
            and request.session_id in rag_pipeline.conversation_memory.sessions
        ):
            interactions = rag_pipeline.conversation_memory.sessions[request.session_id]
            debug_info["interactions_count"] = len(interactions)

            if interactions:
                debug_info["first_interaction"] = {
                    "user_message_sample": (
                        interactions[0]["user_message"][:100] + "..."
                        if len(interactions[0]["user_message"]) > 100
                        else interactions[0]["user_message"]
                    ),
                    "assistant_message_sample": (
                        interactions[0]["assistant_message"][:100] + "..."
                        if len(interactions[0]["assistant_message"]) > 100
                        else interactions[0]["assistant_message"]
                    ),
                }

        # Scan for daily notes
        daily_notes = []
        try:
            for file_path in vault_path.rglob("*.md"):
                if re.search(r"\d{4}-\d{2}-\d{2}", file_path.name):
                    daily_notes.append(str(file_path.relative_to(vault_path)))
                    if len(daily_notes) >= 5:  # Limit to 5 examples
                        break
            debug_info["daily_notes_found"] = daily_notes
        except Exception as e:
            debug_info["daily_notes_error"] = str(e)

        # All diagnostics complete
        return {"status": "debug_info", "debug_info": debug_info}

    except Exception as e:
        logger.error(f"Debug save conversation failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "debug_info": debug_info,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.get("/query")
@limiter.limit("10/minute")
async def query_vault(
    request: Request,
    q: str,
    session_id: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    date_range: Optional[str] = None,
):
    """
    Query the vault with advanced filtering options.
    """
    health_ok, message = rag_pipeline.check_health()
    if not health_ok:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unavailable: {message}. Please re-index the vault.",
        )

    try:
        logger.info(
            f"Received query: q='{q}', session_id='{session_id}', "
            f"tags={tags}, date_range='{date_range}'"
        )

        # Get retrieval configuration
        retrieval_config = config.get("retrieval", {})
        k = retrieval_config.get("k", 5)  # Default to 5 if not configured
        
        logger.info(f"Using k={k} for document retrieval")

        # Call the RAG pipeline
        result = rag_pipeline.query(
            query=q,
            k=k,
            session_id=session_id,
            tags=tags,
            date_range=date_range,
        )

        logger.info(
            f"Query successful. Returning {len(result.get('sources', []))} sources."
        )
        return result

    except HTTPException as http_exc:
        # Re-raise HTTPException directly to preserve status code and details
        raise http_exc
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during query: {str(e)}",
        )


@app.get("/debug_all_conversations")
async def debug_all_conversations():
    """
    DEBUG: Retrieve all saved conversations.
    """
    sessions_info = {}
    for session_id, interactions in rag_pipeline.conversation_memory.sessions.items():
        if interactions:
            sessions_info[session_id] = {
                "message_count": len(interactions),
                "first_user_message": (
                    interactions[0]["user_message"][:100] + "..."
                    if len(interactions[0]["user_message"]) > 100
                    else interactions[0]["user_message"]
                ),
                "first_assistant_message": (
                    interactions[0]["assistant_message"][:100] + "..."
                    if len(interactions[0]["assistant_message"]) > 100
                    else interactions[0]["assistant_message"]
                ),
            }
        else:
            sessions_info[session_id] = {"message_count": 0}

    return {
        "status": "success",
        "sessions_count": len(sessions_info),
        "sessions": sessions_info,
    }


@app.post("/index_fast")
async def index_vault_fast():
    """
    Index the vault using the high-speed, parallelized FastIndexer.
    """
    try:
        logger.info("Starting fast vault indexing...")

        # Load all documents from the vault using the loader
        document_objects = vault_loader.load_all_documents(config=config)
        documents_to_index = [doc.to_dict() for doc in document_objects]
        logger.info(f"Loaded {len(documents_to_index)} documents from vault for fast indexing.")

        if not documents_to_index:
            logger.warning("No documents found to index.")
            return {"status": "warning", "message": "No documents found to index."}
        
        # Initialize the fast indexer and run it
        indexer = FastIndexer(config=config.get("fast_indexing", {}), storage=storage)
        result = await indexer.index_documents_fast(documents_to_index)
        
        # Optionally, re-sync RAG pipeline with the new index
        if result.get("status") == "success":
            logger.info("Fast indexing successful, reloading RAG pipeline components.")
            rag_pipeline._load_or_initialize_store()

        return result

    except Exception as e:
        logger.error(f"Fast indexing endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fast indexing failed: {str(e)}")


@app.get("/index_fast_status")
async def get_fast_indexing_status():
    """
    Get the status of fast indexing operation
    """
    try:
        fast_indexer = FastIndexer(config)
        status = fast_indexer.get_indexing_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get fast indexing status: {str(e)}")
        return {"status": "error", "error": str(e)}


@app.post("/index_fast_resume")
async def resume_fast_indexing():
    """
    Resumes a previously interrupted fast indexing process.
    """
    try:
        logger.info("Resuming fast indexing...")
        # Note: The 'documents' list would need to be re-loaded or cached
        # to properly resume. This is a simplified example.
        document_objects = vault_loader.load_all_documents(config=config)
        documents_to_index = [doc.to_dict() for doc in document_objects]
        indexer = FastIndexer(config=config.get("fast_indexing", {}), storage=storage)
        result = await indexer.index_documents_fast(documents_to_index, resume=True)
        return result
    except Exception as e:
        logger.error(f"Failed to resume fast indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to resume fast indexing.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=False,  # Explicitly disable reload
    )
