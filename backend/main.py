from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import re
import traceback
import time
import json

from .rag_pipeline import RAGPipeline
from .obsidian_loader_v2 import ObsidianLoaderV2
from .utils import load_config, ensure_directories, validate_config

# Load and validate configuration
config = load_config()
validate_config(config)
ensure_directories(config)

# Configure logging
logging.basicConfig(
    level=config["logging"]["level"],
    filename=config["logging"]["file"],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Gnosis RAG API",
    description="API for querying and analyzing Obsidian vaults using hybrid RAG",
    version="1.0.0"
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
        content={"message": "Logo placeholder. In production, this would be an image file."},
        status_code=200
    )

@app.get("/legal", include_in_schema=False)
async def get_legal():
    return {"terms_of_use": "This is a prototype plugin. Use at your own risk."}

# Model for save conversation request
class SaveConversationRequest(BaseModel):
    session_id: str
    conversation_name: str
    messages: Optional[List[Dict[str, str]]] = None  # Add optional messages field

# After the SaveConversationRequest model definition, add a new model for exact content
class SaveExactContentRequest(BaseModel):
    conversation_name: str
    content: str

# Initialize components
try:
    logger.info("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(config)
    
    logger.info("Loading Obsidian vault...")
    vault_loader = ObsidianLoaderV2(config["vault"]["path"])
    
    logger.info("Initialization complete!")
except Exception as e:
    logger.error(f"Failed to initialize: {str(e)}")
    raise

# Restore the original /index endpoint
@app.post("/index")
async def index_vault():
    """
    Index the Obsidian vault content into the vector store.
    This needs to be called before querying.
    """
    try:
        logger.info("Starting vault indexing...")
        
        # Step 1: Load documents from vault
        documents = vault_loader.load_vault(config)
        logger.info(f"Loaded {len(documents)} documents from vault")

        if not documents:
             logger.warning("No documents found to index.")
             return {
                 "status": "warning",
                 "message": "No documents found to index.",
                 "document_count": 0
             }
        
        # Step 2: Create embeddings and index documents
        # Ensure the document structure matches what index_documents expects
        indexed_documents = [
            {
                # Use .get() for safety in case metadata keys are missing
                'id': f"{doc.metadata.get('source', 'unknown_source')}#{doc.metadata.get('chunk_id', 'unknown_chunk')}", 
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in documents
        ]
        rag_pipeline.index_documents(indexed_documents)
        logger.info("Indexing complete!")
        
        return {
            "status": "success",
            "message": f"Indexed {len(documents)} documents from vault",
            "document_count": len(documents)
        }
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index vault: {str(e)}"
        )

@app.post("/save_conversation")
async def save_conversation(request: SaveConversationRequest):
    """
    Save the current conversation to the current day's page in the Obsidian vault
    """
    try:
        # --- BEGIN ADDED LOGGING ---
        logger.info(f"Received save_conversation request: {request.model_dump()}")
        available_sessions = list(rag_pipeline.conversation_memory.sessions.keys())
        logger.info(f"Current sessions in memory: {available_sessions}")
        # --- END ADDED LOGGING ---
        
        logger.info(f"Saving conversation {request.conversation_name} from session {request.session_id}")
        
        # Skip session lookup if conversation messages are provided directly
        if request.messages is not None and len(request.messages) > 0:
            logger.info(f"Using provided messages directly instead of session lookup")
            # Format the conversation with blockquotes and proper Markdown
            conversation_content = []
            conversation_content.append(f"## My Obsidian Helper: {request.conversation_name}")
            conversation_content.append(f"_Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
            
            for i, message in enumerate(request.messages):
                if message.get("role") == "user":
                    # User message processing
                    user_lines = message.get("content", "").split('\n')
                    first_user_line = user_lines[0] if user_lines else ""
                    conversation_content.append(f"> **User ({i//2+1}):** {first_user_line}")
                    
                    # Add remaining lines of user message
                    for line in user_lines[1:]:
                        conversation_content.append(f"> {line}")
                    
                    conversation_content.append("")  # Add blank line between speakers
                elif message.get("role") == "assistant":
                    # Assistant message processing
                    assistant_lines = message.get("content", "").split('\n')
                    first_assistant_line = assistant_lines[0] if assistant_lines else ""
                    conversation_content.append(f"> **Assistant ({i//2+1}):** {first_assistant_line}")
                    
                    # Add remaining lines
                    for line in assistant_lines[1:]:
                        conversation_content.append(f"> {line}")
                    
                    conversation_content.append("")  # Add blank line after exchange
        else:
            # Original session-based retrieval logic
            # Handle special session ID cases
            session_id = request.session_id
            if session_id in ["current", "default"]:
                # Find the most recent session
                if not rag_pipeline.conversation_memory.sessions:
                    logger.error(f"No sessions found when requesting '{session_id}' session")
                    raise HTTPException(
                        status_code=404,
                        detail="No conversations found. Please start a conversation first."
                    )
                    
                # Get the most recent session by checking timestamps
                most_recent_session = None
                most_recent_time = None
                
                for sess_id, interactions in rag_pipeline.conversation_memory.sessions.items():
                    if interactions and "timestamp" in interactions[-1]:
                        last_time = datetime.fromisoformat(interactions[-1]["timestamp"])
                        if most_recent_time is None or last_time > most_recent_time:
                            most_recent_time = last_time
                            most_recent_session = sess_id
                
                if most_recent_session:
                    logger.info(f"Using most recent session {most_recent_session} for '{session_id}' request")
                    session_id = most_recent_session
                    # --- BEGIN ADDED LOGGING ---
                    logger.info(f"Resolved session_id to: {session_id}")
                    # --- END ADDED LOGGING ---
                else:
                    logger.error(f"Could not determine most recent session for '{session_id}' request")
                    raise HTTPException(
                        status_code=404,
                        detail="Could not determine the current conversation. Please start a new conversation."
                    )
                    
            # Check if conversation session exists
            if session_id not in rag_pipeline.conversation_memory.sessions:
                logger.error(f"Session {session_id} not found")
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation session '{session_id}' not found"
                )
                
            # Get conversation data
            interactions = rag_pipeline.conversation_memory.sessions[session_id]
            
            # --- BEGIN ADDED LOGGING ---
            if interactions:
                logger.info(f"First interaction found for session {session_id}: User: {interactions[0]['user_message'][:100]}... | Assistant: {interactions[0]['assistant_message'][:100]}...")
            else:
                logger.warning(f"No interactions found in memory for session {session_id} despite session existing.")
            # --- END ADDED LOGGING ---
            
            if not interactions:
                logger.error(f"No interactions found in session {session_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"No conversation data found for session '{session_id}'"
                )
                
            # Format the conversation with blockquotes and proper Markdown
            conversation_content = []
            conversation_content.append(f"## My Obsidian Helper: {request.conversation_name}")
            conversation_content.append(f"_Saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
            
            for i, interaction in enumerate(interactions):
                # User message - combine first line with the user label
                user_lines = interaction["user_message"].split('\n')
                first_user_line = user_lines[0] if user_lines else ""
                conversation_content.append(f"> **User ({i+1}):** {first_user_line}")
                
                # Add remaining lines of user message with blockquotes
                for line in user_lines[1:]:
                    conversation_content.append(f"> {line}")
                
                conversation_content.append("")  # Add blank line between speakers
                
                # Assistant message - combine first line with the assistant label
                assistant_lines = interaction["assistant_message"].split('\n')
                first_assistant_line = assistant_lines[0] if assistant_lines else ""
                conversation_content.append(f"> **Assistant ({i+1}):** {first_assistant_line}")
                
                # Add remaining lines of assistant message with blockquotes
                for line in assistant_lines[1:]:
                    conversation_content.append(f"> {line}")
                
                conversation_content.append("")  # Add blank line after each complete exchange
        
        # Get current date for daily note (moved out of the else block)
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        month_str = today.strftime("%m")
        
        # Use the vault path from config, which already includes up to the year directory
        vault_path = Path(config["vault"]["path"])
        
        # Only add the month directory since vault_path already includes up to the year
        daily_notes_folder = vault_path / month_str
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
        
        return {
            "status": "success",
            "message": f"Conversation '{request.conversation_name}' saved to {daily_note_path.name}",
            "file_path": str(daily_note_path.relative_to(Path(config["vault"]["path"])))
        }
    except HTTPException as he:
        # Re-raise FastAPI HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Failed to save conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save conversation: {str(e)}"
        )

@app.post("/debug_save_conversation")
async def debug_save_conversation(request: SaveConversationRequest):
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
            "conversation_name": request.conversation_name
        },
        "session_exists": request.session_id in rag_pipeline.conversation_memory.sessions,
        "interactions_count": 0
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
                "readable": os.access(str(vault_path), os.R_OK) if vault_path.exists() else False,
                "writable": os.access(str(vault_path), os.W_OK) if vault_path.exists() else False,
                "executable": os.access(str(vault_path), os.X_OK) if vault_path.exists() else False
            }
        }
        
        # Check for session and conversations
        if request.session_id in rag_pipeline.conversation_memory.sessions:
            interactions = rag_pipeline.conversation_memory.sessions[request.session_id]
            debug_info["interactions_count"] = len(interactions)
            
            if interactions:
                debug_info["first_interaction"] = {
                    "user_message_sample": interactions[0]["user_message"][:100] + "..." if len(interactions[0]["user_message"]) > 100 else interactions[0]["user_message"],
                    "assistant_message_sample": interactions[0]["assistant_message"][:100] + "..." if len(interactions[0]["assistant_message"]) > 100 else interactions[0]["assistant_message"]
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
        return {
            "status": "debug_info",
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Debug save conversation failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "debug_info": debug_info,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/query")
@limiter.limit("10/minute")
async def query_vault(
    request: Request,
    q: str,
    session_id: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    date_range: Optional[str] = None
):
    """
    Query the Obsidian vault using the RAG pipeline
    """
    try:
        logger.info(f"Processing query: {q}")
        
        # Parse date range if specified
        start_date = None
        end_date = None
        if date_range:
            try:
                dates = date_range.split(',')
                if len(dates) >= 1 and dates[0]:
                    start_date = datetime.fromisoformat(dates[0])
                if len(dates) >= 2 and dates[1]:
                    end_date = datetime.fromisoformat(dates[1])
            except ValueError as e:
                logger.warning(f"Invalid date format in range '{date_range}': {e}")
                # Continue with None values for dates
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Generated new session ID: {session_id}")
        
        # Execute query with specified filters
        try:
            return rag_pipeline.query(
                query=q,
                k=5,
                session_id=session_id,
                tags=tags,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error during query execution: {error_msg}")
            
            # Handle rate limit errors specially
            if "RetryError" in error_msg and ("RateLimitError" in error_msg or "429" in error_msg or "quota" in error_msg.lower()):
                logger.error("OpenAI API rate limit or quota exceeded")
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "message": "OpenAI API rate limit or quota exceeded. Please try again later or check your API usage and billing details.",
                        "error": "rate_limit_exceeded"
                    }
                )
            
            # General API error
            raise HTTPException(
                status_code=500,
                detail=f"Failed to query the vault: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )

@app.get("//query")
@limiter.limit("10/minute")
async def query_vault_double_slash(
    request: Request,
    q: str,
    session_id: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    date_range: Optional[str] = None
):
    """
    Alternative route for query to handle double slashes
    """
    return await query_vault(request, q, session_id, tags, date_range)

# Add a new endpoint to see all available conversations
@app.get("/debug_all_conversations")
async def debug_all_conversations():
    """
    Return all available conversation sessions and their first messages
    """
    sessions_info = {}
    for session_id, interactions in rag_pipeline.conversation_memory.sessions.items():
        if interactions:
            sessions_info[session_id] = {
                "message_count": len(interactions),
                "first_user_message": interactions[0]["user_message"][:100] + "..." if len(interactions[0]["user_message"]) > 100 else interactions[0]["user_message"],
                "first_assistant_message": interactions[0]["assistant_message"][:100] + "..." if len(interactions[0]["assistant_message"]) > 100 else interactions[0]["assistant_message"]
            }
        else:
            sessions_info[session_id] = {"message_count": 0}
    
    return {
        "status": "success",
        "sessions_count": len(sessions_info),
        "sessions": sessions_info
    }

# Add a new endpoint that accepts exact content for saving
@app.post("/save_exact_content")
async def save_exact_content(request: SaveExactContentRequest):
    """
    Save explicitly provided content to the current day's page in the Obsidian vault
    """
    try:
        logger.info(f"Saving exact content with name: {request.conversation_name}")
        
        # Get current date for daily note
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        month_str = today.strftime("%m")
        
        # Use the vault path from config, which already includes up to the year directory
        vault_path = Path(config["vault"]["path"])
        
        # Only add the month directory since vault_path already includes up to the year
        daily_notes_folder = vault_path / month_str
        daily_note_filename = f"{today_str}.md"
        
        # Make sure directory exists
        daily_notes_folder.mkdir(parents=True, exist_ok=True)
        
        daily_note_path = daily_notes_folder / daily_note_filename
        logger.info(f"Using daily note path: {daily_note_path}")
        
        # Format the content with headers
        conversation_content = []
        conversation_content.append(f"## My Obsidian Helper: {request.conversation_name}")
        conversation_content.append(f"_Saved on: {today.strftime('%Y-%m-%d %H:%M:%S')}_\n")
        conversation_content.append(request.content)
        
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
        
        return {
            "status": "success",
            "message": f"Content '{request.conversation_name}' saved to {daily_note_path.name}",
            "file_path": str(daily_note_path.relative_to(Path(config["vault"]["path"])))
        }
    except Exception as e:
        logger.error(f"Failed to save exact content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save exact content: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=False  # Explicitly disable reload
    ) 