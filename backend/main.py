from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
    Query the Obsidian vault using hybrid search
    """
    try:
        logger.info(f"Processing query: {q}")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Generated new session ID: {session_id}")
        
        # Parse date range if provided
        start_date = None
        end_date = None
        if date_range:
            if date_range == "last_30_days":
                start_date = datetime.now() - timedelta(days=30)
                end_date = datetime.now()
            # Add more date range parsing options as needed

        # Query the RAG pipeline
        response = rag_pipeline.query(
            query=q,
            session_id=session_id,
            conversation_memory=rag_pipeline.conversation_memory,
            tags=tags,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add session ID to response
        response["session_id"] = session_id
        
        logger.info("Query processed successfully")
        return response
        
    except HTTPException as he:
        # Re-raise FastAPI HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# Additional route to handle double slash issue with ChatGPT
@app.get("//query")
@limiter.limit("10/minute")
async def query_vault_double_slash(
    request: Request,
    q: str,
    session_id: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    date_range: Optional[str] = None
):
    return await query_vault(request, q, session_id, tags, date_range)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=False  # Explicitly disable reload
    ) 