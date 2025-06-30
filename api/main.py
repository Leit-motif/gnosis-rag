"""
Enhanced FastAPI application for the Gnosis RAG API.
Includes rate limiting, authentication, error handling, and structured logging.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import text
from datetime import datetime
from fastapi.openapi.utils import get_openapi

from api.config import settings
from api.middleware import setup_middleware, limiter, security, verify_token
from api.database import init_db, close_db, check_db_connection

# Dropbox integration
try:
    from api.dropbox_client import DropboxClient, DropboxAuthenticationError, DropboxOperationError
    DROPBOX_AVAILABLE = True
    logger_dropbox = logging.getLogger(__name__ + ".dropbox")
    logger_dropbox.info("Dropbox integration available")
except ImportError as e:
    DROPBOX_AVAILABLE = False
    logger_dropbox = logging.getLogger(__name__ + ".dropbox")
    logger_dropbox.warning(f"Dropbox integration not available: {e}")


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan event handler."""
    # Startup
    logger.info("Starting up Gnosis RAG API...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gnosis RAG API...")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Hybrid Graph RAG API for intelligent document retrieval and chat",
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
    servers=[
        {"url": settings.api_url, "description": "Gnosis RAG API server"}
    ],
)

# Ensure the OpenAPI schema includes only the public server URL

def custom_openapi():
    """Generate a custom OpenAPI schema with the correct public server URL."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["servers"] = [{"url": settings.api_url}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up all middleware (rate limiting, logging, error handling)
setup_middleware(app)


# Pydantic models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    debug: bool
    timestamp: str
    database: str


class UploadResponse(BaseModel):
    """Document upload response model."""
    document_ids: list[str]
    message: str
    processed_count: int


class ChatRequest(BaseModel):
    """Chat request model."""
    thread_id: Optional[str] = Field(None, description="Optional conversation thread ID")
    query: str = Field(..., min_length=1, max_length=2000, description="User query")


class ChatResponse(BaseModel):
    """Chat response model."""
    answer: str
    citations: list[str] = []
    thread_id: str


class PluginManifest(BaseModel):
    """AI plugin manifest model."""
    schema_version: str
    name_for_human: str
    name_for_model: str
    description_for_human: str
    description_for_model: str
    auth: dict
    api: dict


# API Endpoints

@app.get("/health", response_model=HealthResponse)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def health_check(request: Request):
    """Health check endpoint with rate limiting."""
    # Check database connectivity
    db_status = "connected" if await check_db_connection() else "disconnected"
    overall_status = "ok" if db_status == "connected" else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.api_version,
        debug=settings.debug,
        timestamp=datetime.utcnow().isoformat(),
        database=db_status
    )


@app.post("/upload", response_model=UploadResponse)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Document upload endpoint with authentication and file size limits."""
    # Verify authentication if token provided
    user = await verify_token(credentials)
    
    logger.info(f"Upload request from {'authenticated' if user else 'anonymous'} user")
    
    # Validate file sizes (4MB limit per file)
    max_size = 4 * 1024 * 1024  # 4MB
    for file in files:
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} exceeds 4MB limit"
            )
    
    # Validate file types
    allowed_types = {"text/markdown", "text/plain", "application/pdf"}
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {file.content_type}"
            )
    
    # Placeholder implementation
    raise HTTPException(
        status_code=501,
        detail="Upload functionality will be implemented in task 21"
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Chat endpoint with rate limiting and authentication."""
    # Verify authentication if token provided
    user = await verify_token(credentials)
    
    logger.info(
        f"Chat request from {'authenticated' if user else 'anonymous'} user",
        extra={"query_length": len(chat_request.query)}
    )
    
    # Placeholder response
    return ChatResponse(
        answer="This is a placeholder response. Full chat functionality will be implemented in task 28.",
        citations=["placeholder-doc-1"],
        thread_id=chat_request.thread_id or "new-thread-id"
    )


@app.get("/.well-known/ai-plugin.json", response_model=PluginManifest)
async def get_plugin_manifest():
    """AI plugin manifest for ChatGPT/Custom GPT integration."""
    logger.info("Plugin manifest requested")
    
    return PluginManifest(
        schema_version="v1",
        name_for_human="Gnosis RAG",
        name_for_model="gnosis_rag",
        description_for_human="Intelligent document retrieval and chat using hybrid graph RAG",
        description_for_model="A hybrid graph RAG system that can upload documents and answer questions using graph-enhanced retrieval",
        auth={
            "type": "none" if not settings.bearer_token else "bearer"
        },
        api={
            "type": "openapi",
            "url": f"{settings.api_url}/openapi.json"
        }
    )


@app.get("/openapi.yaml")
async def get_openapi_yaml():
    """OpenAPI specification in YAML format."""
    logger.info("OpenAPI YAML requested")
    import yaml
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    return yaml.dump(openapi_schema, default_flow_style=False)


@app.post("/admin/init-database")
@limiter.limit("1/minute")  # Very restrictive since this is admin-only
async def initialize_database(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Initialize database with required extensions (admin only).
    This endpoint creates the pgvector extension if it doesn't exist.
    """
    # Verify token (you should set a strong admin token)
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    try:
        from api.database import get_session
        async with get_session() as session:
            # Create pgvector extension
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await session.commit()
            
            logger.info("Database initialized successfully with pgvector extension")
            return {
                "status": "success",
                "message": "Database initialized with pgvector extension",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize database: {str(e)}"
        )


# Dropbox response models
class DropboxStatusResponse(BaseModel):
    """Dropbox status response model."""
    enabled: bool
    is_syncing: bool = False
    last_sync_time: Optional[str] = None
    sync_interval_minutes: int
    local_vault_path: Optional[str] = None
    dropbox_connected: bool
    timestamp: str


class DropboxSyncResponse(BaseModel):
    """Dropbox sync response model."""
    status: str
    message: str
    files_downloaded: int = 0
    files_uploaded: int = 0
    files_conflicted: int = 0
    files_skipped: int = 0
    errors: list[str] = []
    timestamp: str


# Dropbox Endpoints
@app.get("/dropbox/status", response_model=DropboxStatusResponse)
@limiter.limit(f"{settings.rate_limit_requests}/minute")
async def get_dropbox_status(request: Request):
    """Get Dropbox sync status and configuration."""
    logger.info("Dropbox status requested")
    
    # Base response structure
    base_response = {
        "enabled": settings.dropbox_enabled if DROPBOX_AVAILABLE else False,
        "is_syncing": False,  # TODO: Track actual sync state in future tasks
        "last_sync_time": None,  # TODO: Track last sync time in future tasks
        "sync_interval_minutes": settings.dropbox_sync_interval_minutes,
        "local_vault_path": None,  # Will be set if vault path is configured
        "dropbox_connected": False,  # Will be tested below
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not DROPBOX_AVAILABLE:
        logger.warning("Dropbox integration not available")
        return DropboxStatusResponse(**base_response)
    
    if not settings.dropbox_enabled:
        logger.info("Dropbox sync is disabled in configuration")
        return DropboxStatusResponse(**base_response)
    
    try:
        # Test Dropbox connection
        dropbox_client = DropboxClient()
        connected = dropbox_client.verify_connection()
        
        # Update response with connection status
        base_response.update({
            "dropbox_connected": connected,
        })
        
        # Get vault path from environment or config if available
        # TODO: In future tasks, this should come from actual vault configuration
        vault_path = getattr(settings, 'vault_path', None) or "/obsidian-vault"
        base_response["local_vault_path"] = vault_path
        
        logger.info(f"Dropbox status check complete - connected: {connected}")
        return DropboxStatusResponse(**base_response)
        
    except DropboxAuthenticationError as e:
        logger.error(f"Dropbox authentication failed: {e}")
        return DropboxStatusResponse(**base_response)
    except Exception as e:
        logger.error(f"Error checking Dropbox status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Dropbox status: {str(e)}"
        )


@app.post("/dropbox/sync", response_model=DropboxSyncResponse)
@limiter.limit("5/minute")  # More restrictive for sync operations
async def sync_dropbox_vault(request: Request):
    """Perform bidirectional synchronization between Dropbox and local vault."""
    logger.info("Dropbox sync requested")
    
    if not DROPBOX_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="Dropbox integration is not available"
        )
    
    if not settings.dropbox_enabled:
        raise HTTPException(
            status_code=400,
            detail="Dropbox sync is disabled"
        )
    
    try:
        dropbox_client = DropboxClient()
        
        # Verify connection first
        if not dropbox_client.verify_connection():
            raise HTTPException(
                status_code=400,
                detail="Dropbox connection failed"
            )
        
        # For now, return a placeholder response
        # TODO: Implement actual sync logic in future tasks
        logger.info("Dropbox sync placeholder - full implementation pending")
        
        return DropboxSyncResponse(
            status="success",
            message="Dropbox sync placeholder - full implementation will be added in future tasks",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except DropboxAuthenticationError as e:
        logger.error(f"Dropbox authentication failed during sync: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Dropbox authentication failed: {str(e)}"
        )
    except DropboxOperationError as e:
        logger.error(f"Dropbox operation failed during sync: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Dropbox sync operation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during Dropbox sync: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Dropbox sync failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=True,
        reload=settings.debug
    ) 