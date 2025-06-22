"""
Enhanced FastAPI application for the Gnosis RAG API.
Includes rate limiting, authentication, error handling, and structured logging.
"""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from api.config import settings
from api.middleware import setup_middleware, limiter, security, verify_token

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Hybrid Graph RAG API for intelligent document retrieval and chat",
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

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
    from datetime import datetime
    
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        version=settings.api_version,
        debug=settings.debug,
        timestamp=datetime.utcnow().isoformat()
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
            "url": f"http://localhost:{settings.port}/openapi.json"
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