"""
Simplified test version of the API server for testing Dropbox endpoints without database.
"""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# Dropbox integration
try:
    from api.dropbox_client import DropboxClient, DropboxAuthenticationError, DropboxOperationError
    from api.config import settings
    DROPBOX_AVAILABLE = True
    logger_dropbox = logging.getLogger(__name__ + ".dropbox")
    logger_dropbox.info("Dropbox integration available")
except ImportError as e:
    DROPBOX_AVAILABLE = False
    logger_dropbox = logging.getLogger(__name__ + ".dropbox")
    logger_dropbox.warning(f"Dropbox integration not available: {e}")

# Configure logging
logging.basicConfig(
    level="INFO",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Gnosis RAG API - Test Mode",
    version="1.0.0-test",
    description="Test version for Dropbox endpoints without database",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    mode: str
    timestamp: str

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

# Basic endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0-test",
        mode="test",
        timestamp=datetime.utcnow().isoformat()
    )

# Dropbox Endpoints
@app.get("/dropbox/status", response_model=DropboxStatusResponse)
async def get_dropbox_status(request: Request):
    """Get Dropbox sync status and configuration."""
    logger.info("Dropbox status requested")
    
    # Base response structure
    base_response = {
        "enabled": settings.dropbox_enabled if DROPBOX_AVAILABLE else False,
        "is_syncing": False,  # TODO: Track actual sync state in future tasks
        "last_sync_time": None,  # TODO: Track last sync time in future tasks
        "sync_interval_minutes": settings.dropbox_sync_interval_minutes if DROPBOX_AVAILABLE else 15,
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
        "api.main_test:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        reload=True
    ) 