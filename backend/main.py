from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import logging

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

app = FastAPI(
    title="Gnosis RAG API",
    description="API for querying and analyzing Obsidian vaults using hybrid RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chat.openai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

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
        
        # Step 2: Create embeddings and index documents
        indexed_documents = [
            {
                'id': f"{doc.metadata['source']}#{doc.metadata['chunk_id']}",
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
        logger.error(f"Indexing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to index vault: {str(e)}"
        )

class ReflectionRequest(BaseModel):
    mode: str
    agent: str

@app.get("/query")
async def query_vault(
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

@app.get("/themes")
async def get_themes():
    try:
        logger.info("Analyzing themes...")
        themes = rag_pipeline.analyze_themes()
        return {"themes": themes}
    except Exception as e:
        logger.error(f"Theme analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reflect")
async def generate_reflection(request: ReflectionRequest):
    try:
        logger.info(f"Generating {request.mode} reflection with {request.agent} agent...")
        reflection = rag_pipeline.generate_reflection(
            mode=request.mode,
            agent=request.agent
        )
        return reflection
    except Exception as e:
        logger.error(f"Reflection generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=False  # Explicitly disable reload
    ) 