from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import yaml
import os
from datetime import datetime, timedelta

from .rag_pipeline import RAGPipeline
from .obsidian_loader import ObsidianLoader

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

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

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(config)
vault_loader = ObsidianLoader(config["vault"]["path"])

class ReflectionRequest(BaseModel):
    mode: str
    agent: str

@app.get("/query")
async def query_vault(
    q: str,
    tags: Optional[List[str]] = Query(None),
    date_range: Optional[str] = None
):
    try:
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
            tags=tags,
            start_date=start_date,
            end_date=end_date
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/themes")
async def get_themes():
    try:
        themes = rag_pipeline.analyze_themes()
        return {"themes": themes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reflect")
async def generate_reflection(request: ReflectionRequest):
    try:
        reflection = rag_pipeline.generate_reflection(
            mode=request.mode,
            agent=request.agent
        )
        return reflection
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["api"]["debug"]
    ) 