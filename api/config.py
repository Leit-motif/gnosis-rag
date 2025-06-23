"""Configuration management for the Gnosis RAG API."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/gnosis_rag",
        env="DATABASE_URL",
        description="PostgreSQL database URL with async driver"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        env="OPENAI_API_KEY",
        description="OpenAI API key for embeddings and completions"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        env="OPENAI_MODEL",
        description="OpenAI model for chat completions"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL",
        description="OpenAI model for generating embeddings"
    )
    
    # API Configuration
    api_url: str = Field(
        default="http://localhost:8080",
        env="API_URL",
        description="Public URL for the API, used for OpenAPI server spec."
    )
    port: int = Field(
        default=8080,
        env="PORT",
        description="Port for the API server"
    )
    api_title: str = Field(
        default="Gnosis RAG API",
        env="API_TITLE",
        description="API title for OpenAPI documentation"
    )
    api_version: str = Field(
        default="1.0.0",
        env="API_VERSION",
        description="API version"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(
        default=30,
        env="RATE_LIMIT_REQUESTS",
        description="Number of requests allowed per period"
    )
    rate_limit_period: int = Field(
        default=60,
        env="RATE_LIMIT_PERIOD",
        description="Rate limit period in seconds"
    )
    
    # Graph Configuration
    graph_max_hops: int = Field(
        default=2,
        env="GRAPH_MAX_HOPS",
        description="Maximum hops for graph traversal"
    )
    graph_max_documents: int = Field(
        default=10,
        env="GRAPH_MAX_DOCUMENTS",
        description="Maximum documents to retrieve"
    )
    graph_weight_vector: float = Field(
        default=0.6,
        env="GRAPH_WEIGHT_VECTOR",
        description="Weight for vector similarity in hybrid retrieval"
    )
    graph_weight_graph: float = Field(
        default=0.4,
        env="GRAPH_WEIGHT_GRAPH",
        description="Weight for graph similarity in hybrid retrieval"
    )
    
    # Vector Store Configuration
    embedding_dimension: int = Field(
        default=1536,
        env="EMBEDDING_DIMENSION",
        description="Dimension of embedding vectors"
    )
    similarity_threshold: float = Field(
        default=0.7,
        env="SIMILARITY_THRESHOLD",
        description="Minimum similarity threshold for retrieval"
    )
    
    # Security
    bearer_token: Optional[str] = Field(
        default=None,
        env="BEARER_TOKEN",
        description="Optional bearer token for API authentication"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings() 