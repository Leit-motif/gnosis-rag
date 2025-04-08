import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml and environment variables.
    Returns the complete configuration dictionary.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the absolute path to the config file
    config_path = Path(__file__).parent / "config.yaml"
    
    # Read and parse the YAML file
    with open(config_path, "r") as f:
        # Load YAML with environment variable interpolation
        config_str = f.read()
        # Replace environment variables in the YAML string
        for key, value in os.environ.items():
            config_str = config_str.replace(f"${{{key}}}", value)
        
        config = yaml.safe_load(config_str)
    
    return config

def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all necessary directories exist.
    Creates them if they don't exist.
    """
    directories = [
        config["vector_store"]["index_path"],
        os.path.dirname(config["logging"]["file"]),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration.
    Raises ValueError if configuration is invalid.
    """
    required_env_vars = [
        "OPENAI_API_KEY",
        "OBSIDIAN_VAULT_PATH",
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate Obsidian vault path
    vault_path = Path(config["vault"]["path"])
    if not vault_path.exists():
        raise ValueError(f"Obsidian vault path does not exist: {vault_path}")
    
    # Validate embedding configuration
    if config["embeddings"]["provider"] not in ["openai", "local"]:
        raise ValueError("Invalid embedding provider. Must be 'openai' or 'local'")
    
    # Validate vector store configuration
    if config["vector_store"]["type"] not in ["faiss", "chroma", "qdrant"]:
        raise ValueError("Invalid vector store type. Must be 'faiss', 'chroma', or 'qdrant'") 