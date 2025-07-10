import os
import re
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

    # Regex to find all ${VAR:-default} or ${VAR} patterns
    pattern = re.compile(r'\$\{(?P<name>\w+)(?::-(?P<default>.*?))?\}')
    
    def replace_var(match):
        var_name = match.group('name')
        default_value = match.group('default')
        return os.environ.get(var_name, default_value)

    config_str = pattern.sub(replace_var, config_str)
    
    config = yaml.safe_load(config_str)
    
    return config

def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensure all necessary directories exist.
    Creates them if they don't exist.
    """
    directories = [
        os.path.dirname(config["vector_store"]["index_path"]),
        os.path.dirname(config["logging"]["file"]),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration.
    Raises ValueError if configuration is invalid.
    """
    required_env_vars = ["OPENAI_API_KEY"]
    
    storage_provider = config.get("storage", {}).get("provider", "local")

    if storage_provider == "local":
        required_env_vars.append("LOCAL_VAULT_PATH")
    elif storage_provider == "gcs":
        # BUCKET_NAME can be in the config file directly or as an env var
        if not config.get("storage", {}).get("gcs", {}).get("bucket_name"):
            required_env_vars.append("GCS_BUCKET_NAME")
    else:
        raise ValueError(f"Invalid storage provider: {storage_provider}")

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate Obsidian vault path if local
    if storage_provider == "local":
        vault_path = Path(config["storage"]["local"]["vault_path"])
        if not vault_path.exists():
            raise ValueError(f"Obsidian vault path does not exist: {vault_path}")
    
    # Validate embedding configuration
    if config["embeddings"]["provider"] not in ["openai", "local"]:
        raise ValueError("Invalid embedding provider. Must be 'openai' or 'local'")
    
    # Validate vector store configuration
    if config["vector_store"]["type"] not in ["faiss", "chroma", "qdrant"]:
        raise ValueError("Invalid vector store type. Must be 'faiss', 'chroma', or 'qdrant'") 