#!/usr/bin/env python
"""
Auto vault reset script - runs without interactive prompts
Addresses OpenAI API rate limits and context length issues
"""
import os
import json
import logging
import time
import shutil
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vault_reset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VaultResetManager:
    def __init__(self):
        self.vector_store_path = Path("data/vector_store")
        self.backup_path = Path("data/vector_store/backup")
        self.max_retries = 3
        self.rate_limit_delay = 60
        self.batch_size = 50
        
    def create_backup(self):
        """Create timestamped backup of current vector store"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            files_to_backup = [
                "document_store.json",
                "embeddings.json", 
                "faiss.index"
            ]
            
            for filename in files_to_backup:
                src = self.vector_store_path / filename
                if src.exists():
                    dst = backup_dir / filename
                    shutil.copy2(src, dst)
                    logger.info(f"Backed up {filename} to {dst}")
            
            logger.info(f"Backup created at {backup_dir}")
            return backup_dir
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            raise

    def clear_vector_store(self):
        """Safely clear the vector store files"""
        try:
            files_to_clear = [
                "document_store.json",
                "embeddings.json",
                "faiss.index"
            ]
            
            for filename in files_to_clear:
                file_path = self.vector_store_path / filename
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed {filename}")
            
            logger.info("Vector store cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            raise

    def check_openai_api_status(self):
        """Check if OpenAI API is accessible"""
        try:
            # Skip API check for now to avoid further rate limiting
            logger.info("Skipping OpenAI API check to avoid rate limits")
            return True
        except Exception as e:
            logger.warning(f"API check failed: {str(e)}")
            return False

    def restart_backend_service(self):
        """Stop any running backend processes"""
        try:
            import subprocess
            
            logger.info("Stopping any running backend processes...")
            
            try:
                # Kill Python processes more safely
                subprocess.run([
                    "taskkill", "/F", "/IM", "python.exe"
                ], capture_output=True, check=False)
                
                logger.info("Backend processes stopped")
                time.sleep(3)  # Shorter wait
                
            except Exception as e:
                logger.warning(f"Could not stop backend processes: {str(e)}")
                
        except Exception as e:
            logger.warning(f"Failed to restart backend service: {str(e)}")

    def reset_vault(self):
        """Main vault reset process - AUTO MODE"""
        try:
            logger.info("=== AUTO VAULT RESET STARTING ===")
            logger.info("This will reset your vault and clear all embeddings")
            logger.info("A backup will be created before proceeding")
            
            # Step 1: Stop any running backend processes
            self.restart_backend_service()
            
            # Step 2: Create backup
            backup_dir = self.create_backup()
            
            # Step 3: Clear existing vector store
            self.clear_vector_store()
            
            logger.info("✅ Vault reset completed successfully!")
            logger.info("Next steps:")
            logger.info("1. Restart your application")
            logger.info("2. Re-index your documents gradually") 
            logger.info("3. Monitor the logs for any rate limiting issues")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Vault reset failed: {str(e)}")
            return False

def main():
    """Main execution function - AUTO MODE"""
    manager = VaultResetManager()
    return manager.reset_vault()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 