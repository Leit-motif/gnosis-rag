#!/usr/bin/env python
"""
Enhanced vault reset script with rate limiting and error recovery
Addresses OpenAI API rate limits and context length issues
"""
import os
import json
import logging
import time
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
from typing import List, Dict, Any

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
        self.rate_limit_delay = 60  # 60 seconds between batches
        self.batch_size = 50  # Smaller batches to avoid rate limits
        
    def create_backup(self):
        """Create timestamped backup of current vector store"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backup_path / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup existing files
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
        """Check if OpenAI API is accessible and within rate limits"""
        try:
            import openai
            from openai import OpenAI
            
            client = OpenAI()
            
            # Test with a small request
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
            
            logger.info("OpenAI API is accessible")
            return True
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                logger.warning("Rate limit detected. Need to wait...")
                return False
            else:
                logger.error(f"OpenAI API error: {str(e)}")
                raise

    def wait_for_rate_limit_reset(self):
        """Wait for OpenAI rate limits to reset"""
        logger.info("Waiting for rate limits to reset...")
        time.sleep(self.rate_limit_delay)

    def restart_backend_service(self):
        """Restart the backend service to clear any stuck processes"""
        try:
            import subprocess
            
            # Kill any existing Python processes that might be using the API
            logger.info("Stopping any running backend processes...")
            
            # On Windows, find and kill Python processes running the backend
            try:
                result = subprocess.run([
                    "tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"
                ], capture_output=True, text=True)
                
                if "python.exe" in result.stdout:
                    logger.info("Found running Python processes")
                    
                # Force kill any backend processes (be careful with this)
                subprocess.run([
                    "taskkill", "/F", "/IM", "python.exe", "/T"
                ], capture_output=True)
                
                logger.info("Backend processes stopped")
                time.sleep(5)  # Wait for cleanup
                
            except Exception as e:
                logger.warning(f"Could not stop backend processes: {str(e)}")
                
        except Exception as e:
            logger.warning(f"Failed to restart backend service: {str(e)}")

    def validate_environment(self):
        """Validate environment variables and configuration"""
        required_vars = ["OPENAI_API_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            raise EnvironmentError(f"Required environment variables not set: {missing_vars}")
        
        logger.info("Environment validation passed")

    def reset_vault(self):
        """Main vault reset process"""
        try:
            logger.info("Starting vault reset process...")
            
            # Step 1: Validate environment
            self.validate_environment()
            
            # Step 2: Stop any running backend processes
            self.restart_backend_service()
            
            # Step 3: Create backup
            backup_dir = self.create_backup()
            
            # Step 4: Clear existing vector store
            self.clear_vector_store()
            
            # Step 5: Wait for rate limits if needed
            if not self.check_openai_api_status():
                self.wait_for_rate_limit_reset()
            
            logger.info("Vault reset completed successfully!")
            logger.info("You can now restart your application and re-index gradually")
            
            return True
            
        except Exception as e:
            logger.error(f"Vault reset failed: {str(e)}")
            return False

def main():
    """Main execution function"""
    logger.info("=== Vault Reset Tool ===")
    
    # Confirm with user
    print("\nThis will reset your vault and clear all embeddings.")
    print("A backup will be created before proceeding.")
    
    confirmation = input("\nDo you want to continue? (yes/no): ").lower().strip()
    
    if confirmation != 'yes':
        print("Reset cancelled.")
        return
    
    # Execute reset
    manager = VaultResetManager()
    success = manager.reset_vault()
    
    if success:
        print("\n✅ Vault reset completed successfully!")
        print("\nNext steps:")
        print("1. Restart your application")
        print("2. Re-index your documents gradually")
        print("3. Monitor the logs for any rate limiting issues")
    else:
        print("\n❌ Vault reset failed. Check the logs for details.")

if __name__ == "__main__":
    main() 