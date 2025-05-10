#!/usr/bin/env python
import os
import json
import logging
import re
import shutil
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_date_from_path(path: str) -> dict:
    """
    Extract date information from a file path with format YYYY-MM-DD
    Returns a dictionary with extracted date components or empty dict if no match
    """
    date_info = {}
    
    # Try to extract from the last part of the path (filename)
    filename = os.path.basename(path)
    # Remove extension
    filename_without_ext = os.path.splitext(filename)[0]
    
    # Match YYYY-MM-DD pattern in filename
    date_pattern = r'(\d{4})-(\d{2})-(\d{2})'
    match = re.search(date_pattern, filename_without_ext)
    
    if not match:
        # If not found in filename, try to find in directory structure
        # Look for patterns like .../2023/04/... or .../2023/...
        year_pattern = r'[\\/](\d{4})[\\/]'
        month_pattern = r'[\\/](\d{4})[\\/](\d{2})[\\/]'
        
        year_match = re.search(year_pattern, path)
        month_match = re.search(month_pattern, path)
        
        if month_match:
            year, month = month_match.groups()
            date_info['year'] = int(year)
            date_info['month'] = int(month)
            # Create partial ISO date string (YYYY-MM)
            date_info['date'] = f"{year}-{month}"
        elif year_match:
            year = year_match.group(1)
            date_info['year'] = int(year)
            date_info['date'] = year
    else:
        year, month, day = match.groups()
        # Create ISO format date string
        date_str = f"{year}-{month}-{day}"
        date_info = {
            'year': int(year),
            'month': int(month),
            'day': int(day),
            'date': date_str
        }
    
    return date_info

def update_document_store():
    """Update existing document store with date information from file paths"""
    try:
        # Path to document store
        document_store_path = Path("data/vector_store/document_store.json")
        
        if not document_store_path.exists():
            logger.error(f"Document store not found at {document_store_path}")
            return False
        
        # Create backup
        backup_path = document_store_path.with_suffix('.json.bak')
        try:
            shutil.copy2(document_store_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {str(e)}")
        
        # Load document store
        try:
            with open(document_store_path, 'r', encoding='utf-8') as f:
                document_store = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load document store: {str(e)}")
            return False
        
        # Update each document with date information
        update_count = 0
        for doc_id, doc in document_store.items():
            metadata = doc.get('metadata', {})
            path = metadata.get('path', '')
            
            # Skip if no path
            if not path:
                continue
            
            # Skip if already has complete date metadata
            if metadata.get('date') and metadata.get('year') and metadata.get('month') and metadata.get('day'):
                continue
            
            # Extract date from path
            date_info = extract_date_from_path(path)
            
            # Update metadata if date information was found
            if date_info:
                # Preserve existing metadata and add/update date fields
                metadata.update(date_info)
                doc['metadata'] = metadata
                update_count += 1
        
        logger.info(f"Updated date metadata for {update_count} documents")
        
        # Save updated document store
        try:
            with open(document_store_path, 'w', encoding='utf-8') as f:
                json.dump(document_store, f, ensure_ascii=False)
            logger.info(f"Successfully saved updated document store to {document_store_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save updated document store: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating document store: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting document store update")
    result = update_document_store()
    if result:
        logger.info("Document store update completed successfully")
    else:
        logger.error("Document store update failed") 