#!/usr/bin/env python
import os
import json
import logging
import re
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
        # Look for patterns like .../2023/04/... or .../2023-04/...
        dir_pattern1 = r'[/\\](\d{4})[/\\](\d{2})[/\\]'  # .../2023/04/...
        dir_pattern2 = r'[/\\](\d{4})-(\d{2})[/\\]'      # .../2023-04/...
        
        dir_match1 = re.search(dir_pattern1, path)
        if dir_match1:
            year, month = dir_match1.groups()
            
            # Try to extract day from filename if it's numeric
            day_match = re.search(r'^(\d{1,2})', filename_without_ext)
            day = day_match.group(1).zfill(2) if day_match else "01"  # Default to first day if not found
            
            match = (year, month, day)
        else:
            dir_match2 = re.search(dir_pattern2, path)
            if dir_match2:
                year, month = dir_match2.groups()
                # Default to first day of month
                match = (year, month, "01")
    
    if match:
        year, month, day = match if isinstance(match, tuple) else match.groups()
        # Create ISO format date string
        date_str = f"{year}-{month}-{day}"
        
        try:
            # Create datetime object to validate date is legit
            date_obj = datetime.fromisoformat(date_str)
            
            # Add date components to metadata
            date_info['date'] = date_str
            date_info['year'] = year
            date_info['month'] = month
            date_info['day'] = day
            date_info['created'] = date_str
            
            logger.info(f"Extracted date {date_str} from path {path}")
        except ValueError:
            logger.warning(f"Found date-like pattern in {path} but date is invalid")
            
    return date_info

def test_paths():
    """Test the date extraction with various path formats"""
    test_paths = [
        # YYYY-MM-DD filenames
        "2023-04-10.md",
        "notes/2023-04-10.md",
        "journal/entries/2023-04-10.md",
        
        # Directory structure
        "journal/2023/04/10.md",
        "journal/2023/04/entry.md",
        "journal/2023/04/notes.md",
        
        # Mixed
        "journal/2023/04/2023-04-10.md",
        
        # Year only
        "archive/2023/random_notes.md",
        
        # No date information
        "notes/random.md",
        "thoughts.md"
    ]
    
    logger.info("Testing date extraction from various path formats:")
    
    for path in test_paths:
        logger.info(f"\nTesting path: {path}")
        date_info = extract_date_from_path(path)
        if date_info:
            logger.info(f"Found date: {json.dumps(date_info, indent=2)}")
        else:
            logger.info("No date information found")

def test_actual_files():
    """Test with actual files from the data directory"""
    data_dir = Path("data")
    if not data_dir.exists():
        logger.warning("Data directory not found")
        return
        
    file_count = 0
    date_found_count = 0
    
    logger.info("\nTesting with actual files from data directory:")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.md'):
                file_count += 1
                path = os.path.join(root, file)
                date_info = extract_date_from_path(path)
                if date_info:
                    date_found_count += 1
                    logger.info(f"File: {path}")
                    logger.info(f"Found date: {json.dumps(date_info, indent=2)}")
    
    if file_count > 0:
        logger.info(f"\nSummary: Found dates in {date_found_count} out of {file_count} files ({date_found_count/file_count*100:.1f}%)")
    else:
        logger.info("No markdown files found in data directory")

if __name__ == "__main__":
    # Test with example paths
    test_paths()
    
    # Test with actual files
    test_actual_files() 