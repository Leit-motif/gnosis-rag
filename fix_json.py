import json
import os

file_path = 'data/vector_store/document_store.json'
backup_path = 'data/vector_store/document_store.json.bak'

def deduplicate_preserving_order(array):
    """Deduplicate a list while preserving the order of first occurrences."""
    seen = set()
    result = []
    for item in array:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def deduplicate_document_store(data):
    """Process the entire document store to deduplicate arrays in metadata."""
    if "sources" not in data:
        print("Warning: 'sources' key not found in the document store")
        return data
    
    total_sources = len(data["sources"])
    total_deduplications = 0
    
    for i, source in enumerate(data["sources"]):
        if "metadata" in source:
            metadata = source["metadata"]
            
            # Arrays to deduplicate
            arrays_to_deduplicate = [
                "tags", "internal_links", "markdown_links", "link_titles"
            ]
            
            for array_name in arrays_to_deduplicate:
                if array_name in metadata and isinstance(metadata[array_name], list):
                    original_length = len(metadata[array_name])
                    metadata[array_name] = deduplicate_preserving_order(metadata[array_name])
                    new_length = len(metadata[array_name])
                    total_deduplications += (original_length - new_length)
        
        # Progress indicator for large files
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_sources} sources...")
    
    print(f"Deduplication complete. Removed {total_deduplications} duplicate entries.")
    return data

# Create backup
if os.path.exists(file_path):
    with open(file_path, 'rb') as src:
        with open(backup_path, 'wb') as dst:
            dst.write(src.read())
    print(f"Created backup at {backup_path}")

# Try to load and identify the error
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("JSON is valid! Proceeding with deduplication...")
    
    # Deduplicate the document store
    deduplicated_data = deduplicate_document_store(data)
    
    # Save the deduplicated data back to the file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_data, f, indent=2)
        print(f"Successfully saved deduplicated data to {file_path}")
    except Exception as save_e:
        print(f"Error saving deduplicated data: {save_e}")
        print("Original data has not been modified.")

except json.JSONDecodeError as e:
    print(f"JSON error: {e}")
    
    # Get error position details
    error_line = e.lineno
    error_col = e.colno
    error_msg = e.msg
    
    print(f"Error at line {error_line}, column {error_col}: {error_msg}")
    
    # For large files, read as text and debug around the error
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get context around the error
        start_pos = max(0, error_col - 50)
        end_pos = min(len(content), error_col + 50)
        context = content[start_pos:end_pos]
        
        print(f"\nContext around error:\n{repr(context)}\n")
        
        # Simple attempt to fix common issues: unescaped quotes, missing commas, etc.
        print("Attempting to fix the JSON file...")
        
        # Try a basic approach - validate objects
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Count braces to check overall structure
            opening_braces = content.count('{')
            closing_braces = content.count('}')
            opening_brackets = content.count('[')
            closing_brackets = content.count(']')
            
            print(f"Structure check: {{ {opening_braces}:{closing_braces} }}, [ {opening_brackets}:{closing_brackets} ]")
            
            if opening_braces != closing_braces or opening_brackets != closing_brackets:
                print("Unbalanced braces or brackets detected.")
        except Exception as fix_e:
            print(f"Error during fix attempt: {fix_e}")
    except Exception as read_e:
        print(f"Error reading file: {read_e}") 