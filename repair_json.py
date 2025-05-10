import json
import os
from pathlib import Path

# Paths
file_path = 'data/vector_store/document_store.json'
repaired_path = 'data/vector_store/document_store_repaired.json'

print(f"Attempting to repair {file_path}...")

try:
    # Read the file and try to find valid JSON objects
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try to find the start of the document_store structure
    start_marker = '{"status": "success"'
    if start_marker in content:
        # Find the start of the valid JSON
        start_pos = content.find(start_marker)
        partial_content = content[start_pos:]
        
        # Try to parse with increasing lengths to find valid JSON
        valid_json = None
        for i in range(100, len(partial_content), 100):
            try:
                test_json = partial_content[:i]
                # Try to complete the JSON by adding missing braces
                if test_json.count('{') > test_json.count('}'):
                    test_json += '}' * (test_json.count('{') - test_json.count('}'))
                json.loads(test_json)
                valid_json = test_json
                print(f"Found valid JSON structure at position {start_pos} with length {i}")
                break
            except json.JSONDecodeError:
                continue
        
        if valid_json:
            # Write the valid JSON to the repaired file
            with open(repaired_path, 'w', encoding='utf-8') as f:
                f.write(valid_json)
            print(f"Repaired JSON saved to {repaired_path}")
    else:
        print("Could not find valid JSON structure marker")
    
    # If we couldn't repair it that way, let's try to create a new valid structure
    if not os.path.exists(repaired_path):
        print("Attempting to create a new valid document store structure...")
        
        # Create a minimal valid document store structure
        minimal_structure = {
            "status": "success",
            "message": "Indexed 0 documents from vault",
            "document_count": 0,
            "documents": []
        }
        
        with open(repaired_path, 'w', encoding='utf-8') as f:
            json.dump(minimal_structure, f, indent=2)
        print(f"Created new empty document store at {repaired_path}")
    
except Exception as e:
    print(f"Error during repair: {e}")

print("\nSuggested next steps:")
print("1. Verify the repaired file using: python -c \"import json; print(json.load(open('data/vector_store/document_store_repaired.json')))\"")
print("2. If the repaired file is valid, replace the original: copy data\\vector_store\\document_store_repaired.json data\\vector_store\\document_store.json")
print("3. Reindex your documents to rebuild the document store correctly") 