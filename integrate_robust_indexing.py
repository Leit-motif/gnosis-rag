#!/usr/bin/env python
"""
Integration script to add robust indexing endpoints to main.py
This will modify the main.py file to include the new robust indexing endpoints
"""
import os
import shutil
from pathlib import Path

def backup_main_py():
    """Create a backup of the current main.py"""
    main_file = Path("backend/main.py")
    backup_file = Path("backend/main.py.backup")
    
    if main_file.exists():
        shutil.copy2(main_file, backup_file)
        print(f"✅ Created backup: {backup_file}")
        return True
    else:
        print("❌ main.py not found in backend directory")
        return False

def add_robust_endpoints():
    """Add robust indexing endpoints to main.py"""
    main_file = Path("backend/main.py")
    
    if not main_file.exists():
        print("❌ main.py not found")
        return False
    
    # Read the current main.py
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if robust endpoints are already added
    if "from robust_index_endpoint import router as robust_router" in content:
        print("ℹ️ Robust endpoints already integrated")
        return True
    
    # Find where to add the import
    import_lines = []
    other_lines = []
    in_imports = True
    
    for line in content.split('\n'):
        if line.strip().startswith('from ') or line.strip().startswith('import '):
            if in_imports:
                import_lines.append(line)
            else:
                other_lines.append(line)
        else:
            if in_imports and line.strip():  # First non-import line
                in_imports = False
            other_lines.append(line)
    
    # Add the new import
    import_lines.append("from robust_index_endpoint import router as robust_router")
    
    # Find where to add the router include
    new_other_lines = []
    app_included = False
    
    for line in other_lines:
        new_other_lines.append(line)
        
        # Add the router include after app creation
        if "app = FastAPI(" in line and not app_included:
            # Look for the end of the FastAPI constructor
            if ");" in line or line.strip().endswith(")"):
                new_other_lines.append("")
                new_other_lines.append("# Include robust indexing endpoints")
                new_other_lines.append("app.include_router(robust_router)")
                new_other_lines.append("")
                app_included = True
        elif "app = FastAPI()" in line and not app_included:
            new_other_lines.append("")
            new_other_lines.append("# Include robust indexing endpoints")
            new_other_lines.append("app.include_router(robust_router)")
            new_other_lines.append("")
            app_included = True
    
    # Reconstruct the file
    new_content = '\n'.join(import_lines + [''] + new_other_lines)
    
    # Write the updated content
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Added robust indexing endpoints to main.py")
    return True

def update_api_schemas():
    """Update the OpenAPI schemas to include new endpoints"""
    schema_files = [
        "customgpt_schema.yaml",
        "plugin/openapi.yaml"
    ]
    
    new_endpoints = '''
  /index_robust:
    post:
      summary: Robust Index Vault
      description: Index the Obsidian vault content with rate limiting and error recovery. This is the recommended indexing method.
      operationId: index_vault_robust_index_robust_post
      responses:
        '200':
          description: Successful Response
          content:
            application/json:
              schema: {}
  /index_status:
    get:
      summary: Get Index Status
      description: Get the current indexing status and progress information
      operationId: get_index_status_index_status_get
      responses:
        '200':
          description: Successful Response
          content:
            application/json:
              schema: {}
  /resume_index:
    post:
      summary: Resume Indexing
      description: Resume indexing from where it left off after a failure or interruption
      operationId: resume_indexing_resume_index_post
      responses:
        '200':
          description: Successful Response
          content:
            application/json:
              schema: {}'''
    
    for schema_file in schema_files:
        file_path = Path(schema_file)
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if endpoints are already added
                if "/index_robust:" in content:
                    print(f"ℹ️ {schema_file} already includes robust endpoints")
                    continue
                
                # Find the paths section and add the new endpoints
                if "paths:" in content:
                    # Find the location to insert (after existing /index endpoint)
                    if "/index:" in content:
                        # Insert after the /index endpoint
                        index_pos = content.find("/index:")
                        next_endpoint_pos = content.find("\n  /", index_pos + 1)
                        
                        if next_endpoint_pos != -1:
                            # Insert before the next endpoint
                            content = content[:next_endpoint_pos] + new_endpoints + content[next_endpoint_pos:]
                        else:
                            # Insert at the end of paths
                            paths_end = content.rfind("components:")
                            if paths_end != -1:
                                content = content[:paths_end] + new_endpoints + "\n" + content[paths_end:]
                    
                    # Write back
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"✅ Updated {schema_file}")
                
            except Exception as e:
                print(f"❌ Failed to update {schema_file}: {str(e)}")
        else:
            print(f"⚠️ {schema_file} not found")

def create_usage_guide():
    """Create a usage guide for the robust indexing system"""
    guide_content = """# Robust Indexing System Usage Guide

## Overview
The robust indexing system provides reliable document indexing with rate limiting, error recovery, and progress tracking.

## New Endpoints

### 1. POST /index_robust
Replaces the original `/index` endpoint with better error handling.

**Features:**
- Rate limiting to avoid OpenAI API limits
- Progressive saving with checkpoints
- Automatic resume capability
- Better error messages and logging

**Usage:**
```bash
curl -X POST "http://localhost:8000/index_robust"
```

### 2. GET /index_status
Check the current indexing status and progress.

**Usage:**
```bash
curl "http://localhost:8000/index_status"
```

**Response:**
- `not_started`: No indexing has been performed
- `in_progress`: Indexing is currently running or was interrupted
- `completed`: Indexing finished successfully

### 3. POST /resume_index
Resume indexing from where it left off after a failure.

**Usage:**
```bash
curl -X POST "http://localhost:8000/resume_index"
```

## Configuration
The system uses `config_rate_limited.json` for rate limiting settings:

```json
{
  "rate_limiting": {
    "max_requests_per_minute": 50,
    "batch_size": 25,
    "delay_between_batches": 2.0,
    "max_retries": 3,
    "backoff_factor": 2.0
  },
  "api_settings": {
    "max_tokens_per_request": 4000,
    "embedding_timeout": 30,
    "max_concurrent_requests": 2
  }
}
```

## Troubleshooting

### Rate Limiting Issues
- The system automatically handles OpenAI rate limits
- Increase `delay_between_batches` if you still hit limits
- Reduce `batch_size` for more conservative API usage

### Resuming After Failure
1. Check status: `GET /index_status`
2. If status is `in_progress`, run: `POST /resume_index`
3. The system will continue from the last checkpoint

### Monitoring Progress
- Check logs in `vault_reset.log`
- Use `/index_status` to see progress statistics
- Checkpoint files are saved in `data/vector_store/checkpoints/`

## Migration from Original Indexing
1. Use `/index_robust` instead of `/index`
2. Monitor progress with `/index_status`
3. If interrupted, resume with `/resume_index`
4. The system is backward compatible with existing vector stores
"""
    
    with open("ROBUST_INDEXING_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("✅ Created usage guide: ROBUST_INDEXING_GUIDE.md")

def main():
    """Main integration function"""
    print("🔧 Integrating Robust Indexing System")
    print("=" * 50)
    
    # Step 1: Backup main.py
    print("1. Creating backup...")
    if not backup_main_py():
        return False
    
    # Step 2: Add endpoints to main.py
    print("2. Adding robust endpoints...")
    if not add_robust_endpoints():
        return False
    
    # Step 3: Update API schemas
    print("3. Updating API schemas...")
    update_api_schemas()
    
    # Step 4: Create usage guide
    print("4. Creating usage guide...")
    create_usage_guide()
    
    print("\n✅ Integration Complete!")
    print("\nNext steps:")
    print("1. Restart your application")
    print("2. Test the new endpoints:")
    print("   - POST /index_robust (for indexing)")
    print("   - GET /index_status (to check progress)")
    print("   - POST /resume_index (if needed)")
    print("3. Read ROBUST_INDEXING_GUIDE.md for detailed usage")
    print("\nThe original /index endpoint is still available as fallback.")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Integration failed. Check the errors above.")
        exit(1)
    else:
        print("\n🎉 Integration successful!")
        exit(0) 