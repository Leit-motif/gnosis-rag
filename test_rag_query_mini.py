#!/usr/bin/env python
import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date
from backend.enhanced_graph_retriever import EnhancedGraphRetriever

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle date objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

def create_test_files(temp_dir):
    """Create test markdown files in the temporary directory"""
    vault_path = Path(temp_dir)
    
    # File 1: Introduction note with links to others
    with open(vault_path / "introduction.md", "w") as f:
        f.write("""---
title: Introduction to Graph RAG
created: 2023-01-01
tags: [graph-rag, important]
---

# Introduction to Graph RAG

This is the introduction to Graph Retrieval Augmented Generation.

The key components include:
- Graph Builder
- Graph Retriever
- Context Formatter
- Hybrid Retrieval

Links to:
- [[components]]
- [[examples]]

#graph-rag #important
""")
    
    # File 2: Components note with links and tags
    with open(vault_path / "components.md", "w") as f:
        f.write("""---
title: Graph RAG Components
created: 2023-01-02
tags: [graph-rag, components]
---

# Components of Graph RAG

The key components of Graph RAG are:

1. **Graph Builder**: Creates the knowledge graph
2. **Graph Retriever**: Retrieves nodes from the graph
3. **Context Formatter**: Formats retrieved data for the LLM
4. **Hybrid Retriever**: Combines graph and vector retrieval

See also:
- [[examples]]

Referenced from:
- [[introduction]]

#components #graph-rag
""")
    
    # File 3: Examples note
    with open(vault_path / "examples.md", "w") as f:
        f.write("""---
title: Graph RAG Examples
created: 2023-01-03
tags: [examples, graph-rag]
---

# Examples of Graph RAG

Some example applications of Graph RAG:

- Knowledge management
- Technical documentation
- Academic research
- Personal notes

Reference:
- [[components]]
- [[introduction]]

#examples #graph-rag
""")
    
    return vault_path

def test_optimized_metadata():
    """Create a test environment and verify metadata optimization"""
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test files
        vault_path = create_test_files(temp_dir)
        logger.info(f"Created test files in {vault_path}")
        
        # Initialize and build graph with custom config for tag entry points
        retriever = EnhancedGraphRetriever(
            vault_path,
            config={
                "entry_points": {
                    "vector_entry_count": 1,
                    "tag_entry_enabled": True,
                    "entry_weight_vector": 0.5,
                    "entry_weight_tags": 0.5
                }
            }
        )
        retriever.build_graph()
        
        # Test query with tag that exists in our data
        query = "What are the important #graph-rag components?"
        
        # Get results
        results = retriever.query(query)
        
        # Format results
        formatted_context = retriever.format_context_for_llm(results)
        
        # Print results
        print("\n" + "="*50)
        print("OPTIMIZED METADATA TEST RESULTS")
        print("="*50)
        print(f"\nQuery: {query}")
        print(f"\nFound {len(results.get('results', []))} results")
        
        # Print metadata sizes to verify optimization
        total_metadata_bytes = 0
        for i, doc in enumerate(results.get("results", []), 1):
            metadata = doc.get("metadata", {})
            metadata_size = len(json.dumps(metadata, cls=CustomJSONEncoder))
            total_metadata_bytes += metadata_size
            
            print(f"\n[{i}] Document: {doc['id']}")
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  Score: {doc.get('score', 0):.4f}")
            
            # Print all metadata keys to show optimized structure
            print("  Metadata keys:", ", ".join(sorted(metadata.keys())))
            print(f"  Metadata size: {metadata_size} bytes")
            
            # Print specific metadata values to verify essential-only approach
            if "tags" in metadata and metadata["tags"]:
                print(f"  Tags: {metadata['tags']}")
            
            if "created" in metadata:
                print(f"  Created: {metadata['created']}")
                
            if "links" in metadata:
                print(f"  Links: {metadata['links']}")
                
            # Show what frontmatter is NOT included (should be removed by our optimization)
            if "frontmatter" in metadata:
                print(f"  Frontmatter included: {len(metadata['frontmatter'])} fields")
            else:
                print(f"  Frontmatter: Optimized out (good!)")
            
            # Print a content snippet
            content = doc.get("content", "")
            if content:
                print(f"  Content snippet: {content[:75]}...")
        
        print(f"\nTotal metadata size for all results: {total_metadata_bytes} bytes")
        
        # Print formatted context (abbreviated if long)
        print("\n" + "="*50)
        print("FORMATTED CONTEXT FOR LLM:")
        print("="*50)
        context_to_show = formatted_context[:1000] + "..." if len(formatted_context) > 1000 else formatted_context
        print(context_to_show)
        
        return results
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up test directory {temp_dir}")

if __name__ == "__main__":
    test_optimized_metadata() 