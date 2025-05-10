# True Graph RAG Usage Examples

This document provides practical examples of how to use the True Graph RAG implementation.

## Basic Setup

To set up Graph RAG in your existing RAG pipeline:

```python
from backend.rag_pipeline import RAGPipeline
from backend.graph_rag_integration import GraphRAGIntegration

# Initialize your regular RAG pipeline first
config = {
    "vault": {"path": "/path/to/your/obsidian/vault"},
    # Other RAG pipeline configuration...
}
rag_pipeline = RAGPipeline(config)

# Now integrate the enhanced graph retriever
graph_config = {
    "enabled": True,
    "default_mode": "hybrid",  # "vector", "graph", or "hybrid"
    "max_results": 10,
    "graph_retriever": {
        "entry_points": {
            "vector_entry_count": 3,
            "tag_entry_enabled": True
        },
        "traversal": {
            "max_hops": 2,
            "max_documents": 15
        }
        # Additional configuration...
    }
}

# This will enhance your RAG pipeline with graph retrieval
graph_integration = GraphRAGIntegration(rag_pipeline, graph_config)

# Your rag_pipeline.query method is now enhanced with graph retrieval
# You can use it as before, but with added graph capabilities
```

## Querying Examples

### Vector-Only Retrieval

If you want to use the traditional vector-based retrieval:

```python
# Use the vector-only mode
results = rag_pipeline.query(
    query="Tell me about knowledge management",
    retrieval_mode="vector",
    k=5  # Number of vector results to retrieve
)

# Process the results
context = results.get("context", "")
print(f"Found {len(results.get('results', []))} documents")
```

### Graph-Only Retrieval

To use pure graph-based retrieval without vector similarity:

```python
# Use the graph-only mode
results = rag_pipeline.query(
    query="Tell me about #productivity techniques",
    retrieval_mode="graph"
)

# The results include graph traversal information
for doc in results.get("results", []):
    print(f"Document: {doc['metadata'].get('title')}")
    print(f"Distance: {doc['distance']}")
    print(f"Tags: {', '.join(doc['tags'])}")
    
    # Show connections
    if not doc['entry_point'] and doc['connections']:
        for conn in doc['connections']:
            print(f"Connected via: {conn['type']} from {conn['from']}")
    
    print("---")
```

### Hybrid Retrieval (Default)

The hybrid mode combines vector similarity with graph traversal:

```python
# Use hybrid mode (default if not specified)
results = rag_pipeline.query(
    query="How do zettelkasten and linking notes help with creativity?",
    # retrieval_mode="hybrid"  # This is the default
)

# The results are ranked by a combination of graph and vector scores
for i, doc in enumerate(results.get("results", []), 1):
    print(f"{i}. {doc['metadata'].get('title')}")
    print(f"   Combined score: {doc['score']:.3f}")
    print(f"   Graph score: {doc['graph_score']:.3f}")
    if 'vector_similarity' in doc:
        print(f"   Vector similarity: {doc['vector_similarity']:.3f}")
```

## Advanced Configuration

### Adjusting Graph Traversal Parameters

You can fine-tune the graph traversal behavior:

```python
# Create a more expansive graph traversal configuration
expansive_config = {
    "graph_retriever": {
        "traversal": {
            "max_hops": 3,            # Increase hop distance
            "max_documents": 25,       # Return more documents
            "tag_expansion_enabled": True,   # Enable tag-based expansion
            "path_expansion_enabled": True,  # Find paths between entry points
            "min_similarity": 0.3      # Lower threshold for inclusion
        }
    }
}

# Update the configuration
graph_integration._update_config(expansive_config)

# Query with the new config
results = rag_pipeline.query(
    query="How are my project management notes connected to my productivity system?",
)
```

### Customizing Entry Point Strategy

You can adjust how the graph entry points are selected:

```python
# Emphasize tag matching over vector similarity
tag_focus_config = {
    "graph_retriever": {
        "entry_points": {
            "vector_entry_count": 2,     # Use fewer vector results
            "tag_entry_enabled": True,
            "entry_weight_vector": 0.4,  # Lower weight for vector results
            "entry_weight_tags": 0.6     # Higher weight for tag matches
        }
    }
}

# Update the configuration
graph_integration._update_config(tag_focus_config)

# Query focusing more on tag matches
results = rag_pipeline.query(
    query="Show me my #project notes related to #coding",
)
```

### Adjusting Hybrid Ranking Weights

You can change how the hybrid ranking balances graph and vector scores:

```python
# Prioritize graph structure over vector similarity
graph_priority_config = {
    "graph_retriever": {
        "hybrid": {
            "enabled": True,
            "graph_weight": 0.8,       # Higher weight for graph scores
            "vector_weight": 0.2,       # Lower weight for vector similarity
            "recency_bonus": 0.15       # Slight boost for recent documents
        }
    }
}

# Update the configuration
graph_integration._update_config(graph_priority_config)

# Query with graph-prioritized ranking
results = rag_pipeline.query(
    query="What are the main concepts in my digital garden?",
)
```

## Analyzing Graph Statistics

You can get statistics about your knowledge graph:

```python
# Get graph statistics
stats = graph_integration.get_graph_statistics()

print(f"Total nodes: {stats['nodes']}")
print(f"Document nodes: {stats['document_nodes']}")
print(f"Tag nodes: {stats['tag_nodes']}")
print(f"Link connections: {stats['link_edges']}")
print(f"Backlink connections: {stats['backlink_edges']}")
print(f"Tag connections: {stats['tag_edges']}")
``` 