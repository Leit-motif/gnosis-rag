# True Graph RAG Retrieval Strategy

## Overview

True Graph RAG (Retrieval-Augmented Generation) leverages the knowledge graph structure within the Obsidian vault to provide more contextually relevant information to the LLM. Unlike traditional vector-based RAG, which relies solely on embedding similarity, Graph RAG uses the inherent connections between documents (links, tags, backlinks) to retrieve related content.

## Current Implementation vs. True Graph RAG

### Current Implementation

The current system has:
- Vector-based retrieval using FAISS
- Basic graph structure in `GraphRetriever` that builds connections but isn't deeply integrated into retrieval
- Document relationships tracked but not fully utilized in the RAG pipeline

### True Graph RAG Approach

The enhanced approach will:
- Use the knowledge graph as a first-class citizen in retrieval
- Traverse the graph to find contextually related documents based on connection types
- Optionally combine graph-based and vector-based retrieval for a hybrid approach

## Retrieval Strategy

### 1. Entry Points to the Graph

For an incoming query, we'll need to map it to one or more nodes in the knowledge graph:

**Option A: Vector-First Entry Point**
- Use embedding similarity to find the most relevant document(s) as starting node(s)
- Query is embedded and compared to document embeddings
- Top-k documents become entry points into the graph

**Option B: Tag/Keyword Entry Point**
- Extract keywords or tags from the query
- Match against document tags/titles/content
- Documents with matching tags/keywords become entry points

**Selected Approach: Hybrid Entry**
- Start with vector-based similarity to find initial documents
- Enhance with tag/keyword matching for explicit tag mentions
- Use both as entry points with configurable weighting

### 2. Graph Traversal Strategies

Once entry points are identified, we'll expand through the graph using one or more traversal strategies:

**K-Hop Neighborhood Expansion**
- Collect all nodes within k steps from entry points
- Weight nodes by distance (closer = higher relevance)
- Filter by node type (e.g., include only document nodes)

**Path-Based Expansion**
- Find meaningful paths between high-relevance nodes
- Include nodes along these paths as context
- Paths can reveal narrative or logical connections

**Tag/Cluster Expansion**
- Identify relevant tags from entry points
- Expand to include other documents sharing these tags
- Create tag-based document clusters

**Selected Approach: Multi-Strategy Traversal**
- Begin with k-hop neighborhood expansion from entry points
- For documents with common tags, perform tag expansion
- If multiple entry points exist, include path-based connections
- Weight by combination of distance, tag relevance, and original vector similarity

### 3. Hybrid Retrieval: Graph + Vector

To ensure quality and relevance, we'll combine graph-based and vector-based approaches:

- Use graph traversal to generate a candidate set of documents
- Re-rank these candidates using vector similarity to the query
- Apply a weighted scoring function that considers:
  - Graph distance
  - Connection type (direct link > shared tag > distant connection)
  - Vector similarity
  - Recency or other metadata factors

## Implementation Parameters

The following parameters will be configurable:

1. **Entry Point Configuration**
   - `vector_entry_count`: Number of vector-similar documents to use as entry points (default: 3)
   - `tag_entry_enabled`: Whether to use tag/keyword matching for entry points (default: true)
   - `entry_weight_vector`: Weight for vector-similarity in combined ranking (default: 0.7)
   - `entry_weight_tags`: Weight for tag-matching in combined ranking (default: 0.3)

2. **Graph Traversal Configuration**
   - `max_hops`: Maximum distance to traverse from entry points (default: 2)
   - `max_documents`: Maximum total documents to retrieve (default: 15)
   - `tag_expansion_enabled`: Whether to expand via shared tags (default: true)
   - `path_expansion_enabled`: Whether to include paths between entry points (default: true)
   - `min_similarity`: Minimum similarity threshold for inclusion (default: 0.5)

3. **Hybrid Ranking Configuration**
   - `hybrid_enabled`: Whether to re-rank graph results with vector similarity (default: true)
   - `graph_weight`: Weight of graph-based factors in final ranking (default: 0.6)
   - `vector_weight`: Weight of vector similarity in final ranking (default: 0.4)
   - `recency_bonus`: Extra weight for more recent documents (default: 0.1)

## Output Format for LLM

The retrieved context will be structured as follows:

```
CONTEXT:
[1] Document: "Title 1" (Distance: 1, Shared Tags: tag1, tag2)
Content: "Document content..."

[2] Document: "Title 2" (Distance: 2, Path: Entry → Link → Title 2)  
Content: "Document content..."

[Connection] Documents 1 and 2 are connected via shared tag "tag1"

[3] Document: "Title 3" (Distance: 1, Direct Link from Query Entry)
Content: "Document content..."
```

This format:
- Presents documents in ranked order
- Includes metadata about relationships
- Makes explicit connections between documents
- Provides the LLM with graph structure information

## Next Steps

1. Enhance `GraphRetriever` with new traversal methods
2. Create entry point mapping logic
3. Implement hybrid ranking algorithm
4. Update RAG pipeline to use the new retrieval strategy
5. Develop configurable parameters interface
6. Test and tune performance 