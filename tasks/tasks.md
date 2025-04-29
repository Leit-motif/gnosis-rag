# Obsidian ChatGPT Plugin Implementation Tasks

## Phase 1: Setup and Infrastructure

- [x] 1. Review existing codebase and documentation
  - [x] Understand the current RAG pipeline implementation
  - [x] Identify integration points between backend and plugin
  - [x] Document the data flow from Obsidian vault to ChatGPT

- [x] 2. Configure development environment
  - [x] Set up ngrok for local testing
  - [x] Configure ChatGPT developer environment
  - [x] Test existing backend functionality

- [x] 3. Update plugin manifest
  - [x] Update plugin/ai-plugin.json with proper metadata
  - [x] Configure authentication mechanisms (using "none" for prototype)
  - [x] Define appropriate plugin description and instructions

- [x] 4. Create OpenAPI specification
  - [x] Enhance existing plugin/openapi.yaml
  - [x] Define all endpoints required for ChatGPT integration
  - [x] Document request/response formats

- [x] 5. Remove unused API calls
  - [x] Reflection
  - [x] Health
  - [x] Themes
  - [x] Reflect

 - [x] 6. Add ability to write current chatgpt conversation (from within the plugin) to the current day's page in my obsidian vault. It should be under it's own header '## My Obsidian Helper: {conversation name}'
  - [x] If the day's page hasn't been created, create it first.

[ ] 7. Implement True Graph RAG
  - [x] 7.1 Define retrieval strategy for True Graph RAG (graph traversal, hybrid, etc.)  
  - [x] 7.2 Specify output/context format for LLM input
    - [x] 7.2.1 Define structured format with document relationships
    - [x] 7.2.2 Create templates for different connection types (links, tags, paths)
    - [x] 7.2.3 Determine how to present graph traversal information to the LLM
  - [x] 7.3 Enhance graph construction (ensure all relevant relationships are captured)
    - [x] 7.3.1 Add support for backlinks
    - [x] 7.3.2 Include metadata and frontmatter in node properties
    - [x] 7.3.3 Implement relationship types/weights between nodes
    - [x] 7.3.4 Add timestamp/recency information to nodes
  - [x] 7.4 Implement graph-based retriever (traversal, expansion, etc.)
    - [x] 7.4.1 Create entry point mapping (vector-to-graph and keyword-to-graph)
    - [x] 7.4.2 Implement k-hop neighborhood expansion
    - [x] 7.4.3 Add tag/cluster expansion capability
    - [x] 7.4.4 Build path-based document connection discovery
    - [x] 7.4.5 Add configurable traversal parameters
  - [x] 7.5 Implement hybrid retriever (combine graph and vector similarity)
    - [x] 7.5.1 Create candidate set generation via graph traversal
    - [x] 7.5.2 Implement re-ranking with vector similarity
    - [x] 7.5.3 Build weighted scoring function
    - [x] 7.5.4 Add configuration options for hybrid retrieval
  - [x] 7.6 Integrate graph retriever into RAG pipeline
    - [x] 7.6.1 Update RAGPipeline.query() to use graph-based retrieval
    - [x] 7.6.2 Implement graph context formatting for LLM input
    - [x] 7.6.3 Add parameter handling for graph retrieval options
    - [x] 7.6.4 Ensure backward compatibility with vector-only retrieval
  - [x] 7.7 Write unit and integration tests for graph retrieval
    - [x] 7.7.1 Test graph construction with sample vault data
    - [x] 7.7.2 Test different traversal strategies
    - [x] 7.7.3 Test hybrid retrieval and ranking
    - [x] 7.7.4 Test end-to-end RAG pipeline with graph retrieval
  - [ ] 7.8 Evaluate retrieval quality and adjust parameters
    - [ ] 7.8.1 Create evaluation dataset of queries and expected documents
    - [ ] 7.8.2 Measure retrieval precision and recall
    - [ ] 7.8.3 Compare against vector-only baseline
    - [ ] 7.8.4 Fine-tune parameters for optimal performance
  - [x] 7.9 Document design, usage, and configuration
    - [x] 7.9.1 Update README with graph RAG explanation
    - [x] 7.9.2 Document configuration parameters
    - [x] 7.9.3 Create usage examples
    - [x] 7.9.4 Document performance characteristics and trade-offs
