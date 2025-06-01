import unittest
import os
import tempfile
import shutil
import networkx as nx
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock, patch
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_graph_retriever import EnhancedGraphRetriever
from graph_rag_integration import GraphRAGIntegration

class TestEnhancedGraphRetriever(unittest.TestCase):
    """Test suite for the EnhancedGraphRetriever"""
    
    def setUp(self):
        """Set up a temporary directory with test markdown files"""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.vault_path = Path(self.temp_dir)
        
        # Create test markdown files
        self.create_test_files()
        
        # Initialize the graph retriever
        self.graph_retriever = EnhancedGraphRetriever(self.vault_path)
        self.graph_retriever.build_graph()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test markdown files in the temporary directory"""
        # Create a few markdown files with links and tags
        
        # File 1: Introduction note with links to others
        with open(self.vault_path / "introduction.md", "w") as f:
            f.write("""---
title: Introduction
created: 2023-01-01
tags: [start, important]
---

# Introduction

This is the introduction note.

Links to:
- [[concepts]]
- [[examples]]

#important #start
""")
        
        # File 2: Concepts note with links and tags
        with open(self.vault_path / "concepts.md", "w") as f:
            f.write("""---
title: Core Concepts
created: 2023-01-02
tags: [concept, important]
---

# Core Concepts

This note explains the core concepts.

See also:
- [[examples]]

Referenced from:
- [[introduction]]

#concept #important
""")
        
        # File 3: Examples note
        with open(self.vault_path / "examples.md", "w") as f:
            f.write("""---
title: Examples
created: 2023-01-03
tags: [example, practice]
---

# Examples

This note contains examples.

Reference:
- [[concepts]]

#example #practice
""")
        
        # File 4: Standalone note not linked to others
        with open(self.vault_path / "standalone.md", "w") as f:
            f.write("""---
title: Standalone Note
created: 2023-01-04
tags: [unrelated]
---

# Standalone Note

This note is not linked to any others.

#unrelated
""")
    
    def test_graph_construction(self):
        """Test that the graph is constructed correctly with all nodes and edges"""
        # Check nodes
        self.assertEqual(self.graph_retriever.graph.number_of_nodes(), 10)  # 4 docs + 6 unique tags
        
        # Check document nodes
        doc_nodes = [n for n, data in self.graph_retriever.graph.nodes(data=True) if data.get("type") == "document"]
        self.assertEqual(len(doc_nodes), 4)
        self.assertIn("introduction.md", doc_nodes)
        self.assertIn("concepts.md", doc_nodes)
        self.assertIn("examples.md", doc_nodes)
        self.assertIn("standalone.md", doc_nodes)
        
        # Check tag nodes
        tag_nodes = [n for n, data in self.graph_retriever.graph.nodes(data=True) if data.get("type") == "tag"]
        self.assertEqual(len(tag_nodes), 6)
        self.assertIn("tag:important", tag_nodes)
        self.assertIn("tag:start", tag_nodes)
        self.assertIn("tag:concept", tag_nodes)
        self.assertIn("tag:example", tag_nodes)
        self.assertIn("tag:practice", tag_nodes)
        self.assertIn("tag:unrelated", tag_nodes)
        
        # Check links
        self.assertTrue(self.graph_retriever.graph.has_edge("introduction.md", "concepts.md"))
        self.assertTrue(self.graph_retriever.graph.has_edge("introduction.md", "examples.md"))
        self.assertTrue(self.graph_retriever.graph.has_edge("concepts.md", "examples.md"))
        
        # Check backlinks
        self.assertTrue(self.graph_retriever.graph.has_edge("concepts.md", "introduction.md"))
        self.assertTrue(self.graph_retriever.graph.has_edge("examples.md", "introduction.md"))
        self.assertTrue(self.graph_retriever.graph.has_edge("examples.md", "concepts.md"))
        
        # Check edge types
        self.assertEqual(self.graph_retriever.graph.edges["introduction.md", "concepts.md"]["type"], "links_to")
        self.assertEqual(self.graph_retriever.graph.edges["concepts.md", "introduction.md"]["type"], "linked_from")
    
    def test_metadata_extraction(self):
        """Test that metadata is correctly extracted from documents"""
        # Check introduction.md metadata
        metadata = self.graph_retriever.document_cache["introduction.md"]["metadata"]
        self.assertEqual(metadata["title"], "Introduction")
        self.assertIn("start", metadata["tags"])
        self.assertIn("important", metadata["tags"])
        
        # Check concepts.md metadata
        metadata = self.graph_retriever.document_cache["concepts.md"]["metadata"]
        self.assertEqual(metadata["title"], "Core Concepts")
        self.assertIn("concept", metadata["tags"])
        self.assertIn("important", metadata["tags"])
        
        # Check frontmatter
        self.assertEqual(
            self.graph_retriever.document_cache["examples.md"]["metadata"]["frontmatter"]["title"],
            "Examples"
        )
    
    def test_entry_point_finding(self):
        """Test finding entry points into the graph"""
        # Mock vector results
        mock_vector_results = [
            {'id': 'introduction.md', 'score': 0.9},
            {'id': 'concepts.md', 'score': 0.8},
        ]
        
        # Test vector entry points
        entry_points = self.graph_retriever.find_entry_points(
            "Tell me about the introduction",
            vector_results=mock_vector_results
        )
        
        self.assertEqual(len(entry_points), 2)
        self.assertEqual(entry_points[0]["document"], "introduction.md")
        self.assertEqual(entry_points[0]["entry_type"], "vector")
        
        # Test tag entry points
        entry_points = self.graph_retriever.find_entry_points(
            "Tell me about #important concepts"
        )
        
        self.assertGreater(len(entry_points), 0)
        # At least one entry should be tag-based
        self.assertTrue(any(e["entry_type"] == "tag" for e in entry_points))
    
    def test_graph_expansion(self):
        """Test graph expansion from entry points"""
        # Create mock entry points
        entry_points = [
            {
                "document": "introduction.md", 
                "similarity": 0.9, 
                "entry_type": "vector",
                "weight": 0.7
            }
        ]
        
        # Expand graph
        expanded_docs = self.graph_retriever.expand_graph(entry_points)
        
        # Check that all documents are found
        self.assertGreater(len(expanded_docs), 1)  # Should find more than just the entry point
        
        # Introduction should be the highest score (entry point)
        intro_doc = next((d for d in expanded_docs if d["document"] == "introduction.md"), None)
        self.assertIsNotNone(intro_doc)
        self.assertTrue(intro_doc["entry_point"])
        
        # Concepts and examples should be included (direct links)
        self.assertTrue(any(d["document"] == "concepts.md" for d in expanded_docs))
        self.assertTrue(any(d["document"] == "examples.md" for d in expanded_docs))
    
    def test_hybrid_reranking(self):
        """Test hybrid reranking with graph and vector scores"""
        # Mock documents from graph expansion
        docs = [
            {"document": "introduction.md", "score": 0.9, "distance": 0, "entry_point": True, "connections": [], "tags": ["important", "start"]},
            {"document": "concepts.md", "score": 0.7, "distance": 1, "entry_point": False, "connections": [], "tags": ["concept", "important"]},
            {"document": "examples.md", "score": 0.5, "distance": 1, "entry_point": False, "connections": [], "tags": ["example", "practice"]}
        ]
        
        # Mock query and embedding function
        query = "Tell me about important concepts"
        
        # Create mock embedding function
        def mock_embed(texts):
            # Return mock embeddings where "concepts" is most similar to query
            if len(texts) == 1:  # Query embedding
                return np.array([[0.1, 0.2, 0.3]])
            else:  # Document embeddings
                return np.array([
                    [0.2, 0.3, 0.1],  # introduction
                    [0.1, 0.2, 0.3],  # concepts (identical to query)
                    [0.3, 0.1, 0.2]   # examples
                ])
        
        # Mock document content
        self.graph_retriever.document_cache = {
            "introduction.md": {"content": "Introduction content", "metadata": {}},
            "concepts.md": {"content": "Concepts content", "metadata": {}},
            "examples.md": {"content": "Examples content", "metadata": {}}
        }
        
        # Test reranking
        reranked_docs = self.graph_retriever.hybrid_rerank(
            docs,
            query,
            embedding_function=mock_embed
        )
        
        # Check that reranking works
        self.assertEqual(len(reranked_docs), 3)
        
        # Check that docs have combined scores
        self.assertTrue(all("combined_score" in doc for doc in reranked_docs))
        
        # In our mock, concepts.md should have the highest vector similarity
        concepts_doc = next(d for d in reranked_docs if d["document"] == "concepts.md")
        self.assertGreater(concepts_doc["vector_similarity"], 
                         next(d for d in reranked_docs if d["document"] == "introduction.md")["vector_similarity"])
    
    def test_context_formatting(self):
        """Test formatting context for LLM"""
        # Mock query result
        query_result = {
            "results": [
                {
                    "id": "introduction.md",
                    "content": "Introduction content",
                    "metadata": {"title": "Introduction", "tags": ["important", "start"]},
                    "score": 0.9,
                    "graph_score": 0.9,
                    "distance": 0,
                    "entry_point": True,
                    "connections": [],
                    "tags": ["important", "start"]
                },
                {
                    "id": "concepts.md",
                    "content": "Concepts content",
                    "metadata": {"title": "Core Concepts", "tags": ["concept", "important"]},
                    "score": 0.8,
                    "graph_score": 0.7,
                    "distance": 1,
                    "entry_point": False,
                    "connections": [{"from": "introduction.md", "type": "links_to", "distance": 1}],
                    "tags": ["concept", "important"]
                }
            ]
        }
        
        # Format context
        context = self.graph_retriever.format_context_for_llm(query_result)
        
        # Check context format
        self.assertIn("CONTEXT:", context)
        self.assertIn("[1] Document: \"Introduction\"", context)
        self.assertIn("[2] Document: \"Core Concepts\"", context)
        self.assertIn("(Tags: ", context)
        
        # Check for connection information
        self.assertIn("[Connection] Documents 1 and 2 share tags", context)


class TestGraphRAGIntegration(unittest.TestCase):
    """Test suite for the GraphRAGIntegration"""
    
    def setUp(self):
        """Set up mocks for the RAG pipeline integration tests"""
        # Mock RAGPipeline
        self.mock_rag_pipeline = MagicMock()
        self.mock_rag_pipeline.config = {
            "vault": {"path": "/mock/vault/path"}
        }
        self.mock_rag_pipeline.embed = MagicMock(return_value=np.array([[0.1, 0.2, 0.3]]))
        self.mock_rag_pipeline.index = MagicMock()
        self.mock_rag_pipeline.index.search = MagicMock(return_value=(
            np.array([[0.9, 0.8]]),  # Scores
            np.array([[0, 1]])       # Indices
        ))
        self.mock_rag_pipeline.document_store = {
            "0": {"content": "Doc 0 content", "metadata": {"title": "Doc 0"}},
            "1": {"content": "Doc 1 content", "metadata": {"title": "Doc 1"}}
        }
        
        # Patch the EnhancedGraphRetriever class before creating GraphRAGIntegration
        self.patcher = patch('backend.graph_rag_integration.EnhancedGraphRetriever')
        self.mock_graph_retriever_class = self.patcher.start()
        self.mock_graph_retriever = self.mock_graph_retriever_class.return_value
        
        # Mock graph building and query
        self.mock_graph_retriever.build_graph = MagicMock()
        self.mock_graph_retriever.query = MagicMock(return_value={
            "results": [
                {
                    "id": "doc1.md",
                    "content": "Doc 1 content",
                    "metadata": {"title": "Doc 1"},
                    "score": 0.9
                }
            ]
        })
        self.mock_graph_retriever.format_context_for_llm = MagicMock(return_value="Formatted context")
        
        # Create integration
        self.graph_rag = GraphRAGIntegration(self.mock_rag_pipeline)
    
    def tearDown(self):
        """Stop the patcher"""
        self.patcher.stop()
    
    def test_initialization(self):
        """Test initialization and patching"""
        # Check that graph retriever was created
        self.mock_graph_retriever_class.assert_called_once()
        
        # Check that build_graph was called
        self.mock_graph_retriever.build_graph.assert_called_once()
        
        # Check that the RAG pipeline was patched
        self.assertTrue(hasattr(self.mock_rag_pipeline, '_original_query'))
        self.assertEqual(self.mock_rag_pipeline.query, self.graph_rag.enhanced_query)
    
    def test_vector_mode_query(self):
        """Test query in vector-only mode"""
        # Set up original query method
        self.mock_rag_pipeline._original_query = MagicMock(return_value={"results": ["original_result"]})
        
        # Call with vector mode
        result = self.graph_rag.enhanced_query(
            "test query",
            retrieval_mode="vector"
        )
        
        # Check that original method was called
        self.mock_rag_pipeline._original_query.assert_called_once()
        self.assertEqual(result["results"], ["original_result"])
    
    def test_hybrid_mode_query(self):
        """Test query in hybrid mode"""
        # Set up original query method in case of fallback
        self.mock_rag_pipeline._original_query = MagicMock(return_value={"results": ["original_result"]})
        
        # Ensure the query method is not throwing exceptions
        self.mock_graph_retriever.query = MagicMock(return_value={
            "results": [
                {
                    "id": "doc1.md",
                    "content": "Doc 1 content",
                    "metadata": {"title": "Doc 1"},
                    "score": 0.9
                }
            ]
        })
        
        # Call with hybrid mode
        result = self.graph_rag.enhanced_query(
            "test query",
            retrieval_mode="hybrid"
        )
        
        # Check that vector search was performed
        self.mock_rag_pipeline.embed.assert_called()
        self.mock_rag_pipeline.index.search.assert_called()
        
        # Check that graph retriever was called with vector results
        self.mock_graph_retriever.query.assert_called_once()
        
        # Check that query is first arg and vector_results is a kwarg
        call_args = self.mock_graph_retriever.query.call_args
        self.assertTrue(call_args is not None, "query method was not called")
        
        if call_args and len(call_args[0]) > 0:
            self.assertEqual(call_args[0][0], "test query")
        
        if call_args and len(call_args[1]) > 0:
            self.assertIn("vector_results", call_args[1])
        
        # Check that context was formatted
        self.mock_graph_retriever.format_context_for_llm.assert_called_once()
        
        # Check results
        self.assertEqual(result["context"], "Formatted context")
        self.assertTrue(result["graph_retrieval"])
        self.assertEqual(result["mode"], "hybrid")
    
    def test_graph_mode_query(self):
        """Test query in graph-only mode"""
        # Set up original query method in case of fallback
        self.mock_rag_pipeline._original_query = MagicMock(return_value={"results": ["original_result"]})
        
        # Ensure the query method is not throwing exceptions
        self.mock_graph_retriever.query = MagicMock(return_value={
            "results": [
                {
                    "id": "doc1.md",
                    "content": "Doc 1 content",
                    "metadata": {"title": "Doc 1"},
                    "score": 0.9
                }
            ]
        })
        
        # Call with graph mode
        result = self.graph_rag.enhanced_query(
            "test query",
            retrieval_mode="graph"
        )
        
        # Check that vector search was NOT performed
        self.mock_rag_pipeline.index.search.assert_not_called()
        
        # Graph retriever should be called without vector results
        self.mock_graph_retriever.query.assert_called_once()
        
        # Check call arguments
        call_args = self.mock_graph_retriever.query.call_args
        self.assertTrue(call_args is not None, "query method was not called")
        
        # Check vector_results in kwargs
        if call_args and len(call_args[1]) > 0:
            vector_results = call_args[1].get("vector_results")
            self.assertIsNone(vector_results)
        
        # Check results
        self.assertEqual(result["context"], "Formatted context")
        self.assertTrue(result["graph_retrieval"])
        self.assertEqual(result["mode"], "graph")
    
    def test_fallback_on_error(self):
        """Test fallback to original query on error"""
        # Make graph retriever raise an exception
        self.mock_graph_retriever.query = MagicMock(side_effect=Exception("Test error"))
        
        # Set up original query method
        self.mock_rag_pipeline._original_query = MagicMock(return_value={"results": ["original_result"]})
        
        # Call query
        result = self.graph_rag.enhanced_query("test query")
        
        # Should fall back to original method
        self.mock_rag_pipeline._original_query.assert_called_once()
        self.assertEqual(result["results"], ["original_result"])


if __name__ == '__main__':
    unittest.main() 