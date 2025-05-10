from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np
import faiss
import json
import time
from datetime import datetime, timedelta

from .enhanced_graph_retriever import EnhancedGraphRetriever
from .rag_pipeline import RAGPipeline

class GraphRAGIntegration:
    """
    Integration class that connects the EnhancedGraphRetriever with the RAGPipeline.
    This allows the RAG pipeline to use graph-based retrieval while maintaining
    backward compatibility with vector-based retrieval.
    """
    
    def __init__(
        self, 
        rag_pipeline: RAGPipeline,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Graph RAG integration
        
        Args:
            rag_pipeline: Existing RAGPipeline instance
            config: Configuration for graph retrieval (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.rag_pipeline = rag_pipeline
        
        # Default configuration
        self.config = {
            "enabled": True,
            "default_mode": "hybrid",  # "vector", "graph", or "hybrid"
            "max_results": 10,
            "graph_retriever": {
                "entry_points": {
                    "vector_entry_count": 3,
                    "tag_entry_enabled": True,
                    "entry_weight_vector": 0.7,
                    "entry_weight_tags": 0.3
                },
                "traversal": {
                    "max_hops": 2,
                    "max_documents": 15,
                    "tag_expansion_enabled": True,
                    "path_expansion_enabled": True,
                    "min_similarity": 0.5
                },
                "hybrid": {
                    "enabled": True,
                    "graph_weight": 0.6,
                    "vector_weight": 0.4,
                    "recency_bonus": 0.1
                }
            }
        }
        
        # Update with user configuration if provided
        if config:
            self._update_config(config)
            
        # Initialize the enhanced graph retriever
        # Note: EnhancedGraphRetriever is imported at module level for patch testing
        vault_path = rag_pipeline.config.get("vault", {}).get("path", "")
        self.graph_retriever = EnhancedGraphRetriever(vault_path=vault_path, config=self.config["graph_retriever"])
        
        # Build the graph
        self.logger.info("Building enhanced knowledge graph...")
        self.graph_retriever.build_graph()
        self.logger.info("Enhanced knowledge graph built successfully")
        
        # Patch the RAGPipeline query method to use graph retrieval
        self._patch_rag_pipeline()
        
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration with user provided values"""
        for section, section_config in config.items():
            if section in self.config:
                if isinstance(self.config[section], dict) and isinstance(section_config, dict):
                    # Recursive update for nested dictionaries
                    self._recursive_update(self.config[section], section_config)
                else:
                    # Direct update for non-dict values
                    self.config[section] = section_config
    
    def _recursive_update(self, target: Dict, source: Dict) -> None:
        """Recursively update nested dictionaries"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._recursive_update(target[key], value)
            else:
                target[key] = value
    
    def _patch_rag_pipeline(self) -> None:
        """
        Patch the RAGPipeline.query method to use graph-based retrieval.
        This preserves the existing method while adding graph functionality.
        """
        # Store the original query method
        self.rag_pipeline._original_query = self.rag_pipeline.query
        
        # Replace with our enhanced query method
        self.rag_pipeline.query = self.enhanced_query
        
        self.logger.info("RAGPipeline query method patched to use graph retrieval")
    
    def enhanced_query(
        self,
        query: str,
        k: int = 5,
        session_id: Optional[str] = None,
        conversation_memory = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        retrieval_mode: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced query method that supports graph-based retrieval
        
        Args:
            query: User query
            k: Number of results to return (for vector search)
            session_id: Conversation session ID
            conversation_memory: ConversationMemory instance
            tags: List of tags to filter results
            start_date: Start date for filtering
            end_date: End date for filtering
            retrieval_mode: "vector", "graph", or "hybrid" (overrides default_mode)
            **kwargs: Additional arguments for backward compatibility
            
        Returns:
            Query result dictionary
        """
        # Use default mode if not specified
        mode = retrieval_mode or self.config["default_mode"]
        
        # If graph retrieval is disabled or mode is "vector", use original method
        if not self.config["enabled"] or mode == "vector":
            return self.rag_pipeline._original_query(
                query=query,
                k=k,
                session_id=session_id,
                conversation_memory=conversation_memory,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
        
        try:
            start_time = time.time()
            
            # Use instance conversation memory if none provided
            conversation_memory = conversation_memory or getattr(self.rag_pipeline, 'conversation_memory', None)
            
            # Generate session ID if none provided
            if not session_id:
                session_id = str(int(time.time()))
            
            # Get conversation context
            conversation_context = ""
            if conversation_memory and session_id:
                conversation_context = conversation_memory.get_context_window(session_id)
                self.logger.info(f"Session [{session_id}] Context window retrieved")
            
            # Get vector search results if in hybrid mode
            vector_results = None
            if mode == "hybrid":
                self.logger.info(f"Session [{session_id}] Performing vector search for: '{query}'")
                query_vector = self.rag_pipeline.embed([query])[0]
                D, I = self.rag_pipeline.index.search(query_vector.reshape(1, -1), k)
                
                # Format vector results
                vector_results = []
                for i, idx in enumerate(I[0]):
                    if idx == -1 or str(idx) not in self.rag_pipeline.document_store:
                        continue
                    doc = self.rag_pipeline.document_store[str(idx)]
                    vector_results.append({
                        'id': str(idx),
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'score': float(D[0][i])
                    })
            
            # Perform graph-based retrieval
            self.logger.info(f"Session [{session_id}] Performing graph retrieval for: '{query}'")
            
            # Create embedding function closure for the graph retriever
            def embedding_function(texts):
                return self.rag_pipeline.embed(texts)
            
            graph_results = self.graph_retriever.query(
                query=query,
                vector_results=vector_results,
                embedding_function=embedding_function,
                max_results=self.config["max_results"]
            )
            
            # Format results for the LLM
            context = self.graph_retriever.format_context_for_llm(graph_results)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Session [{session_id}] Graph retrieval completed in {elapsed:.2f}s")
            
            # Return results
            return {
                "query": query,
                "results": graph_results.get("results", []),
                "context": context,
                "conversation_context": conversation_context,
                "graph_retrieval": True,
                "mode": mode
            }
            
        except Exception as e:
            self.logger.error(f"Graph retrieval failed: {str(e)}")
            # Fall back to original query method
            self.logger.info("Falling back to vector retrieval")
            return self.rag_pipeline._original_query(
                query=query,
                k=k,
                session_id=session_id,
                conversation_memory=conversation_memory,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Dictionary with graph statistics
        """
        graph = self.graph_retriever.graph
        
        # Count node types
        doc_nodes = sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "document")
        tag_nodes = sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "tag")
        
        # Count edge types
        link_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get("type") == "links_to")
        backlink_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get("type") == "linked_from")
        tag_edges = sum(1 for _, _, data in graph.edges(data=True) if data.get("type") in ["has_tag", "tagged_document"])
        
        return {
            "nodes": {
                "total": graph.number_of_nodes(),
                "documents": doc_nodes,
                "tags": tag_nodes,
                "other": graph.number_of_nodes() - doc_nodes - tag_nodes
            },
            "edges": {
                "total": graph.number_of_edges(),
                "links": link_edges,
                "backlinks": backlink_edges,
                "tags": tag_edges,
                "other": graph.number_of_edges() - link_edges - backlink_edges - tag_edges
            }
        } 