from typing import List, Dict, Any, Optional
import networkx as nx
from pathlib import Path
import json
import logging
from datetime import datetime

class GraphRetriever:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.graph = nx.Graph()
        self.logger = logging.getLogger(__name__)
        
    def build_graph(self) -> None:
        """Build a graph from the Obsidian vault's markdown files and their connections"""
        try:
            # Load all markdown files
            markdown_files = list(self.vault_path.rglob("*.md"))
            self.logger.info(f"Found {len(markdown_files)} markdown files")
            
            # Add nodes for each file
            for file_path in markdown_files:
                relative_path = file_path.relative_to(self.vault_path)
                self.graph.add_node(str(relative_path))
                
                # Extract links and tags from the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract links ([[link]] format)
                    links = self._extract_links(content)
                    for link in links:
                        # Add edge for each link
                        self.graph.add_edge(str(relative_path), link)
                    
                    # Extract tags (#tag format)
                    tags = self._extract_tags(content)
                    for tag in tags:
                        # Add tag node and connect to file
                        self.graph.add_node(f"tag:{tag}")
                        self.graph.add_edge(str(relative_path), f"tag:{tag}")
            
            self.logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to build graph: {str(e)}")
            raise
            
    def _extract_links(self, content: str) -> List[str]:
        """Extract Obsidian-style links from content"""
        import re
        # Match [[link]] or [[link|alias]]
        pattern = r'\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]'
        matches = re.findall(pattern, content)
        return [match[0] for match in matches]
        
    def _extract_tags(self, content: str) -> List[str]:
        """Extract Obsidian-style tags from content"""
        import re
        # Match #tag or #tag/subtag
        pattern = r'#([a-zA-Z0-9_/-]+)'
        return re.findall(pattern, content)
        
    def get_related_documents(
        self,
        source_doc: str,
        max_distance: int = 2,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get documents related to the source document through the graph"""
        try:
            if source_doc not in self.graph:
                self.logger.warning(f"Source document {source_doc} not found in graph")
                return []
                
            # Get all nodes within max_distance
            related_nodes = nx.single_source_shortest_path_length(
                self.graph, source_doc, cutoff=max_distance
            )
            
            # Convert to list of documents with their distance
            related_docs = []
            for node, distance in related_nodes.items():
                if node.startswith("tag:"):
                    continue  # Skip tag nodes
                    
                # Calculate similarity score (inverse of distance)
                similarity = 1.0 / (distance + 1)
                
                if similarity >= min_similarity:
                    related_docs.append({
                        "document": node,
                        "similarity": similarity,
                        "distance": distance
                    })
            
            # Sort by similarity
            related_docs.sort(key=lambda x: x["similarity"], reverse=True)
            return related_docs
            
        except Exception as e:
            self.logger.error(f"Failed to get related documents: {str(e)}")
            return []
            
    def get_common_tags(self, doc1: str, doc2: str) -> List[str]:
        """Get tags that are common between two documents"""
        try:
            if doc1 not in self.graph or doc2 not in self.graph:
                return []
                
            # Get neighbors of both documents
            doc1_tags = {n for n in self.graph.neighbors(doc1) if n.startswith("tag:")}
            doc2_tags = {n for n in self.graph.neighbors(doc2) if n.startswith("tag:")}
            
            # Get intersection
            common_tags = doc1_tags.intersection(doc2_tags)
            
            # Remove 'tag:' prefix
            return [tag[4:] for tag in common_tags]
            
        except Exception as e:
            self.logger.error(f"Failed to get common tags: {str(e)}")
            return []
            
    def get_document_metadata(self, doc_path: str) -> Dict[str, Any]:
        """Get metadata for a document"""
        try:
            full_path = self.vault_path / doc_path
            if not full_path.exists():
                return {}
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract frontmatter if present
                frontmatter = self._extract_frontmatter(content)
                
                # Get tags
                tags = self._extract_tags(content)
                
                # Get links
                links = self._extract_links(content)
                
                return {
                    "frontmatter": frontmatter,
                    "tags": tags,
                    "links": links,
                    "path": str(doc_path)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get document metadata: {str(e)}")
            return {}
            
    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from content"""
        import yaml
        import re
        
        pattern = r'^---\n(.*?)\n---\n'
        match = re.match(pattern, content, re.DOTALL)
        
        if match:
            try:
                return yaml.safe_load(match.group(1))
            except:
                return {}
        return {} 