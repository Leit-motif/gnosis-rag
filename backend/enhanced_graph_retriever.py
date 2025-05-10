from typing import List, Dict, Any, Optional, Tuple, Set, Union
import networkx as nx
from pathlib import Path
import json
import logging
from datetime import datetime
import re
import yaml
import numpy as np
from collections import defaultdict

class EnhancedGraphRetriever:
    """
    Enhanced Graph Retriever for True Graph RAG implementation.
    This class extends the basic graph retrieval capabilities to support
    sophisticated graph traversal strategies, hybrid retrieval, and
    structured context output for LLMs.
    """
    
    def __init__(
        self, 
        vault_path: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the enhanced graph retriever"""
        self.vault_path = Path(vault_path)
        self.graph = nx.DiGraph()  # Use directed graph to distinguish between links and backlinks
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
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
        
        # Update with user configuration if provided
        if config:
            self._update_config(config)
            
        # Cache for document content and metadata
        self.document_cache = {}
            
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update the configuration with user-provided values"""
        for section, section_config in config.items():
            if section in self.config:
                self.config[section].update(section_config)
                
    def build_graph(self) -> None:
        """
        Build an enhanced graph from the Obsidian vault's markdown files.
        Includes links, backlinks, tags, and metadata.
        """
        try:
            # Load all markdown files
            markdown_files = list(self.vault_path.rglob("*.md"))
            self.logger.info(f"Found {len(markdown_files)} markdown files")
            
            # Map to normalize filenames to handle case sensitivity and extension variations
            # This will map both lowercase filename and filename without extension to the actual path
            filename_map = {}
            for file_path in markdown_files:
                relative_path = str(file_path.relative_to(self.vault_path))
                filename = file_path.name.lower()
                stem = file_path.stem.lower()
                filename_map[filename] = relative_path
                filename_map[stem] = relative_path
            
            # First pass: add nodes and extract links/tags
            for file_path in markdown_files:
                relative_path = str(file_path.relative_to(self.vault_path))
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract metadata
                metadata = self._extract_metadata(content, relative_path, file_path)
                
                # Add node with metadata
                self.graph.add_node(
                    relative_path, 
                    type="document",
                    title=metadata.get("title", relative_path),
                    created=metadata.get("created"),
                    modified=metadata.get("modified"),
                    tags=metadata.get("tags", []),
                    frontmatter=metadata.get("frontmatter", {}),
                )
                
                # Cache document content and metadata
                self.document_cache[relative_path] = {
                    "content": content,
                    "metadata": metadata
                }
                
                # Use deduplicated links from metadata
                raw_links = metadata.get("links", [])
                # Normalize links to handle relative paths, case sensitivity, etc.
                normalized_links = []
                for link in raw_links:
                    # Try to find the actual file path for this link
                    link_lower = link.lower()
                    # Try with and without .md extension
                    link_candidates = [
                        link_lower,
                        link_lower + ".md",
                        link_lower.replace(".md", "")
                    ]
                    
                    matched_link = None
                    for candidate in link_candidates:
                        if candidate in filename_map:
                            matched_link = filename_map[candidate]
                            break
                    
                    if matched_link:
                        normalized_links.append(matched_link)
                
                metadata["links"] = normalized_links
                
                # Add tag nodes and connections
                for tag in metadata.get("tags", []):
                    tag_node = f"tag:{tag}"
                    if not self.graph.has_node(tag_node):
                        self.graph.add_node(tag_node, type="tag")
                    self.graph.add_edge(relative_path, tag_node, type="has_tag")
                    self.graph.add_edge(tag_node, relative_path, type="tagged_document")
            
            # Second pass: add link edges and backlinks
            for doc_path, data in self.document_cache.items():
                metadata = data["metadata"]
                for link in metadata.get("links", []):
                    # Add link edge
                    if self.graph.has_node(link):  # Only add if target exists
                        self.graph.add_edge(doc_path, link, type="links_to")
                        self.graph.add_edge(link, doc_path, type="linked_from")
            
            self.logger.info(f"Built enhanced graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            self.logger.error(f"Failed to build enhanced graph: {str(e)}")
            raise
    
    def _extract_metadata(self, content: str, relative_path: str, file_path: Path) -> Dict[str, Any]:
        """Extract essential metadata from a document"""
        # Extract frontmatter
        frontmatter = self._extract_frontmatter(content)
        
        # Extract title (from frontmatter or filename)
        title = frontmatter.get("title", file_path.stem)
        
        # Extract tags from frontmatter and content
        tags_from_frontmatter = frontmatter.get("tags", [])
        if isinstance(tags_from_frontmatter, str):
            tags_from_frontmatter = [tags_from_frontmatter]
        
        # Deduplicate frontmatter tags
        frontmatter_tags_set = set(t.strip() for t in tags_from_frontmatter)
        
        # Extract tags from content and deduplicate
        tags_from_content = self._extract_tags(content)
        content_tags_set = set(t.strip() for t in tags_from_content)
        
        # Combine all tags
        all_tags_raw = list(frontmatter_tags_set.union(content_tags_set))
        
        # Deduplicate tags case-insensitively
        tags_lower_dict = {}
        for tag in all_tags_raw:
            tag_lower = tag.lower()
            # If we already have this tag in a different case, prefer the frontmatter version
            # or the first one we encountered
            if tag_lower not in tags_lower_dict:
                tags_lower_dict[tag_lower] = tag
                
        # Convert back to a list of unique tags
        all_tags = list(tags_lower_dict.values())
        
        # Extract links and deduplicate
        raw_links = set(self._extract_links(content))  # Use set to deduplicate
        
        # Deduplicate - ensure links don't duplicate tags
        deduped_links = []
        for link in raw_links:
            if link.lower() not in tags_lower_dict:
                deduped_links.append(link)
        
        # Get timestamps only if recency-based features are enabled
        created = None
        modified = None
        if self.config["hybrid"]["recency_bonus"] > 0:
            created = frontmatter.get("created")
            if not created:
                try:
                    created = datetime.fromtimestamp(file_path.stat().st_ctime)
                except:
                    created = None
                    
            modified = frontmatter.get("modified")
            if not modified:
                try:
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                except:
                    modified = None
        
        # Extract only essential frontmatter fields rather than storing the whole object
        essential_frontmatter = {}
        for key in frontmatter:
            # Skip fields we already have dedicated metadata for
            if key not in ['title', 'tags', 'created', 'modified']:
                essential_frontmatter[key] = frontmatter[key]
        
        return {
            "title": title,
            "tags": all_tags,
            "links": deduped_links,
            "path": relative_path,
            # Only include timestamps if they exist
            **({"created": created} if created else {}),
            **({"modified": modified} if modified else {}),
            # Only include essential frontmatter fields
            **({"frontmatter": essential_frontmatter} if essential_frontmatter else {})
        }
            
    def _extract_links(self, content: str) -> List[str]:
        """Extract Obsidian-style links from content"""
        # Match [[link]] or [[link|alias]] and extract just the link part
        pattern = r'\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]'
        matches = re.findall(pattern, content)
        
        # Deduplicate links case-insensitively
        links_lower_dict = {}
        for link in matches:
            link_stripped = link.strip()
            link_lower = link_stripped.lower()
            # If we already have this link in a different case, prefer the first one we encountered
            if link_lower not in links_lower_dict:
                links_lower_dict[link_lower] = link_stripped
                
        # Return deduplicated links
        return list(links_lower_dict.values())
        
    def _extract_tags(self, content: str) -> List[str]:
        """Extract Obsidian-style tags from content"""
        # Skip tags inside code blocks or URLs
        # First, remove code blocks
        content_no_code = re.sub(r'`[^`]*`', '', content)
        
        # Remove URLs
        content_filtered = re.sub(r'https?://\S+', '', content_no_code)
        
        # Now match tags
        pattern = r'(?<!\w)#([a-zA-Z0-9_/-]+)'
        raw_tags = re.findall(pattern, content_filtered)
        
        # Deduplicate tags case-insensitively
        tags_lower_dict = {}
        for tag in raw_tags:
            tag_stripped = tag.strip()
            tag_lower = tag_stripped.lower()
            # If we already have this tag in a different case, prefer the first one we encountered
            if tag_lower not in tags_lower_dict:
                tags_lower_dict[tag_lower] = tag_stripped
                
        # Return deduplicated tags
        return list(tags_lower_dict.values())
        
    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from content"""
        pattern = r'^---\n(.*?)\n---\n'
        match = re.match(pattern, content, re.DOTALL)
        
        if match:
            try:
                return yaml.safe_load(match.group(1))
            except:
                return {}
        return {}
    
    def find_entry_points(
        self, 
        query: str,
        vector_results: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find entry points into the graph based on the query.
        Can use vector search results and/or tag/keyword matching.
        
        Args:
            query: The user query
            vector_results: Optional vector search results to use as entry points
            top_k: Number of entry points to return
            
        Returns:
            List of entry point documents with metadata
        """
        entry_points = []
        
        # Use vector results if provided
        if vector_results and self.config["entry_points"]["vector_entry_count"] > 0:
            for result in vector_results[:self.config["entry_points"]["vector_entry_count"]]:
                doc_id = result.get("id") or result.get("document")
                if doc_id and self.graph.has_node(doc_id):
                    entry_points.append({
                        "document": doc_id,
                        "similarity": result.get("score", 0.8),  # Default if not provided
                        "entry_type": "vector",
                        "weight": self.config["entry_points"]["entry_weight_vector"]
                    })
        
        # Use tag/keyword matching if enabled
        if self.config["entry_points"]["tag_entry_enabled"]:
            # Extract potential tags from query
            potential_tags = self._extract_tags(query)
            
            # Find documents with matching tags
            for tag in potential_tags:
                tag_node = f"tag:{tag}"
                if self.graph.has_node(tag_node):
                    # Get documents with this tag
                    for edge_data in self.graph.in_edges(tag_node, data=True):
                        # Correctly unpack - NetworkX returns (source, target, data_dict)
                        doc_node, _, edge_attrs = edge_data
                        if edge_attrs.get("type") == "has_tag" and self.graph.nodes[doc_node]["type"] == "document":
                            entry_points.append({
                                "document": doc_node,
                                "similarity": 0.9,  # High similarity for exact tag matches
                                "entry_type": "tag",
                                "tag": tag,
                                "weight": self.config["entry_points"]["entry_weight_tags"]
                            })
            
            # TODO: Add keyword matching for title/content
            
        # Deduplicate and sort by weighted similarity
        unique_entries = {}
        for entry in entry_points:
            doc = entry["document"]
            if doc not in unique_entries or entry["similarity"] * entry["weight"] > unique_entries[doc]["similarity"] * unique_entries[doc]["weight"]:
                unique_entries[doc] = entry
        
        # Convert to list and sort
        sorted_entries = sorted(
            unique_entries.values(), 
            key=lambda x: x["similarity"] * x["weight"], 
            reverse=True
        )
        
        return sorted_entries[:top_k]
    
    def expand_graph(
        self, 
        entry_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Expand from entry points using multiple traversal strategies.
        
        Args:
            entry_points: List of entry point documents
            
        Returns:
            List of related documents with scores and relationship metadata
        """
        max_hops = self.config["traversal"]["max_hops"]
        max_documents = self.config["traversal"]["max_documents"]
        
        # Track discovered documents and their scores
        discovered_docs = {}
        
        # Extract just the document IDs from entry points
        entry_doc_ids = [entry["document"] for entry in entry_points]
        
        # 1. K-hop Neighborhood Expansion
        for entry in entry_points:
            doc_id = entry["document"]
            entry_score = entry["similarity"] * entry["weight"]
            
            # Include the entry point itself
            if doc_id not in discovered_docs:
                discovered_docs[doc_id] = {
                    "document": doc_id,
                    "score": entry_score,
                    "distance": 0,
                    "entry_point": True,
                    "connections": [],
                    "tags": self.graph.nodes[doc_id].get("tags", [])
                }
            
            # Get k-hop neighborhood
            self._expand_neighborhood(doc_id, entry_score, 1, max_hops, discovered_docs)
        
        # 2. Tag/Cluster Expansion (if enabled)
        if self.config["traversal"]["tag_expansion_enabled"]:
            self._expand_by_tags(discovered_docs, entry_doc_ids)
            
        # 3. Path-based Expansion (if enabled and multiple entry points)
        if self.config["traversal"]["path_expansion_enabled"] and len(entry_points) > 1:
            self._expand_by_paths(discovered_docs, entry_doc_ids)
            
        # Convert to list and sort by score
        result_list = list(discovered_docs.values())
        result_list.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to max_documents
        return result_list[:max_documents]
    
    def _expand_neighborhood(
        self, 
        doc_id: str, 
        entry_score: float,
        current_distance: int,
        max_distance: int,
        discovered_docs: Dict[str, Dict[str, Any]],
        visited: set = None
    ) -> None:
        """Recursively expand the neighborhood around a document"""
        # Initialize visited set if not provided (first call)
        if visited is None:
            visited = set()
            
        # Stop if we've reached max distance or already visited this node
        if current_distance > max_distance or doc_id in visited:
            return
            
        # Mark this node as visited
        visited.add(doc_id)
            
        # Get outgoing links - NetworkX returns (source, target, data_dict)
        for source, target, edge_attrs in self.graph.out_edges(doc_id, data=True):
            # Skip tag nodes unless tag expansion is explicitly enabled
            if self.graph.nodes[target].get("type") == "tag" and not self.config["traversal"]["tag_expansion_enabled"]:
                continue
                
            # Calculate score based on distance and connection type
            connection_score = entry_score * (1.0 / current_distance)
            
            # Adjust score based on connection type
            if edge_attrs.get("type") == "links_to":
                connection_score *= 1.0  # Direct link (full weight)
            elif edge_attrs.get("type") == "linked_from":
                connection_score *= 0.8  # Backlink (slightly lower weight)
            elif edge_attrs.get("type") == "has_tag":
                connection_score *= 0.7  # Tag connection (slightly lower than backlink)
                # Skip further processing for tag nodes
                continue
            else:
                connection_score *= 0.5  # Other connection types
                
            # Add or update discovered document
            if target not in discovered_docs:
                # Check that target is a document node (not a tag or other type)
                if self.graph.nodes[target].get("type") != "document":
                    continue
                    
                discovered_docs[target] = {
                    "document": target,
                    "score": connection_score,
                    "distance": current_distance,
                    "entry_point": False,
                    "connections": [{
                        "from": doc_id,
                        "type": edge_attrs.get("type", "unknown"),
                        "distance": current_distance
                    }],
                    "tags": self.graph.nodes[target].get("tags", [])
                }
            else:
                # Update score if higher
                if connection_score > discovered_docs[target]["score"]:
                    discovered_docs[target]["score"] = connection_score
                    discovered_docs[target]["distance"] = min(current_distance, discovered_docs[target]["distance"])
                
                # Add connection if from a different source
                connection = {
                    "from": doc_id,
                    "type": edge_attrs.get("type", "unknown"),
                    "distance": current_distance
                }
                
                # Check if this specific connection already exists
                if not any(c["from"] == connection["from"] and c["type"] == connection["type"] 
                          for c in discovered_docs[target]["connections"]):
                    discovered_docs[target]["connections"].append(connection)
                
            # Continue expanding from this node
            self._expand_neighborhood(target, connection_score, current_distance + 1, max_distance, discovered_docs, visited)
    
    def _expand_by_tags(
        self, 
        discovered_docs: Dict[str, Dict[str, Any]],
        entry_doc_ids: List[str]
    ) -> None:
        """Expand the discovered documents using shared tags"""
        # Collect tags from all discovered documents
        all_tags = set()
        for doc_data in discovered_docs.values():
            all_tags.update(doc_data.get("tags", []))
            
        # Find documents with these tags that aren't already discovered
        for tag in all_tags:
            tag_node = f"tag:{tag}"
            if not self.graph.has_node(tag_node):
                continue
                
            # Find documents with this tag
            for edge_data in self.graph.in_edges(tag_node, data=True):
                # Correctly unpack - NetworkX returns (source, target, data_dict)
                doc_node, _, edge_attrs = edge_data
                
                if (edge_attrs.get("type") == "has_tag" and 
                    self.graph.nodes[doc_node]["type"] == "document" and
                    doc_node not in discovered_docs):
                    
                    # Calculate score based on tag importance
                    # (Tags from entry points get higher scores)
                    tag_score = 0.5  # Base score for tag connection
                    
                    # Boost score if it's an entry point tag
                    for entry_id in entry_doc_ids:
                        if entry_id in discovered_docs and tag in discovered_docs[entry_id].get("tags", []):
                            tag_score = 0.7
                            break
                    
                    # Add to discovered documents
                    discovered_docs[doc_node] = {
                        "document": doc_node,
                        "score": tag_score,
                        "distance": 1,  # Consider tag connections as distance 1
                        "entry_point": False,
                        "connections": [{
                            "from": "tag:" + tag,
                            "type": "shared_tag",
                            "tag": tag,
                            "distance": 1
                        }],
                        "tags": self.graph.nodes[doc_node].get("tags", [])
                    }
    
    def _expand_by_paths(
        self, 
        discovered_docs: Dict[str, Dict[str, Any]],
        entry_doc_ids: List[str]
    ) -> None:
        """Find paths between entry points and add intermediate nodes"""
        # Find paths between all pairs of entry points
        for i, source in enumerate(entry_doc_ids):
            for target in entry_doc_ids[i+1:]:
                try:
                    # Find shortest path between source and target
                    path = nx.shortest_path(self.graph, source, target)
                    
                    # Add intermediate nodes to discovered docs
                    for j, node in enumerate(path[1:-1], 1):  # Skip source and target
                        if self.graph.nodes[node].get("type") != "document":
                            continue
                            
                        path_position = j / len(path)  # Relative position in path
                        path_score = 0.6 * (1 - path_position)  # Higher score for nodes closer to start
                        
                        if node not in discovered_docs:
                            discovered_docs[node] = {
                                "document": node,
                                "score": path_score,
                                "distance": j,
                                "entry_point": False,
                                "connections": [{
                                    "from": path[j-1],
                                    "to": path[j+1],
                                    "type": "path",
                                    "path": f"{source} → ... → {target}",
                                    "distance": j
                                }],
                                "tags": self.graph.nodes[node].get("tags", [])
                            }
                        else:
                            # Update score if higher
                            if path_score > discovered_docs[node]["score"]:
                                discovered_docs[node]["score"] = path_score
                            
                            # Add path connection
                            discovered_docs[node]["connections"].append({
                                "from": path[j-1],
                                "to": path[j+1],
                                "type": "path",
                                "path": f"{source} → ... → {target}",
                                "distance": j
                            })
                except nx.NetworkXNoPath:
                    # No path exists between these entry points
                    continue
    
    def hybrid_rerank(
        self, 
        documents: List[Dict[str, Any]],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        embedding_function = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid reranking of graph search results with vector similarity
        
        Args:
            documents: List of documents from graph retrieval
            query: The user query
            query_embedding: Optional pre-computed query embedding
            embedding_function: Function to compute embeddings if needed
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
            
        # Get document IDs and contents for embedding
        doc_ids = []
        doc_contents = []
        
        for doc in documents:
            doc_id = doc["document"]
            doc_ids.append(doc_id)
            
            content = ""
            if doc_id in self.document_cache:
                content = self.document_cache[doc_id]["content"]
            doc_contents.append(content)
        
        # Get embeddings if needed
        if embedding_function:
            # Compute document embeddings
            doc_embeddings = embedding_function(doc_contents)
            
            # Get query embedding if not provided
            if query_embedding is None:
                query_embedding = embedding_function([query])[0]
                
            # Compute vector similarities
            vector_similarities = []
            for doc_emb in doc_embeddings:
                # Cosine similarity
                similarity = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                vector_similarities.append(float(similarity))
        else:
            # Default to 0 vector similarity if no embedding function
            vector_similarities = [0] * len(documents)
        
        # Rerank using combined score
        graph_weight = self.config["hybrid"]["graph_weight"]
        vector_weight = self.config["hybrid"]["vector_weight"]
        recency_bonus = self.config["hybrid"]["recency_bonus"]
        
        for i, doc in enumerate(documents):
            # Combine graph score with vector similarity
            doc_id = doc["document"]
            graph_score = doc["score"]
            vector_score = vector_similarities[i] if i < len(vector_similarities) else 0
            
            # Add recency bonus if available
            recency_score = 0
            if recency_bonus > 0 and doc_id in self.document_cache:
                metadata = self.document_cache[doc_id].get("metadata", {})
                modified = metadata.get("modified")
                if modified and isinstance(modified, datetime):
                    # Calculate days since modification (max 30 days)
                    days_old = min(30, (datetime.now() - modified).days)
                    recency_score = recency_bonus * (1 - days_old/30)
            
            # Calculate combined score
            combined_score = (
                graph_score * graph_weight +
                vector_score * vector_weight +
                recency_score
            )
            
            # Update score
            doc["combined_score"] = combined_score
            doc["vector_similarity"] = vector_score
        
        # Sort by combined score
        documents.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return documents
    
    def query(
        self,
        query: str,
        vector_results: Optional[List[Dict[str, Any]]] = None,
        embedding_function = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Query the graph for relevant documents
        
        Args:
            query: User query
            vector_results: Optional vector search results to use as entry points
            embedding_function: Function to compute embeddings (for hybrid reranking)
            max_results: Maximum number of results to return
            
        Returns:
            Dict with results and metadata
        """
        try:
            # 1. Find entry points
            entry_points = self.find_entry_points(query, vector_results)
            
            # 2. Expand graph from entry points
            graph_results = self.expand_graph(entry_points)
            
            # 3. Hybrid reranking (if enabled and embedding function provided)
            if self.config["hybrid"]["enabled"] and embedding_function:
                query_embedding = embedding_function([query])[0]
                reranked_results = self.hybrid_rerank(
                    graph_results,
                    query,
                    query_embedding,
                    embedding_function
                )
            else:
                reranked_results = graph_results
                
            # 4. Limit results
            final_results = reranked_results[:max_results]
            
            # 5. Format results with content
            formatted_results = []
            for result in final_results:
                doc_id = result["document"]
                doc_data = {
                    "id": doc_id,
                    "score": result.get("combined_score", result["score"]),
                    "graph_score": result["score"],
                    "distance": result["distance"],
                    "entry_point": result["entry_point"],
                    "connections": result["connections"],
                    "tags": result["tags"]
                }
                
                # Add content if available
                if doc_id in self.document_cache:
                    doc_data["content"] = self.document_cache[doc_id]["content"]
                    doc_data["metadata"] = self.document_cache[doc_id]["metadata"]
                else:
                    # Try to read the file
                    try:
                        with open(self.vault_path / doc_id, 'r', encoding='utf-8') as f:
                            content = f.read()
                            doc_data["content"] = content
                            # Extract basic metadata
                            doc_data["metadata"] = self._extract_metadata(content, doc_id, self.vault_path / doc_id)
                    except:
                        doc_data["content"] = ""
                        doc_data["metadata"] = {}
                
                formatted_results.append(doc_data)
            
            # Return results with metadata
            return {
                "results": formatted_results,
                "entry_points": entry_points,
                "total_candidates": len(graph_results),
                "query": query,
                "config": self.config
            }
            
        except Exception as e:
            self.logger.error(f"Graph query failed: {str(e)}")
            return {"results": [], "error": str(e)}
    
    def format_context_for_llm(self, query_result: Dict[str, Any]) -> str:
        """
        Format the query results into a concise, information-rich context string for the LLM
        
        Args:
            query_result: Result from the query method
            
        Returns:
            Formatted context string with only essential metadata
        """
        results = query_result.get("results", [])
        if not results:
            return "No relevant documents found."
            
        context_parts = ["CONTEXT:"]
        
        # Helper to get document title
        def get_title(doc_id):
            for result in results:
                if result["id"] == doc_id:
                    return result["metadata"].get("title", doc_id)
            return doc_id
        
        # Add each document with essential metadata only
        for i, doc in enumerate(results, 1):
            # Document header with only critical metadata
            doc_header = f"[{i}] Document: \"{doc['metadata'].get('title', doc['id'])}\""
            
            # Add tags if present and not empty
            if doc.get("tags") and len(doc["tags"]) > 0:
                tag_str = ", ".join(doc["tags"][:3])  # Limit to 3 tags for brevity
                doc_header += f" (Tags: {tag_str})"
                
            # Only include connection info if it adds valuable context
            if not doc["entry_point"] and doc["connections"]:
                conn = doc["connections"][0]  # Use the first connection
                if conn["type"] == "links_to":
                    doc_header += f" (Linked from: {get_title(conn['from'])})"
                elif conn["type"] == "linked_from":
                    doc_header += f" (Links to: {get_title(conn['from'])})"
            
            # Add document content
            context_parts.append(doc_header)
            context_parts.append(f"Content: {doc['content']}\n")
        
        # Add only the most relevant connections between documents
        # Focus on direct links as they're more valuable than shared tags
        added_connections = set()
        direct_links_found = False
        
        # First try to find direct links
        for i, doc1 in enumerate(results):
            for j, doc2 in enumerate(results):
                if i >= j:
                    continue  # Avoid duplicates and self-connections
                
                # Check for direct links in the graph
                doc1_id = doc1["id"]
                doc2_id = doc2["id"]
                
                # Only add if there's a direct link
                if self.graph.has_edge(doc1_id, doc2_id) or self.graph.has_edge(doc2_id, doc1_id):
                    connection_key = f"{doc1_id}-{doc2_id}-link"
                    if connection_key not in added_connections:
                        direction = "→" if self.graph.has_edge(doc1_id, doc2_id) else "←"
                        context_parts.append(
                            f"[Connection] Document {i+1} {direction} Document {j+1}"
                        )
                        added_connections.add(connection_key)
                        direct_links_found = True
        
        # Only add tag connections if we didn't find direct links
        if not direct_links_found:
            for i, doc1 in enumerate(results):
                for j, doc2 in enumerate(results):
                    if i >= j:
                        continue  # Avoid duplicates and self-connections
                        
                    # Check for shared tags (but limit to most relevant)
                    doc1_tags = set(doc1.get("tags", []))
                    doc2_tags = set(doc2.get("tags", []))
                    common_tags = doc1_tags.intersection(doc2_tags)
                    
                    if common_tags:
                        # Only add up to 3 tag connections total and only if they share important tags
                        connection_key = f"{doc1['id']}-{doc2['id']}-tags"
                        if len(added_connections) < 3 and connection_key not in added_connections:
                            # Sort by tag name to get consistent results
                            tag_list = sorted(list(common_tags))
                            tag_str = ", ".join(f'"{tag}"' for tag in tag_list[:2])  # Show at most 2 tags
                            context_parts.append(
                                f"[Connection] Documents {i+1} and {j+1} share tags: {tag_str}"
                            )
                            added_connections.add(connection_key)
        
        return "\n\n".join(context_parts) 