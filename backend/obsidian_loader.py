import os
from typing import List, Dict, Any, Optional, Set, Generator
from datetime import datetime
import re
import frontmatter
from pathlib import Path
import yaml
import logging

class ObsidianNote:
    def __init__(
        self,
        title: str,
        content: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        created: Optional[datetime] = None,
        modified: Optional[datetime] = None
    ):
        self.title = title
        self.content = content
        self.path = path
        self.metadata = metadata or {}
        self.created = created
        self.modified = modified
        
        # Extract features
        self.tags = self._extract_tags()
        self.links = self._extract_links()
        self.chunks = []  # Will be populated during processing
        
    def _extract_tags(self) -> Set[str]:
        """Extract both frontmatter and inline tags"""
        tags = set()
        
        # Frontmatter tags
        if "tags" in self.metadata:
            if isinstance(self.metadata["tags"], list):
                tags.update(self.metadata["tags"])
            elif isinstance(self.metadata["tags"], str):
                tags.add(self.metadata["tags"])
                
        # Inline tags
        inline_tags = re.findall(r'#([a-zA-Z0-9_-]+)', self.content)
        tags.update(inline_tags)
        
        return tags
        
    def _extract_links(self) -> Set[str]:
        """Extract both wiki-style and markdown links"""
        links = set()
        
        # Wiki-style links [[Page Name]]
        wiki_links = re.findall(r'\[\[([^\]]+)\]\]', self.content)
        links.update(wiki_links)
        
        # Markdown links [Text](link)
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', self.content)
        links.update(link for _, link in md_links)
        
        return links

class ObsidianLoader:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.logger = logging.getLogger(__name__)
        self.notes: Dict[str, ObsidianNote] = {}
        self.tags: Set[str] = set()
        self.link_graph: Dict[str, Set[str]] = {}
        self.documents = []
        
    def load_vault(self, config: Dict[str, Any]) -> None:
        """Load all markdown files from the vault"""
        try:
            self.logger.info(f"Loading vault from {self.vault_path}")
            
            # Get excluded folders and allowed extensions
            exclude_folders = set(config["vault"]["exclude_folders"])
            allowed_extensions = set(config["vault"]["file_extensions"])
            
            # Walk through the vault
            for root, dirs, files in os.walk(self.vault_path):
                # Remove excluded folders
                dirs[:] = [d for d in dirs if d not in exclude_folders]
                
                for file in files:
                    if any(file.endswith(ext) for ext in allowed_extensions):
                        file_path = Path(root) / file
                        try:
                            self.process_file(file_path, config)
                        except Exception as e:
                            self.logger.error(f"Error processing file {file_path}: {str(e)}")
                            
            self.logger.info(f"Loaded {len(self.documents)} documents from vault")
            
        except Exception as e:
            self.logger.error(f"Error loading vault: {str(e)}")
            raise
            
    def process_file(self, file_path: Path, config: Dict[str, Any]) -> None:
        """Process a single markdown file"""
        try:
            relative_path = file_path.relative_to(self.vault_path)
            self.logger.debug(f"Processing file: {relative_path}")
            
            # Read file content
            content = frontmatter.load(file_path)
            
            # Extract metadata
            metadata = {
                'title': content.get('title', file_path.stem),
                'tags': content.get('tags', []),
                'created': content.get('created', None),
                'path': str(relative_path),
            }
            
            # Process content into chunks
            chunks = self.chunk_content(
                content.content,
                config["chunking"]["chunk_size"],
                config["chunking"]["chunk_overlap"],
                config["chunking"]["split_by"]
            )
            
            # Store document chunks
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    'id': f"{relative_path}#{i}",
                    'content': chunk,
                    'metadata': metadata
                })
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
            
    def chunk_content(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int,
        split_by: str
    ) -> List[str]:
        """Split content into overlapping chunks"""
        if split_by == "paragraph":
            # Split by double newline to preserve paragraph structure
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            chunks = []
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para.split())
                if current_size + para_size > chunk_size and current_chunk:
                    # Store current chunk
                    chunks.append(' '.join(current_chunk))
                    # Keep last part for overlap
                    overlap_size = 0
                    overlap_chunk = []
                    for p in reversed(current_chunk):
                        p_size = len(p.split())
                        if overlap_size + p_size <= chunk_overlap:
                            overlap_chunk.insert(0, p)
                            overlap_size += p_size
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_size = overlap_size
                
                current_chunk.append(para)
                current_size += para_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
        else:  # split_by == "sentence"
            # Simple sentence splitting - can be improved
            sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence.split())
                if current_size + sentence_size > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Calculate overlap
                    overlap_size = 0
                    overlap_chunk = []
                    for s in reversed(current_chunk):
                        s_size = len(s.split())
                        if overlap_size + s_size <= chunk_overlap:
                            overlap_chunk.insert(0, s)
                            overlap_size += s_size
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_size = overlap_size
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
        
    def get_documents(self) -> List[Dict[str, Any]]:
        """Return all processed documents"""
        return self.documents
        
    def _load_note(self, file_path: Path) -> Optional[ObsidianNote]:
        """Load and parse a single note"""
        try:
            # Read frontmatter and content
            post = frontmatter.load(file_path)
            
            # Get file metadata
            stats = file_path.stat()
            created = datetime.fromtimestamp(stats.st_ctime)
            modified = datetime.fromtimestamp(stats.st_mtime)
            
            # Create note object
            note = ObsidianNote(
                title=file_path.stem,
                content=post.content,
                path=str(file_path.relative_to(self.vault_path)),
                metadata=post.metadata,
                created=created,
                modified=modified
            )
            
            return note
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
            
    def _build_backlinks(self) -> None:
        """Build bidirectional link relationships"""
        backlinks: Dict[str, Set[str]] = {}
        
        # Collect backlinks
        for source, links in self.link_graph.items():
            for target in links:
                if target not in backlinks:
                    backlinks[target] = set()
                backlinks[target].add(source)
                
        # Update notes with backlinks
        for path, note in self.notes.items():
            if path in backlinks:
                note.metadata["backlinks"] = list(backlinks[path])
                
    def get_notes_by_tag(self, tag: str) -> List[ObsidianNote]:
        """Get all notes with a specific tag"""
        return [
            note for note in self.notes.values()
            if tag in note.tags
        ]
        
    def get_notes_in_date_range(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None
    ) -> List[ObsidianNote]:
        """Get notes created within a date range"""
        if end_date is None:
            end_date = datetime.now()
            
        return [
            note for note in self.notes.values()
            if start_date <= note.created <= end_date
        ]
        
    def get_connected_notes(self, note_path: str, depth: int = 1) -> Set[str]:
        """Get notes connected via links up to a certain depth"""
        if depth < 1:
            return set()
            
        connected = set()
        to_visit = {note_path}
        
        for _ in range(depth):
            next_level = set()
            for path in to_visit:
                if path in self.link_graph:
                    next_level.update(self.link_graph[path])
            connected.update(next_level)
            to_visit = next_level
            
        return connected 