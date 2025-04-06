import os
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import re
import frontmatter
from pathlib import Path
import yaml

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
        self.notes: Dict[str, ObsidianNote] = {}
        self.tags: Set[str] = set()
        self.link_graph: Dict[str, Set[str]] = {}
        
    def load_vault(self, config: Dict[str, Any]) -> None:
        """Load all notes from the vault"""
        exclude_folders = set(config["vault"]["exclude_folders"])
        allowed_extensions = set(config["vault"]["file_extensions"])
        
        for file_path in self.vault_path.rglob("*"):
            # Skip excluded folders
            if any(parent.name in exclude_folders for parent in file_path.parents):
                continue
                
            # Check file extension
            if file_path.suffix not in allowed_extensions:
                continue
                
            try:
                note = self._load_note(file_path)
                if note:
                    self.notes[note.path] = note
                    self.tags.update(note.tags)
                    
                    # Update link graph
                    self.link_graph[note.path] = note.links
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        # Build bidirectional link graph
        self._build_backlinks()
        
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