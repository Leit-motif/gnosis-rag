from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import re
import frontmatter
import time

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_core.documents import Document

class ObsidianDocument:
    """Represents a processed Obsidian document with metadata"""
    def __init__(
        self,
        page_content: str,
        metadata: Dict[str, Any],
        source: str,
        chunk_id: Optional[int] = None
    ):
        self.page_content = page_content
        self.metadata = metadata
        self.source = source
        self.chunk_id = chunk_id

    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format"""
        return Document(
            page_content=self.page_content,
            metadata={
                **self.metadata,
                "source": self.source,
                "chunk_id": self.chunk_id
            }
        )

class ObsidianLoaderV2:
    """Enhanced Obsidian vault loader with change detection."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).resolve()
        self.logger = logging.getLogger(__name__)
        self.documents: List[ObsidianDocument] = []
        
        # Initialize text splitters with smaller chunk sizes for testing
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=500,  # Smaller chunk size
            chunk_overlap=50  # Smaller overlap
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunk size
            chunk_overlap=50,  # Smaller overlap
            length_function=len,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )
    
    def extract_metadata(self, file_path: Path, content: frontmatter.Post) -> Dict[str, Any]:
        """Extract metadata from an Obsidian markdown file"""
        # Get frontmatter metadata
        metadata = dict(content.metadata)
        
        # Extract tags from content and metadata
        tags = set()
        if 'tags' in metadata:
            if isinstance(metadata['tags'], list):
                tags.update(t.strip() for t in metadata['tags'])
            elif isinstance(metadata['tags'], str):
                tags.update(t.strip() for t in metadata['tags'].split(","))
        
        # Extract hashtags from content - filter out code blocks and URLs
        content_text = content.content
        # Remove code blocks
        content_no_code = re.sub(r'`[^`]*`', '', content_text)
        # Remove URLs
        content_filtered = re.sub(r'https?://\S+', '', content_no_code)
        
        # Extract tags
        hashtag_pattern = r'(?<!\w)#([a-zA-Z0-9_/-]+)'
        content_tags = re.findall(hashtag_pattern, content_filtered)
        tags.update(t.strip() for t in content_tags)
        
        # Deduplicate tags case-insensitively
        tags_lower_dict = {}
        for tag in tags:
            tag_lower = tag.lower()
            # If we already have this tag in a different case, prefer the frontmatter version
            # or the first one we encountered
            if tag_lower not in tags_lower_dict:
                tags_lower_dict[tag_lower] = tag
        
        # Convert back to a list of unique tags
        unique_tags = list(tags_lower_dict.values())
        
        # Extract links (deduplicated)
        internal_link_pattern = r'\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]'
        markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        # Extract internal links
        internal_links = re.findall(internal_link_pattern, content.content)
        
        # Deduplicate internal links case-insensitively
        links_lower_dict = {}
        for link in internal_links:
            link_stripped = link.strip()
            link_lower = link_stripped.lower()
            # Only add if not already added in a different case
            if link_lower not in links_lower_dict:
                links_lower_dict[link_lower] = link_stripped
        
        # Extract markdown links
        markdown_links = re.findall(markdown_link_pattern, content.content)
        
        # Clean up internal links that are duplicated in tags
        internal_links_deduped = []
        for link in links_lower_dict.values():
            # Only add if not already a tag (case-insensitive comparison)
            if link.lower() not in tags_lower_dict:
                internal_links_deduped.append(link)
        
        # Build metadata dict with deduplicated lists
        return {
            'title': metadata.get('title', file_path.stem),
            'tags': unique_tags,
            'created': metadata.get('created', None),
            'modified': metadata.get('modified', None),
            'internal_links': internal_links_deduped,  # Deduplicated internal links
            'markdown_links': [link[1] for link in markdown_links],
            'link_titles': [link[0] for link in markdown_links],
            'path': str(file_path.relative_to(self.vault_path)),
            **{k: v for k, v in metadata.items() if k not in ['title', 'tags', 'created', 'modified']}
        }
    
    def process_file(self, file_path: Path) -> List[ObsidianDocument]:
        """Process a single markdown file into ObsidianDocuments"""
        try:
            self.logger.debug(f"Processing file: {file_path}")
            
            # Read file content with frontmatter
            content = frontmatter.load(file_path)
            
            # Extract metadata
            metadata = self.extract_metadata(file_path, content)
            
            # Extract date from filename if present
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", file_path.name)
            if date_match:
                date_str = date_match.group(0)
                metadata["date"] = date_str
                metadata["year"] = date_str[:4]
                metadata["month"] = date_str[5:7]
                metadata["day"] = date_str[8:10]
            
            # We'll use the existing tags from extract_metadata and not re-extract them
            # The tags are already deduplicated and processed in the extract_metadata method
            
            # Split content into chunks using markdown splitter
            try:
                chunks = self.markdown_splitter.split_text(content.content)
            except Exception as e:
                self.logger.warning(f"Markdown splitter failed, falling back to recursive splitter: {e}")
                chunks = self.text_splitter.split_text(content.content)
            
            # Create ObsidianDocuments
            documents = []
            for i, chunk in enumerate(chunks):
                doc = ObsidianDocument(
                    page_content=chunk,
                    metadata=metadata,
                    source=Path(file_path).relative_to(self.vault_path).as_posix(),
                    chunk_id=i
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def load_changed_documents(
        self, last_indexed_state: Dict[str, float]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Loads only the documents that have been added, modified, or deleted
        since the last indexing run.

        Args:
            last_indexed_state: A dictionary mapping file paths (as strings) to
                                their last modification timestamps.

        Returns:
            A dictionary with three keys: 'added', 'updated', and 'deleted'.
            - 'added' and 'updated' contain lists of document dictionaries.
            - 'deleted' contains a list of source file paths to remove.
        """
        self.logger.info(f"Scanning for changes in vault: {self.vault_path}")
        current_file_state = {
            str(p.relative_to(self.vault_path)): p.stat().st_mtime
            for p in self.vault_path.rglob("*.md")
        }

        added_files = [
            p for p, mtime in current_file_state.items() if p not in last_indexed_state
        ]
        updated_files = [
            p
            for p, mtime in current_file_state.items()
            if p in last_indexed_state and mtime > last_indexed_state[p]
        ]
        deleted_files = [
            p for p in last_indexed_state if p not in current_file_state
        ]

        docs_to_add = self._process_files(added_files, "Adding new file")
        docs_to_update = self._process_files(updated_files, "Updating modified file")

        if docs_to_add or docs_to_update or deleted_files:
            self.logger.info(
                f"Found {len(docs_to_add)} new, {len(docs_to_update)} updated, "
                f"and {len(deleted_files)} deleted documents."
            )

        return {
            "added": docs_to_add,
            "updated": docs_to_update,
            "deleted": deleted_files,
        }

    def _process_files(self, file_paths: List[str], log_message: str) -> List[Dict[str, Any]]:
        """Helper to process a list of files and return document dicts."""
        processed_docs = []
        for file_path_str in file_paths:
            file_path = self.vault_path / file_path_str
            self.logger.info(f"{log_message}: {file_path_str}")
            try:
                # Using UnstructuredMarkdownLoader as it's efficient
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()
                # We expect one document per file from UnstructuredMarkdownLoader
                if docs:
                    doc_content = docs[0]
                    # Add file modification time to metadata for future checks
                    doc_content.metadata["last_modified"] = file_path.stat().st_mtime
                    processed_docs.append(doc_content)
            except Exception as e:
                self.logger.error(
                    f"Error processing file {file_path_str}: {e}", exc_info=True
                )
        # Convert to a list of dicts for serialization
        return [self._document_to_dict(doc) for doc in processed_docs]

    def _document_to_dict(self, doc: Document) -> Dict[str, Any]:
        """Converts a LangChain Document to a dictionary."""
        return {"page_content": doc.page_content, "metadata": doc.metadata}

    # Keep the original load_vault for full re-indexing if needed
    def load_vault(self, config: Dict[str, Any]) -> List[Document]:
        """Load all markdown files from the vault (for full re-index)."""
        try:
            self.logger.info(f"Performing full load of vault from {self.vault_path}")
            
            exclude_folders = set(config["vault"]["exclude_folders"])
            
            all_documents = []
            for file_path in self.vault_path.rglob("*.md"):
                if any(parent.name in exclude_folders for parent in file_path.parents):
                    continue
                
                try:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                    docs = loader.load()
                    if docs:
                        # Add modification time for the new state tracking
                        docs[0].metadata["last_modified"] = file_path.stat().st_mtime
                        all_documents.extend(docs)
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}", exc_info=True)

            self.logger.info(f"Loaded {len(all_documents)} documents from vault for full re-index.")
            self.documents = all_documents
            
            # Convert to LangChain Documents
            return [doc.to_langchain_document() for doc in all_documents]
            
        except Exception as e:
            self.logger.error(f"Error loading vault: {str(e)}")
            raise
    
    def get_documents(self) -> List[Document]:
        """Return all processed documents in LangChain format"""
        return [doc.to_langchain_document() for doc in self.documents] 