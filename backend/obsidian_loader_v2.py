from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import re
import frontmatter

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_core.documents import Document

from backend.storage.base import VaultStorage


class ObsidianDocument:
    """Represents a processed Obsidian document with metadata"""

    def __init__(
        self,
        page_content: str,
        metadata: Dict[str, Any],
        source: str,
        chunk_id: Optional[int] = None,
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
                "chunk_id": self.chunk_id,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ObsidianDocument to a dictionary."""
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "source": self.source,
            "chunk_id": self.chunk_id,
        }


class ObsidianLoaderV2:
    """Enhanced Obsidian vault loader with change detection."""

    def __init__(self, storage: VaultStorage):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        self.documents: List[ObsidianDocument] = []

        # Initialize text splitters with smaller chunk sizes for testing
        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=500, chunk_overlap=50  # Smaller chunk size  # Smaller overlap
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunk size
            chunk_overlap=50,  # Smaller overlap
            length_function=len,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""],
        )

    def extract_metadata(
        self, file_path_str: str, content: frontmatter.Post, metadata_from_storage: dict
    ) -> Dict[str, Any]:
        """Extract metadata from an Obsidian markdown file"""
        # Get frontmatter metadata
        metadata = dict(content.metadata)

        # Extract tags from content and metadata
        tags = set()
        if "tags" in metadata:
            if isinstance(metadata["tags"], list):
                tags.update(t.strip() for t in metadata["tags"])
            elif isinstance(metadata["tags"], str):
                tags.update(t.strip() for t in metadata["tags"].split(","))

        # Extract hashtags from content - filter out code blocks and URLs
        content_text = content.content
        # Remove code blocks
        content_no_code = re.sub(r"`[^`]*`", "", content_text)
        # Remove URLs
        content_filtered = re.sub(r"https?://\S+", "", content_no_code)

        # Extract tags
        hashtag_pattern = r"(?<!\w)#([a-zA-Z0-9_/-]+)"
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
        internal_link_pattern = r"\[\[([^\]\|]+)(?:\|[^\]]+)?\]\]"
        markdown_link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

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
            "title": metadata.get("title", Path(file_path_str).stem),
            "tags": unique_tags,
            "created": metadata.get("created", None),
            "modified": metadata.get("modified", None),
            "last_modified_time": metadata_from_storage.get("last_modified"),
            "internal_links": internal_links_deduped,  # Deduplicated internal links
            "markdown_links": [link[1] for link in markdown_links],
            "link_titles": [link[0] for link in markdown_links],
            "path": file_path_str,
            **{
                k: v
                for k, v in metadata.items()
                if k not in ["title", "tags", "created", "modified"]
            },
        }

    def process_file(self, file_path_str: str) -> List[ObsidianDocument]:
        """Process a single markdown file into ObsidianDocuments"""
        try:
            self.logger.debug(f"Processing file: {file_path_str}")

            # Read file content with frontmatter
            file_content_str = self.storage.read_file(file_path_str)
            content = frontmatter.loads(file_content_str)
            
            # Get metadata from storage
            metadata_from_storage = self.storage.get_metadata(file_path_str)

            # Extract metadata
            metadata = self.extract_metadata(file_path_str, content, metadata_from_storage)

            # Extract date from filename if present
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", Path(file_path_str).name)
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
                self.logger.warning(
                    f"Markdown splitter failed, falling back to recursive splitter: {e}"
                )
                chunks = self.text_splitter.split_text(content.content)

            # Create ObsidianDocuments
            documents = []
            for i, chunk in enumerate(chunks):
                doc = ObsidianDocument(
                    page_content=chunk,
                    metadata=metadata,
                    source=file_path_str,
                    chunk_id=i,
                )
                documents.append(doc)

            return documents

        except Exception as e:
            self.logger.error(f"Error processing file {file_path_str}: {str(e)}")
            return []

    def load_vault(
        self,
        last_indexed_state: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Loads documents from the Obsidian vault, detecting changes since the
        last indexing run if a state is provided.

        Args:
            last_indexed_state: A dictionary mapping file paths to their last
                                modification timestamps. If None, performs a
                                full re-index by treating all files as new.

        Returns:
            A dictionary with 'added', 'updated', and 'deleted' lists,
            and the 'new_state' of the vault.
            'added' and 'updated' contain ObsidianDocument objects.
            'deleted' contains relative file paths to remove.
        """
        if last_indexed_state is None:
            last_indexed_state = {}

        self.logger.info("Scanning for changes in vault...")

        # Get the current state of all markdown files in the vault
        try:
            all_files = self.storage.list_files(file_extension=".md")
            current_file_state = {
                file_path: self.storage.get_metadata(file_path)["last_modified"]
                for file_path in all_files
            }
        except Exception as e:
            self.logger.error(f"Could not list or get metadata for vault files: {e}")
            return {"added": [], "updated": [], "deleted": [], "new_state": {}}

        # Identify files that are new or have been updated
        added_files = [p for p in current_file_state if p not in last_indexed_state]
        updated_files = [
            p
            for p, mtime in current_file_state.items()
            if p in last_indexed_state and mtime > last_indexed_state[p]
        ]

        # Identify files that have been deleted
        deleted_files = [p for p in last_indexed_state if p not in current_file_state]

        unchanged_file_count = (
            len(current_file_state) - len(added_files) - len(updated_files)
        )

        processed_docs = {"added": [], "updated": []}

        # Process new and updated files
        files_to_process = added_files + updated_files
        if files_to_process:
            self.logger.info(f"Processing {len(files_to_process)} changed files...")
            for file_path_str in files_to_process:
                docs = self.process_file(file_path_str)
                if file_path_str in added_files:
                    processed_docs["added"].extend(docs)
                else:
                    processed_docs["updated"].extend(docs)

        self.logger.info(
            f"Found {len(added_files)} new, {len(updated_files)} updated, "
            f"{len(deleted_files)} deleted, and {unchanged_file_count} unchanged files."
        )

        self.documents = processed_docs["added"] + processed_docs["updated"]

        return {
            "added": processed_docs["added"],
            "updated": processed_docs["updated"],
            "deleted": deleted_files,
            "new_state": current_file_state,
        }

    def get_obsidian_documents(self) -> List[ObsidianDocument]:
        """Returns the loaded documents as ObsidianDocument objects."""
        return self.documents

    def get_documents(self) -> List[Document]:
        """Returns the loaded documents as LangChain Document objects."""
        return [doc.to_langchain_document() for doc in self.documents]

    def load_all_documents(
        self, config: Optional[Dict[str, Any]] = None
    ) -> List[ObsidianDocument]:
        """Loads all documents from the vault without change detection"""
        self.logger.info("Loading all documents from vault...")
        all_docs = []
        try:
            file_paths = self.storage.list_files(file_extension=".md")
            for file_path_str in file_paths:
                processed_file_docs = self.process_file(file_path_str)
                all_docs.extend(processed_file_docs)
        except Exception as e:
            self.logger.error(f"Error loading all documents: {e}")

        self.documents = all_docs
        return self.get_obsidian_documents()
