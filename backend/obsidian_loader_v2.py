from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import re
import frontmatter

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
    """Enhanced Obsidian vault loader using LangChain components"""
    
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
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
        tags = set(metadata.get('tags', []))
        hashtag_pattern = r'#([^\s#]+)'
        content_tags = re.findall(hashtag_pattern, content.content)
        tags.update(content_tags)
        
        # Extract links
        internal_link_pattern = r'\[\[(.*?)\]\]'
        markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        internal_links = re.findall(internal_link_pattern, content.content)
        markdown_links = re.findall(markdown_link_pattern, content.content)
        
        # Build metadata dict
        return {
            'title': metadata.get('title', file_path.stem),
            'tags': list(tags),
            'created': metadata.get('created', None),
            'modified': metadata.get('modified', None),
            'internal_links': internal_links,
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
            
            # Extract tags from content (both frontmatter and inline)
            tags = set()
            
            # Get tags from frontmatter
            if "tags" in content.metadata:
                if isinstance(content.metadata["tags"], list):
                    tags.update(content.metadata["tags"])
                elif isinstance(content.metadata["tags"], str):
                    tags.update(tag.strip() for tag in content.metadata["tags"].split(","))
            
            # Get inline tags
            inline_tags = re.findall(r'\[\[(.*?)\]\]', content.content)
            tags.update(tag.strip() for tag in inline_tags)
            
            # Add tags to metadata
            metadata["tags"] = list(tags)
            
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
    
    def load_vault(self, config: Dict[str, Any]) -> List[Document]:
        """Load all markdown files from the vault"""
        try:
            self.logger.info(f"Loading vault from {self.vault_path}")
            
            # Get excluded folders and allowed extensions
            exclude_folders = set(config["vault"]["exclude_folders"])
            allowed_extensions = set(config["vault"]["file_extensions"])
            
            # Process all files
            all_documents = []
            for file_path in self.vault_path.rglob("*"):
                # Skip excluded folders
                if any(parent.name in exclude_folders for parent in file_path.parents):
                    continue
                    
                # Check file extension
                if not any(file_path.name.endswith(ext) for ext in allowed_extensions):
                    continue
                
                # Process file
                documents = self.process_file(file_path)
                all_documents.extend(documents)
            
            self.logger.info(f"Loaded {len(all_documents)} documents from vault")
            self.documents = all_documents
            
            # Convert to LangChain Documents
            return [doc.to_langchain_document() for doc in all_documents]
            
        except Exception as e:
            self.logger.error(f"Error loading vault: {str(e)}")
            raise
    
    def get_documents(self) -> List[Document]:
        """Return all processed documents in LangChain format"""
        return [doc.to_langchain_document() for doc in self.documents] 