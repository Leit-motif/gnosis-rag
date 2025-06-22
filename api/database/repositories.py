"""Repository pattern implementation for database operations."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update, and_, or_, func
from sqlalchemy.orm import selectinload

from .models import Document, Embedding, GraphEdge, Conversation

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session


class DocumentRepository(BaseRepository):
    """Repository for document operations."""
    
    async def create(
        self,
        title: str,
        content: str,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_size: Optional[int] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a new document."""
        document = Document(
            id=str(uuid4()),
            title=title,
            content=content,
            file_path=file_path,
            file_name=file_name,
            file_size=file_size,
            content_type=content_type,
            metadata=metadata or {}
        )
        
        self.session.add(document)
        await self.session.flush()
        logger.info(f"Created document: {document.id}")
        return document
    
    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        stmt = (
            select(Document)
            .options(selectinload(Document.embeddings))
            .where(Document.id == document_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        content_type: Optional[str] = None
    ) -> List[Document]:
        """Get all documents with optional filtering."""
        stmt = select(Document).offset(offset).limit(limit)
        
        if content_type:
            stmt = stmt.where(Document.content_type == content_type)
        
        stmt = stmt.order_by(Document.created_at.desc())
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def update(self, document_id: str, **kwargs) -> Optional[Document]:
        """Update document fields."""
        kwargs['updated_at'] = datetime.utcnow()
        
        stmt = (
            update(Document)
            .where(Document.id == document_id)
            .values(**kwargs)
            .returning(Document)
        )
        result = await self.session.execute(stmt)
        updated_doc = result.scalar_one_or_none()
        
        if updated_doc:
            logger.info(f"Updated document: {document_id}")
        
        return updated_doc
    
    async def delete(self, document_id: str) -> bool:
        """Delete document by ID."""
        stmt = delete(Document).where(Document.id == document_id)
        result = await self.session.execute(stmt)
        
        if result.rowcount > 0:
            logger.info(f"Deleted document: {document_id}")
            return True
        return False
    
    async def search_by_title(self, title_query: str, limit: int = 10) -> List[Document]:
        """Search documents by title."""
        stmt = (
            select(Document)
            .where(Document.title.ilike(f"%{title_query}%"))
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()


class EmbeddingRepository(BaseRepository):
    """Repository for embedding operations with vector similarity search."""
    
    async def create(
        self,
        document_id: str,
        embedding_vector: List[float],
        model_name: str = "text-embedding-3-small",
        chunk_index: int = 0,
        chunk_text: Optional[str] = None
    ) -> Embedding:
        """Create a new embedding."""
        embedding = Embedding(
            document_id=document_id,
            embedding_vector=embedding_vector,
            model_name=model_name,
            chunk_index=chunk_index,
            chunk_text=chunk_text
        )
        
        self.session.add(embedding)
        await self.session.flush()
        logger.info(f"Created embedding for document: {document_id}")
        return embedding
    
    async def get_by_document_id(self, document_id: str) -> List[Embedding]:
        """Get all embeddings for a document."""
        stmt = (
            select(Embedding)
            .where(Embedding.document_id == document_id)
            .order_by(Embedding.chunk_index)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def similarity_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        model_name: Optional[str] = None
    ) -> List[tuple[Embedding, float]]:
        """
        Perform vector similarity search using pgvector operators.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)
            model_name: Filter by embedding model
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        # Convert similarity threshold to distance threshold
        # For cosine similarity: distance = 1 - similarity
        distance_threshold = 1.0 - threshold
        
        stmt = (
            select(
                Embedding,
                (1 - Embedding.embedding_vector.cosine_distance(query_vector)).label('similarity')
            )
            .where(Embedding.embedding_vector.cosine_distance(query_vector) < distance_threshold)
            .order_by(Embedding.embedding_vector.cosine_distance(query_vector))
            .limit(limit)
        )
        
        if model_name:
            stmt = stmt.where(Embedding.model_name == model_name)
        
        result = await self.session.execute(stmt)
        return [(row.Embedding, row.similarity) for row in result]
    
    async def get_document_similarities(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[tuple[Document, float]]:
        """
        Get documents with their best similarity scores.
        
        Returns documents with their highest embedding similarity score.
        """
        # Subquery to get best similarity per document
        subquery = (
            select(
                Embedding.document_id,
                func.max(1 - Embedding.embedding_vector.cosine_distance(query_vector)).label('max_similarity')
            )
            .where(Embedding.embedding_vector.cosine_distance(query_vector) < (1.0 - threshold))
            .group_by(Embedding.document_id)
            .subquery()
        )
        
        # Join with documents to get full document info
        stmt = (
            select(Document, subquery.c.max_similarity)
            .join(subquery, Document.id == subquery.c.document_id)
            .order_by(subquery.c.max_similarity.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        return [(row.Document, row.max_similarity) for row in result]
    
    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all embeddings for a document."""
        stmt = delete(Embedding).where(Embedding.document_id == document_id)
        result = await self.session.execute(stmt)
        logger.info(f"Deleted {result.rowcount} embeddings for document: {document_id}")
        return result.rowcount


class GraphRepository(BaseRepository):
    """Repository for graph operations."""
    
    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphEdge:
        """Create a new graph edge."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            metadata=metadata or {}
        )
        
        self.session.add(edge)
        await self.session.flush()
        logger.info(f"Created edge: {source_id} -> {target_id} ({relationship_type})")
        return edge
    
    async def get_neighbors(
        self,
        document_id: str,
        relationship_types: Optional[List[str]] = None,
        max_hops: int = 2
    ) -> List[Document]:
        """Get neighboring documents through graph traversal."""
        if max_hops == 1:
            # Direct neighbors only
            stmt = (
                select(Document)
                .join(GraphEdge, or_(
                    and_(GraphEdge.source_id == document_id, Document.id == GraphEdge.target_id),
                    and_(GraphEdge.target_id == document_id, Document.id == GraphEdge.source_id)
                ))
            )
            
            if relationship_types:
                stmt = stmt.where(GraphEdge.relationship_type.in_(relationship_types))
            
        else:
            # Multi-hop traversal using recursive CTE
            # This is a simplified version - full implementation would use WITH RECURSIVE
            stmt = (
                select(Document)
                .join(GraphEdge, or_(
                    and_(GraphEdge.source_id == document_id, Document.id == GraphEdge.target_id),
                    and_(GraphEdge.target_id == document_id, Document.id == GraphEdge.source_id)
                ))
            )
            
            if relationship_types:
                stmt = stmt.where(GraphEdge.relationship_type.in_(relationship_types))
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_edges_by_document(
        self,
        document_id: str,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[GraphEdge]:
        """Get edges connected to a document."""
        if direction == "outgoing":
            stmt = select(GraphEdge).where(GraphEdge.source_id == document_id)
        elif direction == "incoming":
            stmt = select(GraphEdge).where(GraphEdge.target_id == document_id)
        else:  # both
            stmt = select(GraphEdge).where(
                or_(
                    GraphEdge.source_id == document_id,
                    GraphEdge.target_id == document_id
                )
            )
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def delete_edges_by_document(self, document_id: str) -> int:
        """Delete all edges connected to a document."""
        stmt = delete(GraphEdge).where(
            or_(
                GraphEdge.source_id == document_id,
                GraphEdge.target_id == document_id
            )
        )
        result = await self.session.execute(stmt)
        logger.info(f"Deleted {result.rowcount} edges for document: {document_id}")
        return result.rowcount


class ConversationRepository(BaseRepository):
    """Repository for conversation operations."""
    
    async def create(
        self,
        thread_id: str,
        user_query: str,
        assistant_response: str,
        citations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation entry."""
        conversation = Conversation(
            id=str(uuid4()),
            thread_id=thread_id,
            user_query=user_query,
            assistant_response=assistant_response,
            citations=citations or [],
            metadata=metadata or {}
        )
        
        self.session.add(conversation)
        await self.session.flush()
        logger.info(f"Created conversation entry for thread: {thread_id}")
        return conversation
    
    async def get_by_thread_id(
        self,
        thread_id: str,
        limit: int = 50
    ) -> List[Conversation]:
        """Get conversation history for a thread."""
        stmt = (
            select(Conversation)
            .where(Conversation.thread_id == thread_id)
            .order_by(Conversation.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all() 