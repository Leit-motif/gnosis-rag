"""SQLAlchemy database models for Gnosis RAG API."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey, JSON, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Document(Base):
    """Document model for storing processed document content and metadata."""
    
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    title = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    file_path = Column(String, nullable=True)
    file_name = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    content_type = Column(String, nullable=True)
    doc_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    embeddings = relationship("Embedding", back_populates="document", cascade="all, delete-orphan")
    source_edges = relationship("GraphEdge", foreign_keys="GraphEdge.source_id", back_populates="source_document")
    target_edges = relationship("GraphEdge", foreign_keys="GraphEdge.target_id", back_populates="target_document")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_file_name', 'file_name'),
        Index('idx_documents_content_type', 'content_type'),
    )
    
    def __repr__(self):
        return f"<Document(id='{self.id}', title='{self.title}', created_at='{self.created_at}')>"


class Embedding(Base):
    """Embedding model for storing document vector embeddings."""
    
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    embedding_vector = Column(Vector(1536), nullable=False)  # OpenAI text-embedding-3-small dimension
    model_name = Column(String, nullable=False, default="text-embedding-3-small")
    chunk_index = Column(Integer, nullable=True, default=0)  # For document chunking
    chunk_text = Column(Text, nullable=True)  # Store the text that was embedded
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="embeddings")
    
    # Indexes for performance - pgvector specific
    __table_args__ = (
        Index('idx_embeddings_document_id', 'document_id'),
        Index('idx_embeddings_model_name', 'model_name'),
        # Vector similarity indexes - will be created via migration
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, document_id='{self.document_id}', model='{self.model_name}')>"


class GraphEdge(Base):
    """Graph edge model for storing relationships between documents."""
    
    __tablename__ = "graph_edges"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String, nullable=False, index=True)
    weight = Column(Float, nullable=False, default=1.0)
    doc_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    source_document = relationship("Document", foreign_keys=[source_id], back_populates="source_edges")
    target_document = relationship("Document", foreign_keys=[target_id], back_populates="target_edges")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_graph_edges_source_id', 'source_id'),
        Index('idx_graph_edges_target_id', 'target_id'),
        Index('idx_graph_edges_relationship_type', 'relationship_type'),
        Index('idx_graph_edges_weight', 'weight'),
        Index('idx_graph_edges_source_target', 'source_id', 'target_id'),  # Composite index
    )
    
    def __repr__(self):
        return f"<GraphEdge(id={self.id}, source='{self.source_id}', target='{self.target_id}', type='{self.relationship_type}')>"


# Optional: Conversation history model for chat functionality
class Conversation(Base):
    """Conversation model for storing chat history."""
    
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    thread_id = Column(String, nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True, default=list)
    doc_metadata = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_conversations_thread_id', 'thread_id'),
        Index('idx_conversations_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Conversation(id='{self.id}', thread_id='{self.thread_id}', created_at='{self.created_at}')>" 