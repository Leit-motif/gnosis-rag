"""Initial migration: Create pgvector extension and base tables

Revision ID: 001
Revises: 
Create Date: 2024-12-22 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create pgvector extension and all database tables."""
    
    # Create pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=True),
        sa.Column('file_name', sa.String(), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('content_type', sa.String(), nullable=True),
        sa.Column('doc_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for documents table
    op.create_index('idx_documents_created_at', 'documents', ['created_at'])
    op.create_index('idx_documents_file_name', 'documents', ['file_name'])
    op.create_index('idx_documents_content_type', 'documents', ['content_type'])
    op.create_index('ix_documents_title', 'documents', ['title'])
    
    # Create embeddings table
    op.create_table(
        'embeddings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('embedding_vector', Vector(1536), nullable=False),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=True),
        sa.Column('chunk_text', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for embeddings table
    op.create_index('idx_embeddings_document_id', 'embeddings', ['document_id'])
    op.create_index('idx_embeddings_model_name', 'embeddings', ['model_name'])
    
    # Create HNSW index for vector similarity search
    op.execute(
        "CREATE INDEX idx_embeddings_vector_cosine ON embeddings "
        "USING hnsw (embedding_vector vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    
    # Create graph_edges table
    op.create_table('graph_edges',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('source_id', sa.String(), nullable=False),
        sa.Column('target_id', sa.String(), nullable=False),
        sa.Column('relationship_type', sa.String(), nullable=False),
        sa.Column('weight', sa.Float(), nullable=False),
        sa.Column('doc_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['source_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for graph_edges table
    op.create_index('idx_graph_edges_source_id', 'graph_edges', ['source_id'])
    op.create_index('idx_graph_edges_target_id', 'graph_edges', ['target_id'])
    op.create_index('idx_graph_edges_relationship_type', 'graph_edges', ['relationship_type'])
    op.create_index('idx_graph_edges_weight', 'graph_edges', ['weight'])
    op.create_index('idx_graph_edges_source_target', 'graph_edges', ['source_id', 'target_id'])
    op.create_index('ix_graph_edges_relationship_type', 'graph_edges', ['relationship_type'])
    
    # Create conversations table
    op.create_table('conversations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('thread_id', sa.String(), nullable=False),
        sa.Column('user_query', sa.Text(), nullable=False),
        sa.Column('assistant_response', sa.Text(), nullable=False),
        sa.Column('citations', sa.JSON(), nullable=True),
        sa.Column('doc_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for conversations table
    op.create_index('idx_conversations_thread_id', 'conversations', ['thread_id'])
    op.create_index('idx_conversations_created_at', 'conversations', ['created_at'])


def downgrade() -> None:
    """Drop all tables and pgvector extension."""
    
    # Drop tables in reverse order
    op.drop_table('conversations')
    op.drop_table('graph_edges')
    op.drop_table('embeddings')
    op.drop_table('documents')
    
    # Drop pgvector extension
    op.execute("DROP EXTENSION IF EXISTS vector") 