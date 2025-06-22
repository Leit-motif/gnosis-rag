# Database Setup Guide

This guide explains how to set up PostgreSQL with pgvector for the Gnosis RAG API.

## Prerequisites

1. PostgreSQL 12+ installed
2. pgvector extension installed
3. Python dependencies installed (`pip install -r requirements-api.txt`)

## Environment Variables

Set the following environment variables in your `.env` file:

```bash
# Database Configuration (Required)
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/gnosis_rag

# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# API Configuration
PORT=8080
DEBUG=false

# Rate Limiting
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_PERIOD=60

# Graph Configuration
GRAPH_MAX_HOPS=2
GRAPH_MAX_DOCUMENTS=10
GRAPH_WEIGHT_VECTOR=0.6
GRAPH_WEIGHT_GRAPH=0.4

# Vector Store Configuration
EMBEDDING_DIMENSION=1536
SIMILARITY_THRESHOLD=0.7

# Security (Optional)
BEARER_TOKEN=your_optional_bearer_token

# Logging
LOG_LEVEL=INFO
```

## Database URL Examples

### Local PostgreSQL
```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/gnosis_rag
```

### Supabase
```bash
DATABASE_URL=postgresql+asyncpg://user:password@host.supabase.co:5432/postgres
```

### Railway
```bash
DATABASE_URL=postgresql+asyncpg://user:password@containers-us-west-xx.railway.app:5432/railway
```

### Render
```bash
DATABASE_URL=postgresql+asyncpg://user:password@dpg-xxx.oregon-postgres.render.com:5432/database_name
```

## Local Setup Instructions

### 1. Install PostgreSQL and pgvector

#### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-14-pgvector  # Adjust version as needed
```

#### On macOS with Homebrew:
```bash
brew install postgresql
brew install pgvector
```

#### On Windows:
1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Install pgvector from https://github.com/pgvector/pgvector

### 2. Create Database and User

```sql
-- Connect to PostgreSQL as superuser
sudo -u postgres psql

-- Create database
CREATE DATABASE gnosis_rag;

-- Create user (optional)
CREATE USER gnosis_user WITH PASSWORD 'your_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE gnosis_rag TO gnosis_user;

-- Connect to the database
\c gnosis_rag

-- Create pgvector extension
CREATE EXTENSION vector;

-- Verify extension
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Run Database Migrations

```bash
# Initialize Alembic (only needed once)
alembic init alembic

# Generate and run migrations
alembic upgrade head
```

### 4. Verify Setup

Start the API server and check the health endpoint:

```bash
# Start the server
cd api && python -m uvicorn main:app --host 0.0.0.0 --port 8080

# Check health (should show database: "connected")
curl http://localhost:8080/health
```

## Database Schema

### Documents Table
- Stores document content and metadata
- Includes file information (path, name, size, type)
- Supports JSON metadata field

### Embeddings Table
- Stores vector embeddings using pgvector
- Supports chunked documents
- Includes HNSW index for fast similarity search

### Graph Edges Table
- Stores relationships between documents
- Supports weighted edges with metadata
- Enables graph-based retrieval

### Conversations Table
- Stores chat history and responses
- Includes citations and metadata
- Supports thread-based conversations

## Performance Considerations

### Vector Index Configuration

The migration creates an HNSW index with these parameters:
- `m = 16`: Number of connections per layer
- `ef_construction = 64`: Search width during construction

You can adjust these based on your data size and performance requirements:

```sql
-- For larger datasets, consider higher values
CREATE INDEX idx_embeddings_vector_cosine ON embeddings 
USING hnsw (embedding_vector vector_cosine_ops) 
WITH (m = 32, ef_construction = 128);
```

### Connection Pool Settings

The connection pool is configured for containerized environments:
- Pool size: 10 connections
- Max overflow: 20 additional connections
- Connection recycling: Every hour
- Pre-ping: Enabled for connection validation

Adjust these in `api/database/connection.py` based on your load requirements.

## Troubleshooting

### Common Issues

1. **pgvector extension not found**
   ```
   ERROR: extension "vector" is not available
   ```
   Solution: Install pgvector extension for your PostgreSQL version

2. **Connection refused**
   ```
   ERROR: could not connect to server
   ```
   Solution: Check DATABASE_URL and ensure PostgreSQL is running

3. **Permission denied**
   ```
   ERROR: permission denied for schema public
   ```
   Solution: Grant proper privileges to your database user

4. **Migration fails**
   ```
   ERROR: relation already exists
   ```
   Solution: Check if tables already exist, run `alembic downgrade base` if needed

### Checking Database Status

```sql
-- Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check tables
\dt

-- Check indexes
\di

-- Check vector index details
SELECT schemaname, tablename, indexname, indexdef 
FROM pg_indexes 
WHERE indexdef LIKE '%vector%';
```

## Production Deployment

### Cloud Database Services

Most cloud PostgreSQL services support pgvector:
- **Supabase**: Built-in pgvector support
- **AWS RDS**: Install from source or use Aurora
- **Google Cloud SQL**: Install from source
- **Azure Database**: Install from source
- **Railway**: pgvector available
- **Render**: pgvector available

### Docker Deployment

Use the provided Dockerfile which includes all necessary dependencies.

### Environment Security

- Never commit `.env` files with real credentials
- Use secrets management in production
- Enable SSL/TLS for database connections
- Restrict database access by IP when possible 