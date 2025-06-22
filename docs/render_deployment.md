# Deploying Gnosis RAG API to Render.io

This guide walks you through deploying the Gnosis RAG API to Render.io with a PostgreSQL database that supports pgvector.

## Prerequisites

1. Render.io account
2. GitHub repository with your code
3. OpenAI API key

## Step 1: Create PostgreSQL Database on Render

1. **Log into Render.io Dashboard**
   - Go to https://dashboard.render.com
   - Click "New +" → "PostgreSQL"

2. **Configure Database**
   - **Name**: `gnosis-rag-db`
   - **Database**: `gnosis_rag`
   - **User**: `gnosis_user` (or leave default)
   - **Region**: Choose your preferred region
   - **Plan**: Free or Starter (depending on your needs)

3. **Enable pgvector Extension**
   After the database is created:
   - Go to the database dashboard
   - Click "Connect" and use the External Database URL
   - Connect via psql or a database client:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Save Database URL**
   - Copy the "Internal Database URL" from the database dashboard
   - It should look like: `postgresql://user:pass@dpg-xxx-a.oregon-postgres.render.com/dbname`

## Step 2: Deploy the API Service

1. **Create Web Service**
   - In Render dashboard, click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository containing this code

2. **Configure Service Settings**
   - **Name**: `gnosis-rag-api`
   - **Region**: Same as your database
   - **Branch**: `main` (or your deployment branch)
   - **Runtime**: `Docker`
   - **Plan**: Free or Starter

3. **Set Environment Variables**
   In the Environment section, add these variables:

   ```bash
   # Database (Required)
   DATABASE_URL=postgresql+asyncpg://user:pass@host/database
   # ^ Use the Internal Database URL from Step 1, but change postgresql:// to postgresql+asyncpg://

   # OpenAI (Required)
   OPENAI_API_KEY=sk-your-openai-api-key-here
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small

   # API Configuration
   PORT=10000
   DEBUG=false
   LOG_LEVEL=INFO

   # Rate Limiting
   RATE_LIMIT_REQUESTS=30
   RATE_LIMIT_PERIOD=60

   # Graph Configuration
   GRAPH_MAX_HOPS=2
   GRAPH_MAX_DOCUMENTS=10
   GRAPH_WEIGHT_VECTOR=0.6
   GRAPH_WEIGHT_GRAPH=0.4

   # Vector Configuration
   EMBEDDING_DIMENSION=1536
   SIMILARITY_THRESHOLD=0.7

   # Security (Optional)
   BEARER_TOKEN=your-optional-bearer-token
   ```

4. **Advanced Settings**
   - **Build Command**: Leave empty (Docker handles this)
   - **Start Command**: Leave empty (Docker handles this)
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: Yes (recommended)

## Step 3: Database Migration

After deployment, you need to run the database migrations:

1. **Using Render Shell**
   - Go to your web service dashboard
   - Click "Shell" tab
   - Run the migration:
   ```bash
   alembic upgrade head
   ```

2. **Alternative: Local Migration**
   If you have access to the database URL locally:
   ```bash
   # Set the database URL
   export DATABASE_URL="postgresql+asyncpg://user:pass@host/database"
   
   # Run migrations
   alembic upgrade head
   ```

## Step 4: Test Deployment

1. **Check Health Endpoint**
   ```bash
   curl https://your-app-name.onrender.com/health
   ```
   
   Should return:
   ```json
   {
     "status": "ok",
     "version": "1.0.0",
     "debug": false,
     "timestamp": "2024-12-22T...",
     "database": "connected"
   }
   ```

2. **Test Upload Endpoint** (placeholder)
   ```bash
   curl -X POST https://your-app-name.onrender.com/upload \
     -H "Content-Type: multipart/form-data" \
     -F "files=@test.md"
   ```

3. **Test Chat Endpoint** (placeholder)
   ```bash
   curl -X POST https://your-app-name.onrender.com/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Hello, how are you?"}'
   ```

## Local Testing (if Docker is available)

If you have Docker installed locally, you can test before deploying:

1. **Start Local Test Environment**
   ```bash
   # Build and start services
   docker compose -f docker-compose.test.yml up --build
   
   # Wait for services to be healthy, then test
   curl http://localhost:8080/health
   ```

2. **Run Migrations Locally**
   ```bash
   # In another terminal
   docker compose -f docker-compose.test.yml exec api alembic upgrade head
   ```

3. **Clean Up**
   ```bash
   docker compose -f docker-compose.test.yml down -v
   ```

## Troubleshooting

### Common Issues

1. **Database Connection Fails**
   ```
   ERROR: could not connect to server
   ```
   - Verify DATABASE_URL is correct
   - Ensure you're using the Internal Database URL (not External)
   - Check that the URL starts with `postgresql+asyncpg://`

2. **pgvector Extension Missing**
   ```
   ERROR: type "vector" does not exist
   ```
   - Connect to your database and run: `CREATE EXTENSION IF NOT EXISTS vector;`

3. **Migration Fails**
   ```
   ERROR: relation already exists
   ```
   - Check if tables were created manually
   - Run: `alembic downgrade base` then `alembic upgrade head`

4. **Health Check Shows Database Disconnected**
   - Check database is running and accessible
   - Verify DATABASE_URL environment variable
   - Check database logs in Render dashboard

### Performance Optimization

1. **Database Plan**: Consider upgrading to a paid plan for better performance
2. **Connection Pooling**: Already configured in the application
3. **Indexes**: The migrations create optimal indexes for vector search

### Security Considerations

1. **Environment Variables**: Never commit real API keys to Git
2. **HTTPS**: Render provides HTTPS by default
3. **Database Access**: Use Internal Database URL for better security
4. **Bearer Token**: Set BEARER_TOKEN for additional API security

## Monitoring and Logs

1. **Application Logs**: Available in Render dashboard under "Logs" tab
2. **Database Logs**: Available in database dashboard
3. **Health Monitoring**: Render monitors the `/health` endpoint automatically
4. **Custom Monitoring**: The API includes structured logging for debugging

## Scaling Considerations

1. **Horizontal Scaling**: Enable auto-scaling in service settings
2. **Database Scaling**: Upgrade database plan as needed
3. **Connection Limits**: Monitor database connection usage
4. **Vector Index Performance**: Monitor query performance and adjust indexes if needed

## Cost Estimation

**Free Tier**:
- Web Service: Free (750 hours/month)
- PostgreSQL: Free (90 days, then expires)

**Paid Plans**:
- Web Service: $7/month (Starter)
- PostgreSQL: $7/month (Starter, 1GB storage)

For production use, consider at least the Starter plans for both services. 