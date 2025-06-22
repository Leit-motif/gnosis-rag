# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and use non-root user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies in optimized order
COPY requirements-api.txt .

# Switch to app user before installing packages
USER app

# Install CPU-only PyTorch first for better caching
RUN pip install --user --no-warn-script-location \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu

# Install remaining dependencies
RUN pip install --user --no-warn-script-location -r requirements-api.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:$PATH" \
    PORT=8080

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Copy Python packages from builder stage
COPY --from=builder --chown=app:app /home/app/.local /home/app/.local

# Set work directory and change ownership
WORKDIR /app
RUN chown app:app /app

# Switch to non-root user
USER app

# Copy application code
COPY --chown=app:app ./api ./api
COPY --chown=app:app ./backend/enhanced_graph_retriever.py ./backend/
COPY --chown=app:app ./plugin ./plugin
COPY --chown=app:app ./alembic ./alembic
COPY --chown=app:app ./alembic.ini ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health')" || exit 1

# Run the application
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT --access-log --log-config api/logging.yaml 