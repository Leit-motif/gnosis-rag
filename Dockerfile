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

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Switch to app user before installing packages
USER app

# Install all dependencies for the full backend
RUN pip install --user --no-warn-script-location -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:$PATH" \
    PORT=8080 \
    OBSIDIAN_VAULT_PATH=/app/data/vault \
    OPENAI_SYSTEM_PROMPT="You are Gnosis, an AI assistant that helps users query their Obsidian vault."

# Install runtime dependencies (libpq + zstd for compression)
RUN apt-get update && apt-get install -y \
    libpq5 \
    zstd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# AWS CLI for S3 upload/download (small footprint)
RUN pip install --no-cache-dir awscli==1.29.*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Copy Python packages from builder stage
COPY --from=builder --chown=app:app /home/app/.local /home/app/.local

# Set work directory and change ownership
WORKDIR /app
RUN chown app:app /app

# Switch to non-root user
USER app

# Copy application code - deploy the full backend with all functionality
COPY --chown=app:app ./backend ./backend
COPY --chown=app:app ./plugin ./plugin
COPY --chown=app:app ./data ./data

# Create necessary directories and minimal vault structure
RUN mkdir -p /app/logs /app/data/conversations /app/data/vector_store /app/data/vault && \
    echo "# Welcome to Gnosis\n\nThis is a placeholder vault for the deployed API.\nConnect your real Obsidian vault to use the full functionality." > /app/data/vault/README.md

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:$PORT/health')" || exit 1

# Copy bootstrap script and make it executable
COPY --chown=app:app ./scripts/boot.sh /app/scripts/boot.sh
RUN chmod +x /app/scripts/boot.sh

# Start via bootstrapper (downloads existing index or builds & uploads)
CMD ["/app/scripts/boot.sh"] 