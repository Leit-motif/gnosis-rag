#!/usr/bin/env bash
# Bootstrap for Render free plan:
#  • tries to download a compressed FAISS index from S3
#  • if not found, calls /index to build it, then uploads the archive
set -euo pipefail

INDEX_DIR="/app/data/vector_store"
ARCHIVE="/app/data/vector_store.tar.zst"

download_index() {
  echo "🔄  Downloading index from S3…"
  aws s3 cp "s3://${S3_BUCKET}/vector_store.tar.zst" "${ARCHIVE}" --only-show-errors
  zstd -d --rm "${ARCHIVE}" -o /app/data          # extracts vector_store/*
}

upload_index() {
  echo "⬆️  Uploading updated index to S3…"
  tar --use-compress-program="zstd -T0 -3" -cf "${ARCHIVE}" -C /app/data vector_store
  aws s3 cp "${ARCHIVE}" "s3://${S3_BUCKET}/vector_store.tar.zst" --only-show-errors
  rm -f "${ARCHIVE}"
}

# ---------- boot sequence ----------
mkdir -p "${INDEX_DIR}"

if [ ! -f "${INDEX_DIR}/faiss.index" ]; then
  download_index || echo "⚠️  No existing index found – will build a fresh one"
fi

# Start API in the background
uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}" &
API_PID=$!

# Allow API to start accepting requests
sleep 5

if [ ! -f "${INDEX_DIR}/faiss.index" ]; then
  echo "🛠   Building index from scratch…"
  curl -sf -X POST "http://localhost:${PORT:-8000}/index" \
       || echo "⚠️  /index call failed—check logs"
  upload_index || echo "⚠️  Upload failed – continuing anyway"
fi

wait "${API_PID}" 