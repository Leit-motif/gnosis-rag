# Robust Indexing System Usage Guide

## Overview
The robust indexing system provides reliable document indexing with rate limiting, error recovery, and progress tracking.

## New Endpoints

### 1. POST /index_robust
Replaces the original `/index` endpoint with better error handling.

**Features:**
- Rate limiting to avoid OpenAI API limits
- Progressive saving with checkpoints
- Automatic resume capability
- Better error messages and logging

**Usage:**
```bash
curl -X POST "http://localhost:8000/index_robust"
```

### 2. GET /index_status
Check the current indexing status and progress.

**Usage:**
```bash
curl "http://localhost:8000/index_status"
```

**Response:**
- `not_started`: No indexing has been performed
- `in_progress`: Indexing is currently running or was interrupted
- `completed`: Indexing finished successfully

### 3. POST /resume_index
Resume indexing from where it left off after a failure.

**Usage:**
```bash
curl -X POST "http://localhost:8000/resume_index"
```

## Configuration
The system uses `config_rate_limited.json` for rate limiting settings:

```json
{
  "rate_limiting": {
    "max_requests_per_minute": 50,
    "batch_size": 25,
    "delay_between_batches": 2.0,
    "max_retries": 3,
    "backoff_factor": 2.0
  },
  "api_settings": {
    "max_tokens_per_request": 4000,
    "embedding_timeout": 30,
    "max_concurrent_requests": 2
  }
}
```

## Troubleshooting

### Rate Limiting Issues
- The system automatically handles OpenAI rate limits
- Increase `delay_between_batches` if you still hit limits
- Reduce `batch_size` for more conservative API usage

### Resuming After Failure
1. Check status: `GET /index_status`
2. If status is `in_progress`, run: `POST /resume_index`
3. The system will continue from the last checkpoint

### Monitoring Progress
- Check logs in `vault_reset.log`
- Use `/index_status` to see progress statistics
- Checkpoint files are saved in `data/vector_store/checkpoints/`

## Migration from Original Indexing
1. Use `/index_robust` instead of `/index`
2. Monitor progress with `/index_status`
3. If interrupted, resume with `/resume_index`
4. The system is backward compatible with existing vector stores
