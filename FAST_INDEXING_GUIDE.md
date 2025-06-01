# Fast Indexing System Guide

## 🚀 Overview

The Fast Indexing System is designed to maximize embedding generation speed for large vaults while maintaining reliability. It uses aggressive optimization techniques including:

- **Concurrent API requests** (up to 20 simultaneous)
- **Large batch processing** (up to 200 documents per batch)
- **Streaming memory management** for massive vaults
- **Automatic checkpoint/resume** functionality
- **Pickle serialization** for faster I/O

## ⚡ Quick Start

### 1. Use the Fast Indexing Endpoint

```bash
# Ultra-fast indexing (recommended for large vaults)
curl -X POST "http://localhost:8000/index_fast"

# Check indexing status
curl "http://localhost:8000/index_fast_status"

# Resume if interrupted
curl -X POST "http://localhost:8000/index_fast_resume"
```

### 2. Environment Variable Configuration

Add these to your `.env` file for custom optimization:

```env
# Basic settings
OPENAI_API_KEY=your_api_key_here

# Fast indexing optimizations
FAST_BATCH_SIZE=150                    # Documents per API request
FAST_CONCURRENT_REQUESTS=15            # Simultaneous API calls
FAST_CHECKPOINT_INTERVAL=500           # Save progress every N docs
FAST_EMBEDDING_TIMEOUT=90              # API timeout in seconds
FAST_USE_STREAMING=true                # Memory-efficient processing
```

## 📊 Performance Presets

Choose based on your vault size:

### Small Vault (<1,000 documents)
```env
FAST_PRESET=small_vault
```
- **Expected Speed**: 30-60 docs/sec
- **Memory Usage**: Low
- **API Usage**: Conservative

### Medium Vault (1,000-5,000 documents)
```env
FAST_PRESET=medium_vault
```
- **Expected Speed**: 50-100 docs/sec
- **Memory Usage**: Moderate
- **API Usage**: Balanced

### Large Vault (5,000-20,000 documents)
```env
FAST_PRESET=large_vault
```
- **Expected Speed**: 100-200 docs/sec
- **Memory Usage**: High
- **API Usage**: Aggressive

### Massive Vault (>20,000 documents)
```env
FAST_PRESET=massive_vault
```
- **Expected Speed**: 200-400 docs/sec
- **Memory Usage**: Very High
- **API Usage**: Maximum

## 🎯 Optimization Recommendations

### For Your Massive Vault

Based on your description of having a "massive amount of items," I recommend:

1. **Start with the `large_vault` preset**:
   ```env
   FAST_PRESET=large_vault
   FAST_BATCH_SIZE=150
   FAST_CONCURRENT_REQUESTS=15
   ```

2. **Monitor and adjust based on results**:
   - If you hit rate limits → reduce concurrent requests
   - If you run out of memory → enable streaming and reduce memory limit
   - If it's working well → gradually increase settings

3. **Check your OpenAI tier**:
   ```bash
   # Check your current usage at: https://platform.openai.com/usage
   # Tier 1: 500 RPM, 30K TPM → use conservative settings
   # Tier 2: 5000 RPM, 150K TPM → use medium/large settings  
   # Tier 3: 10000 RPM, 1M TPM → use massive settings
   ```

## 🔧 Manual Configuration

For fine-tuned control, set these environment variables:

```env
# Throughput settings
FAST_BATCH_SIZE=200                    # Higher = faster, more API usage
FAST_CONCURRENT_REQUESTS=20            # Higher = faster, more rate limit risk

# Reliability settings
FAST_CHECKPOINT_INTERVAL=1000          # Lower = more frequent saves
FAST_EMBEDDING_TIMEOUT=120             # Higher = more tolerance for slow requests

# Memory management
FAST_USE_STREAMING=true                # Always recommended for large vaults
FAST_MAX_MEMORY_MB=4000               # Adjust based on available RAM
```

## 📈 Expected Performance

| Configuration | Speed (docs/sec) | Memory Usage | Risk Level |
|--------------|------------------|--------------|------------|
| Conservative | 10-30           | Low          | Very Safe  |
| Medium Vault | 50-100          | Moderate     | Safe       |
| Large Vault  | 100-200         | High         | Moderate   |
| Massive Vault| 200-400         | Very High    | Higher     |

## 🛠️ Troubleshooting

### Rate Limit Errors
**Symptoms**: `429 Too Many Requests`, `Rate limit exceeded`

**Solutions**:
```env
FAST_BATCH_SIZE=25
FAST_CONCURRENT_REQUESTS=3
FAST_DELAY_BETWEEN_BATCHES=2.0
```

### Memory Errors
**Symptoms**: `MemoryError`, System becomes unresponsive

**Solutions**:
```env
FAST_USE_STREAMING=true
FAST_MAX_MEMORY_MB=2000
FAST_BATCH_SIZE=50
```

### Timeout Errors
**Symptoms**: `TimeoutError`, `Request timeout`

**Solutions**:
```env
FAST_EMBEDDING_TIMEOUT=180
FAST_BATCH_SIZE=75
```

### Progress Lost/Interrupted
**Solution**: The system automatically saves checkpoints. Simply run:
```bash
curl -X POST "http://localhost:8000/index_fast_resume"
```

## 📊 Monitoring Progress

### Check Status
```bash
curl "http://localhost:8000/index_fast_status"
```

**Response**:
```json
{
  "status": "in_progress",
  "total_documents": 15420,
  "processed_documents": 8350,
  "progress_percent": 54.1,
  "estimated_time_remaining": "12.3 minutes"
}
```

### Watch Logs
```bash
tail -f logs/gnosis.log | grep -E "(Fast indexing|Progress:|Speed:)"
```

## 🎯 Performance Testing

Run the performance test to find optimal settings for your system:

```bash
python test_fast_indexing.py
```

This will:
- Test different configurations
- Measure actual speed on your system
- Recommend optimal settings
- Validate resume functionality

## 🔄 Comparison with Standard Indexing

| Feature | Standard `/index` | Fast `/index_fast` |
|---------|-------------------|-------------------|
| Speed | 5-15 docs/sec | 50-400 docs/sec |
| Memory Usage | High | Configurable |
| Resume Support | Basic | Advanced |
| Batch Size | 10-25 | 50-200 |
| Concurrent Requests | 1-3 | 5-20 |
| Checkpoints | None | Automatic |
| Optimization | Conservative | Aggressive |

## 🚀 Best Practices

1. **Start Conservative**: Begin with `medium_vault` preset and scale up
2. **Monitor API Usage**: Check OpenAI dashboard for rate limit status
3. **Use Streaming**: Always enable for vaults >1000 documents
4. **Save Progress**: Don't disable checkpoints for large operations
5. **Test First**: Run `test_fast_indexing.py` to find optimal settings
6. **Monitor Memory**: Watch system memory usage during large operations
7. **Plan for Interruptions**: The system can resume from any checkpoint

## 🔧 Advanced Configuration

### Custom Preset Creation

Create your own preset in `config_fast_indexing.json`:

```json
{
  "fast_indexing": {
    "presets": {
      "my_custom": {
        "description": "My optimized settings",
        "batch_size": 125,
        "max_concurrent_requests": 12,
        "embedding_timeout": 75,
        "checkpoint_interval": 300,
        "use_streaming": true,
        "max_memory_mb": 2500
      }
    }
  }
}
```

### Dynamic Adjustment

The system automatically adjusts based on errors:
- Rate limits → reduces concurrency
- Timeouts → increases timeout
- Memory pressure → enables streaming

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Run the performance test**: `python test_fast_indexing.py`
3. **Review logs**: Look for specific error messages
4. **Start with conservative settings**: Use `FAST_PRESET=conservative`
5. **Monitor OpenAI usage**: Check your rate limits and billing

---

**Remember**: The goal is to find the balance between speed and reliability for your specific vault size and API limits. Start conservative and gradually increase settings based on your results! 