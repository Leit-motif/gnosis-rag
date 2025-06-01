# Environment Configuration Guide

## 🔧 Your Current Setup

You have a hybrid configuration system:
- **`.env` file**: Contains actual values (API keys, paths, etc.)
- **`config.yaml`**: References environment variables using `${VARIABLE_NAME}` syntax

## 📝 Required `.env` File Configuration

Add these variables to your existing `.env` file:

```env
# =============================================================================
# REQUIRED BASIC CONFIGURATION
# =============================================================================
OPENAI_API_KEY=sk-your-openai-api-key-here
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault

# =============================================================================
# EMBEDDING & MODEL CONFIGURATION  
# =============================================================================
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LOCAL_MODEL=all-mpnet-base-v2
VECTOR_STORE_PATH=data/vector_store

CHAT_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_RESPONSE_TOKENS=2000
OPENAI_SYSTEM_PROMPT=You are a helpful AI assistant that provides accurate information based on the user's Obsidian vault content.

# =============================================================================
# API & SYSTEM CONFIGURATION
# =============================================================================
API_HOST=localhost
API_PORT=8000
DEBUG_MODE=false
DOCS_DIR=docs

# =============================================================================
# ⚡ FAST INDEXING OPTIMIZATION (NEW!)
# =============================================================================

# OPTION 1: Use a preset (recommended for massive vaults)
FAST_PRESET=massive_vault

# OPTION 2: Manual configuration (overrides preset)
FAST_BATCH_SIZE=200                    # Documents per API request
FAST_CONCURRENT_REQUESTS=20            # Simultaneous API calls  
FAST_CHECKPOINT_INTERVAL=1000          # Save progress every N docs
FAST_EMBEDDING_TIMEOUT=120             # API timeout in seconds
FAST_USE_STREAMING=true                # Memory-efficient processing
FAST_MAX_MEMORY_MB=4000               # Memory limit in MB
```

## 🎯 Preset Recommendations for Your Massive Vault

### Start Here (Recommended):
```env
FAST_PRESET=large_vault
```
- **Speed**: 100-200 docs/sec
- **Safe for most API limits**
- **Good balance of speed and reliability**

### If That Works Well, Upgrade To:
```env
FAST_PRESET=massive_vault
```
- **Speed**: 200-400 docs/sec
- **Maximum performance**
- **Requires higher OpenAI API tier**

### If You Hit Rate Limits, Use:
```env
FAST_PRESET=conservative
```
- **Speed**: 10-30 docs/sec
- **Very safe, won't hit limits**
- **Good for Tier 1 OpenAI accounts**

## 🚀 Quick Start for Your Massive Vault

1. **Add to your `.env` file**:
   ```env
   # Your existing variables...
   
   # Add these new fast indexing variables:
   FAST_PRESET=large_vault
   ```

2. **Test the fast indexing**:
   ```bash
   curl -X POST "http://localhost:8000/index_fast"
   ```

3. **Monitor progress**:
   ```bash
   curl "http://localhost:8000/index_fast_status"
   ```

4. **If it works well, upgrade**:
   ```env
   FAST_PRESET=massive_vault
   ```

## 🔍 Performance Testing

Run this to find your optimal settings:
```bash
python test_fast_indexing.py
```

This will test different configurations and recommend the best settings for your system and API limits.

## 🛠️ Troubleshooting

### Rate Limit Errors
Add these to `.env`:
```env
FAST_PRESET=conservative
FAST_BATCH_SIZE=25
FAST_CONCURRENT_REQUESTS=3
```

### Memory Issues
Add these to `.env`:
```env
FAST_USE_STREAMING=true
FAST_MAX_MEMORY_MB=2000
FAST_BATCH_SIZE=50
```

### Want Maximum Speed (if you have high API limits)
Add these to `.env`:
```env
FAST_PRESET=massive_vault
FAST_BATCH_SIZE=200
FAST_CONCURRENT_REQUESTS=20
FAST_CHECKPOINT_INTERVAL=1000
```

## 📊 Expected Results

For a massive vault, you should see:
- **Standard `/index`**: 5-15 docs/sec
- **Fast `/index_fast`**: 50-400 docs/sec (depending on settings)

## 🔧 How It Works

1. Your `config.yaml` stays the same
2. The fast indexer reads these new environment variables
3. If variables aren't set, it uses safe defaults
4. The system automatically handles checkpoints and resume functionality

This setup gives you the best of both worlds: your existing stable configuration plus the new speed optimizations! 