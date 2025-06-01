# Simplified Environment Setup

## 🎯 New `.env` File (Sensitive Data Only)

Replace your current `.env` file with this simplified version:

```env
# =============================================================================
# API KEYS & SENSITIVE DATA ONLY
# =============================================================================

# Required: Your OpenAI API key
OPENAI_API_KEY=your_actual_openai_api_key_here

# Your custom system prompt (keep private as it defines AI behavior)
OPENAI_SYSTEM_PROMPT="Your custom system prompt here"

# User-specific vault path (varies per machine/user)
OBSIDIAN_VAULT_PATH=C:/Users/Rando/Sync/Users/Rando/Sync/My Obsidian Vault/5.0 Journal/5.1 Daily/
```

## ✅ What Moved to `config.yaml`

All these settings are now in your `config.yaml` file:

- ✅ **Model Settings**: `gpt-4o-mini`, `all-mpnet-base-v2`, `text-embedding-3-small`
- ✅ **API Settings**: `127.0.0.1:8000`, `debug: false`
- ✅ **Embedding Provider**: `local` (using all-mpnet-base-v2)
- ✅ **Vector Store**: `384` dimensions (for local embeddings)
- ✅ **Fast Indexing**: `large_vault` preset for your massive vault
- ✅ **Response Tokens**: `128,000` (matches your current setting)

## 🚀 Fast Indexing Configuration

Your `config.yaml` now includes fast indexing with the `large_vault` preset, which gives you:

- **Batch Size**: 150 documents per request
- **Concurrent Requests**: 15 simultaneous API calls
- **Memory Streaming**: Enabled for large vaults
- **Checkpoints**: Every 500 documents
- **Expected Speed**: 100-200 docs/sec (vs 5-15 with standard indexing)

## 🔧 How to Use Fast Indexing

1. **Update your `.env`** with the simplified version above
2. **Restart your server** to load the new config
3. **Use the fast endpoint**:
   ```bash
   curl -X POST "http://127.0.0.1:8000/index_fast"
   ```

## 📊 Benefits of This Setup

- ✅ **Security**: Only sensitive data in `.env`
- ✅ **Version Control**: `config.yaml` can be committed to git
- ✅ **Maintainability**: Easy to adjust settings without touching environment
- ✅ **Team Sharing**: Others can use same config with their own `.env`
- ✅ **Speed**: Optimized for your massive vault

## 🎯 For Your Massive Vault

Since you're using **local embeddings** (`all-mpnet-base-v2`), the fast indexing will be even more efficient because:

- No OpenAI API rate limits for embeddings
- Faster local processing
- No API costs for embedding generation
- Can use even more aggressive settings

If you want to try **maximum speed** for local embeddings, you can change in `config.yaml`:

```yaml
fast_indexing:
  preset: massive_vault  # Even faster for local embeddings
```

This would give you 200-400 docs/sec since you're not limited by API calls!

## 🔄 Migration Steps

1. **Backup** your current `.env` file
2. **Replace** your `.env` with the simplified version above
3. **Add your actual OpenAI API key**
4. **Update your vault path** if needed
5. **Restart** your server
6. **Test** with: `curl -X POST "http://127.0.0.1:8000/index_fast"` 