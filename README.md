# Gnosis RAG - Obsidian-ChatGPT Plugin

A plugin for querying your Obsidian vault from ChatGPT and saving conversations back to your vault.

## Features

- **Query your Obsidian vault** using natural language
- **Save conversations** back to your daily notes
- **Advanced RAG retrieval** using hybrid search and semantic understanding
- **Session management** for multi-turn conversations

## How Save Conversation Works

The plugin provides multiple ways to save conversations to your Obsidian vault:

### Method 1: Standard Session-Based Saving 

The default method uses session IDs to track conversations:

```json
{
  "session_id": "session123",
  "conversation_name": "My Conversation"
}
```

### Method 2: Direct Message Saving (Recommended)

This method sends the actual conversation content directly, bypassing session ID lookup:

```json
{
  "session_id": "any-value",
  "conversation_name": "My Conversation",
  "messages": [
    {
      "role": "user",
      "content": "What's the difference between X and Y?"
    },
    {
      "role": "assistant",
      "content": "X has features A and B, while Y has features C and D..."
    }
  ]
}
```

### Method 3: Direct Content Saving

This method allows saving arbitrary content directly:

```json
{
  "conversation_name": "My Custom Content",
  "content": "Any text you want to save to your vault"
}
```

## Debugging Endpoints

For troubleshooting, there are several debugging endpoints:

- `/debug_all_conversations` - Lists all available sessions and their first messages
- `/debug_save_conversation` - Provides detailed debug info for save requests
- `/save_exact_content` - Allows saving exact content directly

## Using Test Scripts

Several test scripts are provided to demonstrate the API:

- `test_save_conversation.py` - Tests the standard session-based saving
- `test_save_with_messages.py` - Tests the direct message saving method
- `test_direct_save.py` - Tests saving exact content directly

## Editing the Plugin for Custom GPT

If you're creating a Custom GPT, make sure the plugin uses the direct message saving method for more reliable operation.

When saving conversations from ChatGPT, you should construct the request with:

1. A conversation name
2. The actual messages from the current conversation
3. Any session ID (it will be ignored when messages are provided)

This ensures that the exact content you see in ChatGPT is what gets saved to your vault.

## Troubleshooting

If the wrong content is being saved to your vault, try one of these approaches:

1. Use the direct message saving method instead of session-based saving
2. Use `/debug_all_conversations` to see what sessions are available
3. Use `test_save_with_messages.py` to demonstrate correct saving
4. Check the server logs for detailed information about save requests

## Prerequisites

- Python 3.8+
- An OpenAI API key (for GPT-4 and embeddings)
- An Obsidian vault
- Cloudflare Tunnel (for local development with a stable URL) or a hosting service

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/obsidian-chatgpt-plugin.git
cd obsidian-chatgpt-plugin
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
OBSIDIAN_VAULT_PATH=/path/to/your/vault
```

5. Update `config.yaml`:
```yaml
vault:
  path: ${OBSIDIAN_VAULT_PATH}
  # ... other settings ...
```

6. Install Cloudflare Tunnel:
   - Windows: `choco install cloudflared` or download from [Cloudflare's GitHub](https://github.com/cloudflare/cloudflared/releases)
   - Mac: `brew install cloudflared`
   - Linux: Check [Cloudflare documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation) for your distribution

## Usage

### Quick Start
Use the provided startup scripts to launch both the FastAPI server and Cloudflare Tunnel:

- **Windows Batch**: Double-click `start-gnosis.bat`
- **PowerShell**: Right-click `start-gnosis.ps1` and select "Run with PowerShell"

The script will start the FastAPI server and Cloudflare Tunnel in separate windows and display the URLs.

### Manual Start

1. Start the FastAPI server:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

2. Use Cloudflare Tunnel for a stable URL:
```bash
cloudflared tunnel --url http://localhost:8000
```

3. Install the plugin in ChatGPT:
   - Go to ChatGPT Plugin Store
   - Choose "Develop your own plugin"
   - Enter your Cloudflare Tunnel URL
   - Follow the installation prompts

## Features

### Querying Your Vault
The plugin enables natural language search of your Obsidian vault with optional filters:
```
"What notes did I write about machine learning in the last month?"
```

### Saving Conversations
You can save your ChatGPT conversations to your daily notes in Obsidian:
```
"Save this conversation to my daily notes with the title 'Machine Learning Discussion'"
```

This creates or appends to your daily note with a formatted version of the conversation.

## API Endpoints

### GET /query
Query your vault using natural language:
```bash
curl "http://localhost:8000/query?q=What%20did%20I%20learn%20about%20psychology?&tags=learning,psychology&date_range=last_30_days"
```

### POST /save_conversation
Save the current conversation to your daily notes:
```bash
curl -X POST "http://localhost:8000/save_conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id",
    "conversation_name": "Important Discussion"
  }'
```

### POST /index
Index your Obsidian vault:
```bash
curl -X POST "http://localhost:8000/index"
```

## Configuration

### Embedding Models

The plugin currently uses:
- OpenAI (text-embedding-3-small) for embeddings
- GPT-4 for chat completions

Configure in `.env`:
```env
OPENAI_API_KEY=your_api_key_here
OBSIDIAN_VAULT_PATH=/path/to/your/vault
```

### Vector Store

The plugin uses FAISS as the vector store for efficient similarity search. The index is automatically built and updated when the service starts.

Configure paths in `config.yaml`:
```yaml
vault:
  path: ${OBSIDIAN_VAULT_PATH}
  index_path: "./data/faiss_index"
  cache_path: "./data/cache"
```

## Development

### Project Structure
```
obsidian-chatgpt-plugin/
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── rag_pipeline.py         # Core RAG logic
│   ├── obsidian_loader.py      # Vault parsing
│   └── config.yaml             # Configuration
├── plugin/
│   ├── openapi.yaml           # API specification
│   └── ai-plugin.json         # Plugin manifest
├── start-gnosis.bat           # Windows startup script
├── start-gnosis.ps1           # PowerShell startup script
└── requirements.txt
```

### Running Tests
```bash
pytest backend/tests/
```

## Deployment

### Local Development
For the simplest setup, use the provided scripts:

- **Windows**: `start-gnosis.bat`
- **PowerShell**: `start-gnosis.ps1`

These scripts will:
1. Activate your virtual environment
2. Start the FastAPI server
3. Start Cloudflare Tunnel with a stable URL
4. Display the URLs for both services

### Production
For a production environment, you can:
1. Deploy to a cloud hosting service (Azure, AWS, etc.)
2. Update `plugin/ai-plugin.json` with your production URL
3. Share your plugin with other ChatGPT users

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI for ChatGPT Plugin SDK
- Obsidian for the amazing note-taking app
- Facebook Research for FAISS
- Cloudflare for Cloudflare Tunnel 