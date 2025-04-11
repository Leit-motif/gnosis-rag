# Obsidian ChatGPT Plugin

A powerful ChatGPT plugin that enables natural conversation with your Obsidian vault using hybrid RAG (Retrieval Augmented Generation). Combine semantic search with Obsidian's rich linking and tagging features to explore your personal knowledge system.

## Features

- **Hybrid RAG Pipeline**: Combines semantic search (embeddings) with symbolic search (tags, links, dates)
- **Multiple Vector Store Options**: FAISS, Chroma, or Qdrant
- **Flexible Embedding Models**: OpenAI or local models (e.g., all-MiniLM, instructor-xl)
- **Rich Query Capabilities**: Search by natural language, tags, and date ranges
- **Theme Analysis**: Discover recurring patterns and themes in your notes
- **Conversation Memory**: Maintains context across chat sessions for more coherent interactions

## Prerequisites

- Python 3.8+
- An OpenAI API key (for GPT-4 and embeddings)
- An Obsidian vault
- ngrok (for local development) or a hosting service

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

## Usage

1. Start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```

2. (Local Development) Start ngrok:
```bash
ngrok http 8000
```

3. Install the plugin in ChatGPT:
   - Go to ChatGPT Plugin Store
   - Choose "Develop your own plugin"
   - Enter your ngrok URL or hosted domain
   - Follow the installation prompts

## API Endpoints

### POST /chat
Chat with your vault using natural language:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What did I learn about psychology?"}
    ],
    "filters": {
      "tags": ["learning", "psychology"],
      "date_range": "last_30_days"
    }
  }'
```

### GET /health
Check the service health:
```bash
curl "http://localhost:8000/health"
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
└── requirements.txt
```

### Running Tests
```bash
pytest backend/tests/
```

## Deployment

### Local Development
1. Start the FastAPI server:
```bash
uvicorn backend.main:app --reload
```

2. Use ngrok for tunneling:
```bash
ngrok http 8000
```

3. Install in ChatGPT:
- Go to ChatGPT Plugin Store
- Choose "Develop your own plugin"
- Enter your ngrok URL
- Follow the installation prompts

### Production
Deploy to your preferred platform and update `plugin/ai-plugin.json` with your production URL.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI for ChatGPT Plugin SDK
- Obsidian for the amazing note-taking app
- Facebook Research for FAISS 