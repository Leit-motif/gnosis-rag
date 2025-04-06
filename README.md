# Gnosis-RAG: Obsidian ChatGPT Plugin

A powerful ChatGPT plugin that enables natural conversation with your Obsidian vault using hybrid RAG (Retrieval Augmented Generation). Combine semantic search with Obsidian's rich linking and tagging features to explore your personal knowledge system.

## Features

- **Hybrid RAG Pipeline**: Combines semantic search (embeddings) with symbolic search (tags, links, dates)
- **Multiple Vector Store Options**: FAISS, Chroma, or Qdrant
- **Flexible Embedding Models**: OpenAI or local models (e.g., all-MiniLM, instructor-xl)
- **Rich Query Capabilities**: Search by natural language, tags, and date ranges
- **Theme Analysis**: Discover recurring patterns and themes in your notes
- **Reflective Agents**: Generate insights from different perspectives (Gnosis, Anima, Archivist)

## Prerequisites

- Python 3.8+
- An OpenAI API key (for GPT-4 and embeddings)
- An Obsidian vault
- ngrok (for local development) or a hosting service

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gnosis-rag.git
cd gnosis-rag
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

### GET /query
Query your vault with natural language:
```bash
curl "http://localhost:8000/query?q=What+did+I+learn+about+psychology&tags=learning,psychology&date_range=last_30_days"
```

### GET /themes
Analyze recurring themes:
```bash
curl "http://localhost:8000/themes"
```

### POST /reflect
Generate reflections:
```bash
curl -X POST "http://localhost:8000/reflect" \
  -H "Content-Type: application/json" \
  -d '{"mode": "weekly", "agent": "gnosis"}'
```

## Configuration

### Embedding Models

Choose between:
- OpenAI (text-embedding-3-small)
- Local models:
  - all-MiniLM-L6-v2 (fast, lightweight)
  - instructor-xl (more accurate)

### Vector Stores

Available options:
- FAISS (default, in-memory)
- Chroma (persistent, metadata-rich)
- Qdrant (distributed, scalable)

Configure in `config.yaml`:
```yaml
vector_store:
  type: "faiss"  # or "chroma" or "qdrant"
  # ... other settings ...
```

## Development

### Project Structure
```
gnosis-rag/
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
Use ngrok for tunneling:
```bash
ngrok http 8000
```

### Production
Deploy to your preferred platform:
- Render
- Fly.io
- Your own VPS

Update `plugin/ai-plugin.json` with your production URL.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- OpenAI for ChatGPT Plugin SDK
- Obsidian for the amazing note-taking app
- FAISS, Chroma, and Qdrant teams 