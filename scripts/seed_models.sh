#!/usr/bin/env bash
set -euo pipefail
curl -s http://localhost:11434/api/pull -d '{"name": "nomic-embed-text:v1.5"}'
curl -s http://localhost:11434/api/pull -d '{"name": "gemma3:1b"}'


# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Phoenix (optional)
python -m src.observability.start_phoenix.py

# Terminal 3: Start FastAPI backend
uvicorn src.app.api:app --reload --port 8000

# Terminal 4: Start Open WebUI
OPENAI_API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=anything open-webui serve --host 0.0.0.0 --port 3000
