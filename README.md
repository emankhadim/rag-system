# RAG System — Retrieval-Augmented Generation API

A production-ready, local-only RAG (Retrieval-Augmented Generation) system with FastAPI, FAISS, and HuggingFace Transformers.

## Features

-  **Fully Local**: No API keys required - runs entirely on your machine
- **Docker Ready**: Containerized deployment with docker-compose
- **Fast Retrieval**: FAISS vector database with semantic search
- **Local LLM**: FLAN-T5 for answer generation
- **Multiple Strategies**: Configurable retrieval and prompting strategies
- **Production Ready**: Health checks, error handling and comprehensive logging
- **REST API**: FastAPI with automatic OpenAPI documentation

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- 8GB+ RAM recommended
- 10GB+ free disk space

### Option 1: Docker (Recommended)

```bash
# 1. Build database locally (REQUIRED FIRST STEP)
python scripts/prebuild.py

# 2. Start Docker containers
docker-compose up -d

# 3. Check health
curl http://localhost:8000/health

# 4. Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Currrent AI challenges in Healthcare sector?", "top_k": 3, "return_sources": true}'
```

**Access Points:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501 (if enabled)

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build database
python -m scripts.prebuild_index

# 4. Start API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
rag/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   └── models.py            # Pydantic request/response models
├── core/
│   ├── chunker.py           # Document chunking with overlap
│   ├── embeddings.py        # SentenceTransformer wrapper
│   ├── vector_db.py         # FAISS vector database
│   ├── retrieval.py         # Retrieval strategies (semantic/MMR)
│   ├── prompts.py           # Prompt templates
│   ├── llm.py               # Local LLM (FLAN-T5)
│   ├── pipeline.py          # RAG pipeline orchestration
│   └── rag_system.py        # High-level RAG interface
├── data/
│   ├── raw/                 # Input documents
│   │   └── wikipedia_documents.json
│   └── processed/
│       └── vector_db/       # FAISS index + metadata
├── scripts/
│   └── prebuild.py          # Build vector database
├── Dockerfile               # API container
├── docker-compose.yml       # Orchestration
├── streamlit_app.py        # Web UI
├── requirements.txt         # Python dependencies
├── .env                     # Configuration
└── README.md               # This file
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "vector_db_loaded": true,
  "num_vectors": 125,
  "num_documents": 10,
  "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model_name": "google/flan-t5-base"
}
```

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc_1",
        "title": "Artificial Intelligence",
        "text": "AI is the simulation of human intelligence...",
        "metadata": {}
      }
    ],
    "chunk_size": 512,
    "chunk_overlap": 50
  }'
```

### Query Documents

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the applications of AI in healthcare?",
    "top_k": 5,
    "max_new_tokens": 150,
    "temperature": 0.0,
    "return_sources": true
  }'
```

**Response:**
```json
{
  "question": "What are the applications of AI in healthcare?",
  "answer": "AI in healthcare includes diagnostic imaging, drug discovery...",
  "sources": [
    {
      "title": "Healthcare AI",
      "score": 0.89,
      "text": "AI technologies are transforming..."
    }
  ],
  "num_sources": 5,
  "processing_time": 1.234,
  "retrieval_strategy": "semantic",
  "prompt_strategy": "few_shot",
  "fallback_used": false
}
```

---

## Architecture

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI | REST API with automatic docs |
| **Vector DB** | FAISS | Fast similarity search |
| **Embeddings** | sentence-transformers | Semantic encoding |
| **LLM** | FLAN-T5 | Answer generation |
| **Web UI** | Streamlit | Optional chat interface |
| **Containerization** | Docker | Deployment |

### Data Flow

```
User Query
    ↓
[1] Embed Query (SentenceTransformer)
    ↓
[2] Retrieve Similar Chunks (FAISS)
    ↓
[3] Build Context with Top-K Results
    ↓
[4] Generate Answer (FLAN-T5)
    ↓
[5] Return Answer + Sources
```

### Models Used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
  - 384 dimensions
  - Fast, good quality
  - 80MB model size
  
- **LLM**: `google/flan-t5-base`
  - 247M parameters
  - Instruction-tuned
  - ~1GB model size

---
### Adding New Documents

```bash
# Option 1: Via API
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d @your_documents.json

# Option 2: Rebuild database
# 1. Add documents to data/raw/
# 2. Run: python scripts/prebuild.py
# 3. Restart Docker: docker-compose restart
```

---

## Docker Deployment

### Build and Run

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Stop services
docker-compose down
```

### Resource Limits

Default limits in `docker-compose.yml`:
- Memory: 10GB limit, 8GB reserved
- CPU: No limit (uses available cores)

**See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed Docker documentation.**

---

## Troubleshooting

### Container Crashes on Query

**Symptom:** `curl: (52) Empty reply from server` or `exit code 139`

**Solution:** 
- PyTorch CPU incompatibility in Docker
- Already fixed in current Dockerfile with proper system libraries
- If still issues, increase memory limit in `docker-compose.yml`

### Database Not Found

**Symptom:** "System not ready. Please ingest documents first"

**Solution:**
```bash
# Build database locally
python -m scripts.prebuild_index.py

# Verify files exist
ls -lh data/processed/vector_db/
# Should show: faiss_index.bin, vector_db_metadata.pkl
```

### Slow First Query

**Symptom:** First query takes 30+ seconds

**Cause:** Models downloading from HuggingFace

**Solution:**
- Pre-download models or use cached directory
- Mount `hf-cache` volume in Docker (already configured)

### Out of Memory

**Symptom:** Container killed or slow performance

**Solution:**
```yaml
# In docker-compose.yml, increase limits:
deploy:
  resources:
    limits:
      memory: 12G  # Increase from 10G
```

### Port Already in Use

**Symptom:** "Address already in use" error

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# OR change port in docker-compose.yml
ports:
  - "8080:8000"  # Use 8080 instead
```

---
## Advanced Usage

### Custom Retrieval Strategy

Use MMR (Maximal Marginal Relevance) for diverse results:

```python
from core.rag_system import RAGSystem

rag = RAGSystem(
    vector_db_dir='data/processed/vector_db',
    llm_model_name='google/flan-t5-base',
    retrieval_strategy='mmr',  # Instead of 'semantic'
    embedder=embedder
)
```

### Custom Prompts

Use different prompt strategies:

```python
rag = RAGSystem(
    # ...
    prompt_strategy='chain_of_thought'  # Instead of 'few_shot'
)
```

Available strategies:
- `simple`: Direct question-context-answer
- `few_shot`: Includes examples
- `chain_of_thought`: Step-by-step reasoning

--

**Built with ❤️ for local, privacy-focused AI applications.**
