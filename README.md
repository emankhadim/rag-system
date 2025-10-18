# RAG System — Retrieval-Augmented Generation API

A local RAG system using FAISS vector search and FLAN-T5 for answer generation. Includes semantic retrieval, sentence-aware chunking, and few-shot prompting.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB RAM minimum

### Setup & Run
```bash
# Clone repository
git clone https://github.com/emankhadim/rag-system.git
cd rag-system
```
### Option A: Docker (Recommended)

```bash

# 1. Build vector database (required first)
python -m scripts.prebuild_index

# 2. Start containers
docker-compose up -d

# 3. Test
curl http://localhost:8000/health
```
### Option B: Local Setup (If Docker Doesn't Work)

**Step 1: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Build Vector Database**
```bash
python -m scripts.prebuild_index
```
**Step 4: Run API (Terminal 1)**
```bash
# Make sure venv is activated
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run Streamlit
streamlit run streamlit_app.py
# Opens automatically in browser at http://localhost:8501

```
**Step 6: Test**
```bash
# In another terminal (Terminal 3)
curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "AI challenges in healthcare?", "top_k": 3}'
```

---

## Project Structure

```
rag/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings (models, paths)
│   └── models.py            # Request/response schemas
├── core/
│   ├── chunker.py           # Sentence-aware chunking
│   ├── embeddings.py        # SentenceTransformer encoding
│   ├── vector_db.py         # FAISS IndexFlatIP
│   ├── retrieval.py         # Semantic + MMR strategies
│   ├── prompts.py           # Few-shot prompt templates
│   ├── llm.py               # FLAN-T5 generation
│   ├── pipeline.py          # RAG orchestration
│   └── rag_system.py        # High-level interface
├── data/
│   ├── raw/                 # Input documents
│   │   └── wikipedia_documents.json
│   └── processed/
│       └── vector_db/       # FAISS index + metadata
├── scripts/
│   └── prebuild.py          # Build database script
├── docker-compose.yml
└── requirements.txt
```

---

## Dataset Description

**Source:** Wikipedia articles (AI domain)

**Content:** 20 curated Wikipedia documents covering:
- Artificial Intelligence fundamentals
- Machine Learning concepts and algorithms
- Deep Learning architectures
- Natural Language Processing
- Computer Vision
- AI ethics and challenges
- AI applications in various industries
- Historical development of AI
- Current AI research trends

**Format:** `data/raw/wikipedia_documents.json`
```json
[
  {
    "id": "doc_1",
    "title": "Artificial Intelligence",
    "content": "Full article text...",
    "source": "Wikipedia",
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"
  }
]
```

**Use Case:** This dataset enables Q&A about AI/ML concepts, making it perfect for demonstrating RAG capabilities on technical domain knowledge.


## RAG Pipeline Components

### 1. Document Chunking (`core/chunker.py`)

**Strategy:** Sentence-aware chunking with overlap

```python
chunk_size = 512      # characters per chunk
chunk_overlap = 50    # overlap between chunks
```

**How it works:**
- Split on sentence boundaries (`. `, `! `, `? `)
- Maintain context with overlap
- Preserve semantic coherence

**Example:**
```
Document: "AI is powerful. It transforms industries. Machine learning enables predictions."

Chunk 1: "AI is powerful. It transforms industries."
Chunk 2: "It transforms industries. Machine learning enables predictions."
         ↑ Overlap maintains context
```

---

### 2. Embeddings (`core/embeddings.py`)

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

```python
embedding_dim = 384
batch_size = 32
normalization = L2 (for cosine similarity)
```
**Why this model:**
- Fast inference on CPU
- Good semantic understanding
- Compact 80MB size

---

### 3. Vector Search (`core/vector_db.py`)

**Index:** FAISS `IndexFlatIP` (Inner Product)

```python
# Normalized embeddings → inner product ≈ cosine similarity
cosine_sim = dot(query_vec, doc_vec) / (||query|| * ||doc||)
```

**Search process:**
```python
1. Query: "WHat are the challenges of AI in healthcare?"
2. Embed query → [0.15, -0.22, ...] (384-dim)
3. FAISS search -> top-k similar chunks
4. Return ranked results with scores
```

**Storage:**
- `faiss_index.bin`: Vector index
- `vector_db_metadata.pkl`: Chunk text + metadata

---

### 4. Retrieval Strategies (`core/retrieval.py`)

#### **Semantic Retrieval (Default)**
```python
# Simple top-k by cosine similarity
results = vector_db.query(query, k=5)
# Returns: [doc1 (score=0.89), doc2 (0.85), doc3 (0.82), ...]
```

#### **MMR Retrieval (Optional)**
```python
# Maximal Marginal Relevance - balances relevance + diversity
mmr_score = λ * relevance - (1-λ) * max_similarity_to_selected
```

**Use case:** Avoid redundant results

---

### 5. Prompting (`core/prompts.py`)

**Available Strategies:**

#### **Few-Shot Prompt (Default)**
- **When to use:** Most queries, especially when you want consistent answer format
- **Why:** Provides examples that guide the LLM to generate grounded, concise answers
- **Best for:** General Q&A, factual queries, preventing hallucinations

#### **Simple Prompt**
- **When to use:** Straightforward questions with clear context
- **Why:** Minimal prompt overhead, faster processing
- **Best for:** Short answers, when context is very clear

#### **Chain-of-Thought**
- **When to use:** Complex reasoning, multi-step questions
- **Why:** Encourages step-by-step thinking before answering
- **Best for:** Math problems, logical reasoning, analytical questions


### 6. LLM Generation (`core/llm.py`)

**Model:** `google/flan-t5-base` (248M parameters)

```python
generation_config = {
    "max_new_tokens": 150,
    "min_new_tokens": 40,
    "temperature": 0.0,          
    "no_repeat_ngram_size": 3,    
    "repetition_penalty": 1.05
}
```

**Post-processing:**
- Remove artifacts (quotes, brackets)
- Normalize whitespace
- Fallback if output is weak

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can you explain me the challenges of AI?",
    "top_k": 3,
    "return_sources": true
  }'
```

**Response:**
```json
{
  "answer": "Artificial intelligence is the capability of computational systems...",
  "sources": [
    {"title": "AI Overview", "score": 0.89, "text": "..."},
    {"title": "ML Basics", "score": 0.85, "text": "..."}
  ],
  "retrieval_strategy": "semantic",
  "prompt_strategy": "few_shot"
}
```

### Ingest Documents
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc1", "title": "AI", "text": "AI researchers have adapted and integrated techniques including search and mathematical optimization, formal logic."}
    ]
  }'
```

---



## Complete Flow Example

```
User: "What are the applications of AI?"
                
[1] Embed query with SentenceTransformer
     [0.15, -0.22, 0.08, ..., 0.31] (384-dim)
                
[2] FAISS search for top-5 similar chunks
    Chunk 1: "AI in healthcare..." (score: 0.89)
    Chunk 2: "AI in finance..." (score: 0.87)
    Chunk 3: "AI in education..." (score: 0.84)
                
[3] Build few-shot prompt with context
    "Answer based on context... [examples] ... Question: What are..."
                
[4] Generate with FLAN-T5
    "AI applications include healthcare diagnostics, financial..."
                
[5] Return answer + sources
```

---

## Docker Commands

```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f rag-api

# Stop
docker-compose down

# Rebuild
docker-compose build --no-cache
```

---

## Troubleshooting

**Database not found:**
```bash
python scripts/prebuild.py
ls data/processed/vector_db/  # Should show files
```

**Out of memory:**
```yaml
# Edit docker-compose.yml
memory: 12G  # Increase from 10G
```

**Port in use:**
```bash
lsof -i :8000  # Find process
# OR change port in docker-compose.yml
```

---

## Tech Stack

- **Chunking**: Sentence-aware with overlap
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: FAISS IndexFlatIP (cosine similarity)
- **Retrieval**: Semantic search (+ MMR option)
- **Prompting**: Few-shot templates
- **LLM**: google/flan-t5-base
- **API**: FastAPI

---
