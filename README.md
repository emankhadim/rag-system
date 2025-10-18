RAG: Simple Retrieval-Augmented Generation API

Minimal, production-lean RAG system that:

Ingests documents, chunks them, generates embeddings, and stores them in a FAISS vector index.

Retrieves relevant chunks for a user question (semantic similarity).

Generates a context-aware answer with a local HuggingFace LLM (no commercial keys).

Fully containerized with FastAPI + Uvicorn, prebuild utility, and simple shell tests.
