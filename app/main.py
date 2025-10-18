""" RAG System API """
from __future__ import annotations

import os
HF_HOME = os.getenv('HF_HOME', './hf-cache')
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', './hf-cache/transformers')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './hf-cache/sentence_transformers')
os.environ['HF_DATASETS_CACHE'] = f'{HF_HOME}/datasets'
os.environ['TORCH_HOME'] = f'{HF_HOME}/torch'

import uvicorn
import time
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    HealthResponse, Source
)
from core.chunker import DocumentChunker
from core.embeddings import EmbeddingGenerator
from core.vector_db import VectorDatabase
from core.rag_system import RAGSystem


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rag_system: Optional[RAGSystem] = None
embedder: Optional[EmbeddingGenerator] = None
startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """load models once at startup"""
    global rag_system, embedder, startup_time
    
    startup_time = time.time()
    vdb_dir = str(settings.vector_db_dir)
    
    try:
        embedder = EmbeddingGenerator(
            model_name=settings.embedding_model,
            batch_size=getattr(settings, "batch_size", 32),
            device=settings.device
        )
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        embedder = None
    
    # Load vector database if exists
    index_file = os.path.join(vdb_dir, "faiss_index.bin")
    meta_file = os.path.join(vdb_dir, "vector_db_metadata.pkl")
    
    if os.path.exists(index_file) and os.path.exists(meta_file):
        try:
            logger.info("Initializing RAG system (loading LLM)...")
            rag_system = RAGSystem(
                vector_db_dir=vdb_dir,
                llm_model_name=settings.llm_model,
                device=settings.device,
                retrieval_strategy="semantic",
                prompt_strategy="few_shot",
                embedder=embedder
            )
            stats = rag_system.vector_db.get_stats()
            logger.info(f"Database loaded: {stats['num_vectors']} vectors")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            import traceback
            traceback.print_exc()
            rag_system = None
    
    yield
    
    # Cleanup
    if embedder:
        del embedder
    if rag_system:
        del rag_system

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
allowed = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed if allowed else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)}
    )

@app.get("/")
async def root():
    return {
        "message": "RAG System API",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - startup_time if startup_time > 0 else 0.0
    
    if rag_system is None:
        return HealthResponse(
            status="ready",
            message="System ready for document ingestion",
            vector_db_loaded=False,
            num_vectors=0,
            num_documents=0,
            embedding_model_name=settings.embedding_model,
            llm_model_name=settings.llm_model,
            uptime_seconds=uptime
        )
    
    try:
        stats = rag_system.vector_db.get_stats()
        return HealthResponse(
            status="healthy",
            message="RAG system operational",
            vector_db_loaded=True,
            num_vectors=stats["num_vectors"],
            num_documents=stats.get("num_documents", 0),
            embedding_model_name=stats["model_name"],
            llm_model_name=settings.llm_model,
            uptime_seconds=uptime
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            message=f"System error: {str(e)}",
            vector_db_loaded=False,
            num_vectors=0,
            num_documents=0,
            embedding_model_name=settings.embedding_model,
            llm_model_name=settings.llm_model,
            uptime_seconds=uptime
        )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents into the vector database"""
    global rag_system, embedder
    
    if embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not loaded"
        )
    
    t0 = time.time()
    try:
        documents = [doc.model_dump() for doc in request.documents]
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents provided"
            )
        
        # Chunk documents
        chunker = DocumentChunker(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        chunks = chunker.process_documents(documents)
        
        # Generate embeddings
        result = embedder.process_chunks(chunks)
        
        # Load or create database
        vdb_dir = str(settings.vector_db_dir)
        index_file = os.path.join(vdb_dir, "faiss_index.bin")
        
        if os.path.exists(index_file):
            vdb = VectorDatabase(
                embedding_dim=result["embedding_dim"],
                model_name=result["model_name"],
                device=settings.device
            )
            vdb.load(vdb_dir)
            vdb.add_embeddings(result["embeddings"], result["chunks"])
        else:
            vdb = VectorDatabase(
                embedding_dim=result["embedding_dim"],
                model_name=result["model_name"],
                device=settings.device
            )
            vdb.create_index(
                result["embeddings"],
                result["chunks"],
                result["model_name"]
            )
        
        # Save database
        os.makedirs(vdb_dir, exist_ok=True)
        vdb.save(vdb_dir)
        
        # Reload RAG system
        rag_system = RAGSystem(
            vector_db_dir=vdb_dir,
            llm_model_name=settings.llm_model,
            device=settings.device,
            retrieval_strategy="semantic",
            prompt_strategy="few_shot",
            embedder=embedder
        )
        
        elapsed = time.time() - t0
        logger.info(f"Ingested {len(documents)} documents, total vectors: {vdb.index.ntotal}")
        
        return IngestResponse(
            status="success",
            message=f"Successfully ingested {len(documents)} documents",
            num_documents=len(documents),
            num_chunks=len(chunks),
            processing_time=round(elapsed, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System not ready. Please ingest documents first"
        )
    
    t0 = time.time()
    try:
        result = rag_system.query(
            question=request.query,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            return_sources=request.return_sources
        )
        
        elapsed = time.time() - t0
        
        # Build sources list
        sources: Optional[List[Source]] = None
        if request.return_sources and "retrieved_documents" in result:
            sources = []
            for doc in result["retrieved_documents"]:
                text = doc.get("text", "")
                if len(text) > 300:
                    text = text[:300] + "..."
                
                sources.append(Source(
                    title=doc.get("metadata", {}).get("title", "Unknown"),
                    score=float(doc.get("score", 0.0)),
                    text=text
                ))
        
        logger.info(f"Query processed in {elapsed:.2f}s")
        
        return QueryResponse(
            question=result.get("question", request.query),
            answer=result.get("answer", "").strip(),
            sources=sources,
            num_sources=len(sources) if sources else 0,
            processing_time=round(elapsed, 3),
            retrieval_strategy=result.get("retrieval_strategy"),
            prompt_strategy=result.get("prompt_strategy"),
            fallback_used=result.get("fallback_used", False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )