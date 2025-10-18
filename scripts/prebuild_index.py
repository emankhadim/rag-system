"""
Vector Database Pre-builder
One-shot script to chunk documents, generate embeddings and build FAISS index.

Usage:
    python -m scripts.prebuild_index
"""

from __future__ import annotations
import os
import sys
import time
import logging
from pathlib import Path

HF_HOME = os.getenv('HF_HOME', './hf-cache')
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', './hf-cache/transformers')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './hf-cache/sentence_transformers')
os.environ['HF_DATASETS_CACHE'] = f'{HF_HOME}/datasets'
os.environ['TORCH_HOME'] = f'{HF_HOME}/torch'

from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.config import settings
from core.chunker import DocumentChunker, load_documents
from core.embeddings import EmbeddingGenerator, save_embeddings
from core.vector_db import VectorDatabase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def validate_environment() -> None:
    """Validate required directories exist."""
    os.makedirs(str(settings.embeddings_dir), exist_ok=True)
    os.makedirs(str(settings.vector_db_dir), exist_ok=True)


def load_and_validate_documents(raw_json_path: Optional[str] = None) -> list:
    """Load and validate documents from JSON file."""
    if raw_json_path:
        raw_path = Path(raw_json_path)
    else:
        raw_path = settings.raw_data_dir / "wikipedia_documents.json"
    
    logger.info(f"Loading: {raw_path}")
    
    if not raw_path.exists():
        raise FileNotFoundError(f"File not found: {raw_path}")
    
    docs = load_documents(str(raw_path))
    
    if not docs:
        raise ValueError("No documents loaded")
    
    logger.info(f"Loaded {len(docs)} documents")
    return docs


def chunk_documents(docs: list) -> list:
    """Chunk documents into smaller pieces."""
    logger.info("Chunking documents...")
    chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
    chunks = chunker.process_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def generate_embeddings(chunks: list) -> dict:
    """Generate embeddings for chunks."""
    logger.info("Generating embeddings...")
    
    embedder = EmbeddingGenerator(
        model_name=settings.embedding_model,
        batch_size=getattr(settings, "batch_size", 64),
        device=settings.device,
    )
    
    result = embedder.process_chunks(chunks)
    logger.info(f"Generated {len(chunks)} embeddings")
    return result


def save_embedding_data(result: dict) -> None:
    """Save embeddings to disk."""
    save_embeddings(result, str(settings.embeddings_dir))
    logger.info("Embeddings saved")


def build_vector_database(result: dict) -> VectorDatabase:
    """Build FAISS vector database."""
    logger.info("Building index...")
    
    vdb = VectorDatabase(
        embedding_dim=result["embedding_dim"],
        model_name=result["model_name"],
        device=settings.device,
    )
    
    vdb.create_index(
        result["embeddings"], 
        result["chunks"], 
        result["model_name"]
    )
    
    return vdb


def save_vector_database(vdb: VectorDatabase) -> None:
    """Save vector database to disk."""
    vdb.save(str(settings.vector_db_dir))
    logger.info(f"Database saved: {settings.vector_db_dir}")


def print_summary(total_docs: int, total_chunks: int, total_time: float) -> None:
    """Print summary of prebuild process."""
    print("Prebuild Complete")
    print(f"Documents: {total_docs}")
    print(f"Chunks: {total_chunks}")
    print(f"Time: {total_time:.1f}s")
    print(f"Ready: {settings.vector_db_dir}\n")


def prebuild(raw_json_path: Optional[str] = None) -> None:
    """Main prebuild pipeline."""
    overall_start = time.time()
    
    try:
        validate_environment()
        docs = load_and_validate_documents(raw_json_path)
        chunks = chunk_documents(docs)
        result = generate_embeddings(chunks)
        save_embedding_data(result)
        vdb = build_vector_database(result)
        save_vector_database(vdb)
        
        total_time = time.time() - overall_start
        print_summary(len(docs), len(chunks), total_time)
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def main():

    print("\nRAG SYSTEM - DATABASE BUILDER\n")
    custom_path = sys.argv[1] if len(sys.argv) > 1 else None
    prebuild(custom_path)

if __name__ == "__main__":
    main()