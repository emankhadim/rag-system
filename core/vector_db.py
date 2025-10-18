"""
Vector Storage and Retrieval Module 
Uses FAISS for efficient similarity search on normalized embeddings (cosine via inner product).
"""

from __future__ import annotations
import os
import pickle
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import faiss
from app.config import settings

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    FAISS-based vector database for document retrieval.
    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        embedder=None,
    ):
        self.embedding_dim: Optional[int] = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []  
        self.embedder = embedder
        self.model_name: Optional[str] = model_name or (settings.embedding_model if settings else None)
        self.device: str = (device or (settings.device if settings else "cpu")).lower()

    def set_embedder(self, embedder) -> None:
        """Set the embedder instance for reusing loaded models)"""
        self.embedder = embedder

    def create_index(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
        model_name: str,
    ) -> None:
        """
        Initialize a new index with the provided vectors and chunk metadata.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")

        if self.embedding_dim is None:
            self.embedding_dim = int(embeddings.shape[1])
        elif embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim} dim, got {embeddings.shape[1]}")

        emb = np.asarray(embeddings, dtype="float32", order="C")

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(emb)

        self.chunks = chunks
        self.model_name = model_name
        
        logger.info(f"Index created: {len(chunks)} vectors")

    def add_embeddings(
        self,
        new_embeddings: np.ndarray,
        new_chunks: List[Dict[str, Any]]
    ) -> None:
        """
        Add new embeddings to existing FAISS index.
        This allows appending documents without recreating the entire database.
        
        Args:
            new_embeddings: New embedding vectors to add
            new_chunks: Corresponding chunk data with text and metadata
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index or load first.")
        
        if len(new_embeddings) != len(new_chunks):
            raise ValueError(f"Mismatch: {len(new_embeddings)} embeddings vs {len(new_chunks)} chunks")
        
        if new_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim} dim, got {new_embeddings.shape[1]}")
        
        # Convert to proper format
        emb = np.asarray(new_embeddings, dtype="float32", order="C")
        
        # Add vectors to FAISS index
        self.index.add(emb)
        
        # Append metadata
        self.chunks.extend(new_chunks)
        
        logger.info(f"Added {len(new_embeddings)} vectors, total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
      
        if self.index is None:
            raise ValueError("Index not created. Call create_index() or load().")
        if k <= 0:
            raise ValueError("k must be positive.")

        k = min(k, self.index.ntotal)
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)

        # Normalize query vector
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.clip(norm, 1e-12, None)

        D, I = self.index.search(q, k)

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            if 0 <= idx < len(self.chunks):
                ch = self.chunks[idx]
                results.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "chunk_row": int(idx),
                        "text": ch["text"],
                        "metadata": ch["metadata"],
                        "embedding": ch.get("embedding"),
                    }
                )
        return results

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Encode a natural-language query and search the index.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty.")
        
        if self.embedder is None:
            raise ValueError(
                "No embedder provided. VectorDatabase needs a pre-loaded embedder "
                "to avoid loading models on every query."
            )
        
        qv = self.embedder.embed_single(query_text)
        return self.search(qv, k)

    def save(self, directory: str) -> None:
        """
        Save FAISS index and chunk metadata to disk.
        """
        if self.index is None:
            raise ValueError("No index to save. Call create_index() first.")
        os.makedirs(directory, exist_ok=True)

        index_path = os.path.join(directory, "faiss_index.bin")
        faiss.write_index(self.index, index_path)

        meta = {
            "chunks": self.chunks,  
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "num_vectors": int(self.index.ntotal),
            "index_type": "IndexFlatIP",
        }
        meta_path = os.path.join(directory, "vector_db_metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved index: {self.index.ntotal} vectors")

    def load(self, directory: str) -> None:
        """
        Load FAISS index and chunk metadata from directory.
        """
        index_path = os.path.join(directory, "faiss_index.bin")
        meta_path = os.path.join(directory, "vector_db_metadata.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.chunks = meta["chunks"]
        self.embedding_dim = int(meta["embedding_dim"])
        self.model_name = meta.get("model_name", self.model_name)
        
        logger.info(f"Loaded index: {self.index.ntotal} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        unique_docs = set()
        for chunk in self.chunks:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("doc_id", "")
            if doc_id:
                unique_docs.add(doc_id)
        return {
            "num_vectors": int(self.index.ntotal) if self.index else 0,
            "num_chunks": len(self.chunks),
            "num_documents": len(unique_docs),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "index_type": "IndexFlatIP (cosine)",
        }