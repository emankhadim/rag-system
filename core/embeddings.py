"""
Embedding Generator for RAG System
"""
import gc
import json
import os
import time
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingGenerator:
    def __init__(self, model_name: str | None = None, batch_size: int | None = None, device: str | None = None):
    
        self.model_name = model_name or (settings.embedding_model if settings else "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = int(batch_size or (settings.batch_size if settings else 32))
        self.device = (device or (settings.device if settings else "cpu")).lower()

        print(f"\nLoading embedding model: {self.model_name} (device={self.device}, batch_size={self.batch_size})")
        t0 = time.time()

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded in {time.time() - t0:.2f}s | dim={self.embedding_dim}")

    def embed_single(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        vec = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
       
        return vec.astype(np.float32, copy=False)

    def embed_batch(self, texts: List[str], batch_size: int | None = None, show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        bsz = int(batch_size or self.batch_size)
        print(f"\nGenerating embeddings for {len(texts)} chunks (batch_size={bsz})")
        t0 = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=bsz,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        gc.collect() 
        elapsed = max(time.time() - t0, 1e-9)
        print(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s ({len(texts)/elapsed:.1f} chunks/s)")
        return embeddings

    def process_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        if not chunks:
            raise ValueError("chunks cannot be empty")
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_batch(texts)
        assert embeddings.shape[0] == len(chunks), "Embedding count mismatch"
        assert embeddings.shape[1] == self.embedding_dim, "Embedding dimension mismatch"
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(chunks),
        }


def load_chunks(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(result: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    np.save(emb_path, result["embeddings"])

    chunks_path = os.path.join(output_dir, "chunks_with_metadata.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(result["chunks"], f, indent=2, ensure_ascii=False)

    meta = {
        "model_name": result["model_name"],
        "embedding_dim": result["embedding_dim"],
        "num_chunks": result["num_chunks"],
        "embeddings_shape": list(result["embeddings"].shape),
    }
    meta_path = os.path.join(output_dir, "embedding_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings to {output_dir}/")


def load_embeddings(embeddings_dir: str) -> Dict:
    embeddings = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    with open(os.path.join(embeddings_dir, "chunks_with_metadata.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(embeddings_dir, "embedding_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "model_name": metadata["model_name"],
        "embedding_dim": metadata["embedding_dim"],
        "num_chunks": metadata["num_chunks"],
    }


