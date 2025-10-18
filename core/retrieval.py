"""
Retrieval strategies for RAG system.
Includes semantic search and MMR re-ranking 
"""

from __future__ import annotations
from typing import List, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class RetrievalStrategy:
    """Base class for retrieval strategies."""
    
    def retrieve(self, vector_db, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve documents from vector database."""
        raise NotImplementedError


class SemanticRetrieval(RetrievalStrategy):
    """Standard semantic similarity search."""
    
    def retrieve(self, vector_db, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve top-k most similar documents."""
        docs = vector_db.query(query, k=k)
        logger.info(f"Retrieved {len(docs)} documents via semantic search")
        return docs


class MMRRetrieval(RetrievalStrategy):
    """
    Maximal Marginal Relevance (MMR) retrieval.
    
    Balances relevance to query with diversity among results.
    Prevents redundant information in retrieved documents.
    
    Algorithm:
    1. Retrieve initial candidates (more than needed)
    2. Select most relevant document first
    3. Iteratively select documents that maximize:
       MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    """
    
    def __init__(self, lambda_param: float = 0.7, candidate_multiplier: int = 2):
        """
        Initialize MMR retrieval.
        
        Args:
            lambda_param: Trade-off between relevance and diversity (0-1)
                         1.0 = only relevance, 0.0 = only diversity
            candidate_multiplier: Retrieve k * multiplier candidates for re-ranking
        """
        self.lambda_param = lambda_param
        self.candidate_multiplier = candidate_multiplier
    
    def retrieve(self, vector_db, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieve documents using MMR re-ranking.
        
        Args:
            vector_db: Vector database instance
            query: Search query
            k: Number of final documents to return
            
        Returns:
            Re-ranked list of k documents
        """
        initial_k = k * self.candidate_multiplier
        candidates = vector_db.query(query, k=initial_k)
        
        logger.info(f"Retrieved {len(candidates)} candidates for MMR re-ranking")
        
        reranked = self._mmr_rerank(candidates, k)
        logger.info(f"Re-ranked to {len(reranked)} diverse documents (λ={self.lambda_param})")
        
        return reranked
    
    def _mmr_rerank(self, docs: List[Dict], final_k: int) -> List[Dict]:
        """
        Apply MMR algorithm to re-rank documents.
        
        Args:
            docs: Candidate documents with scores and embeddings
            final_k: Number of documents to select
            
        Returns:
            Re-ranked documents
        """
        if len(docs) <= final_k:
            return docs
        
        selected = []
        remaining = docs.copy()
        
        # Always select most relevant document first
        selected.append(remaining.pop(0))
        
        # Iteratively select documents maximizing MMR score
        while len(selected) < final_k and remaining:
            best_score = -float('inf')
            best_idx = 0
            
            for idx, doc in enumerate(remaining):
                # Relevance component (from similarity search)
                relevance = doc.get('score', 0.0)
                
                # Diversity component (max similarity to already selected)
                max_sim = max(
                    self._cosine_similarity(doc, sel_doc) 
                    for sel_doc in selected
                )
                
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
            logger.debug(f"MMR selected: {selected[-1]['metadata'].get('title')} (score: {best_score:.3f})")
        
        return selected
    
    @staticmethod
    def _cosine_similarity(doc1: Dict, doc2: Dict) -> float:
        """
        Calculate cosine similarity between document embeddings.
        """
        emb1 = doc1.get('embedding')
        emb2 = doc2.get('embedding')
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        return float(dot_product / norm_product) if norm_product > 0 else 0.0


def get_retrieval_strategy(strategy: str = "semantic", **kwargs) -> RetrievalStrategy:
    """
    Factory function to get retrieval strategy.
    """
    if strategy == "mmr":
        return MMRRetrieval(
            lambda_param=kwargs.get('lambda_param', 0.7),
            candidate_multiplier=kwargs.get('candidate_multiplier', 2)
        )
    else:  # Default to semantic
        return SemanticRetrieval()