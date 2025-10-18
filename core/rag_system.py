"""
Complete RAG System.
Loads FAISS vector DB and orchestrates RAG pipeline.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging

from app.config import settings
from core.vector_db import VectorDatabase
from core.llm import LocalLLM
from core.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class RAGSystem:
   
    def __init__(
        self,
        vector_db_dir: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        device: Optional[str] = None,
        retrieval_strategy: str = "semantic",
        prompt_strategy: str = "few_shot",
        embedder=None,
        **strategy_kwargs
    ):
        """
        Initialize RAG system.
        
        Args:
            vector_db_dir: Path to FAISS index directory
            llm_model_name: HuggingFace model name
            device: Device for LLM ('cpu' or 'cuda')
            retrieval_strategy: 'semantic' or 'mmr'
            prompt_strategy: 'simple', 'few_shot', or 'chain_of_thought'
            **strategy_kwargs: Additional parameters (e.g., lambda_param for MMR)
        """
        # Load vector database
        vdb_path = vector_db_dir or str(settings.vector_db_dir)
        self.vector_db = VectorDatabase(embedder=embedder)
        self.vector_db.load(vdb_path)
        logger.info(f"Vector DB loaded: {vdb_path}")
        
        # Initialize LLM
        self.llm = LocalLLM(
            model_name=(llm_model_name or settings.llm_model), 
            device=device
        )
        
        # Initialize pipeline with strategies
        self.pipeline = RAGPipeline(
            self.vector_db,
            self.llm,
            retrieval_strategy=retrieval_strategy,
            prompt_strategy=prompt_strategy,
            **strategy_kwargs
        )

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents without generating an answer.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with metadata and scores
        """
        k = int(top_k or settings.top_k)
        
        try:
            docs = self.vector_db.query(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def query(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        return_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end RAG query: retrieve + generate answer.
        """
        if not question or len(question.strip()) < 3:
            return {
                "question": question,
                "answer": "Please provide a more detailed question.",
                "error": "Query too short",
            }
        
        try:
            k = int(top_k or settings.top_k)
            result = self.pipeline.query(
                question,
                top_k=k,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            if not return_sources:
                result.pop("retrieved_documents", None)
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": "An error occurred while processing your question.",
                "error": str(e),
                "fallback_used": True,
            }

    def format_response(self, result: Dict[str, Any]) -> str:
        return self.pipeline.format_response(result)