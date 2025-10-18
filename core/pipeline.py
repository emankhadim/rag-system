"""
RAG Pipeline orchestration.
Combines retrieval, prompting and generation into end-to-end workflow.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import logging

from app.config import settings
from core.llm import LocalLLM
from core.prompts import get_prompt_strategy, PromptStrategy
from core.retrieval import get_retrieval_strategy, RetrievalStrategy

logger = logging.getLogger(__name__)


def _normalize_query(q: str) -> str:
    """Normalize query to fix common typos and formatting."""
    q = (q or "").strip()
    q = q.replace("heathcare", "healthcare")
    q = q.replace("Ai", "AI")
    return q


class RAGPipeline:
    """
    Complete RAG pipeline with configurable strategies.

    Orchestrates:
    - Query normalization
    - Document retrieval (semantic or MMR)
    - Context building within a token budget
    - Prompt construction (simple, few_shot, chain_of_thought)
    - LLM generation with validation + graceful fallback
    """

    def __init__(
        self,
        vector_db,
        llm: LocalLLM,
        retrieval_strategy: str = "semantic",
        prompt_strategy: str = "few_shot",
        **strategy_kwargs
    ):
        """
        Initialize RAG pipeline with strategies.

        Args:
            vector_db: Vector store with .query(text, k) -> List[docs]
            llm: Local language model instance
            retrieval_strategy: 'semantic' or 'mmr'
            prompt_strategy: 'simple', 'few_shot', or 'chain_of_thought'
            **strategy_kwargs: Additional parameters for strategies
        """
        self.vector_db = vector_db
        self.llm = llm

        # Strategy plug-ins
        self.retrieval: RetrievalStrategy = get_retrieval_strategy(retrieval_strategy, **strategy_kwargs)
        self.prompt:    PromptStrategy    = get_prompt_strategy(prompt_strategy)

        self.retrieval_strategy_name = retrieval_strategy
        self.prompt_strategy_name = prompt_strategy

        logger.info(f"Pipeline: {retrieval_strategy}/{prompt_strategy}")

    def _build_context(self, docs: List[Dict[str, Any]], budget_tokens: int = 420) -> str:
        """
        Build a compact context string from retrieved docs within a token budget.
        Uses the LLM tokenizer for a rough token estimate.
        """
        parts: List[str] = []
        used_tokens = 0

        for i, doc in enumerate(docs, 1):
            title = doc.get("metadata", {}).get("title", f"Doc {i}")
            text  = (doc.get("text", "") or "").strip()
            if not text:
                continue

            fragment = f"[{title}]: {text}\n"
            est_tokens = len(self.llm.tokenizer.encode(fragment, add_special_tokens=False))

            if used_tokens + est_tokens > budget_tokens:
                break

            parts.append(fragment)
            used_tokens += est_tokens

        return "\n".join(parts)

    def query(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        max_new_tokens: int = 150,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end RAG query.

        Args:
            question: User's question
            top_k: documents to retrieve
            max_new_tokens: generation cap
            temperature: 0.0 for deterministic

        """
        k = int(top_k or settings.top_k)
        q_norm = _normalize_query(question)

        try:
            # 1) Retrieve
            retrieved_docs = self.retrieval.retrieve(self.vector_db, q_norm, k=k)
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return self._empty_result(question)

            # 2) Context
            context = self._build_context(retrieved_docs, budget_tokens=420)

            # 3) Prompt 
            prompt = self.prompt.build_prompt(q_norm, context)

            # 4) Generate 
            raw_answer = self.llm.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            ).strip()

            # 5) Validate
            answer, is_fallback = self._validate_answer(raw_answer, retrieved_docs)

            return {
                "question": question,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "num_documents_used": len(retrieved_docs),
                "model": self.llm.model_name,
                "fallback_used": is_fallback,
                "retrieval_strategy": self.retrieval_strategy_name,
                "prompt_strategy": self.prompt_strategy_name,
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "error": str(e),
                "fallback_used": True,
            }

    #helper

    def _validate_answer(
        self,
        raw_answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Tuple[str, bool]:
        """
        Basic quality check. If the model outputs a very short or degenerate
        string, build a fallback paragraph from the top docs.
        """
        too_short = len(raw_answer) < 25
        non_alpha = not any(c.isalpha() for c in raw_answer)
        repetitive = False
        if raw_answer:
            first = raw_answer.split()[0]
            repetitive = raw_answer.count(first) > 5

        if too_short or non_alpha or repetitive:
            logger.warning("Using fallback answer")
            return self._create_fallback_paragraph(retrieved_docs), True

        return raw_answer, False

    def _create_fallback_paragraph(self, docs: List[Dict[str, Any]]) -> str:
        """
        Concise paragraph fallback derived from the top retrieved docs.
        """
        pieces: List[str] = []
        for i, doc in enumerate(docs[:3], 1):
            title = doc.get("metadata", {}).get("title", f"Doc {i}")
            txt = (doc.get("text", "") or "").replace("\n", " ").strip()
            if not txt:
                continue
            snippet = txt[:220].rstrip()
            if len(txt) > 220:
                snippet += "..."
            pieces.append(f"From {title}: {snippet}")

        if not pieces:
            return "I couldn't extract enough context to answer confidently."

        return " ".join(pieces)

    def _empty_result(self, question: str) -> Dict[str, Any]:
        return {
            "question": question,
            "answer": "I couldn't find relevant information to answer your question.",
            "retrieved_documents": [],
            "num_documents_used": 0,
            "fallback_used": True,
        }

    @staticmethod
    def format_response(result: Dict[str, Any]) -> str:
        """
        Pretty format for console logs or CLI demo (no bullets in the answer).
        """
        lines = []
        lines.append(f"\nQ: {result.get('question','')}")
        lines.append(f"\nA: {result.get('answer','')}\n")

        if result.get("fallback_used"):
            lines.append("(Note: fallback answer used)\n")

        if result.get("retrieval_strategy"):
            lines.append(f"Retrieval: {result['retrieval_strategy']}")
        if result.get("prompt_strategy"):
            lines.append(f"Prompt: {result['prompt_strategy']}")

        lines.append("\nSources:")
        for i, doc in enumerate(result.get("retrieved_documents", []), 1):
            title = doc.get("metadata", {}).get("title", "Unknown")
            score = doc.get("score", 0.0)
            lines.append(f"  Doc {i} — {title} (score: {score:.3f})")

        return "\n".join(lines)