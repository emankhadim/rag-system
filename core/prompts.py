"""
Prompt engineering strategies for RAG system.
"""

from __future__ import annotations
from typing import List, Dict, Any


class PromptStrategy:
    """Base class for prompt strategies."""
    
    def build_prompt(self, query: str, context: str) -> str:
        """Build prompt from query and context."""
        raise NotImplementedError


class SimplePrompt(PromptStrategy):
    """Simple, direct prompt without examples."""
    
    def build_prompt(self, query: str, context: str) -> str:
        return f"""Answer the question based on the context below. Write 3-4 complete sentences.

Context:
{context}

Question: {query}

Answer:"""


class FewShotPrompt(PromptStrategy):
    """
    Few-shot prompting with examples.
    Teaches the model desired response format through demonstration.
    """
    
    def build_prompt(self, query: str, context: str) -> str:
        return f"""Answer questions using the provided context. Write 3-4 complete sentences in a natural paragraph.

Example 1:
Context: [Machine Learning] Machine learning is a branch of AI that enables systems to learn from data without explicit programming.
Question: What is machine learning?
Answer: Machine learning is a subset of artificial intelligence that focuses on enabling computer systems to learn and improve from experience. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns. This technology powers applications ranging from recommendation systems to autonomous vehicles.

Example 2:
Context: [Python] Python is a high-level programming language known for its simplicity and extensive libraries.
Question: Describe Python programming language.
Answer: Python is a high-level, interpreted programming language that emphasizes code readability and simplicity. It features dynamic typing and automatic memory management, making it accessible for beginners while powerful enough for complex applications. Python is widely used in web development, data science, artificial intelligence, and scientific computing.

Now answer this question:

Context:
{context}

Question: {query}

Answer:"""


class ChainOfThoughtPrompt(PromptStrategy):
    """
    Chain-of-thought prompting for complex reasoning.
    Encourages step-by-step thinking.
    """
    
    def build_prompt(self, query: str, context: str) -> str:
        return f"""Answer the question using the context below. Think step-by-step and provide a clear, logical answer in 3-4 sentences.

Context:
{context}

Question: {query}

Let's think step by step:
Answer:"""


def get_prompt_strategy(strategy: str = "few_shot") -> PromptStrategy:
    """
    Factory function to get prompt strategy.
    
    Args:
        strategy: One of 'simple', 'few_shot', 'chain_of_thought'
        
    Returns:
        PromptStrategy instance
    """
    strategies = {
        "simple": SimplePrompt(),
        "few_shot": FewShotPrompt(),
        "chain_of_thought": ChainOfThoughtPrompt(),
    }
    return strategies.get(strategy, FewShotPrompt())