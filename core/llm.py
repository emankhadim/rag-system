"""
Local LLM wrapper for RAG system.
Uses HuggingFace Transformers.
"""

from __future__ import annotations
from typing import Optional
import logging

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import settings

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Local LLM using HuggingFace Transformers.

    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize local LLM.
        
        Args:
            model_name: HuggingFace model name (defaults to settings.llm_model)
            device: Device for inference ('cpu' or 'cuda')
        """
        self.model_name = model_name or settings.llm_model
        self.device = (
            device or ("cuda" if torch.cuda.is_available() else settings.device or "cpu")
        ).lower()

        logger.info(f"Loading LLM: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"LLM loaded: {param_count:.1f}M parameters")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        min_new_tokens: int = 40,
        temperature: float = 0.0,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.05,
    ) -> str:
        """
        Generate text from prompt with post-processing cleanup.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            no_repeat_ngram_size: Prevent n-gram repetition
            repetition_penalty: Penalize token repetition
            
        Returns:
            Generated text with artifacts cleaned
        """
        # Tokenize input
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(512, self.tokenizer.model_max_length),
        )
        
        # Check if truncation happened
        if enc['input_ids'].shape[1] >= 512:
            logger.warning("Prompt truncated")
        
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # Configure generation
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )
        if do_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=temperature))
        else:
            gen_kwargs.update(dict(do_sample=False))

        # Generate and decode
        outputs = self.model.generate(**enc, **gen_kwargs)
        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to clean FLAN-T5 artifacts
        return self._clean_generated_text(raw_text)
    
    @staticmethod
    def _clean_generated_text(text: str) -> str:
        """
        Clean up common LLM generation artifacts.
        Removes spurious quotes, brackets, and normalizes whitespace.
        """
        cleaned = text.strip()
        cleaned = cleaned.replace('"""', '').replace("'''", '')
        cleaned = cleaned.replace('[', '').replace(']', '')
        cleaned = ' '.join(cleaned.split()) 
        return cleaned