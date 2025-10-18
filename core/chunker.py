"""
Data Loading and Chunking Module
"""

import json
import re
from typing import List, Dict, Optional, Any
from app.config import settings


class DocumentChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = int(chunk_size or settings.chunk_size)
        self.chunk_overlap = int(chunk_overlap or settings.chunk_overlap)
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not text.strip():
            return []
        
        sentences = self.split_into_sentences(text)
        if not sentences:
            sentences = [text]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            sentence_cost = sentence_len + (1 if current_chunk else 0)
            # handle really long sentences
            
            if sentence_cost > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk, current_size = [], 0
                words = sentence.split()
                temp, temp_size = [], 0
                for w in words:
                    w_cost = len(w) + (1 if temp else 0)
                    if temp and temp_size + w_cost > self.chunk_size:
                        chunks.append(" ".join(temp))
                        temp, temp_size = [], 0
                    temp.append(w)
                    temp_size += w_cost
                if temp:
                    current_chunk = temp
                    current_size = temp_size
                continue
            
            if current_chunk and current_size + sentence_cost > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                # build overlap window by adding sentences from the end
                overlap: List[str] = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    add_cost = len(sent) + (1 if overlap else 0)
                    if overlap_size + add_cost <= self.chunk_overlap:
                        overlap.insert(0, sent)
                        overlap_size += add_cost
                    else:
                        break
                current_chunk, current_size = overlap, overlap_size

            # add sentence
            current_chunk.append(sentence)
            current_size += sentence_cost

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        out: List[Dict[str, Any]] = []
        base_meta = metadata.copy() if metadata else {}
        for i, ch in enumerate(chunks):
            md = {**base_meta, "chunk_id": i, "chunk_size": len(ch), "text": ch}
            out.append({"text": ch, "chunk_id": i, "chunk_size": len(ch), "metadata": md})
        return out
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get('text', '')
            
            if not text:
                continue
            
            metadata = {
                'doc_id': doc.get('id', f'doc_{doc_idx}'),
                'title': doc.get('title', 'Unknown'),
                'url': doc.get('url', ''),
                'category': doc.get('category', 'Unknown')
            }
            
            doc_chunks = self.chunk_text(text, metadata)
            
            for chunk in doc_chunks:
                chunk['global_chunk_id'] = len(all_chunks)
                all_chunks.append(chunk)
        
        return all_chunks


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return documents


def save_chunks(chunks: List[Dict[str, Any]], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def analyze_chunks(chunks: List[Dict[str, Any]]):
    if not chunks:
        print("No chunks to analyze")
        return
    
    total_chunks = len(chunks)
    total_chars = sum(chunk['chunk_size'] for chunk in chunks)
    avg_size = total_chars / total_chunks
    
    doc_counts = {}
    for chunk in chunks:
        doc_id = chunk['metadata']['doc_id']
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    
    print("\nChunking Analysis.")
    print(f"\nTotal chunks: {total_chunks}")
    print(f"Average chunk size: {avg_size:.0f} characters")
    print(f"Min: {min(chunk['chunk_size'] for chunk in chunks)}, Max: {max(chunk['chunk_size'] for chunk in chunks)}")
    
    print(f"\nChunks per document:")
    for doc_id, count in sorted(doc_counts.items()):
        doc_title = 'Unknown'
        for chunk in chunks:
            if chunk['metadata']['doc_id'] == doc_id:
                doc_title = chunk['metadata']['title']
                break
        print(f"  {doc_title[:50]:50} {count:3} chunks")


def preview_chunks(chunks: List[Dict[str, Any]], num_preview: int = 2):
    print("\nChunk Preview.")
    
    for i, chunk in enumerate(chunks[:num_preview]):
        print(f"\nChunk {i+1}:")
        print(f"  Document: {chunk['metadata']['title']}")
        print(f"  Size: {chunk['chunk_size']} characters")
        preview_text = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
        print(f"  Text: {preview_text}")
    
    if len(chunks) > num_preview:
        print(f"\n... and {len(chunks) - num_preview} more chunks")
