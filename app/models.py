"""
Pydantic models for request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from app.config import settings


class Document(BaseModel):
    
    id: str = Field(
        ..., 
        description="Unique document identifier",
        min_length=1,
        max_length=255,
        example="doc_001"
    )
    title: str = Field(
        ..., 
        description="Document title",
        min_length=1,
        max_length=500,
        example="Artificial Intelligence Overview"
    )
    text: str = Field(
        ..., 
        description="Document content",
        min_length=10,
        example="Artificial intelligence is the simulation of human intelligence.."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (optional)",
        example={"source": "Wikipedia", "category": "AI"}
    )
    
    @field_validator('text')
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        """Ensure text is not empty or too short."""
        if len(v.strip()) < 10:
            raise ValueError('Document text must be at least 10 characters')
        return v


class IngestRequest(BaseModel):
    """Request model for /ingest endpoint."""
    
    documents: List[Document] = Field(
        ..., 
        description="List of documents to ingest",
        min_length=1,
        max_length=1000
    )
    chunk_size: int = Field(
        default=settings.chunk_size,
        description="Size of text chunks in tokens",
        ge=50,
        le=2000,
        example=500
    )
    chunk_overlap: int = Field(
        default=settings.chunk_overlap,
        description="Overlap between chunks in tokens",
        ge=0,
        le=500,
        example=50
    )
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get('chunk_size', settings.chunk_size)
        if v >= chunk_size:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": "doc_001",
                        "title": "Machine Learning Basics",
                        "text": "Machine learning is a subset of AI that enables systems to learn from data...",
                        "metadata": {"category": "AI", "source": "Internal"}
                    }
                ],
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }


class IngestResponse(BaseModel):
    """Response model for /ingest endpoint."""
    
    status: str = Field(
        ...,
        description="Status of the operation",
        example="success"
    )
    message: str = Field(
        ...,
        description="Human-readable message",
        example="Successfully ingested 5 documents"
    )
    num_documents: int = Field(
        ...,
        description="Number of documents processed",
        ge=0,
        example=5
    )
    num_chunks: int = Field(
        ...,
        description="Total number of chunks created",
        ge=0,
        example=127
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds",
        ge=0.0,
        example=2.45
    )


class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    
    query: str = Field(
        ...,
        description="User query or question",
        min_length=3,
        max_length=1000,
        example="What is artificial intelligence?"
    )
    top_k: int = Field(
        default=settings.top_k,
        description="Number of documents to retrieve",
        ge=1,
        le=20,
        example=4
    )
    return_sources: bool = Field(
        default=True,
        description="Include source documents in response",
        example=True
    )
    max_new_tokens: int = Field(
        default=150,
        description="Maximum tokens to generate in answer",
        ge=50,
        le=512,
        example=150
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature (0.0 = deterministic)",
        ge=0.0,
        le=1.0,
        example=0.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main applications of AI in healthcare?",
                "top_k": 4,
                "return_sources": True,
                "max_new_tokens": 150,
                "temperature": 0.0
            }
        }


class Source(BaseModel):
    """Source document in query response."""
    
    title: str = Field(
        ...,
        description="Document title",
        example="AI in Healthcare"
    )
    score: float = Field(
        ...,
        description="Similarity score",
        ge=0.0,
        le=1.0,
        example=0.847
    )
    text: str = Field(
        ...,
        description="Relevant text chunk",
        example="AI is being used in medical imaging to detect diseases earlier..."
    )


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    
    question: str = Field(
        ...,
        description="Original question",
        example="What is artificial intelligence?"
    )
    answer: str = Field(
        ...,
        description="Generated answer",
        example="Artificial intelligence is the capability of computational systems..."
    )
    sources: Optional[List[Source]] = Field(
        default=None,
        description="Source documents used for answer"
    )
    num_sources: int = Field(
        default=0,
        description="Number of source documents",
        ge=0,
        example=4
    )
    processing_time: float = Field(
        ...,
        description="Query processing time in seconds",
        ge=0.0,
        example=1.23
    )
    retrieval_strategy: Optional[str] = Field(
        default=None,
        description="Retrieval strategy used",
        example="semantic"
    )
    prompt_strategy: Optional[str] = Field(
        default=None,
        description="Prompt strategy used",
        example="few_shot"
    )
    fallback_used: bool = Field(
        default=False,
        description="Whether fallback answer was used",
        example=False
    )


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    
    status: str = Field(
        ...,
        description="Health status",
        example="healthy"
    )
    message: str = Field(
        ...,
        description="Status message",
        example="RAG system is operational"
    )
    vector_db_loaded: bool = Field(
        ...,
        description="Whether vector database is loaded",
        example=True
    )
    num_vectors: int = Field(
        ...,
        description="Number of vectors in database",
        ge=0,
        example=1247
    )
    num_documents: int = Field(  
        default=0,
        description="Number of unique original documents",
        ge=0
    )
    embedding_model_name: str = Field(
        ...,
        description="Name of embedding model",
        example="sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model_name: str = Field(
        ...,
        description="Name of LLM model",
        example="google/flan-t5-small"
    )
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="System uptime in seconds",
        ge=0.0,
        example=3600.5
    )


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    
    status: str = Field(
        default="error",
        description="Error status",
        example="error"
    )
    message: str = Field(
        ...,
        description="Error message",
        example="Query failed: Vector database not loaded"
    )
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information",
        example="FileNotFoundError: vector_db directory not found"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Application-specific error code",
        example="VDB_NOT_FOUND"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid query: text too short",
                "detail": "Query must be at least 3 characters",
                "error_code": "INVALID_INPUT"
            }
        }