"""
Application configuration using Pydantic Settings.
Supports environment variables via .env file.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with automatic environment variable loading.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Base directories
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Project root directory"
    )
    data_dir: Optional[Path] = Field(
        default=None,
        description="Main data directory"
    )
    raw_data_dir: Optional[Path] = Field(
        default=None,
        description="Raw documents directory"
    )
    processed_data_dir: Optional[Path] = Field(
        default=None,
        description="Processed data directory"
    )
    embeddings_dir: Optional[Path] = Field(
        default=None,
        description="Embeddings storage directory"
    )
    vector_db_dir: Optional[Path] = Field(
        default=None,
        description="FAISS vector database directory"
    )
    
    # Model configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model name"
    )
    llm_model: str = Field(
        default="google/flan-t5-base",
        description="HuggingFace LLM model name"
    )
    device: str = Field(
        default="cpu",
        description="Device for model inference (cpu or cuda)"
    )
    
    # Chunking configuration
    chunk_size: int = Field(
        default=600,
        ge=50,
        le=2000,
        description="Size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=80,
        ge=0,
        le=500,
        description="Overlap between chunks in characters"
    )
    
    # Retrieval configuration
    similarity: str = Field(
        default="cosine",
        description="Similarity metric for vector search"
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation",
        alias="embed_batch"
    )
    faiss_index: str = Field(
        default="IndexFlatIP",
        description="FAISS index type"
    )
    
    # API configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port"
    )
    api_reload: bool = Field(
        default=False,
        description="Enable hot reload for development"
    )
    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="CORS allowed origins (comma-separated)"
    )
    
    use_extractive_baseline: bool = Field(
        default=True,
        description="Enable extractive baseline fallback"
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 600)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string."""
        v = v.lower()
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"device must be 'cpu', 'cuda', or 'mps', got '{v}'")
        return v
    
    @model_validator(mode="after")
    def initialize_paths(self) -> "Settings":
        """Initialize directory paths and create them if they don't exist."""
        # Set default paths if not provided
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_dir / "raw"
        
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_dir / "processed"
        
        if self.embeddings_dir is None:
            self.embeddings_dir = self.processed_data_dir / "embeddings"
        
        if self.vector_db_dir is None:
            self.vector_db_dir = self.processed_data_dir / "vector_db"
        
        # Create directories if they don't exist
        for directory in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.embeddings_dir,
            self.vector_db_dir
        ]:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
        
        return self
    
    def get_origins_list(self) -> list[str]:
        """Parse allowed origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


# Global settings instance
settings = Settings()