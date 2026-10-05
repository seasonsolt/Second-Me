from typing import Optional, List
import os
import chromadb
from lpm_kernel.configs.logging import get_train_process_logger

logger = get_train_process_logger()


def get_embedding_dimension(embedding: List[float]) -> int:
    """
    Get the dimension of an embedding vector
    
    Args:
        embedding: The embedding vector
        
    Returns:
        The dimension of the embedding vector
    """
    return len(embedding)


def detect_embedding_model_dimension(model_name: str) -> Optional[int]:
    """
    Detect the dimension of an embedding model based on its name
    This is a fallback method when we can't get a sample embedding
    
    Args:
        model_name: The name of the embedding model
        
    Returns:
        The dimension of the embedding model, or None if unknown
    """
    # Common embedding model dimensions
    model_dimensions = {
        # OpenAI models
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        
        "bge-m3": 1024,
        "@cf/baai/bge-m3": 1024,

        # Ollama models
        "snowflake-arctic-embed": 768,
        "snowflake-arctic-embed:110m": 768,
        "nomic-embed-text": 768,
        "nomic-embed-text:v1.5": 768,
        "mxbai-embed-large": 1024,
        "mxbai-embed-large:v1": 1024,
    }
    
    # Try to find exact match
    if model_name in model_dimensions:
        return model_dimensions[model_name]
    
    # Try to find partial match
    for model, dimension in model_dimensions.items():
        if model in model_name:
            return dimension
    
    # Default to OpenAI dimension if unknown
    logger.warning(f"Unknown embedding model: {model_name}, defaulting to 1536 dimensions")
    return 1536


def embedding_space_key(model_name: str, provider_type: str = "", endpoint: str = "") -> str:
    """Identify a vector space without persisting credentials or endpoint URLs."""
    import hashlib
    import json
    from urllib.parse import urlsplit
    parsed = urlsplit(endpoint or "")
    # Query strings/userinfo can contain credentials and are not part of identity.
    endpoint_identity = (parsed.scheme.lower(), (parsed.hostname or "").lower(),
                         parsed.port, parsed.path.rstrip("/"))
    identity = json.dumps([model_name, provider_type or "", endpoint_identity], separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


def reinitialize_chroma_collections(dimension: int = 1536, model_name: Optional[str] = None,
                                   provider_type: str = "", endpoint: str = "") -> bool:
    """Prepare isolated model collections, preserving every existing index.

    Old calls without an embedding model are refused: dimension alone cannot
    identify a vector space and must never trigger legacy collection deletion.
    """
    if not model_name or dimension < 1:
        logger.error("An embedding model and positive dimension are required for reindexing")
        return False
    try:
        client = chromadb.PersistentClient(path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"))
        model_key = embedding_space_key(model_name, provider_type, endpoint)
        metadata = {"hnsw:space": "cosine", "dimension": dimension,
                    "embedding_model": model_name, "embedding_space": model_key}
        for prefix in ("documents", "document_chunks"):
            collection = client.get_or_create_collection(
                name=f"{prefix}_{model_key}_{dimension}", metadata=metadata)
            if collection.metadata != metadata:
                raise ValueError("Model collection metadata is incompatible")
        return True
    except Exception:
        logger.exception("Failed to prepare isolated embedding collections")
        return False
