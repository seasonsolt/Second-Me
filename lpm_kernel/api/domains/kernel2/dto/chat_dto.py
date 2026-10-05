"""
Chat-related DTO objects
"""
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, model_validator

class ChatRequest(BaseModel):
    """Chat request in OpenAI-compatible format"""
    # Core OpenAI API fields
    messages: List[Dict[str, str]]  # OpenAI compatible messages array
    model: Optional[str] = None  # Model identifier
    temperature: float = 0.1  # Temperature parameter for controlling randomness
    max_tokens: int = 2000  # Maximum tokens to generate
    stream: bool = True  # Whether to stream response
    
    # Metadata for request processing - contains extension parameters
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)  # Additional parameters for LLM request processing


    @model_validator(mode="before")
    @classmethod
    def normalize_metadata(cls, values):
        """Accept legacy flags while keeping metadata as the canonical contract."""
        if not isinstance(values, dict):
            return values
        values = dict(values)
        metadata = dict(values.get("metadata") or {})
        for name in ("role_id", "enable_l0_retrieval", "enable_l1_retrieval"):
            if name in values and name not in metadata:
                metadata[name] = values.pop(name)
        values["metadata"] = metadata
        return values
