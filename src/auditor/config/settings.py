import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    api_key: Optional[str]
    llm_model: str = "deepseek-ai/deepseek-v3.1"
    embedding_model: str = "nvidia/nv-embed-v1"
    temperature: float = 0.0
    top_k_retrieval: int = 3
    use_local_embedding: bool = False
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("NVIDIA_API_KEY")
        use_local = os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true"
        local_model = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        if not use_local and not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable not set (or set USE_LOCAL_EMBEDDING=true)")
        
        return cls(
            api_key=api_key,
            use_local_embedding=use_local,
            local_embedding_model=local_model
        )

    @classmethod
    def from_default(cls, api_key: str) -> "Settings":
        return cls(api_key=api_key)