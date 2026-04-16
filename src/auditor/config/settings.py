import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    api_key: Optional[str]
    llm_model: str = "deepseek-ai/deepseek-v3.1"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    temperature: float = 0.0
    top_k_retrieval: int = 3
    embedding_cache_folder: str = "./model_cache"
    provider: str = "openai"
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_cache = os.getenv("EMBEDDING_CACHE_FOLDER", "./model_cache")
        provider = os.getenv("LLM_PROVIDER", "openai")
        base_url = os.getenv("LLM_BASE_URL")
        llm_model = os.getenv("LLM_MODEL", "google/gemma-4-e4b")
        
        if not api_key and provider != "ollama":
            raise ValueError("OPENAI_API_KEY environment variable not set (or use LLM_PROVIDER=ollama)")
        
        return cls(
            api_key=api_key,
            embedding_model=embedding_model,
            embedding_cache_folder=embedding_cache,
            provider=provider,
            base_url=base_url,
            llm_model=llm_model
        )

    @classmethod
    def from_default(cls, api_key: str) -> "Settings":
        return cls(api_key=api_key, provider="openai")