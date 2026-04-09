from typing import Optional, Union
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nvidia/nv-embed-v1",
        use_local: bool = False,
        local_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self._api_key = api_key
        self._model = model
        self._use_local = use_local
        self._local_model_name = local_model_name
        self._client: Optional[Union[NVIDIAEmbeddings, HuggingFaceEmbeddings]] = None

    @property
    def client(self) -> Union[NVIDIAEmbeddings, HuggingFaceEmbeddings]:
        if self._client is None:
            if self._use_local:
                self._client = HuggingFaceEmbeddings(
                    model_name=self._local_model_name,
                    cache_folder='./model_cache'
                )
            else:
                if not self._api_key:
                    raise ValueError("API key required for NVIDIA embeddings")
                self._client = NVIDIAEmbeddings(
                    model=self._model,
                    nvidia_api_key=self._api_key
                )
        return self._client