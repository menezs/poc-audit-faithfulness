from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingService:
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: str = "./model_cache"
    ):
        self._model = model
        self._cache_folder = cache_folder
        self._client: HuggingFaceEmbeddings = None

    @property
    def client(self) -> HuggingFaceEmbeddings:
        if self._client is None:
            self._client = HuggingFaceEmbeddings(
                model_name=self._model,
                cache_folder=self._cache_folder
            )
        return self._client