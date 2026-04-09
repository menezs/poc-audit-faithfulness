from typing import List
from langchain_core.documents import Document
from ..services.vector_store import VectorStoreService


class Retriever:
    def __init__(self, vector_store: VectorStoreService, top_k: int = 3):
        self._vector_store = vector_store
        self._top_k = top_k

    @property
    def vector_store(self) -> VectorStoreService:
        return self._vector_store

    def retrieve(self, query: str, k: int = None) -> List[tuple[str, float]]:
        k = k or self._top_k
        results = self._vector_store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score) for doc, score in results]

    def retrieve_texts(self, query: str, k: int = None) -> List[str]:
        results = self.retrieve(query, k)
        return [text for text, score in results]

    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        k = k or self._top_k
        return self._vector_store.similarity_search(query, k=k)