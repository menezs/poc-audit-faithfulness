import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from .embedding_service import EmbeddingService


class VectorStoreService:
    def __init__(self, embedding_service: EmbeddingService):
        self._embedding_service = embedding_service
        self._vectorstore: Optional[FAISS] = None

    @property
    def is_loaded(self) -> bool:
        return self._vectorstore is not None

    def add_documents(self, documents: List[str], metadatas: List[dict] = None) -> "VectorStoreService":
        docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas or [{}] * len(documents))
        ]
        
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self._embedding_service.client
            )
        else:
            self._vectorstore.add_documents(docs)
        
        return self

    def create_from_texts(self, texts: List[str], metadatas: List[dict] = None) -> "VectorStoreService":
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        
        self._vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=self._embedding_service.client
        )
        return self

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.is_loaded:
            raise ValueError("Vector store not loaded. Add documents first.")
        return self._vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        if not self.is_loaded:
            raise ValueError("Vector store not loaded. Add documents first.")
        return self._vectorstore.similarity_search_with_score(query, k=k)

    def similarity_search_by_vector(self, query_embedding: List[float], k: int = 4) -> List[Document]:
        if not self.is_loaded:
            raise ValueError("Vector store not loaded. Add documents first.")
        return self._vectorstore.similarity_search_by_vector(query_embedding, k=k)

    def save(self, path: str) -> "VectorStoreService":
        if not self.is_loaded:
            raise ValueError("Vector store not loaded.")
        self._vectorstore.save_local(path)
        return self

    @classmethod
    def load(cls, path: str, embedding_service: EmbeddingService) -> "VectorStoreService":
        instance = cls(embedding_service)
        instance._vectorstore = FAISS.load_local(
            path,
            embedding_service.client,
            allow_dangerous_deserialization=True
        )
        return instance

    @classmethod
    def from_documents(cls, documents: List[str], embedding_service: EmbeddingService, metadatas: List[dict] = None) -> "VectorStoreService":
        instance = cls(embedding_service)
        instance.create_from_texts(documents, metadatas)
        return instance