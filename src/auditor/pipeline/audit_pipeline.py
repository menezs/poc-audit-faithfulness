from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from ..models.claim import Claim
from ..models.verification_result import VerificationResult, VerificationLabel, VerificationLabelNew, VerificationResultNew
from ..core import ClaimExtractor, Retriever, Verifier, Scorer
from ..config.settings import Settings
from ..services import LLMService, EmbeddingService, VectorStoreService


@dataclass
class AuditResult:
    score: float
    results: List[VerificationResultNew]
    claims: List[Claim]
    total_supported: int
    total_partial: int
    total_not_supported: int

    @property
    def is_fully_supported(self) -> bool:
        return self.total_not_supported == 0 and self.total_partial == 0

    def summary(self) -> str:
        lines = [
            f"Score: {self.score:.2f}",
            f"Supported: {self.total_supported}",
            f"Partial: {self.total_partial}",
            f"Not Supported: {self.total_not_supported}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AuditResult(score={self.score:.2f}, claims={len(self.claims)})"


class AuditPipeline:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._llm_service = LLMService(
            api_key=settings.api_key,
            model=settings.llm_model,
            temperature=settings.temperature,
            provider=settings.provider,
            base_url=settings.base_url
        )
        self._embedding_service = EmbeddingService(
            model=settings.embedding_model,
            cache_folder=settings.embedding_cache_folder
        )
        self._vector_store = VectorStoreService(self._embedding_service)

        self._extractor = ClaimExtractor(self._llm_service)
        self._retriever: Optional[Retriever] = None
        self._verifier = Verifier(self._llm_service)
        self._scorer = Scorer()

    @classmethod
    def create(cls, api_key: str) -> "AuditPipeline":
        settings = Settings.from_default(api_key)
        return cls(settings)

    @classmethod
    def create_from_env(cls) -> "AuditPipeline":
        settings = Settings.from_env()
        return cls(settings)

    def _ensure_retriever(self, documents: List[str]) -> Retriever:
        self._vector_store.create_from_texts(documents)
        self._retriever = Retriever(self._vector_store, top_k=self._settings.top_k_retrieval)
        return self._retriever

    def _process_answer(self, answer: str) -> str:
        if Path(answer).suffix == ".md" and Path(answer).exists():
            with open(answer, "r", encoding="utf-8") as f:
                return f.read()
        return answer

    def _process_documents(self, documents: List[str]) -> List[str]:
        processed = []
        for doc in documents:
            if Path(doc).suffix == ".md" and Path(doc).exists():
                with open(doc, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():
                        processed.append(content)
            else:
                processed.append(doc)
        return processed

    def audit(self, answer: str, documents: List[str]) -> AuditResult:
        answer_text = self._process_answer(answer)
        processed_docs = self._process_documents(documents)

        claims = self._extractor.extract(answer_text)

        retriever = self._ensure_retriever(processed_docs)
        verification_results = []

        print("\nVerifying claims...")
        for claim in tqdm(claims, desc="Verifying claims", unit="claim"):
            passages = retriever.retrieve_texts(claim.text)
            result = self._verifier.verify(claim, passages)
            verification_results.append(result)

        score = self._scorer.compute(verification_results)
        return self._build_result(claims, verification_results, score)

    def _build_result(
        self,
        claims: List[Claim],
        verification_results: List[VerificationResultNew],
        score: float
    ) -> AuditResult:
        total_supported = sum(
            1 for r in verification_results
            if r.label == VerificationLabelNew.EXPLICIT
        )
        total_partial = sum(
            1 for r in verification_results
            if r.label == VerificationLabelNew.INFERRED
        )
        total_not_supported = sum(
            1 for r in verification_results
            if r.label == VerificationLabelNew.NOT_SUPPORTED
        )

        return AuditResult(
            score=score,
            results=verification_results,
            claims=claims,
            total_supported=total_supported,
            total_partial=total_partial,
            total_not_supported=total_not_supported,
        )

    @property
    def vector_store(self) -> VectorStoreService:
        return self._vector_store

    def save_vector_store(self, path: str) -> "AuditPipeline":
        self._vector_store.save(path)
        return self

    def load_vector_store(self, path: str) -> "AuditPipeline":
        self._vector_store = VectorStoreService.load(path, self._embedding_service)
        self._retriever = Retriever(self._vector_store, top_k=self._settings.top_k_retrieval)
        return self