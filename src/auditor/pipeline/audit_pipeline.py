from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
from ..models.claim import Claim
from ..models.verification_result import VerificationResult, VerificationLabel
from ..core import ClaimExtractor, Retriever, Verifier, Scorer
from ..config.settings import Settings
from ..services import LLMService, EmbeddingService, VectorStoreService


@dataclass
class AuditResult:
    score: float
    results: List[VerificationResult]
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
        self._llm_service = LLMService(settings.api_key, settings.llm_model, settings.temperature)
        self._embedding_service = EmbeddingService(
            api_key=settings.api_key,
            model=settings.embedding_model,
            use_local=settings.use_local_embedding,
            local_model_name=settings.local_embedding_model
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

    def audit(self, answer: str, documents: List[str]) -> AuditResult:
        claims = self._extractor.extract(answer)
        
        retriever = self._ensure_retriever(documents)
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
        verification_results: List[VerificationResult],
        score: float
    ) -> AuditResult:
        total_supported = sum(
            1 for r in verification_results
            if r.label == VerificationLabel.SUPPORTED
        )
        total_partial = sum(
            1 for r in verification_results
            if r.label == VerificationLabel.PARTIAL
        )
        total_not_supported = sum(
            1 for r in verification_results
            if r.label == VerificationLabel.NOT_SUPPORTED
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