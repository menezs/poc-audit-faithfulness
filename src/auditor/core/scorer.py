from typing import List
from ..models.verification_result import VerificationResult, VerificationResultNew, VerificationResultNew2


class Scorer:
    def __init__(self, partial_weight: float = 0.5):
        self._partial_weight = partial_weight

    def compute(self, results: List[VerificationResultNew2]) -> float:
        if not results:
            return 0.0
        
        total = len(results)
        score = 0.0
        
        for result in results:
            if result.is_supported:
                score += 1.0
            elif result.is_unsupported:
                score += self._partial_weight
        
        return score / total