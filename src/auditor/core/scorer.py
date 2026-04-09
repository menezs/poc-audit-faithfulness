from typing import List
from ..models.verification_result import VerificationResult


class Scorer:
    def __init__(self, partial_weight: float = 0.5):
        self._partial_weight = partial_weight

    def compute(self, results: List[VerificationResult]) -> float:
        if not results:
            return 0.0
        
        total = len(results)
        score = 0.0
        
        for result in results:
            if result.is_supported:
                score += 1.0
            elif result.is_partial:
                score += self._partial_weight
        
        return score / total