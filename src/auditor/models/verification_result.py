from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from .claim import Claim


class VerificationLabel(Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    NOT_SUPPORTED = "NOT_SUPPORTED"

    @classmethod
    def from_string(cls, value: str) -> "VerificationLabel":
        normalized = value.strip().upper()
        for label in cls:
            if label.value == normalized:
                return label
        return cls.NOT_SUPPORTED


@dataclass
class VerificationResult:
    claim: Claim
    label: VerificationLabel
    justification: str
    passages: List[str]

    @property
    def is_supported(self) -> bool:
        return self.label == VerificationLabel.SUPPORTED

    @property
    def is_partial(self) -> bool:
        return self.label == VerificationLabel.PARTIAL

    @property
    def is_not_supported(self) -> bool:
        return self.label == VerificationLabel.NOT_SUPPORTED

    def __repr__(self) -> str:
        return f"VerificationResult(claim={self.claim.index}, label={self.label.value})"