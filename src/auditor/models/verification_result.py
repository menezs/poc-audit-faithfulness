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


class VerificationLabelNew(Enum):
    EXPLICIT = "EXPLICIT"
    INFERRED = "INFERRED"
    NOT_SUPPORTED = "NOT_SUPPORTED"

    @classmethod
    def from_string(cls, value: str) -> "VerificationLabelNew":
        normalized = value.strip().upper()
        for label in cls:
            if label.value == normalized:
                return label
        return cls.NOT_SUPPORTED
    

class VerificationLabelNew2(Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"

    @classmethod
    def from_string(cls, value: str) -> "VerificationLabelNew2":
        normalized = value.strip().upper()
        for label in cls:
            if label.value == normalized:
                return label
        return cls.CONTRADICTED


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


@dataclass
class VerificationResultNew:
    claim: Claim
    label: VerificationLabelNew
    justification: str
    passages: List[str]

    @property
    def is_explicit(self) -> bool:
        return self.label == VerificationLabelNew.EXPLICIT

    @property
    def is_inferred(self) -> bool:
        return self.label == VerificationLabelNew.INFERRED

    @property
    def is_not_supported(self) -> bool:
        return self.label == VerificationLabelNew.NOT_SUPPORTED

    def __repr__(self) -> str:
        return f"VerificationResultNew(claim={self.claim.index}, label={self.label.value})"
    

@dataclass
class VerificationResultNew2:
    claim: Claim
    label: VerificationLabelNew2
    justification: str
    passages: List[str]

    @property
    def is_supported(self) -> bool:
        return self.label == VerificationLabelNew2.SUPPORTED

    @property
    def is_unsupported(self) -> bool:
        return self.label == VerificationLabelNew2.UNSUPPORTED

    @property
    def is_contradicted(self) -> bool:
        return self.label == VerificationLabelNew2.CONTRADICTED

    def __repr__(self) -> str:
        return f"VerificationResultNew2(claim={self.claim.index}, label={self.label.value})"