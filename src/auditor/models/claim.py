from dataclasses import dataclass
from typing import Optional


@dataclass
class Claim:
    text: str
    index: int
    source_answer: Optional[str] = None

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Claim(index={self.index}, text='{self.text[:50]}...')"