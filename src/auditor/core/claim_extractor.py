import json
from typing import List, Optional
from ..models.claim import Claim
from ..services.llm_service import LLMService


class ClaimExtractor:
    def __init__(self, llm_service: LLMService, system_message: Optional[str] = None):
        self._llm = llm_service
        self._system_message = system_message or "You are a precise assistant."

    def extract(self, answer: str) -> List[Claim]:
        prompt = self._build_prompt(answer)
        
        try:
            claims_data = self._llm.complete_json(prompt, self._system_message)
            if isinstance(claims_data, list):
                claims_list = claims_data
            else:
                raise ValueError("Expected list")
        except Exception:
            claims_list = self._fallback_extract(answer)
        
        return [
            Claim(text=claim, index=i, source_answer=answer)
            for i, claim in enumerate(claims_list)
        ]

    def _build_prompt(self, answer: str) -> str:
        return f"""
You are an expert in factual decomposition and information extraction.

Your task is to extract atomic, verifiable factual claims from the text.

Definition of a claim:
- A claim is a single factual statement that can be directly checked against evidence.

Instructions:
- Each claim must express ONE fact only
- Split complex sentences into multiple claims
- Preserve the original meaning

STRICT RULES:
- Prefer claims that are explicitly stated in the text
- Avoid high-level generalizations (e.g., "has strong capabilities", "is important")
- Avoid subjective or interpretative language
- Avoid claims that require interpretation or summarization
- Avoid vague qualifiers (e.g., "advanced", "significant", "remarkable") unless explicitly defined

- Each claim must be:
  - concrete
  - specific
  - verifiable against text spans

- Rewrite pronouns (e.g., "it", "they") into explicit entities

Text:
\"\"\"{answer}\"\"\"

Output:
- Return ONLY a valid JSON array of strings
- No explanations
- No markdown
"""

    def _fallback_extract(self, answer: str) -> List[str]:
        fallback = [s.strip() for s in answer.split(". ") if s.strip()]
        return fallback if fallback else [answer]