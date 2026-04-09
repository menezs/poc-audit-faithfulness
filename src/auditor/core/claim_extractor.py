import json
from typing import List
from ..models.claim import Claim
from ..services.llm_service import LLMService


class ClaimExtractor:
    def __init__(self, llm_service: LLMService, system_message: str = None):
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

Your task is to extract a list of atomic, verifiable factual claims from the text.

Definition of a claim:
- A claim is a single, independent factual statement that can be verified against evidence.
- Each claim must express ONE fact only.

Instructions:
- Split complex sentences into multiple claims if needed
- Preserve the original meaning
- Avoid vague, generic, or subjective statements
- Avoid combining multiple facts into a single claim
- Do not include opinions, speculation, or implications
- Prefer explicit statements over inferred ones
- Keep each claim concise but complete

Important:
- Each claim must be independently verifiable
- Each claim must be self-contained (do not rely on context from other claims)

Text:
\"\"\"{answer}\"\"\"

Output format:
- Return ONLY a valid JSON array of strings
- Do not include explanations, comments, or markdown
"""

    def _fallback_extract(self, answer: str) -> List[str]:
        fallback = [s.strip() for s in answer.split(". ") if s.strip()]
        return fallback if fallback else [answer]