from typing import List, Optional
from ..models.claim import Claim
from ..models.verification_result import VerificationResult, VerificationLabel
from ..services.llm_service import LLMService


class Verifier:
    def __init__(self, llm_service: LLMService, system_message: Optional[str] = None):
        self._llm = llm_service
        self._system_message = system_message or "You are an expert fact-checker."

    def verify(self, claim: Claim, passages: List[str]) -> VerificationResult:
        prompt = self._build_prompt(claim.text, passages)
        
        try:
            data = self._llm.complete_json(prompt, self._system_message)
            label = VerificationLabel.from_string(data.get("label", "NOT_SUPPORTED"))
            justification = data.get("justification", "")
        except Exception as e:
            print(e)
            label = VerificationLabel.NOT_SUPPORTED
            justification = "Parsing error"
        
        return VerificationResult(
            claim=claim,
            label=label,
            justification=justification,
            passages=passages
        )

    def _build_prompt(self, claim: str, passages: List[str]) -> str:
        joined_passages = "\n\n".join(passages)
        return f"""
You are an expert fact-checker.

Claim:
\"{claim}\"

Evidence:
\"\"\"
{joined_passages}
\"\"\"

Classify the claim as:
- SUPPORTED
- PARTIAL
- NOT_SUPPORTED

IMPORTANT:
- Respond ONLY with valid JSON
- Do not include any extra text
- Do not use markdown
- "label": one of SUPPORTED, PARTIAL, NOT_SUPPORTED
- "justification": brief explanation of the classification

Format Example:
{{
  "label": "SUPPORTED | PARTIAL | NOT_SUPPORTED",
  "justification": "short explanation"
}}
"""