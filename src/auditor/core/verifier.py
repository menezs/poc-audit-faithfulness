from typing import List, Optional
from ..models.claim import Claim
from ..models.verification_result import VerificationResult, VerificationLabel, VerificationResultNew, VerificationLabelNew, VerificationResultNew2, VerificationLabelNew2
from ..services.llm_service import LLMService


class Verifier:
    def __init__(self, llm_service: LLMService, system_message: Optional[str] = None):
        self._llm = llm_service
        self._system_message = system_message or "You are an expert fact-checker."

    def verify(self, claim: Claim, passages: List[str]) -> VerificationResultNew2:
        prompt = self._build_prompt(claim.text, passages)
        
        try:
            data = self._llm.complete_json(prompt, self._system_message)
            label = VerificationLabelNew2.from_string(data.get("label", "CONTRADICTED"))
            justification = data.get("justification", "")
        except Exception as e:
            print(e)
            label = VerificationLabelNew2.CONTRADICTED
            justification = "Parsing error"
        
        return VerificationResultNew2(
            claim=claim,
            label=label,
            justification=justification,
            passages=passages
        )

    def _build_prompt(self, claim: str, passages: List[str]) -> str:
        joined_passages = "\n\n".join(passages)
        return f"""
You are an expert in evidence-based verification.

Your task is to determine whether the claim is supported by the provided evidence.

Claim:
\"{claim}\"

Evidence:
\"\"\"
{joined_passages}
\"\"\"

Definitions:

- SUPPORTED:
  The claim can be justified using ONLY the provided evidence.
  It may require light reasoning, but no external knowledge.

- UNSUPPORTED:
  The claim cannot be justified using the provided evidence.
  Even if it is true in the real world, it is not grounded in the evidence.

- CONTRADICTED:
  The evidence contradicts the claim.

Instructions:
- Respond ONLY with valid JSON
- Do not include any extra text
- Do NOT use external knowledge
- Only rely on the provided evidence
- If the evidence is insufficient → UNSUPPORTED

Output:
{{
  "label": "SUPPORTED | UNSUPPORTED | CONTRADICTED",
  "justification": "short explanation"
}}
"""