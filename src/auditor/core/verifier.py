from typing import List, Optional
from ..models.claim import Claim
from ..models.verification_result import VerificationResult, VerificationLabel, VerificationResultNew, VerificationLabelNew
from ..services.llm_service import LLMService


class Verifier:
    def __init__(self, llm_service: LLMService, system_message: Optional[str] = None):
        self._llm = llm_service
        self._system_message = system_message or "You are an expert fact-checker."

    def verify(self, claim: Claim, passages: List[str]) -> VerificationResultNew:
        prompt = self._build_prompt(claim.text, passages)
        
        try:
            data = self._llm.complete_json(prompt, self._system_message)
            label = VerificationLabelNew.from_string(data.get("label", "NOT_SUPPORTED"))
            justification = data.get("justification", "")
        except Exception as e:
            print(e)
            label = VerificationLabelNew.NOT_SUPPORTED
            justification = "Parsing error"
        
        return VerificationResultNew(
            claim=claim,
            label=label,
            justification=justification,
            passages=passages
        )

    def _build_prompt(self, claim: str, passages: List[str]) -> str:
        joined_passages = "\n\n".join(passages)
        return f"""
You are an expert in evidence-based factual verification.

Your task is to determine how a claim is supported by the provided evidence.

Claim:
\"{claim}\"

Evidence:
\"\"\"
{joined_passages}
\"\"\"

Definitions:

- EXPLICIT:
  The claim is directly stated in the evidence.
  The meaning appears clearly without needing interpretation.

- INFERRED:
  The claim can be logically derived from the evidence,
  but is not explicitly stated.
  Requires interpretation, generalization, or combining information.

- NOT_SUPPORTED:
  The claim is not supported or is contradicted by the evidence.

Instructions:
- Be strict.
- Only use EXPLICIT if the claim is clearly present in the evidence.
- If any reasoning is required → use INFERRED.
- Do NOT assume missing information.
- Do NOT rely on outside knowledge.

Output:
Return ONLY valid JSON:

{{
  "label": "EXPLICIT | INFERRED | NOT_SUPPORTED",
  "justification": "short explanation"
}}
"""