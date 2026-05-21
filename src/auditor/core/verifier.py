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
You are a strict evidence-grounded verification system.

Your task is to classify the relationship between the CLAIM and the PROVIDED EVIDENCE.

You **MUST** evaluate the claim using **ONLY** the provided evidence.
You MUST completely ignore:
- prior conversation context
- chat history
- world knowledge
- assumptions
- common sense not explicitly supported by the evidence
- information outside the evidence block

Claim:
\"\"\"
{claim}
\"\"\"

Evidence:
\"\"\"
{joined_passages}
\"\"\"

Definitions:

- SUPPORTED:
  The claim is explicitly stated or can be directly inferred ONLY from the provided evidence.

- UNSUPPORTED:
  The evidence does not contain enough information to verify the claim.
  This includes cases where:
  - the claim may be true in reality, but is not grounded in the evidence
  - the evidence is incomplete, vague, or insufficient
  - verification would require external knowledge or assumptions

- CONTRADICTED:
  The evidence explicitly conflicts with or disproves the claim.

Critical Rules:
- Treat the evidence as the ONLY source of truth
- Never use external knowledge
- Never use information from previous messages or conversation history
- Never infer missing facts unless they are directly supported by the evidence
- If the evidence is ambiguous or incomplete, return UNSUPPORTED
- Prefer UNSUPPORTED over SUPPORTED when uncertain
- Only classify as CONTRADICTED when the evidence clearly conflicts with the claim

Instructions:
- Respond ONLY with valid JSON
- Do not include markdown
- Do not include explanations outside the JSON
- The JSON must contain exactly these fields:
  - "label"
  - "justification"

Output format:
{{
  "label": "SUPPORTED | UNSUPPORTED | CONTRADICTED",
  "justification": "brief evidence-grounded explanation"
}}
"""