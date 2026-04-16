import os
import json
from datetime import datetime
from typing import List
from ..models.verification_result import VerificationResult
from ..pipeline.audit_pipeline import AuditResult


def load_api_key(env_var: str = "OPENAI_API_KEY") -> str:
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"{env_var} environment variable not set")
    return api_key


def print_result(result: AuditResult) -> None:
    print("\n" + "=" * 50)
    print("AUDIT RESULT")
    print("=" * 50)
    print(f"\nScore: {result.score:.2f}")
    print(f"Supported: {result.total_supported}")
    print(f"Partial: {result.total_partial}")
    print(f"Not Supported: {result.total_not_supported}")
    print("\n" + "-" * 50)
    print("DETAILS")
    print("-" * 50)
    for r in result.results:
        print(f"\nClaim: {r.claim.text}")
        print(f"Label: {r.label.value}")
        print(f"Justification: {r.justification}")


def save_result_json(result: AuditResult, output_dir: str = "./results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    data = {
        "timestamp": timestamp,
        "score": result.score,
        "total_supported": result.total_supported,
        "total_partial": result.total_partial,
        "total_not_supported": result.total_not_supported,
        "is_fully_supported": result.is_fully_supported,
        "claims": [
            {
                "index": claim.index,
                "text": claim.text,
                "source_answer": claim.source_answer
            }
            for claim in result.claims
        ],
        "results": [
            {
                "claim_index": r.claim.index,
                "claim_text": r.claim.text,
                "label": r.label.value,
                "justification": r.justification,
                "passages": r.passages
            }
            for r in result.results
        ]
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath