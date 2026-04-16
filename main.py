#!/usr/bin/env python
"""
AuditorFidelidade - Sistema de Auditoria Factual

Entry point principal do projeto refatorado.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from src.auditor import AuditPipeline
from src.auditor.utils import save_result_json


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    provider = os.getenv("LLM_PROVIDER", "openai")

    if not api_key and provider != "ollama":
        print("Erro: OPENAI_API_KEY não configurada", file=sys.stderr)
        print("Defina a variável de ambiente OPENAI_API_KEY ou use LLM_PROVIDER=ollama", file=sys.stderr)
        sys.exit(1)

    pipeline = AuditPipeline.create_from_env()

    answer = """
    Brazil was colonized by Portugal in the 16th century.
    It became independent in 1822.
    The capital of Brazil is Rio de Janeiro.
    """

    documents = [
        "Brazil was colonized by Portugal during the 1500s. It gained independence in 1822.",
        "The capital of Brazil is Brasília, not Rio de Janeiro."
    ]

    result = pipeline.audit(answer, documents)

    filepath = save_result_json(result)
    print(f"\nResultado salvo em: {filepath}")

    print("\n" + "=" * 50)
    print("Vector Store (FAISS)")
    print("=" * 50)
    print(f"Is loaded: {pipeline.vector_store.is_loaded}")


if __name__ == "__main__":
    main()