#!/usr/bin/env python
import os
import sys
import json
from pathlib import Path

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

from src.auditor import AuditPipeline
from src.auditor.utils import save_result_json
from src.auditor.services.file_converter import FileConverter
from src.auditor.services.llm_service import LLMService
from src.auditor.services.reference_extractor import ReferenceExtractor

ANSWERS_DIR = Path("./answers")
DOCUMENTS_DIR = Path("./documents")
REGISTER_FILE = Path("./register/register.json")

llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1/")
llm_model = os.getenv("LLM_MODEL", "google/gemma-4-e4b")

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
Path("./register").mkdir(parents=True, exist_ok=True)


def load_register() -> list:
    if REGISTER_FILE.exists():
        with open(REGISTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_register(register: list):
    with open(REGISTER_FILE, "w", encoding="utf-8") as f:
        json.dump(register, f, ensure_ascii=False, indent=2)


def download_references(answer_path: Path) -> dict:
    llm = LLMService(
        api_key="",
        model=llm_model,
        provider="openai",
        base_url=llm_base_url
    )

    extractor = ReferenceExtractor(llm_service=llm)
    converter = FileConverter()

    print(f"\nProcessando: {answer_path}")

    result_path = extractor.extract_from_markdown(str(answer_path))
    with open(result_path, "r", encoding="utf-8") as f:
        result_json = json.load(f)

    references = result_json.get("references", [])
    documents = []
    errors = []

    for ref in references:
        url = ref.get("url")
        if not url:
            continue

        doc_id = ref.get("id", 0)
        safe_name = answer_path.stem.replace(" ", "_")
        output_path = DOCUMENTS_DIR / f"{safe_name}_doc_{doc_id}.md"

        print(f"  Baixando: {url}")
        try:
            converter.convert(url=url, output_path=output_path)
            documents.append(str(output_path))
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"    ERRO: {error_type} - {error_msg}")
            errors.append({
                "url": url,
                "error_type": error_type,
                "error_message": error_msg,
                "reference_id": doc_id
            })

    return {
        "answer": str(answer_path),
        "documents": documents,
        "errors": errors
    }


def main() -> None:
    pipeline = AuditPipeline.create_from_env()
    register = load_register()
    register_map = {entry["answer"]: entry for entry in register}

    answer_files = list(ANSWERS_DIR.glob("*.md"))

    skipped_answers = []
    processed_answers = []

    for answer_path in answer_files:
        entry = register_map.get(str(answer_path))

        if not entry or not entry.get("documents"):
            print(f"\n>>> Fazendo download das referências: {answer_path}")
            entry = download_references(answer_path)

            if entry["documents"]:
                register.append(entry)
                register_map[str(answer_path)] = entry
                save_register(register)
                print(f"  {len(entry['documents'])} documentos baixados, {len(entry['errors'])} erros")
            else:
                print(f"  ERRO: Nenhum documento baixado")
                skipped_answers.append(str(answer_path))
                continue

        if not entry.get("documents"):
            print(f"  PULADO: Sem documentos associados")
            skipped_answers.append(str(answer_path))
            continue

        answer_text = str(answer_path)
        documents = entry["documents"]

        print(f"\n>>> Executando auditoria: {answer_path}")
        result = pipeline.audit(answer_text, documents)
        processed_answers.append(str(answer_path))

        output_dir = f"./results/{answer_path.stem}"
        filepath = save_result_json(result, output_dir=output_dir, answer_name=answer_path.name)
        print(f"  Salvo: {filepath}")
        print(f"  Score: {result.score:.2f} | Supported: {result.total_supported} | Partial: {result.total_partial} | Not Supported: {result.total_not_supported}")

    print(f"\n{'=' * 50}")
    print(f"RESUMO")
    print(f"{'=' * 50}")
    print(f"Processados: {len(processed_answers)}")
    print(f"Pulados: {len(skipped_answers)}")
    if skipped_answers:
        print(f"\nAnswers pulados:")
        for ans in skipped_answers:
            print(f"  - {ans}")


if __name__ == "__main__":
    main()