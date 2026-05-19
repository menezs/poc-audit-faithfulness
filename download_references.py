#!/usr/bin/env python
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from src.auditor import AuditPipeline
from src.auditor.utils import save_result_json
from src.auditor.services.file_converter import FileConverter
from src.auditor.services.llm_service import LLMService
from src.auditor.services.reference_extractor import ReferenceExtractor

load_dotenv()

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1/")
llm_model = os.getenv("LLM_MODEL", "google/gemma-4-e4b")

DOCUMENTS_DIR = Path("./documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

REGISTER_FILE = Path("./register/register_direito.json")

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
    # result_path = './references/references_lmStudio_20260516_174403.json'
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
    register = load_register()
    register_map = {entry["answer"]: entry for entry in register}

    answer_files = ['./answers/references_chatgpt_direito_e_tecnologia_regulações_e_IA.md']
    skipped_answers = []

    for answer_path in answer_files:
        entry = register_map.get(str(answer_path))

        print(f">>> Fazendo download das referências: {answer_path}")
        entry = download_references(Path(answer_path))

        if entry["documents"]:
            register.append(entry)
            register_map[str(answer_path)] = entry
            save_register(register)
            print(f"  {len(entry['documents'])} documentos baixados, {len(entry['errors'])} erros")
        else:
            print(f"  ERRO: Nenhum documento baixado")
            skipped_answers.append(str(answer_path))
            continue


if __name__ == "__main__":
    main()