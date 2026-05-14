from src.auditor import AuditPipeline
from src.auditor.utils import save_result_json
from dotenv import load_dotenv
from pathlib import Path
import re
import json

load_dotenv()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_doc_id(doc_path):
    """
    Extrai o número do final do arquivo:
    exemplo: doc_12.md -> 12
    """
    match = re.search(r"_doc_(\d+)\.md$", doc_path)
    if match:
        return int(match.group(1))
    return None

def build_mapping(references_data, register_data):
    answer_target = references_data["answer"]

    # encontra o registro correspondente
    register_entry = next(
        (r for r in register_data if r["answer"] == answer_target),
        None
    )

    if not register_entry:
        raise ValueError(f"Nenhum register encontrado para {answer_target}")

    documents = register_entry["documents"]

    # cria mapa id -> documento
    doc_map = {}
    for doc in documents:
        doc_id = extract_doc_id(doc)
        if doc_id is not None:
            doc_map[doc_id] = doc

    # junta com references
    result = []
    for ref in references_data["references"]:
        ref_id = ref["id"]

        if ref_id in doc_map:
            result.append({
                "id": ref_id,
                "paragraph": ref["paragraph"],
                "documents": doc_map[ref_id]
            })

    return result

if __name__ == "__main__":
    pipeline = AuditPipeline.create_from_env()

    REFERENCES_DIR = Path("./references")
    REGISTER_FILE = Path("./register/register.json")

    answer_files = list(REFERENCES_DIR.glob("*.json"))
    quote_with_document = []
    for unit_answer in answer_files:
        references = load_json(unit_answer)
        register = load_json(REGISTER_FILE)

        result = build_mapping(references, register)
        quote_with_document += result

    print(quote_with_document)

    processed_answers = []
    for quote in quote_with_document:
        answer_path = quote['paragraph']
        answer_text = str(answer_path)
        documents = [quote['documents']]

        print(f"\n>>> Executando auditoria: {answer_path}")
        result = pipeline.audit(answer_text, documents)
        processed_answers.append(str(answer_path))

        output_dir = f"./results/claim_for_reference3"
        filepath = save_result_json(result, output_dir=output_dir, answer_name=f'testeUnitario_{quote["id"]}')
        print(f"  Salvo: {filepath}")
        print(f"  Score: {result.score:.2f} | Supported: {result.total_supported} | Unsupported: {result.total_unsupported} | Contradicted: {result.total_contradicted}")

    print(f"{'=' * 50}")
    print(f"Processados: {len(processed_answers)}")


