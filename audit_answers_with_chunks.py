import os
import json
from pathlib import Path
from dotenv import load_dotenv
from src.auditor import AuditPipeline

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy"

load_dotenv()

OUTPUT_BASE = Path("./results/audit_with_chunks")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

REGISTER_FILE = Path("./register/register_copy.json")
CHUNKS_DIR = Path("./chunks")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_chunks_file_for_answer(answer_path):
    answer_name = Path(answer_path).stem
    for chunk_file in CHUNKS_DIR.iterdir():
        print(chunk_file)

    for chunk_file in CHUNKS_DIR.iterdir():
        if chunk_file.stem.startswith(answer_name.split("_")[0]) and chunk_file.suffix == ".json":
            return chunk_file
    return None


def get_or_create_doc_entry(answer_data, doc_path):
    for doc in answer_data.get("documents", []):
        if doc["doc_path"] == doc_path:
            return doc

    doc_entry = {
        "doc_path": doc_path,
        "chunks_audit": []
    }
    answer_data["documents"].append(doc_entry)
    return doc_entry


def is_chunk_processed(doc_entry, chunk_index):
    for chunk_audit in doc_entry.get("chunks_audit", []):
        if chunk_audit["chunk_index"] == chunk_index:
            return True
    return False


def main():
    pipeline = AuditPipeline.create_from_env()

    register_data = load_json(REGISTER_FILE)

    for entry in register_data:
        answer_path = entry["answer"]
        documents = entry["documents"]

        answer_name = Path(answer_path).stem
        output_file = OUTPUT_BASE / f"answers_{answer_name}.json"

        print(f"\n=== Processando: {answer_path} ===")

        chunks_file = get_chunks_file_for_answer(answer_path)
        if not chunks_file:
            print(f"  ERRO: Arquivo de chunks nao encontrado para {answer_name}")
            continue

        print(f"  Arquivo de chunks: {chunks_file.name}")

        chunks_data = load_json(chunks_file)
        chunks = [item["text"] for item in chunks_data]
        chunk_refs = [item.get("references", []) for item in chunks_data]

        print(f"  {len(chunks)} chunks carregados, {len(documents)} documentos")

        if output_file.exists():
            print(f"  Arquivo existente, carregando...")
            answer_data = load_json(output_file)
        else:
            answer_data = {
                "answer_path": answer_path,
                "documents": []
            }

        for doc_path in documents:
            print(f"  >> Documento: {doc_path}")

            doc_entry = get_or_create_doc_entry(answer_data, doc_path)

            for idx, chunk in enumerate(chunks):
                if is_chunk_processed(doc_entry, idx):
                    print(f"     Chunk {idx + 1}/{len(chunks)} ja processado, pulando")
                    continue

                print(f"     Chunk {idx + 1}/{len(chunks)}")

                result = pipeline.audit(chunk, [doc_path])

                chunk_result = {
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "references": chunk_refs[idx] if idx < len(chunk_refs) else [],
                    "score": result.score,
                    "total_supported": result.total_supported,
                    "total_unsupported": result.total_unsupported,
                    "total_contradicted": result.total_contradicted,
                    "results": [
                        {
                            "label": r.label.value,
                            "claim": r.claim.text,
                            "justification": r.justification,
                            "passages": r.passages
                        }
                        for r in result.results
                    ]
                }

                doc_entry["chunks_audit"].append(chunk_result)

                save_json(output_file, answer_data)
                print(f"        Salvo chunk {idx + 1}")

        print(f"  Concluido: {output_file}")

    print(f"\n=== Concluido ===")


if __name__ == "__main__":
    main()