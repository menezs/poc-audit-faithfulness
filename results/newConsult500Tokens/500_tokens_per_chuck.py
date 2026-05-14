import os
import re
import json
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from src.auditor import AuditPipeline

if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "dummy"

load_dotenv()

TOKEN_LIMIT = 500
OUTPUT_BASE = Path("./results/newConsult500Tokens")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

REGISTER_FILE = Path("./register/register_copy.json")


def remove_all_links(text):
    text = re.sub(r'https?://[^\s\)\]】]+', '', text)
    text = re.sub(r'\[(\d+)\]', '', text)
    text = re.sub(r'【\d+[^\]】]*】', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    return text


def count_tokens(text, encoder):
    return len(encoder.encode(text))


def chunk_text(text, encoder, max_tokens=500):
    tokens = encoder.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk = encoder.decode(chunk_tokens).strip()
        chunks.append(chunk)

    return chunks


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
    encoder = tiktoken.get_encoding("cl100k_base")
    pipeline = AuditPipeline.create_from_env()

    register_data = load_json(REGISTER_FILE)

    for entry in register_data:
        answer_path = entry["answer"]
        documents = entry["documents"]

        answer_name = Path(answer_path).stem
        output_file = OUTPUT_BASE / f"answers_{answer_name}.json"

        print(f"\n=== Processando: {answer_path} ===")

        with open(answer_path, "r", encoding="utf-8") as f:
            answer_text = f.read()

        clean_text = remove_all_links(answer_text)
        chunks = chunk_text(clean_text, encoder, TOKEN_LIMIT)
        print(f"  {len(chunks)} chunks gerados, {len(documents)} documentos")

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