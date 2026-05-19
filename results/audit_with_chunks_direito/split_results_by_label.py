import json
import os
from pathlib import Path

INPUT_DIR = Path("./results/audit_with_chunks_direito")
OUTPUT_DIR = Path("./results/audit_with_chunks_direito")

FILES = [
    "answers_Gemini_Direito_e_Tecnologia_Regulação_e_IA.json",
    "answers_ChatGPT_Direito_e_Tecnologia_Regulação_e_IA.json"
]

def process_json_file(filepath, label_filter):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    supported_count = 0
    unsupported_count = 0
    total_chunks = 0

    new_documents = []

    for doc in data.get("documents", []):
        new_chunks_audit = []

        for chunk in doc.get("chunks_audit", []):
            results = chunk.get("results", [])
            filtered_results = [r for r in results if r.get("label") == label_filter]

            if filtered_results:
                new_chunk = {
                    "chunk_index": chunk.get("chunk_index"),
                    "chunk_text": chunk.get("chunk_text"),
                    "score": chunk.get("score"),
                    "total_supported": chunk.get("total_supported"),
                    "total_unsupported": chunk.get("total_unsupported"),
                    "total_contradicted": chunk.get("total_contradicted"),
                    "results": filtered_results
                }
                new_chunks_audit.append(new_chunk)
                total_chunks += 1

                if label_filter == "SUPPORTED":
                    supported_count += len(filtered_results)
                else:
                    unsupported_count += len(filtered_results)

        if new_chunks_audit:
            new_doc = {
                "doc_path": doc.get("doc_path"),
                "chunks_audit": new_chunks_audit
            }
            new_documents.append(new_doc)

    new_data = {
        "answer_path": data.get("answer_path"),
        "documents": new_documents
    }

    return new_data, supported_count, unsupported_count, total_chunks


def main():
    all_metrics = []

    for filename in FILES:
        input_path = INPUT_DIR / filename
        base_name = filename.replace(".json", "")

        for label in ["SUPPORTED", "UNSUPPORTED"]:
            output_filename = f"{base_name}_{label}.json"
            output_path = OUTPUT_DIR / output_filename

            new_data, supported, unsupported, total_chunks = process_json_file(input_path, label)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

            metric = {
                "input_file": filename,
                "output_file": output_filename,
                "label": label,
                "total_results": supported if label == "SUPPORTED" else unsupported,
                "total_chunks": total_chunks
            }
            all_metrics.append(metric)

    print("\n" + "=" * 80)
    print("COMPILADO GERAL - MÉTRICAS DE PROCESSAMENTO")
    print("=" * 80)

    total_supported = 0
    total_unsupported = 0

    for m in all_metrics:
        label = m["label"]
        if label == "SUPPORTED":
            total_supported += m["total_results"]
        else:
            total_unsupported += m["total_results"]

    print(f"\nArquivos processados: {len(FILES)}")
    print(f"Arquivos gerados: {len(all_metrics)}")
    print(f"\nTotal SUPPORTED: {total_supported}")
    print(f"Total UNSUPPORTED: {total_unsupported}")
    print(f"Total geral: {total_supported + total_unsupported}")

    if (total_supported + total_unsupported) > 0:
        pct_supported = (total_supported / (total_supported + total_unsupported)) * 100
        pct_unsupported = (total_unsupported / (total_supported + total_unsupported)) * 100
        print(f"\nPercentual SUPPORTED: {pct_supported:.2f}%")
        print(f"Percentual UNSUPPORTED: {pct_unsupported:.2f}%")

    print("\n" + "-" * 80)
    print("DETALHAMENTO POR ARQUIVO:")
    print("-" * 80)

    current_file = None
    for m in all_metrics:
        if m["input_file"] != current_file:
            current_file = m["input_file"]
            print(f"\n>>> {current_file}")

        print(f"  - {m['label']}: {m['total_results']} resultados ({m['total_chunks']} chunks)")

    print("\n" + "-" * 80)
    print("ARQUIVOS GERADOS:")
    print("-" * 80)

    for m in all_metrics:
        print(f"  - {m['output_file']}")

    print("\n" + "=" * 80)
    print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    main()