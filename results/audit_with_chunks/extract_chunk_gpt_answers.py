import re
import json
from pathlib import Path
from pprint import pprint


REFERENCE_PATTERN = r'【\d+†L\d+(?:-L?\d+)?】'


def clean_text(text: str) -> str:
    """
    Remove:
    - espaços no início/fim
    - pontuação no início/fim
    - caracteres especiais no início/fim
    """

    # remove espaços
    text = text.strip()

    # remove caracteres especiais/pontuação do início e fim
    text = re.sub(r'^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$', '', text)

    return text


def extract_reference_chunks(text: str):

    parts = re.split(f'({REFERENCE_PATTERN})', text)

    chunks = []

    current_text = ""
    current_refs = []

    for part in parts:

        if not part:
            continue

        # Encontrou referência
        if re.fullmatch(REFERENCE_PATTERN, part):

            current_refs.append(part)

        else:

            # Salva chunk anterior
            if current_text.strip() and current_refs:

                chunks.append({
                    "text": clean_text(current_text),
                    "references": current_refs.copy()
                })

                current_text = part
                current_refs = []

            else:
                current_text += part

    # Último chunk
    if current_text.strip():

        chunks.append({
            "text": clean_text(current_text),
            "references": current_refs.copy()
        })

    return chunks


def process_markdown_file(md_file_path: str):

    md_path = Path(md_file_path)

    if not md_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {md_file_path}")

    # Lê markdown
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Extrai chunks
    result = extract_reference_chunks(markdown_text)

    # Nome do JSON de saída
    # output_json = md_path.with_suffix(".json")
    output_json = f'./{md_path.stem}_references.json'

    # Salva JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )

    return result, output_json


if __name__ == "__main__":

    # Arquivo markdown de entrada
    markdown_file = "./answers/chatGPT_Desafios_LLMs_Multilingues_e_Portugues.md"

    result, output_path = process_markdown_file(markdown_file)

    print(f"JSON salvo em: {output_path}")