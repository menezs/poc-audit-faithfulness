import re
import json
from pathlib import Path
from pprint import pprint


# Detecta:
# [6]
# [ 6]
# [6 ]
# [ 6 ]
REFERENCE_PATTERN = r'\[\s*\d+\s*\]'


def clean_text(text: str) -> str:
    """
    Remove:
    - espaços no início/fim
    - pontuação no início/fim
    - caracteres especiais no início/fim
    """

    # remove espaços externos
    text = text.strip()

    # remove caracteres especiais/pontuação do início e fim
    text = re.sub(r'^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$', '', text)

    return text


def normalize_reference(ref: str) -> str:
    """
    Normaliza referências:
    [ 7 ] -> [7]
    """

    number = re.search(r'\d+', ref).group()

    return f'[{number}]'


def extract_reference_chunks(text: str):

    # divide preservando referências
    parts = re.split(f'({REFERENCE_PATTERN})', text)

    chunks = []

    current_text = ""
    current_refs = []

    for part in parts:

        if not part:
            continue

        # É referência
        if re.fullmatch(REFERENCE_PATTERN, part):

            current_refs.append(
                normalize_reference(part)
            )

        else:

            # salva chunk anterior
            if current_text.strip() and current_refs:

                chunks.append({
                    "text": clean_text(current_text),
                    "references": current_refs.copy()
                })

                current_text = part
                current_refs = []

            else:
                current_text += part

    # último chunk
    if current_text.strip():

        chunks.append({
            "text": clean_text(current_text),
            "references": current_refs.copy()
        })

    return chunks


def process_markdown_file(md_file_path: str):

    md_path = Path(md_file_path)

    if not md_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {md_file_path}"
        )

    # lê markdown
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # extrai chunks
    result = extract_reference_chunks(markdown_text)

    # nome do json
    # output_json = md_path.with_name(
    #     md_path.stem + "_references.json"
    # )
    output_json = f'./{md_path.stem}_references.json'

    # salva json
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )

    return result, output_json


if __name__ == "__main__":

    # markdown_file = "./answers/gemini_Desafios_LLMs_Multilingues_e_Portugues.md"
    markdown_file = "./answers/ChatGPT_Direito_e_Tecnologia_Regulação_e_IA.md"

    result, output_path = process_markdown_file(
        markdown_file
    )

    print(f"JSON salvo em: {output_path}")