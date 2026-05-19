from src.auditor.services.file_converter import FileConverter
from pathlib import Path

converter = FileConverter()

print(converter.convert(
    input_path="./answers/chatgpt_ia_no_direito.pdf",
    output_path='./answers/ChatGPT_Direito_e_Tecnologia_Regulação_e_IA.md'
))