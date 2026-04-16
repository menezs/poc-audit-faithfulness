from markitdown import MarkItDown
from pathlib import Path
from typing import Optional, Union


class FileConverter:
    def __init__(self):
        self._client = MarkItDown()

    def convert(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

        if output_path is None:
            output_file = input_file.with_suffix(".md")
        else:
            output_file = Path(output_path)

        result = self._client.convert(str(input_file))

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.markdown)

        return str(output_file)