import trafilatura
import requests
from pathlib import Path
from typing import Optional, Union

import pymupdf4llm


class FileConverter:
    def convert(
        self,
        input_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
    ) -> str:
        if url is not None:
            return self._from_url(url, output_path)

        if input_path is None:
            raise ValueError("Forneça input_path ou url")

        return self._from_file(input_path, output_path)

    def _from_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

        if output_path is None:
            output_file = input_file.with_suffix(".md")
        else:
            output_file = Path(output_path)

        markdown = pymupdf4llm.to_markdown(str(input_file))
        if markdown is None:
            raise ValueError(f"Não foi possível converter o arquivo: {input_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        return str(output_file)

    def _from_url(self, url: str, output_path: Optional[Union[str, Path]] = None) -> str:
        if output_path is None:
            raise ValueError("output_path é obrigatório quando usando url")

        output_file = Path(output_path)

        if url.endswith(".pdf"):
            file_name = url.split('/')[-1].replace('.pdf', '')
            path_to_donwload = Path(output_file.parts[0]) / f'{file_name}.pdf'
            self._download_pdf(url=url, output_path=path_to_donwload)

            self._from_file(path_to_donwload, output_file)

            return str(output_file)

        html = trafilatura.fetch_url(url)
        if html is None:
            raise ValueError(f"Não foi possível acessar a URL: {url}")

        markdown = trafilatura.extract(html, output_format="markdown")
        if markdown is None:
            raise ValueError(f"Não foi possível extrair conteúdo de: {url}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown)

        return str(output_file)
    
    def _download_pdf(self, url: str, output_path: Union[str, Path]):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)