import os
import time
import json
from typing import Optional, Dict, Any
from openai import OpenAI


class OpenAIService:
    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPEN_AI_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPEN_AI_TOKEN or OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=self._api_key, timeout=3600)

    def deep_research(
        self,
        query: str,
        model: str = "o4-mini-deep-research",
        max_tool_calls: int = 10,
        background: bool = True,
    ) -> str:
        try:
            response = self._client.responses.create(
                model=model,
                input=query,
                max_tool_calls=max_tool_calls,
                background=background,
                tools=[{"type": "web_search_preview"}]
            )

            if background:
                print(f"Status: {response.status} - Waiting for research to complete...")
                while response.status in ("in_progress", "queued"):
                    time.sleep(5)
                    response = self._client.responses.retrieve(response_id=response.id)
                    print(f"Status: {response.status} - Still researching...")

                print("Research completed!")

            return response

        except Exception as e:
            print(e)
            print("Rate limit hit. Waiting 60's before retry...")

        return ""

    def save_response_json(self, response: Any, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.model_dump_json(indent=2))

    def save_text(self, response: Any, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.output_text)
