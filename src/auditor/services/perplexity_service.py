import os
import json
from typing import Optional, List
from dataclasses import dataclass, asdict
from perplexity import Perplexity


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    last_updated: Optional[str] = None


class PerplexityService:
    def __init__(self, api_token: Optional[str] = None):
        self._api_token = api_token or os.getenv("PERPLEXITY_TOKEN")
        if not self._api_token:
            raise ValueError("PERPLEXITY_TOKEN not set")
        self._client = Perplexity(api_key=self._api_token)

    def search(
        self,
        query: str,
        max_tokens_per_page: int = 4096
    ) -> List[SearchResult]:
        search = self._client.search.create(
            query=query,
            max_tokens_per_page=max_tokens_per_page
        )
        
        results = []
        for result in search.results:
            results.append(SearchResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                date=getattr(result, 'date', None),
                last_updated=getattr(result, 'last_updated', None)
            ))
        return results

    def deep_research(self, query: str) -> str:
        search = self._client.search.create(
            query=query,
            max_tokens_per_page=8192
        )
        return search.final_output

    def save_results_to_json(self, results: List[SearchResult], filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
