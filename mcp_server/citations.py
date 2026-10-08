from __future__ import annotations

import asyncio
import os
from urllib.parse import quote

import httpx

from app.models import CitationExpansionResult, Paper, deduplicate_papers


class SemanticScholarCitationClient:
    """Expand a seed paper through bounded Semantic Scholar reference and citation calls."""

    def __init__(self, base_url: str = "https://api.semanticscholar.org", timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def expand(self, paper: Paper, limit: int = 5) -> CitationExpansionResult:
        """Return balanced references and citing papers with abstracts when available."""

        identifier = f"DOI:{paper.doi}" if paper.doi else paper.id
        headers = {}
        if api_key := os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip():
            headers["x-api-key"] = api_key
        per_direction = max(1, min(limit, 20))
        fields = "paperId,title,abstract,authors,year,externalIds,venue,url,citationCount"
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            reference_response, citation_response = await self._fetch_both(
                client, identifier, fields, per_direction
            )
        references = self._papers(reference_response, "citedPaper", paper.id, "reference")
        citations = self._papers(citation_response, "citingPaper", paper.id, "citation")
        return CitationExpansionResult(
            seed_paper_id=paper.id,
            papers=deduplicate_papers([*references, *citations]),
            references_found=len(references),
            citations_found=len(citations),
        )

    async def _fetch_both(self, client, identifier: str, fields: str, limit: int):
        encoded = quote(identifier, safe=":")
        references = client.get(
            f"{self.base_url}/graph/v1/paper/{encoded}/references",
            params={"fields": fields, "limit": limit},
        )
        citations = client.get(
            f"{self.base_url}/graph/v1/paper/{encoded}/citations",
            params={"fields": fields, "limit": limit},
        )
        reference_response, citation_response = await asyncio.gather(references, citations)
        reference_response.raise_for_status()
        citation_response.raise_for_status()
        return reference_response.json(), citation_response.json()

    @staticmethod
    def _papers(payload: dict, key: str, seed_id: str, relation: str) -> list[Paper]:
        papers: list[Paper] = []
        for entry in payload.get("data", []) or []:
            item = entry.get(key) if isinstance(entry, dict) else None
            if not item or not item.get("paperId") or not item.get("title"):
                continue
            external_ids = item.get("externalIds") or {}
            papers.append(
                Paper(
                    id=item["paperId"],
                    title=item["title"],
                    abstract=item.get("abstract"),
                    authors=[author.get("name", "") for author in item.get("authors", []) if author.get("name")],
                    year=item.get("year"),
                    doi=external_ids.get("DOI"),
                    venue=item.get("venue"),
                    url=item.get("url"),
                    citation_count=item.get("citationCount"),
                    source=f"semantic-scholar-{relation}",
                    discovered_from=seed_id,
                )
            )
        return papers
