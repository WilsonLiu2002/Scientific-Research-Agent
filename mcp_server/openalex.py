from __future__ import annotations

import os
from typing import Any

import httpx

from app.models import Paper


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class OpenAlexSearchClient:
    """Thin OpenAlex client used behind the MCP server's `search_papers` tool."""

    def __init__(self, timeout: float = 20.0, mailto: str | None = None) -> None:
        self.timeout = timeout
        self.mailto = mailto or os.getenv("OPENALEX_MAILTO")

    async def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        """Search OpenAlex and convert works into the project-wide Paper model."""

        params: dict[str, Any] = {
            "search": query,
            "per-page": max(1, min(limit, 50)),
            "select": "id,display_name,publication_year,authorships,abstract_inverted_index,cited_by_count,doi,primary_location,primary_topic",
        }
        if self.mailto:
            params["mailto"] = self.mailto

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(OPENALEX_WORKS_URL, params=params)
            response.raise_for_status()
            payload = response.json()

        return [paper for item in payload.get("results", []) if (paper := self.parse_work(item))]

    def parse_work(self, item: dict[str, Any]) -> Paper | None:
        """Parse one OpenAlex work into a Paper, skipping records without a usable title."""

        title = (item.get("display_name") or "").strip()
        if not title:
            return None

        authorships = item.get("authorships") or []
        authors = [
            authorship.get("author", {}).get("display_name", "").strip()
            for authorship in authorships
            if authorship.get("author", {}).get("display_name")
        ]
        url = self._best_url(item)
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        return Paper(
            id=str(item.get("id") or item.get("doi") or title),
            title=title,
            abstract=reconstruct_abstract(item.get("abstract_inverted_index")),
            authors=authors,
            year=item.get("publication_year"),
            doi=item.get("doi"),
            venue=source.get("display_name"),
            url=url,
            citation_count=item.get("cited_by_count"),
            source="OpenAlex",
        )

    def _best_url(self, item: dict[str, Any]) -> str | None:
        """Choose the most useful public URL OpenAlex provides for a work."""

        primary_location = item.get("primary_location") or {}
        landing_page_url = primary_location.get("landing_page_url")
        return landing_page_url or item.get("doi") or item.get("id")


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Reconstruct OpenAlex's inverted-index abstract representation into normal text."""

    if not inverted_index:
        return None

    positions: dict[int, str] = {}
    for word, indexes in inverted_index.items():
        for index in indexes:
            positions[index] = word
    if not positions:
        return None
    return " ".join(positions[index] for index in sorted(positions))
