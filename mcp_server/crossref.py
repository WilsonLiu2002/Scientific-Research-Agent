from __future__ import annotations

from urllib.parse import quote

import httpx

from app.models import PaperIntegrityResult


class CrossrefIntegrityClient:
    """Read publication updates from Crossref without treating provider data as instructions."""

    def __init__(self, base_url: str = "https://api.crossref.org", timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def check(self, doi: str) -> PaperIntegrityResult:
        """Normalize retractions, corrections, and expressions of concern for one DOI."""

        normalized = doi.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        url = f"{self.base_url}/works/{quote(normalized, safe='')}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers={"User-Agent": "scientific-literature-agent/0.1"})
            if response.status_code == 404:
                return PaperIntegrityResult(doi=normalized, status="unknown")
            response.raise_for_status()
        message = response.json().get("message", {})
        updates = self._collect_updates(message)
        labels = " ".join(updates).lower()
        if "retract" in labels:
            status = "retracted"
        elif "expression of concern" in labels or "concern" in labels:
            status = "expression_of_concern"
        elif any(term in labels for term in ("correct", "errat", "update")):
            status = "corrected"
        else:
            status = "clear"
        return PaperIntegrityResult(doi=normalized, status=status, updates=updates)

    @staticmethod
    def _collect_updates(message: dict) -> list[str]:
        """Render Crossref update relationships into concise auditable labels."""

        updates: list[str] = []
        for field in ("update-to", "updated-by"):
            for item in message.get(field, []) or []:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or item.get("type") or "publication update")
                related_doi = str(item.get("DOI") or item.get("doi") or "").strip()
                source = str(item.get("source") or "crossref")
                rendered = f"{label}: {related_doi} ({source})" if related_doi else f"{label} ({source})"
                if rendered not in updates:
                    updates.append(rendered)
        return updates
