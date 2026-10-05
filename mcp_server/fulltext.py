from __future__ import annotations

import hashlib
import ipaddress
import io
import json
import os
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from xml.etree import ElementTree

import httpx

from app.models import DocumentSection, FullTextDocument


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024


class ArticleHTMLParser(HTMLParser):
    """Minimal article parser that groups paragraphs under their nearest heading."""

    def __init__(self) -> None:
        super().__init__()
        self.heading = "Article"
        self.capture: str | None = None
        self.buffer: list[str] = []
        self.sections: list[DocumentSection] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "p"}:
            self.capture = tag
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self.capture:
            return
        text = normalize_text(" ".join(self.buffer))
        if tag in {"h1", "h2", "h3"} and text:
            self.heading = text
        elif tag == "p" and len(text) >= 80:
            self.sections.append(DocumentSection(heading=self.heading, text=text))
        self.capture = None
        self.buffer = []


class FullTextService:
    """Acquire, cache, and extract legally accessible scientific articles."""

    def __init__(self, cache_directory: str | Path | None = None, timeout: float = 30.0) -> None:
        configured = cache_directory or os.getenv("FULLTEXT_CACHE_DIR", ".cache/fulltext")
        self.cache_directory = Path(configured)
        self.timeout = timeout

    async def get_full_text(
        self, paper_id: str, doi: str | None = None, url: str | None = None
    ) -> FullTextDocument | None:
        """Resolve an open-access location, download it, and cache structured article text."""

        cache_key = hashlib.sha256((doi or paper_id).lower().encode("utf-8")).hexdigest()
        cached = self._load_cached(cache_key)
        if cached:
            return cached

        metadata = await self._get_openalex_work(paper_id=paper_id, doi=doi)
        candidates = open_access_candidates(metadata)
        if url and metadata.get("open_access", {}).get("is_oa") and is_safe_article_url(url):
            candidates.append((url, metadata.get("best_oa_location", {}).get("license")))
        if not candidates:
            return None

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            for source_url, license_name in candidates:
                try:
                    response = await fetch_safe_redirects(client, source_url)
                    response.raise_for_status()
                    if len(response.content) > MAX_DOWNLOAD_BYTES:
                        continue
                    media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    sections = extract_sections(response.content, media_type, source_url)
                except (httpx.HTTPError, ValueError):
                    continue
                if not sections:
                    continue
                document = FullTextDocument(
                    document_id=cache_key,
                    paper_id=paper_id,
                    title=str(metadata.get("display_name") or paper_id),
                    source_url=source_url,
                    media_type=media_type or infer_media_type(source_url),
                    license=license_name,
                    sections=sections,
                )
                self._save_cached(cache_key, document)
                return document
        return None

    async def _get_openalex_work(self, paper_id: str, doi: str | None) -> dict[str, Any]:
        """Fetch full OpenAlex location metadata for one selected paper."""

        identifier = doi or paper_id
        if identifier.startswith("https://openalex.org/"):
            identifier = identifier.rsplit("/", 1)[-1]
        endpoint = f"{OPENALEX_WORKS_URL}/{quote(identifier, safe='') }"
        params = {"mailto": os.getenv("OPENALEX_MAILTO")} if os.getenv("OPENALEX_MAILTO") else None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()

    def _load_cached(self, document_id: str) -> FullTextDocument | None:
        """Read a validated document from the local full-text cache."""

        path = self.cache_directory / f"{document_id}.json"
        if not path.exists():
            return None
        try:
            return FullTextDocument.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save_cached(self, document_id: str, document: FullTextDocument) -> None:
        """Persist extracted text so repeated deep reads avoid another download."""

        self.cache_directory.mkdir(parents=True, exist_ok=True)
        path = self.cache_directory / f"{document_id}.json"
        path.write_text(document.model_dump_json(indent=2), encoding="utf-8")


def open_access_candidates(metadata: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Return deduplicated URLs explicitly marked open access by OpenAlex."""

    if not metadata.get("open_access", {}).get("is_oa"):
        return []
    locations = [metadata.get("best_oa_location") or {}, *(metadata.get("locations") or [])]
    candidates: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for location in locations:
        if not location.get("is_oa", True):
            continue
        for key in ("pdf_url", "landing_page_url"):
            candidate = location.get(key)
            if candidate and is_safe_article_url(candidate) and candidate not in seen:
                seen.add(candidate)
                candidates.append((candidate, location.get("license")))
    return candidates


def is_safe_article_url(url: str) -> bool:
    """Reject non-web and local targets before the acquisition client follows a URL."""

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host or host == "localhost":
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (address.is_private or address.is_loopback or address.is_link_local)


async def fetch_safe_redirects(
    client: httpx.AsyncClient, url: str, max_redirects: int = 5
) -> httpx.Response:
    """Follow only redirects whose destination remains a safe public HTTPS URL."""

    current_url = url
    for _ in range(max_redirects + 1):
        if not is_safe_article_url(current_url):
            raise ValueError("Unsafe article URL")
        response = await client.get(current_url)
        if not response.is_redirect:
            return response
        if response.next_request is None:
            raise ValueError("Redirect response omitted its destination")
        current_url = str(response.next_request.url)
    raise ValueError("Article download exceeded the redirect limit")


def extract_sections(content: bytes, media_type: str, source_url: str) -> list[DocumentSection]:
    """Dispatch PDF, JATS/XML, or HTML article extraction based on content type."""

    kind = media_type or infer_media_type(source_url)
    if kind == "application/pdf" or content.startswith(b"%PDF"):
        return extract_pdf_sections(content)
    if "xml" in kind or source_url.lower().endswith(".xml"):
        return extract_xml_sections(content)
    if "html" in kind:
        parser = ArticleHTMLParser()
        parser.feed(content.decode("utf-8", errors="ignore"))
        return parser.sections
    return []


def extract_pdf_sections(content: bytes) -> list[DocumentSection]:
    """Extract page-preserving text from a PDF using the optional pypdf dependency."""

    try:
        from pypdf import PdfReader
    except (ImportError, ModuleNotFoundError):
        return []
    reader = PdfReader(io.BytesIO(content))
    return [
        DocumentSection(heading=f"Page {index}", text=text, page=index)
        for index, page in enumerate(reader.pages, start=1)
        if len(text := normalize_text(page.extract_text() or "")) >= 80
    ]


def extract_xml_sections(content: bytes) -> list[DocumentSection]:
    """Extract section titles and paragraphs from JATS-like XML."""

    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return []
    extracted: list[DocumentSection] = []
    for section in root.findall(".//sec"):
        title = normalize_text(" ".join(section.findtext("title", default="").split())) or "Article"
        text = normalize_text(" ".join("".join(p.itertext()) for p in section.findall("./p")))
        if len(text) >= 80:
            extracted.append(DocumentSection(heading=title, text=text))
    return extracted


def normalize_text(text: str) -> str:
    """Collapse extraction whitespace while retaining sentence content."""

    return " ".join(text.split())


def infer_media_type(url: str) -> str:
    """Infer a conservative media type when a server omits its content type."""

    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return "application/pdf"
    if path.endswith(".xml"):
        return "application/xml"
    return "text/html"
