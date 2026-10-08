import asyncio

from app.models import CitationExpansionResult, DocumentSection, FullTextDocument, Paper, PaperIntegrityResult
from mcp_server import server


class FakeSearchClient:
    async def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        return [Paper(id="p1", title=f"Paper for {query}", abstract="Abstract.", source="fake")]


def test_mcp_search_tool_returns_serialized_papers(monkeypatch) -> None:
    monkeypatch.setattr(server, "search_client", FakeSearchClient())

    result = asyncio.run(server.search_papers("crystal graph networks", limit=1))

    assert result[0]["title"] == "Paper for crystal graph networks"


class FakeFullTextService:
    async def get_full_text(self, paper_id: str, doi=None, url=None) -> FullTextDocument:
        return FullTextDocument(
            document_id="doc-1",
            paper_id=paper_id,
            title="Full Paper",
            source_url="https://example.org/paper.xml",
            media_type="application/xml",
            sections=[DocumentSection(heading="Results", text="Detailed result text." * 10)],
        )

def test_mcp_fulltext_tool_returns_serialized_document(monkeypatch) -> None:
    monkeypatch.setattr(server, "fulltext_service", FakeFullTextService())

    document = asyncio.run(server.get_full_text("paper-1"))
    assert document is not None and document["document_id"] == "doc-1"


def test_mcp_material_screening_tool_returns_measured_candidates() -> None:
    """Make property screening callable across the same MCP boundary as search."""

    result = asyncio.run(
        server.screen_material_candidates(
            min_band_gap_ev=1.0,
            max_band_gap_ev=2.0,
            include_elements=["O"],
            top_k=2,
        )
    )

    assert len(result["candidates"]) == 2
    assert all(item["value_type"] == "measured" for item in result["candidates"])
    assert "MATERIAL DATA CONTEXT" in result["rag_context"]


class FakeIntegrityClient:
    async def check(self, doi: str) -> PaperIntegrityResult:
        return PaperIntegrityResult(doi=doi, status="retracted", updates=["Retraction: notice"])


def test_mcp_integrity_tool_returns_normalized_status(monkeypatch) -> None:
    monkeypatch.setattr(server, "integrity_client", FakeIntegrityClient())
    result = asyncio.run(server.check_research_integrity("10.1000/example"))
    assert result["status"] == "retracted"
    assert result["updates"] == ["Retraction: notice"]


class FakeCitationClient:
    async def expand(self, paper: Paper, limit: int = 5) -> CitationExpansionResult:
        discovered = Paper(
            id="cited-1", title="Cited work", abstract="Useful evidence.",
            source="semantic-scholar-reference", discovered_from=paper.id,
        )
        return CitationExpansionResult(seed_paper_id=paper.id, papers=[discovered], references_found=1)


def test_mcp_citation_graph_tool_returns_discovered_papers(monkeypatch) -> None:
    monkeypatch.setattr(server, "citation_client", FakeCitationClient())
    seed = Paper(id="seed-1", title="Seed", abstract="Seed abstract.", source="test")
    result = asyncio.run(server.expand_citation_graph(seed.model_dump(), limit=2))
    assert result["seed_paper_id"] == "seed-1"
    assert result["papers"][0]["discovered_from"] == "seed-1"
