from __future__ import annotations

from mcp_server.fulltext import FullTextService
from mcp_server.openalex import OpenAlexSearchClient
from app.materials_ml import CandidateScreenRequest, MaterialsScreeningTool
from app.observability import log_event, span

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except (ImportError, ModuleNotFoundError):
        FastMCP = None  # type: ignore[assignment]


search_client = OpenAlexSearchClient()
fulltext_service = FullTextService()
materials_tool = MaterialsScreeningTool()
mcp = FastMCP("academic-search", log_level="ERROR") if FastMCP else None


async def search_papers(
    query: str,
    limit: int = 10,
    correlation_run_id: str | None = None,
    correlation_trace_id: str | None = None,
    correlation_parent_span_id: str | None = None,
) -> list[dict]:
    """MCP tool implementation that searches OpenAlex and returns serialized Paper objects."""

    with span("mcp.server.search_papers", kind="tool", export=False, fields={"tool": "search_papers", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id}):
        try:
            papers = await search_client.search_papers(query=query, limit=limit)
        except Exception as exc:
            log_event("mcp.provider_failed", tool="search_papers", failure_type=type(exc).__name__)
            return []
        log_event("mcp.provider_result", tool="search_papers", result_count=len(papers), provider_status="ok")
        return [paper.model_dump() for paper in papers]


async def get_full_text(
    paper_id: str,
    doi: str | None = None,
    url: str | None = None,
    correlation_run_id: str | None = None,
    correlation_trace_id: str | None = None,
    correlation_parent_span_id: str | None = None,
) -> dict | None:
    """MCP tool that acquires and extracts legally accessible open-access article text."""

    with span("mcp.server.get_full_text", kind="tool", export=False, fields={"tool": "get_full_text", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id}):
        try:
            document = await fulltext_service.get_full_text(paper_id=paper_id, doi=doi, url=url)
        except Exception as exc:
            log_event("mcp.provider_failed", tool="get_full_text", failure_type=type(exc).__name__)
            return None
        log_event("mcp.provider_result", tool="get_full_text", result_count=int(document is not None), provider_status="ok")
        return document.model_dump() if document else None


async def screen_material_candidates(
    min_band_gap_ev: float = 1.0,
    max_band_gap_ev: float = 2.5,
    include_elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    top_k: int = 20,
    correlation_run_id: str | None = None,
    correlation_trace_id: str | None = None,
    correlation_parent_span_id: str | None = None,
) -> dict:
    """MCP tool that ranks measured material candidates under explicit constraints."""

    with span("mcp.server.screen_material_candidates", kind="tool", export=False, fields={"tool": "screen_material_candidates", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id}):
        request = CandidateScreenRequest(
            min_band_gap_ev=min_band_gap_ev,
            max_band_gap_ev=max_band_gap_ev,
            include_elements=include_elements or [],
            exclude_elements=exclude_elements or [],
            top_k=top_k,
        )
        result = materials_tool.screen_candidates(request)
        log_event("mcp.provider_result", tool="screen_material_candidates", result_count=len(result.candidates), provider_status="ok")
        return result.model_dump()


if mcp is not None:
    mcp.tool()(search_papers)
    mcp.tool()(get_full_text)
    mcp.tool()(screen_material_candidates)


def main() -> None:
    """Run the MCP server over stdio."""

    if mcp is None:
        raise RuntimeError("The `mcp` package is not installed. Install project dependencies first.")
    mcp.run()


if __name__ == "__main__":
    main()
