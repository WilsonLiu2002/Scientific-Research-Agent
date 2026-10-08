from __future__ import annotations

from mcp_server.fulltext import FullTextService
from mcp_server.openalex import OpenAlexSearchClient
from mcp_server.crossref import CrossrefIntegrityClient
from mcp_server.citations import SemanticScholarCitationClient
from app.models import Paper
from app.materials_ml import CandidateScreenRequest, MaterialsScreeningTool
from app.auth import default_access_grant, resolve_access_token
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
integrity_client = CrossrefIntegrityClient()
citation_client = SemanticScholarCitationClient()
mcp = FastMCP("academic-search", log_level="ERROR") if FastMCP else None


async def search_papers(
    query: str,
    limit: int = 10,
    correlation_run_id: str | None = None,
    correlation_trace_id: str | None = None,
    correlation_parent_span_id: str | None = None,
    simulated_auth_token: str | None = None,
) -> list[dict]:
    """MCP tool implementation that searches OpenAlex and returns serialized Paper objects."""

    grant = resolve_access_token(simulated_auth_token) if simulated_auth_token else default_access_grant()
    if not grant.allows("literature:search"):
        raise PermissionError("Token does not permit literature search")
    with span("mcp.server.search_papers", kind="tool", export=False, fields={"tool": "search_papers", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id, "access_level": grant.level}):
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
    simulated_auth_token: str | None = None,
) -> dict | None:
    """MCP tool that acquires and extracts legally accessible open-access article text."""

    grant = resolve_access_token(simulated_auth_token) if simulated_auth_token else default_access_grant()
    if not grant.allows("fulltext:read"):
        raise PermissionError("Token does not permit full-text access")
    with span("mcp.server.get_full_text", kind="tool", export=False, fields={"tool": "get_full_text", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id, "access_level": grant.level}):
        try:
            document = await fulltext_service.get_full_text(paper_id=paper_id, doi=doi, url=url)
        except Exception as exc:
            log_event("mcp.provider_failed", tool="get_full_text", failure_type=type(exc).__name__)
            return None
        log_event("mcp.provider_result", tool="get_full_text", result_count=int(document is not None), provider_status="ok")
        return document.model_dump() if document else None


async def check_research_integrity(
    doi: str,
    correlation_run_id: str | None = None,
    correlation_trace_id: str | None = None,
    correlation_parent_span_id: str | None = None,
    simulated_auth_token: str | None = None,
) -> dict:
    """MCP tool that checks Crossref for retractions and other publication updates."""

    grant = resolve_access_token(simulated_auth_token) if simulated_auth_token else default_access_grant()
    if not grant.allows("literature:search"):
        raise PermissionError("Token does not permit research-integrity lookup")
    with span("mcp.server.check_research_integrity", kind="tool", export=False, fields={"tool": "check_research_integrity", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id, "access_level": grant.level}):
        try:
            result = await integrity_client.check(doi)
        except Exception as exc:
            log_event("mcp.provider_failed", tool="check_research_integrity", failure_type=type(exc).__name__)
            raise
        log_event("mcp.provider_result", tool="check_research_integrity", result_count=1, provider_status="ok", integrity_status=result.status)
        return result.model_dump()


async def expand_citation_graph(
    paper: dict,
    limit: int = 5,
    correlation_run_id: str | None = None,
    correlation_trace_id: str | None = None,
    correlation_parent_span_id: str | None = None,
    simulated_auth_token: str | None = None,
) -> dict:
    """MCP tool that discovers bounded references and citations around one seed paper."""

    grant = resolve_access_token(simulated_auth_token) if simulated_auth_token else default_access_grant()
    if not grant.allows("literature:search"):
        raise PermissionError("Token does not permit citation-graph expansion")
    seed = Paper.model_validate(paper)
    with span("mcp.server.expand_citation_graph", kind="tool", export=False, fields={"tool": "expand_citation_graph", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id, "access_level": grant.level}):
        try:
            result = await citation_client.expand(seed, limit=min(max(limit, 1), 20))
        except Exception as exc:
            log_event("mcp.provider_failed", tool="expand_citation_graph", failure_type=type(exc).__name__)
            raise
        log_event("mcp.provider_result", tool="expand_citation_graph", result_count=len(result.papers), provider_status="ok")
        return result.model_dump()


async def screen_material_candidates(
    min_band_gap_ev: float = 1.0,
    max_band_gap_ev: float = 2.5,
    include_elements: list[str] | None = None,
    exclude_elements: list[str] | None = None,
    top_k: int = 20,
    correlation_run_id: str | None = None,
    correlation_trace_id: str | None = None,
    correlation_parent_span_id: str | None = None,
    simulated_auth_token: str | None = None,
) -> dict:
    """MCP tool that ranks measured material candidates under explicit constraints."""

    grant = resolve_access_token(simulated_auth_token) if simulated_auth_token else default_access_grant()
    if not grant.allows("materials:read"):
        raise PermissionError("Token does not permit materials database access")
    with span("mcp.server.screen_material_candidates", kind="tool", export=False, fields={"tool": "screen_material_candidates", "correlation_run_id": correlation_run_id, "correlation_trace_id": correlation_trace_id, "correlation_parent_span_id": correlation_parent_span_id, "access_level": grant.level}):
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
    mcp.tool()(check_research_integrity)
    mcp.tool()(expand_citation_graph)


def main() -> None:
    """Run the MCP server over stdio."""

    if mcp is None:
        raise RuntimeError("The `mcp` package is not installed. Install project dependencies first.")
    mcp.run()


if __name__ == "__main__":
    main()
