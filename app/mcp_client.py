from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import Any

from app.models import FullTextDocument, Paper
from app.materials_ml import CandidateScreenRequest, CandidateScreenResult
from app.observability import correlation_fields, log_event, metrics_increment, span


@dataclass
class AcademicMCPClient:
    """Synchronous facade used by LangGraph nodes to call the academic MCP search tool."""

    authorization_token: str | None = None

    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        """Call the MCP `search_papers` tool."""

        return asyncio.run(self.search_papers_async(query=query, limit=limit))

    async def search_papers_async(self, query: str, limit: int = 10) -> list[Paper]:
        """Execute literature search exclusively through the MCP boundary."""

        return await self._search_via_stdio(query=query, limit=limit)

    async def _search_via_stdio(self, query: str, limit: int) -> list[Paper]:
        """Spawn the local MCP server over stdio and invoke its one search tool."""

        result = await self._call_tool("search_papers", {"query": query, "limit": limit})
        raw_items = self._parse_tool_payload(result, default=[])
        if isinstance(raw_items, dict) and "papers" in raw_items:
            raw_items = raw_items["papers"]
        return [Paper.from_mapping(item) for item in raw_items]

    def get_full_text(self, paper: Paper) -> FullTextDocument | None:
        """Acquire one selected article through MCP."""

        return asyncio.run(self.get_full_text_async(paper))

    async def get_full_text_async(self, paper: Paper) -> FullTextDocument | None:
        """Return structured open-access text, or None when no legal copy is available."""

        arguments = {"paper_id": paper.id, "doi": paper.doi, "url": paper.url}
        result = await self._call_tool("get_full_text", arguments)
        payload = self._parse_tool_payload(result, default=None)
        return FullTextDocument.model_validate(payload) if payload else None

    def screen_material_candidates(
        self, request: CandidateScreenRequest
    ) -> CandidateScreenResult:
        """Run property-constrained candidate screening through MCP."""

        return asyncio.run(self.screen_material_candidates_async(request))

    async def screen_material_candidates_async(
        self, request: CandidateScreenRequest
    ) -> CandidateScreenResult:
        """Return ranked measured candidates and RAG-ready material context."""

        result = await self._call_tool("screen_material_candidates", request.model_dump())
        payload = self._parse_tool_payload(result, default={})
        return CandidateScreenResult.model_validate(payload)

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Open one MCP stdio session and invoke a named academic tool."""

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        fields = correlation_fields()
        propagated = {
            **arguments,
            "simulated_auth_token": self.authorization_token,
            "correlation_run_id": fields["run_id"],
            "correlation_trace_id": fields["trace_id"],
            "correlation_parent_span_id": fields["parent_span_id"],
        }
        with span(f"mcp.client.{name}", kind="tool", fields={"tool": name}):
            metrics_increment("tool_calls")
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mcp_server.server"],
            )
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, propagated)
            log_event(
                "mcp.result",
                tool=name,
                retry_count=0,
                provider_status="error" if getattr(result, "isError", False) else "ok",
            )
            return result

    def _parse_tool_payload(self, result: Any, default: Any) -> Any:
        """Normalize structured or text MCP results despite SDK version differences."""

        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError("MCP tool returned an error")
        structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
        if structured:
            if isinstance(structured, dict) and "result" in structured:
                return structured["result"]
            return structured
        content = getattr(result, "content", [])
        if not content:
            return default
        text = getattr(content[0], "text", json.dumps(default))
        return json.loads(text)
