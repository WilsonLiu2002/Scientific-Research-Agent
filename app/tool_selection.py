from __future__ import annotations

import re
from dataclasses import dataclass

from app.materials_ml import CandidateScreenRequest
from app.models import ResearchIntent, ToolChoice, ToolPlan


@dataclass(frozen=True)
class ToolDefinition:
    """Static metadata for one tool the policy may expose to a research run."""

    tool_id: str
    stage: str
    capability: str


TOOL_REGISTRY = (
    ToolDefinition("local_rag", "retrieval", "rag:read"),
    ToolDefinition("search_papers", "discovery", "literature:search"),
    ToolDefinition("check_research_integrity", "discovery", "literature:search"),
    ToolDefinition("expand_citation_graph", "discovery", "literature:search"),
    ToolDefinition("get_full_text", "deep_read", "fulltext:read"),
    ToolDefinition("screen_material_candidates", "discovery", "materials:read"),
)


def select_tools(
    question: str,
    intent: ResearchIntent | None,
    capabilities: set[str],
    enable_full_text: bool = True,
) -> ToolPlan:
    """Choose a bounded tool portfolio from intent, wording, and resolved capabilities."""

    lower = question.lower()
    purpose = intent.purpose if intent else "overview"
    detail_heavy = purpose in {"methods", "quantitative", "datasets", "reproducibility"}
    detail_terms = {
        "architecture", "compare", "comparison", "dataset", "datasets", "detailed",
        "experimental", "exact", "implementation", "limitation", "limitations", "mechanism",
        "method", "methods", "numerical", "performance", "quantitative", "reproduce",
        "reproducibility", "setup", "table",
    }
    detail_heavy = detail_heavy or bool(set(re.findall(r"[a-z]+", lower)) & detail_terms)
    material_screen = (
        any(marker in lower for marker in ("candidate", "screen", "shortlist", "which material"))
        and any(marker in lower for marker in ("band gap", "bandgap", "material", "compound"))
    )
    choices: list[ToolChoice] = []
    for definition in TOOL_REGISTRY:
        allowed = definition.capability in capabilities
        selected = False
        reason = f"Unavailable: authorization lacks {definition.capability}."
        if definition.tool_id == "local_rag" and allowed:
            selected, reason = True, "Retrieve the durable local corpus for every scientific question."
        elif definition.tool_id == "search_papers" and allowed:
            selected, reason = True, "Expand beyond local evidence with current literature metadata."
        elif definition.tool_id == "check_research_integrity" and allowed:
            selected, reason = True, "Validate discovered DOI records before they enter RAG."
        elif definition.tool_id == "expand_citation_graph" and allowed:
            selected, reason = True, "Keep one bounded graph expansion available if retrieval is weak."
        elif definition.tool_id == "get_full_text" and allowed and enable_full_text and detail_heavy:
            selected, reason = True, f"The {purpose} intent requires details abstracts may omit."
        elif definition.tool_id == "get_full_text" and allowed:
            reason = "Not selected: the current intent can normally be answered from passage-level evidence."
        elif definition.tool_id == "screen_material_candidates" and allowed and material_screen:
            selected, reason = True, "The question requests property-constrained material candidates."
        elif definition.tool_id == "screen_material_candidates" and allowed:
            reason = "Not selected: no candidate-screening request was detected."
        choices.append(
            ToolChoice(
                tool_id=definition.tool_id,
                selected=selected,
                reason=reason,
                stage=definition.stage,
                required_capability=definition.capability,
            )
        )
    return ToolPlan(choices=choices)


def material_screen_request(question: str) -> CandidateScreenRequest:
    """Extract a conservative band-gap window from a screening question."""

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*eV",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return CandidateScreenRequest()
    lower, upper = sorted((float(match.group(1)), float(match.group(2))))
    return CandidateScreenRequest(min_band_gap_ev=lower, max_band_gap_ev=upper)


def tool_is_selected(tool_plan: ToolPlan | dict | None, tool_id: str) -> bool:
    """Keep graph routing concise and false for absent legacy state."""

    if isinstance(tool_plan, dict):
        tool_plan = ToolPlan.model_validate(tool_plan)
    return bool(tool_plan and tool_id in tool_plan.selected_ids())
