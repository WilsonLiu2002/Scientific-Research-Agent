from __future__ import annotations

from typing import Any, TypedDict

from app.models import (
    EvidencePassage,
    FullTextDocument,
    Paper,
    PaperReading,
    ResearchGoal,
    ResearchCritique,
    ResearchIntent,
    ResearchSubtask,
)
from app.grounding import ClaimVerification


class ResearchState(TypedDict, total=False):
    """LangGraph state passed between the agent's workflow nodes."""

    question: str
    run_id: str
    trace_id: str
    access_profile: dict[str, Any]
    memory_context: str
    memory_stats: dict[str, int]
    checkpoint_thread_id: str
    run_metrics: dict[str, Any]
    run_trace: dict[str, Any]
    skill: str
    tool_plan: dict[str, Any]
    material_screening_result: dict[str, Any]
    search_queries: list[str]
    research_intent: ResearchIntent
    query_rationales: dict[str, str]
    retrieval_probes: list[str]
    papers: list[Paper]
    citation_expansion_attempted: bool
    citation_expansion_count: int
    integrity_checked_count: int
    retracted_paper_count: int
    indexed_paper_count: int
    local_corpus_paper_count: int
    retrieved_papers: list[Paper]
    retrieved_passages: list[EvidencePassage]
    answer: str | None
    citation_warnings: list[str]
    grounding_verdicts: list[ClaimVerification]
    grounding_warnings: list[str]
    search_iteration: int
    searched_queries: list[str]
    evidence_sufficient: bool
    evidence_coverage: float
    missing_concepts: list[str]
    evidence_notes: list[str]
    critique: ResearchCritique
    critique_history: list[ResearchCritique]
    deep_read_requested: bool
    fulltext_attempted: bool
    fulltext_documents: list[FullTextDocument]
    paper_readings: list[PaperReading]
    human_review_action: str
    human_review_history: list[dict[str, Any]]
    reviewer_instructions: list[str]
    selected_fulltext_paper_ids: list[str]
    synthesis_context: str
    context_stats: dict[str, Any]
    goal: ResearchGoal
    subtasks: list[ResearchSubtask]
    cancelled: bool
    errors: list[str]
