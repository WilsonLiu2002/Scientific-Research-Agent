from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Paper(BaseModel):
    """Structured representation of one academic paper used across MCP, RAG, and synthesis."""

    id: str
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    url: str | None = None
    citation_count: int | None = None
    source: str
    discovered_from: str | None = None
    integrity_status: Literal[
        "unchecked", "clear", "corrected", "expression_of_concern", "retracted", "unknown"
    ] = "unchecked"
    integrity_updates: list[str] = Field(default_factory=list)

    @field_validator("id", "title", "source")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Reject empty identifiers and titles because they are needed for citation and deduplication."""

        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    def document_text(self) -> str:
        """Convert a paper into a complete text document for display or export."""

        abstract = self.abstract or "No abstract available."
        return f"Title: {self.title}\n\nAbstract:\n{abstract}"

    def reference(self) -> str:
        """Render a compact reference line for the final cited answer."""

        authors = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors += " et al."
        if not authors:
            authors = "Unknown authors"
        year = self.year if self.year is not None else "n.d."
        venue = f". {self.venue}" if self.venue else ""
        identifier = self.doi or self.url
        link = f". {identifier}" if identifier else ""
        return f"{authors} ({year}). {self.title}{venue}{link}"

    def dedupe_key(self) -> str:
        """Return a stable key used to remove duplicate papers from multi-query search results."""

        if self.doi:
            return self.doi.lower().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        if self.id:
            return self.id.lower()
        return normalize_title(self.title)


class PaperIntegrityResult(BaseModel):
    """Normalized scholarly-record status returned by the Crossref MCP tool."""

    doi: str
    status: Literal["clear", "corrected", "expression_of_concern", "retracted", "unknown"]
    updates: list[str] = Field(default_factory=list)
    checked_source: str = "crossref"


class CitationExpansionResult(BaseModel):
    """Bounded papers discovered around one seed through a citation graph."""

    seed_paper_id: str
    papers: list[Paper] = Field(default_factory=list)
    references_found: int = 0
    citations_found: int = 0


class ToolChoice(BaseModel):
    """One capability-aware tool decision recorded before execution."""

    tool_id: Literal[
        "local_rag", "search_papers", "check_research_integrity",
        "expand_citation_graph", "get_full_text", "screen_material_candidates",
    ]
    selected: bool
    reason: str
    stage: Literal["discovery", "retrieval", "deep_read"]
    required_capability: str


class ToolPlan(BaseModel):
    """Auditable per-run tool portfolio produced from intent and authorization."""

    choices: list[ToolChoice]

    def selected_ids(self) -> set[str]:
        """Return selected tool identifiers for graph routing."""

        return {choice.tool_id for choice in self.choices if choice.selected}

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Paper":
        """Build a Paper from a plain mapping returned by an MCP tool or test double."""

        return cls.model_validate(data)


class SearchPlan(BaseModel):
    """Small structured output model for planner-generated literature search queries."""

    queries: list[str] = Field(min_length=1, max_length=5)


class ResearchIntent(BaseModel):
    """Planner interpretation of what evidence the research question actually requires."""

    purpose: Literal[
        "overview", "methods", "mechanism", "comparison", "quantitative",
        "datasets", "limitations", "reproducibility",
    ]
    objective: str
    entities: list[str] = Field(default_factory=list)
    properties: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class PlannedSearchQuery(BaseModel):
    """One non-redundant academic query and the evidence angle it targets."""

    query: str
    angle: Literal["core", "methods", "outcomes", "comparison", "limitations", "synonyms"]
    rationale: str


class StrategicSearchPlan(BaseModel):
    """Intent-first search strategy produced before any external literature call."""

    intent: ResearchIntent
    queries: list[PlannedSearchQuery] = Field(min_length=3, max_length=5)


class ResearchGoal(BaseModel):
    """Measurable objective and lifecycle status for one research run."""

    id: str = "primary-research-goal"
    objective: str
    success_criteria: list[str]
    status: Literal[
        "planned", "running", "evidence_ready", "completed", "budget_exhausted", "cancelled"
    ] = "planned"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class ResearchSubtask(BaseModel):
    """Auditable unit of search or deep-reading work within the research goal."""

    id: str
    kind: Literal["literature_search", "deep_read"]
    objective: str
    iteration: int = Field(ge=1)
    query: str | None = None
    paper_id: str | None = None
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    result_count: int = Field(default=0, ge=0)
    result_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class ResearchCritique(BaseModel):
    """Structured reflection over retrieved evidence and the next bounded action."""

    iteration: int = Field(ge=1)
    verdict: Literal["sufficient", "retry", "deep_read", "stop"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    missing_topics: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_queries: list[str] = Field(default_factory=list)
    recommended_paper_ids: list[str] = Field(default_factory=list)
    retry_budget_remaining: int = Field(ge=0)
    signature: str


class EvidencePassage(BaseModel):
    """A citable excerpt retrieved from a paper rather than an entire abstract."""

    id: str
    paper: Paper
    text: str
    score: float = 0.0
    section: str | None = None
    page: int | None = None
    source_url: str | None = None

    @field_validator("id", "text")
    @classmethod
    def non_empty_passage_text(cls, value: str) -> str:
        """Ensure every evidence item can be referenced and inspected."""

        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value


class DocumentSection(BaseModel):
    """A section extracted from an open-access article with optional page provenance."""

    heading: str
    text: str
    page: int | None = None


class FullTextDocument(BaseModel):
    """Structured, cached representation of a legally accessible full-text article."""

    document_id: str
    paper_id: str
    title: str
    source_url: str
    media_type: str
    license: str | None = None
    sections: list[DocumentSection]


class EvidenceClaim(BaseModel):
    """Extractive full-text evidence selected for one research question."""

    text: str
    section: str
    page: int | None = None
    source_url: str


class PaperReading(BaseModel):
    """Question-focused reading result whose entries remain traceable to article text."""

    document_id: str
    research_question: str
    methods: list[EvidenceClaim] = Field(default_factory=list)
    datasets: list[EvidenceClaim] = Field(default_factory=list)
    results: list[EvidenceClaim] = Field(default_factory=list)
    limitations: list[EvidenceClaim] = Field(default_factory=list)
    conclusions: list[EvidenceClaim] = Field(default_factory=list)


def normalize_title(title: str) -> str:
    """Normalize a title enough for simple duplicate detection without fuzzy matching."""

    return " ".join(title.lower().split())


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Remove obvious duplicate papers while preserving the first occurrence order."""

    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        key = paper.dedupe_key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique
