from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

# Regression evaluation is deterministic and must never consume credentials from `.env`.
os.environ["SCIENTIFIC_AGENT_OFFLINE"] = "1"

from pydantic import BaseModel, Field, model_validator

from app.graph import build_research_graph
from app.models import DocumentSection, EvidencePassage, FullTextDocument, Paper
from app.rag import create_vector_store, lexical_overlap


DEFAULT_CASES_PATH = Path(__file__).with_name("cases.json")
SCIENTIFIC_HEADINGS = {
    "Overview", "Main Approaches", "Applications / Predicted Properties", "Key Observations", "Limitations",
}
META_CLAIM_MARKERS = (
    "the evidence below", "see the cited evidence", "only abstracts were searched",
    "selected open-access full-text",
)
DISTRACTOR_PAPERS = [
    Paper(id="distractor-scheduling", title="Optimization of Laboratory Scheduling", abstract="A scheduling algorithm allocates laboratory rooms and staff across weekly time slots.", source="fixture-distractor"),
    Paper(id="distractor-astronomy", title="Galaxy Morphology in Deep Sky Surveys", abstract="Astronomical image processing classifies spiral and elliptical galaxy morphology.", source="fixture-distractor"),
    Paper(id="distractor-language", title="Neural Language Translation", abstract="Transformer language models translate multilingual text using attention mechanisms.", source="fixture-distractor"),
    Paper(id="distractor-finance", title="Financial Time Series Forecasting", abstract="Regression models forecast market volatility from historical financial observations.", source="fixture-distractor"),
]


class EvaluationCase(BaseModel):
    """One human-labeled research scenario with recorded provider responses."""

    id: str
    question: str
    expect_deep_read: bool
    expected_paper_ids: list[str] = Field(min_length=1)
    expected_answer_terms: list[str] = Field(min_length=1)
    expected_fulltext_terms: list[str] = Field(default_factory=list)
    papers: list[Paper]
    fulltext_claims: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_labels(self) -> "EvaluationCase":
        """Reject fixtures whose relevance or deep-reading labels cannot be exercised."""

        paper_ids = [paper.id for paper in self.papers]
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("paper IDs must be unique within an evaluation case")
        missing = set(self.expected_paper_ids) - set(paper_ids)
        if missing:
            raise ValueError(f"expected papers are absent from fixtures: {sorted(missing)}")
        if self.expect_deep_read and not self.fulltext_claims:
            raise ValueError("deep-reading cases require recorded full-text claims")
        if self.expect_deep_read and not self.expected_fulltext_terms:
            raise ValueError("deep-reading cases require expected full-text terms")
        return self


class CaseResult(BaseModel):
    """Retrieval, grounding, content, routing, and reliability metrics for one case."""

    id: str
    retrieval_recall_at_k: float
    retrieval_precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    concept_coverage: float
    expected_content_coverage: float
    citation_coverage: float
    citation_correctness: float
    citation_labels_valid: bool
    deep_evidence_coverage: float
    route_correct: bool
    error_free: bool
    retrieved_paper_ids: list[str]
    missing_answer_terms: list[str]
    missing_fulltext_terms: list[str]
    context_stats: dict[str, Any]
    errors: list[str]

    @property
    def retrieval_quality(self) -> float:
        """Summarize ranking quality without allowing recall alone to hide poor ordering."""

        return (self.retrieval_recall_at_k + self.retrieval_precision_at_k + self.reciprocal_rank + self.ndcg_at_k) / 4

    @property
    def citation_quality(self) -> float:
        """Combine claim citation coverage, passage support, and label integrity."""

        return (self.citation_coverage + self.citation_correctness + float(self.citation_labels_valid)) / 3

    @property
    def score(self) -> float:
        """Compute a balanced score across independently useful quality dimensions."""

        return (
            self.retrieval_quality + self.expected_content_coverage + self.citation_quality
            + self.deep_evidence_coverage + float(self.route_correct) + float(self.error_free)
        ) / 6


class RecordedResearchClient:
    """Offline MCP-compatible client backed by versioned responses plus distractors."""

    def __init__(self, case: EvaluationCase) -> None:
        self.case = case

    def search_papers(self, query: str, limit: int = 10) -> list[Paper]:
        """Return recorded candidates, including irrelevant papers needed to test ranking."""

        return [*self.case.papers, *DISTRACTOR_PAPERS][:limit]

    def get_full_text(self, paper: Paper) -> FullTextDocument | None:
        """Construct a recorded open-access document when the fixture includes claims."""

        claims = self.case.fulltext_claims.get(paper.id)
        if not claims:
            return None
        document = FullTextDocument(
            document_id=f"eval-{self.case.id}-{paper.id}", paper_id=paper.id, title=paper.title,
            source_url=f"https://fixtures.invalid/{paper.id}.xml", media_type="application/xml",
            license="cc-by", sections=[DocumentSection(heading="Methods and Results", text=" ".join(claims))],
        )
        return document


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvaluationCase]:
    """Load and validate the versioned offline evaluation dataset."""

    cases = [EvaluationCase.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("evaluation case IDs must be unique")
    return cases


def ranking_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    """Calculate recall, precision, reciprocal rank, and binary nDCG at the label-set size."""

    k = len(relevant_ids)
    hits = [paper_id in relevant_ids for paper_id in ranked_ids[:k]]
    first_rank = next((index for index, paper_id in enumerate(ranked_ids, start=1) if paper_id in relevant_ids), None)
    dcg = sum(float(hit) / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(k))
    return {
        "recall": sum(hits) / len(relevant_ids),
        "precision": sum(hits) / max(k, 1),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def term_coverage(text: str, expected_terms: list[str]) -> tuple[float, list[str]]:
    """Measure human-labeled content coverage with case-insensitive phrase matching."""

    normalized = " ".join(text.lower().split())
    missing = [term for term in expected_terms if " ".join(term.lower().split()) not in normalized]
    coverage = (len(expected_terms) - len(missing)) / len(expected_terms) if expected_terms else 1.0
    return coverage, missing


def scientific_claim_lines(answer: str) -> list[str]:
    """Extract scientific claim lines while excluding process and limitation boilerplate."""

    body = answer.split("References", 1)[0]
    claims: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped in SCIENTIFIC_HEADINGS:
            continue
        if any(marker in stripped.lower() for marker in META_CLAIM_MARKERS):
            continue
        claims.append(stripped)
    return claims


def citation_metrics(answer: str, passages: list[EvidencePassage], warnings: list[str]) -> dict[str, Any]:
    """Verify claim citation coverage and lexical support against cited passages."""

    claims = scientific_claim_lines(answer)
    valid_labels = {f"P{index}" for index in range(1, len(passages) + 1)}
    cited_claims = 0
    supported_claims = 0
    labels_valid = not warnings
    for claim in claims:
        labels = re.findall(r"\[(P\d+)\]", claim)
        cited_claims += bool(labels)
        if any(label not in valid_labels for label in labels):
            labels_valid = False
        claim_text = re.sub(r"\[P\d+\]", "", claim)
        cited_passages = [passages[int(label[1:]) - 1] for label in labels if label in valid_labels]
        supported = any(
            passage.text.lower() in claim_text.lower() or lexical_overlap(claim_text, passage.text) >= 0.2
            for passage in cited_passages
        )
        supported_claims += supported
    denominator = len(claims) or 1
    return {
        "coverage": cited_claims / denominator,
        "correctness": supported_claims / denominator,
        "labels_valid": labels_valid,
    }


def evaluate_case(case: EvaluationCase) -> CaseResult:
    """Run one case through the real graph with only external providers replaced."""

    graph = build_research_graph(
        search_client=RecordedResearchClient(case), vector_store=create_vector_store(force_memory=True),
        max_search_iterations=1, min_relevant_papers=1, min_evidence_passages=1,
        min_concept_coverage=0.0, max_fulltext_papers=3,
    )
    previous_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        state = graph.invoke({"question": case.question, "papers": [], "retrieved_passages": [], "errors": []})
    finally:
        if previous_key is not None:
            os.environ["OPENAI_API_KEY"] = previous_key

    passages = state.get("retrieved_passages", [])
    ranked_ids = list(dict.fromkeys(paper.id for paper in state.get("retrieved_papers", [])))
    ranking = ranking_metrics(ranked_ids, set(case.expected_paper_ids))
    answer = state.get("answer") or ""
    content_coverage, missing_answer_terms = term_coverage(answer, case.expected_answer_terms)
    fulltext = " ".join(passage.text for passage in passages if passage.section or passage.page)
    deep_coverage, missing_fulltext_terms = term_coverage(fulltext, case.expected_fulltext_terms)
    citations = citation_metrics(answer, passages, state.get("citation_warnings", []))
    route_correct = state.get("deep_read_requested", False) == case.expect_deep_read
    if case.expect_deep_read:
        route_correct = route_correct and state.get("fulltext_attempted", False)

    return CaseResult(
        id=case.id, retrieval_recall_at_k=ranking["recall"], retrieval_precision_at_k=ranking["precision"],
        reciprocal_rank=ranking["reciprocal_rank"], ndcg_at_k=ranking["ndcg"],
        concept_coverage=float(state.get("evidence_coverage", 0.0)),
        expected_content_coverage=content_coverage, citation_coverage=citations["coverage"],
        citation_correctness=citations["correctness"], citation_labels_valid=citations["labels_valid"],
        deep_evidence_coverage=deep_coverage, route_correct=route_correct,
        error_free=not state.get("errors"), retrieved_paper_ids=ranked_ids,
        missing_answer_terms=missing_answer_terms, missing_fulltext_terms=missing_fulltext_terms,
        context_stats=state.get("context_stats", {}),
        errors=state.get("errors", []),
    )


def run_evaluation(cases: list[EvaluationCase]) -> dict[str, Any]:
    """Evaluate all cases and return a detailed, JSON-serializable benchmark report."""

    results = [evaluate_case(case) for case in cases]
    count = len(results) or 1

    def average(values: list[float]) -> float:
        return sum(values) / count

    aggregate = {
        "case_count": len(results),
        "overall_score": average([result.score for result in results]),
        "retrieval_quality": average([result.retrieval_quality for result in results]),
        "retrieval_recall_at_k": average([result.retrieval_recall_at_k for result in results]),
        "retrieval_precision_at_k": average([result.retrieval_precision_at_k for result in results]),
        "mean_reciprocal_rank": average([result.reciprocal_rank for result in results]),
        "ndcg_at_k": average([result.ndcg_at_k for result in results]),
        "concept_coverage": average([result.concept_coverage for result in results]),
        "expected_content_coverage": average([result.expected_content_coverage for result in results]),
        "citation_quality": average([result.citation_quality for result in results]),
        "citation_coverage": average([result.citation_coverage for result in results]),
        "citation_correctness": average([result.citation_correctness for result in results]),
        "deep_evidence_coverage": average([result.deep_evidence_coverage for result in results]),
        "route_accuracy": average([float(result.route_correct) for result in results]),
        "error_free_rate": average([float(result.error_free) for result in results]),
        "average_context_tokens": average(
            [float(result.context_stats.get("estimated_tokens", 0)) for result in results]
        ),
        "average_context_papers": average(
            [float(result.context_stats.get("distinct_papers", 0)) for result in results]
        ),
        "total_duplicate_passages_removed": sum(
            int(result.context_stats.get("duplicate_passages_removed", 0)) for result in results
        ),
        "total_budget_passages_dropped": sum(
            int(result.context_stats.get("budget_passages_dropped", 0)) for result in results
        ),
    }
    fixture_payload = json.dumps(
        [case.model_dump(mode="json") for case in cases], sort_keys=True, separators=(",", ":")
    )
    return {
        "schema_version": 2,
        "benchmark_fingerprint": hashlib.sha256(fixture_payload.encode("utf-8")).hexdigest(),
        "scoring": {
            "retrieval_cutoff": "number of relevant labels for each case",
            "citation_support_threshold": 0.2,
            "overall_dimensions": [
                "retrieval_quality", "expected_content_coverage", "citation_quality",
                "deep_evidence_coverage", "route_accuracy", "error_free_rate",
            ],
        },
        "aggregate": aggregate,
        "cases": [result.model_dump() | {"retrieval_quality": result.retrieval_quality, "citation_quality": result.citation_quality, "score": result.score} for result in results],
    }


def print_report(report: dict[str, Any]) -> None:
    """Print high-signal aggregate metrics and flag low-scoring cases."""

    aggregate = report["aggregate"]
    print(f"Evaluation cases: {aggregate['case_count']}")
    print(f"Overall score: {aggregate['overall_score']:.1%}")
    print(f"Retrieval quality: {aggregate['retrieval_quality']:.1%}")
    print(f"  Recall@K: {aggregate['retrieval_recall_at_k']:.1%}")
    print(f"  Precision@K: {aggregate['retrieval_precision_at_k']:.1%}")
    print(f"  MRR: {aggregate['mean_reciprocal_rank']:.1%}")
    print(f"  nDCG@K: {aggregate['ndcg_at_k']:.1%}")
    print(f"Expected content: {aggregate['expected_content_coverage']:.1%}")
    print(f"Citation quality: {aggregate['citation_quality']:.1%}")
    print(f"  Claim coverage: {aggregate['citation_coverage']:.1%}")
    print(f"  Passage support: {aggregate['citation_correctness']:.1%}")
    print(f"Deep evidence: {aggregate['deep_evidence_coverage']:.1%}")
    print(f"Route accuracy: {aggregate['route_accuracy']:.1%}")
    print(f"Error-free runs: {aggregate['error_free_rate']:.1%}")
    print(
        f"Context: {aggregate['average_context_tokens']:.0f} average tokens across "
        f"{aggregate['average_context_papers']:.1f} papers"
    )
    print(
        f"Context filtering: {aggregate['total_duplicate_passages_removed']} duplicates, "
        f"{aggregate['total_budget_passages_dropped']} budget drops"
    )
    for result in report["cases"]:
        marker = "PASS" if result["score"] >= 0.8 else "REVIEW"
        print(f"- {marker} {result['id']}: {result['score']:.1%}")


def main() -> None:
    """CLI entry point with overall and critical-dimension quality gates."""

    parser = argparse.ArgumentParser(description="Evaluate the research agent offline.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-under", type=float, default=0.0)
    parser.add_argument("--min-retrieval", type=float, default=0.0)
    parser.add_argument("--min-citation", type=float, default=0.0)
    parser.add_argument("--min-route", type=float, default=0.0)
    args = parser.parse_args()

    report = run_evaluation(load_cases(args.cases))
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    aggregate = report["aggregate"]
    failed = (
        aggregate["overall_score"] < args.fail_under
        or aggregate["retrieval_quality"] < args.min_retrieval
        or aggregate["citation_quality"] < args.min_citation
        or aggregate["route_accuracy"] < args.min_route
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
