from app.llm import build_rag_context, validate_citations
from app.models import EvidencePassage, Paper


def make_passage() -> EvidencePassage:
    """Create one realistic evidence passage for citation tests."""

    return EvidencePassage(
        id="W1::p1",
        paper=Paper(
            id="W1",
            title="Crystal Graph Networks",
            abstract="Graph networks predict material properties.",
            authors=["A. Researcher"],
            year=2024,
            doi="https://doi.org/10.1000/test",
            venue="Materials Journal",
            url="https://example.org/paper",
            source="test",
        ),
        text="Graph networks predict material properties.",
    )


def test_context_contains_traceable_metadata_and_passage_label() -> None:
    context = build_rag_context([make_passage()])

    assert "[P1]" in context
    assert "10.1000/test" in context
    assert "Materials Journal" in context
    assert "Graph networks predict material properties." in context


def test_validator_removes_uncited_and_unknown_claims() -> None:
    answer = (
        "Overview\n"
        "Supported claim. [P1]\n"
        "Uncited claim.\n"
        "Invented evidence. [P9]\n\n"
        "References\n"
        "[P1] A. Researcher (2024). Crystal Graph Networks"
    )

    cleaned, warnings = validate_citations(answer, [make_passage()])

    assert "Supported claim" in cleaned
    assert "Uncited claim" not in cleaned
    assert "Invented evidence" not in cleaned
    assert len(warnings) == 2
