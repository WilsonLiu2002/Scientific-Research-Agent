from app.grounding import GroundingVerifier, repair_unsupported_claims
from app.models import EvidencePassage, Paper


def make_passage() -> EvidencePassage:
    """Create one stable cited passage for grounding tests."""

    return EvidencePassage(
        id="p1",
        paper=Paper(id="paper-1", title="Crystal Model", source="fixture"),
        text="The model uses message passing over crystal graphs with three interaction layers.",
    )


def test_grounding_verifier_distinguishes_supported_and_unsupported_claims(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = (
        "Main Approaches\n"
        "- The model uses message passing over crystal graphs with three interaction layers. [P1]\n"
        "- The model achieved 99 percent accuracy in clinical diagnosis. [P1]\n\n"
        "References\n[P1] Crystal Model"
    )

    verdicts = GroundingVerifier().verify(answer, [make_passage()])

    assert [item.verdict for item in verdicts] == ["entailed", "unsupported"]


def test_grounding_repair_removes_only_unsupported_claim_line(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = (
        "Main Approaches\n"
        "- The model uses message passing over crystal graphs with three interaction layers. [P1]\n"
        "- The model cured every disease. [P1]"
    )
    verdicts = GroundingVerifier().verify(answer, [make_passage()])

    repaired, warnings = repair_unsupported_claims(answer, verdicts)

    assert "three interaction layers" in repaired
    assert "cured every disease" not in repaired
    assert len(warnings) == 1
